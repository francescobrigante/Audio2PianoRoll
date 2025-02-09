import os
import librosa
import torch
import torch.utils.data as data
import jams
import numpy as np
from sklearn.decomposition import NMF


from midiNjams2pianoroll import jams_to_pianoroll

AUDIO_DIRS = [
    "/Users/francesco/Desktop/audio2midi/guitarset/audio_hex-pickup_debleeded/",
    "/Users/francesco/Desktop/audio2midi/guitarset/audio_hex-pickup_original/",
    "/Users/francesco/Desktop/audio2midi/guitarset/audio_mono-mic/",
    "/Users/francesco/Desktop/audio2midi/guitarset/audio_mono-pickup_mix/"
]
JAMS_DIR = "/Users/francesco/Desktop/audio2midi/guitarset/annotation/"

# Constants
SR = 22050
N_NOTES = 52                                    # 4 octaves (E2 to E6) + 4 notes or semitones (C2 to E2) = 4*12 + 4 = 52
#52 notes however on MIDI there are 53 classes from c2 to e6
BINS_PER_SEMITONE = 3
BINS_PER_OCTAVE = 36                            # 3*12 = 36
N_BINS = BINS_PER_SEMITONE * N_NOTES            # 3*52 = 156
FMIN = librosa.midi_to_hz(36)                   # MIDI 36 = C2


#windows and overlappings
WINDOW_SIZE = (156, 86)
WINDOW_FREQ, WINDOW_TIME = WINDOW_SIZE
OVERLAP_PERCENTAGE = 0.5
OVERLAP_FRAMES = int(WINDOW_TIME * OVERLAP_PERCENTAGE)




class GuitarSetDataset(data.Dataset):
    def __init__(self, audio_dirs=AUDIO_DIRS, jams_dir=JAMS_DIR, apply_nmf=False, apply_sections=True):
        # Ensure audio_dirs is a list
        if isinstance(audio_dirs, str):
            audio_dirs = [audio_dirs]
        self.audio_dirs = audio_dirs
        self.jams_dir = jams_dir
        self.apply_nmf = apply_nmf
        
        #return output as sections or whole piano roll
        self.apply_sections = apply_sections
        
        self.samples = self._get_file_pairs()

    def _get_file_pairs(self):
        '''Returns a list of (audio_path, jams_path) tuples for all audio directories.'''
        jams_files = [f for f in os.listdir(self.jams_dir) if f.endswith(".jams")]
        samples = []
        for jams_file in jams_files:
            prefix = jams_file.replace(".jams", "")
            for audio_dir in self.audio_dirs:
                if not os.path.isdir(audio_dir):
                    continue  # Skip invalid directories
                # Find all matching audio files in this directory
                matching_wavs = [f for f in os.listdir(audio_dir) 
                                 if f.startswith(prefix) and f.endswith(".wav")]
                for wav_file in matching_wavs:
                    audio_path = os.path.join(audio_dir, wav_file)
                    jams_path = os.path.join(self.jams_dir, jams_file)
                    samples.append((audio_path, jams_path))
        return samples
    

    
    def __len__(self):
        return len(self.samples)
    
    def nmf(CQT_norm, n_components=10):
        model = NMF(n_components=n_components, init='nndsvd', random_state=0)
        W = model.fit_transform(CQT_norm)  # Spectral bases
        H = model.components_  # Activations
        CQT_nmf = np.dot(W, H)  # Reconstruct the cleaned CQT
        return CQT_nmf

    def _load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=SR)
        CQT = librosa.cqt(audio, sr=SR, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS)
        CQT_dB = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
        CQT_norm = (CQT_dB - CQT_dB.min()) / (CQT_dB.max() - CQT_dB.min())
        if self.apply_nmf:
            CQT_norm = self.nmf(CQT_norm)
        return CQT_norm.astype(np.float32)

    def _load_jams(self, file_path, num_time_bins):
        jam = jams.load(file_path)
        piano_roll, time_axis, midi_pitches = jams_to_pianoroll(jam, num_time_bins)
        return piano_roll, time_axis, midi_pitches

    def __getitem__(self, idx):
        audio_path, jams_path = self.samples[idx]
        cqt_data = self._load_audio(audio_path)
        cqt_data = np.expand_dims(cqt_data, axis=0)
        num_time_bins = cqt_data.shape[2]
        piano_roll, time_axis, midi_pitches = self._load_jams(jams_path, num_time_bins)
        cqt_tensor = torch.tensor(cqt_data)
        piano_roll_tensor = torch.tensor(piano_roll)
        
        if self.apply_sections:
            cqt_sections = get_overlap_windows(cqt_tensor, WINDOW_TIME, OVERLAP_FRAMES)
            piano_roll_sections = get_pianoroll_windows(piano_roll_tensor, WINDOW_TIME, OVERLAP_FRAMES)
            return cqt_sections, piano_roll_sections, num_time_bins
        else:
            return cqt_tensor, piano_roll_tensor
    

#functions to get the windows
def get_overlap_windows(cqt, window_time, overlap_frames):
    channels, n_freq, n_time = cqt.shape
    step_size = window_time - overlap_frames
    sections = []
    
    for start_time in range(0, n_time, step_size):
        end_time = min(start_time + window_time, n_time)
        section = cqt[:, :, start_time:end_time]
        
        pad_size = window_time - (end_time - start_time)
        if pad_size > 0:
            pad_tensor = torch.zeros((channels, n_freq, pad_size), device=cqt.device)
            section = torch.cat([section, pad_tensor], dim=2)
        
        sections.append(section)
    
    return torch.stack(sections, dim=0)

def get_pianoroll_windows(pianoroll, window_time, overlap_frames):
    n_notes, n_time = pianoroll.shape
    step_size = window_time - overlap_frames
    sections = []
    
    for start_time in range(0, n_time, step_size):
        end_time = min(start_time + window_time, n_time)
        section = pianoroll[:, start_time:end_time]
        
        pad_size = window_time - (end_time - start_time)
        if pad_size > 0:
            pad_tensor = torch.zeros((n_notes, pad_size), device=pianoroll.device)
            section = torch.cat([section, pad_tensor], dim=1)
        
        sections.append(section)
    
    return torch.stack(sections, dim=0)

#handling inference where we need to reconstruct the piano roll
def patches2pianoroll(pianoroll_sections, overlap_frames, original_cqt_time_bins):
    """
    Reconstructs a piano roll from overlapping patches by averaging overlapping regions.

    Args:
    - pianoroll_sections: Tensor of shape (num_patches, n_notes, window_time)
    - overlap_frames: Number of overlapping time steps between consecutive windows.
    - original_cqt_time_bins: Total number of time bins in the original piano roll.

    Returns:
    - Reconstructed piano roll of shape (n_notes, original_cqt_time_bins)
    """
    num_patches, n_notes, window_time = pianoroll_sections.shape
    step_size = window_time - overlap_frames  # Shift per patch
    
    # Initialize storage for accumulation and count of contributions
    reconstructed = torch.zeros((n_notes, original_cqt_time_bins), device=pianoroll_sections.device)
    count = torch.zeros((n_notes, original_cqt_time_bins), device=pianoroll_sections.device)
    
    for i, start_time in enumerate(range(0, original_cqt_time_bins, step_size)):
        end_time = min(start_time + window_time, original_cqt_time_bins)
        valid_length = end_time - start_time  # Valid non-padded length
        
        # Add the valid portion of the patch to the reconstructed piano roll
        reconstructed[:, start_time:end_time] += pianoroll_sections[i, :, :valid_length]
        count[:, start_time:end_time] += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    
    return reconstructed / count