# ==============================================================================
# GuitarSet dataset: audio → normalized CQT, JAMS annotations → piano roll,
# with optional overlapping windowing. Also provides the persistent
# train/test split (portable JSON of file basenames, no absolute paths).
# ==============================================================================

import json
import os

import librosa
import numpy as np
import torch
import torch.utils.data as data
from sklearn.decomposition import NMF

from src.config import (
    ANNOTATION_SUBDIR,
    AUDIO_SUBDIRS,
    BINS_PER_OCTAVE,
    FMIN,
    N_BINS,
    OVERLAP_FRAMES,
    SR,
    WINDOW_TIME,
)
from src.data.midi_utils import jams_to_pianoroll


class GuitarSetDataset(data.Dataset):
    """Pairs every GuitarSet audio version with its JAMS annotation.

    Each item is either the full (CQT, piano roll) pair or their overlapping
    sections, depending on apply_sections.
    """

    def __init__(self, data_dir="guitarset", audio_subdirs=None, apply_nmf=False, apply_sections=True):
        self.data_dir = data_dir
        self.audio_dirs = [
            os.path.join(data_dir, subdir)
            for subdir in (audio_subdirs or AUDIO_SUBDIRS)
        ]
        self.jams_dir = os.path.join(data_dir, ANNOTATION_SUBDIR)
        self.apply_nmf = apply_nmf

        # return output as sections or whole piano roll
        self.apply_sections = apply_sections

        self.samples = self._get_file_pairs()

    def _get_file_pairs(self):
        '''Returns a list of (audio_path, jams_path) tuples for all audio directories.'''
        if not os.path.isdir(self.jams_dir):
            raise FileNotFoundError(
                f"GuitarSet annotations not found at '{self.jams_dir}'. "
                "Set --data-dir / GUITARSET_DIR to the dataset root (see README)."
            )
        jams_files = [f for f in os.listdir(self.jams_dir) if f.endswith(".jams")]
        samples = []
        for jams_file in jams_files:
            prefix = jams_file.replace(".jams", "")
            for audio_dir in self.audio_dirs:
                if not os.path.isdir(audio_dir):
                    continue  # skip missing audio versions
                matching_wavs = [f for f in os.listdir(audio_dir)
                                 if f.startswith(prefix) and f.endswith(".wav")]
                for wav_file in matching_wavs:
                    audio_path = os.path.join(audio_dir, wav_file)
                    jams_path = os.path.join(self.jams_dir, jams_file)
                    samples.append((audio_path, jams_path))
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def nmf(CQT_norm, n_components=10):
        model = NMF(n_components=n_components, init='nndsvd', random_state=0)
        W = model.fit_transform(CQT_norm)  # spectral bases
        H = model.components_              # activations
        CQT_nmf = np.dot(W, H)             # reconstruct the cleaned CQT
        return CQT_nmf

    def _load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=SR)
        CQT = librosa.cqt(audio, sr=SR, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS)

        # logarithmic scaling + min-max normalization
        CQT_dB = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
        CQT_norm = (CQT_dB - CQT_dB.min()) / (CQT_dB.max() - CQT_dB.min())

        if self.apply_nmf:
            CQT_norm = self.nmf(CQT_norm)
        return CQT_norm.astype(np.float32)

    def _load_jams(self, file_path, num_time_bins):
        import jams
        jam = jams.load(file_path)
        piano_roll, time_axis, midi_pitches = jams_to_pianoroll(jam, num_time_bins)
        return piano_roll, time_axis, midi_pitches

    def __getitem__(self, idx):
        audio_path, jams_path = self.samples[idx]

        cqt_data = self._load_audio(audio_path)
        cqt_data = np.expand_dims(cqt_data, axis=0)          # (1, 156, T)
        num_time_bins = cqt_data.shape[2]

        piano_roll, time_axis, midi_pitches = self._load_jams(jams_path, num_time_bins)

        cqt_tensor = torch.tensor(cqt_data)                  # (1, 156, T)
        piano_roll_tensor = torch.tensor(piano_roll)         # (53, T)

        if self.apply_sections:
            cqt_sections = get_overlap_windows(cqt_tensor, WINDOW_TIME, OVERLAP_FRAMES)             # (S, 1, 156, 86)
            piano_roll_sections = get_pianoroll_windows(piano_roll_tensor, WINDOW_TIME, OVERLAP_FRAMES)  # (S, 53, 86)
            return cqt_sections, piano_roll_sections, num_time_bins
        else:
            return cqt_tensor, piano_roll_tensor


# ------------------------------------------------------------------ windowing

def get_overlap_windows(cqt, window_time, overlap_frames):
    """Splits a CQT (C, F, T) into overlapping windows → (S, C, F, window_time)."""
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
    """Splits a piano roll (N, T) into overlapping windows → (S, N, window_time)."""
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
    step_size = window_time - overlap_frames  # shift per patch

    reconstructed = torch.zeros((n_notes, original_cqt_time_bins), device=pianoroll_sections.device)
    count = torch.zeros((n_notes, original_cqt_time_bins), device=pianoroll_sections.device)

    for i, start_time in enumerate(range(0, original_cqt_time_bins, step_size)):
        end_time = min(start_time + window_time, original_cqt_time_bins)
        valid_length = end_time - start_time  # valid non-padded length

        reconstructed[:, start_time:end_time] += pianoroll_sections[i, :, :valid_length]
        count[:, start_time:end_time] += 1

    count[count == 0] = 1  # avoid division by zero

    return reconstructed / count


# ------------------------------------------------------- persistent test split

def _sample_key(sample):
    """Portable identity of an (audio_path, jams_path) pair: no absolute paths."""
    audio_path, jams_path = sample
    return [
        os.path.basename(os.path.dirname(audio_path)),
        os.path.basename(audio_path),
        os.path.basename(jams_path),
    ]


def load_or_create_split(samples, split_path="test_split.json", test_size=0.2, seed=42):
    """Returns (train_samples, test_samples), persisting the test set to JSON.

    The split file stores [audio_subdir, wav_name, jams_name] triples, so it
    stays valid regardless of where GuitarSet lives on disk.
    """
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            test_keys = {tuple(entry) for entry in json.load(f)}
        test_samples = [s for s in samples if tuple(_sample_key(s)) in test_keys]
        train_samples = [s for s in samples if tuple(_sample_key(s)) not in test_keys]
        print(f"Test split loaded from {split_path} ({len(test_samples)} test samples)")
    else:
        from sklearn.model_selection import train_test_split
        train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=seed)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump([_sample_key(s) for s in test_samples], f, indent=1)
        print(f"Test split created and saved to {split_path} ({len(test_samples)} test samples)")

    return train_samples, test_samples
