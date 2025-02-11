import pretty_midi
import numpy as np
from music21 import converter, environment
import IPython.display as ipd
import librosa
import pathlib
import matplotlib.pyplot as plt
import jams
import matplotlib.lines as mlines

def midi_to_pianoroll(midi, midi_path=None, fs=100):
    '''converts a midi file to a piano roll '''
    if midi_path:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
    piano_roll = midi.get_piano_roll(fs=fs)
    
    # Limit to Guitar Range (C2 - E6) = MIDI 36 to 88
    guitar_piano_roll = piano_roll[36:89, :]
    return guitar_piano_roll

##pianoroll to midi TODO

def play_midi(midi, midi_path=None, fs=44100):
    '''plays a midi file and returns its audio'''
    
    if midi_path:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
    audio = midi.synthesize(fs=fs)
    ipd.Audio(audio, rate=fs)
    return audio

def jams_to_pianoroll(jam, num_time_bins, sr=22050, hop_length=512, min_midi=36, max_midi=88):
    """
    Converts note annotations from a JAMS file into a piano roll with a fixed number of time bins.
    
    Parameters:
    - jam (jams.JAMS): Loaded JAMS object.
    - num_time_bins (int): Fixed number of time bins (should match input tensor).
    - sr (int): Sample rate (default = 22050).
    - hop_length (int): Hop size for discretization (default = 512).
    - min_midi (int): Minimum MIDI pitch to include in the piano roll (default = 36, C2).
    - max_midi (int): Maximum MIDI pitch to include (default = 88, E6).

    Returns:
    - piano_roll (numpy.ndarray): (num_pitches, num_time_bins) matrix with intensity values.
    - time_axis (numpy.ndarray): Time in seconds corresponding to each time bin.
    - midi_pitches (numpy.ndarray): Array of included MIDI pitches.
    """

    # Initialize piano roll
    num_pitches = max_midi - min_midi + 1
    piano_roll = np.zeros((num_pitches, num_time_bins), dtype=np.float32)

    # Compute frame duration
    frame_duration = hop_length / sr 
    time_axis = np.arange(num_time_bins) * frame_duration  # Ensure consistent time axis

    # Get MIDI note annotations
    annos = jam.search(namespace='note_midi')
    if not annos:
        annos = jam.search(namespace='pitch_midi')
    if not annos:
        print("No note annotations found in the JAMS file.")
        return None, None, None

    # Collect note data
    notes = []
    intensities = []
    
    for string_tran in annos:
        for note in string_tran.data:
            start_time = note.time
            duration = note.duration
            midi_note = int(round(note.value))  # MIDI pitch as integer
            intensity = getattr(note, 'velocity', 1.0)  # Default to 1 if no velocity info

            notes.append((start_time, duration, midi_note))
            intensities.append(intensity)

    # Normalize intensity values to [0, 1]
    if intensities:
        min_intensity = min(intensities)
        max_intensity = max(intensities)
        intensity_range = max_intensity - min_intensity
        intensities = [(i - min_intensity) / intensity_range if intensity_range > 0 else 1.0 for i in intensities]
    else:
        intensities = [1.0] * len(notes)  # Default to 1 if no intensity data is found

    # Populate piano roll
    for i, (onset, duration, pitch) in enumerate(notes):
        if pitch < min_midi or pitch > max_midi:
            continue

        start_idx = int(round(onset / frame_duration))
        end_idx = int(round((onset + duration) / frame_duration))
        pitch_idx = pitch - min_midi  # Convert MIDI pitch to row index

        # Clip indices to valid range
        start_idx = max(0, min(start_idx, num_time_bins - 1))
        end_idx = max(start_idx, min(end_idx, num_time_bins))

        # Fill piano roll with intensity
        piano_roll[pitch_idx, start_idx:end_idx] = np.maximum(
            piano_roll[pitch_idx, start_idx:end_idx], intensities[i]
        )
        
    midi_pitches = np.arange(min_midi, max_midi + 1)

    return piano_roll, time_axis, midi_pitches


def plot_piano_roll(piano_roll, time_axis, midi_pitches):
    """
    Plots a piano roll representation.

    Parameters:
    - piano_roll (numpy.ndarray): Binary matrix of shape (num_pitches, num_time_bins).
    - time_axis (numpy.ndarray): Time in seconds corresponding to each time bin.
    - midi_pitches (numpy.ndarray): Array of included MIDI pitches.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll, aspect='auto', cmap='Greys', origin='lower',
               extent=[time_axis[0], time_axis[-1], midi_pitches[0], midi_pitches[-1]])

    plt.colorbar(label="Note Activation (1 = On, 0 = Off)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("MIDI Pitch")
    plt.title("Piano Roll Visualization")
    plt.show()
