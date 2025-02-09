import numpy as np
import pretty_midi
from matplotlib import lines as mlines, pyplot as plt
import tempfile
import librosa
import sox
import os
import pandas as pd


def save_small_wav(out_path, y, fs):
    '''  save audio signal y as a wav file with 16 bit depth '''
    fhandle, tmp_file = tempfile.mkstemp(suffix='.wav')

    librosa.output.write_wav(tmp_file, y, fs)

    tfm = sox.Transformer()
    tfm.convert(bitdepth=16)
    tfm.build(tmp_file, out_path)
    os.close(fhandle)
    os.remove(tmp_file)

def jams_to_midi(jam, q=1):
    # q = 1: with pitch bend. q = 0: without pitch bend.
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for anno in annos:
        midi_ch = pretty_midi.Instrument(program=25)
        for note in anno:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st + dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
            midi_ch.notes.append(n)
            midi_ch.pitch_bends.append(pb)
        if len(midi_ch.notes) != 0:
            midi.instruments.append(midi_ch)
    return midi


def sonify_jams(jam, fpath=None, q=1):
    ''' jams to audio '''
    midi = jams_to_midi(jam, q) # q=1 : with pitchbend
    signal_out = midi.fluidsynth()
    if fpath != None:
        save_small_wav(fpath, signal_out, 44100)
    return signal_out, 44100


def visualize_jams_note(jam, save_path=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    fig = plt.figure()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            dur = note[1]
            plt.plot([start_time, start_time + dur],
                     [midi_note, midi_note],
                     style_dict[s], label=string_dict[s])
        s += 1
    plt.xlabel('Time (sec)')
    plt.ylabel('Pitch (midi note number)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.title(jam.file_metadata.title)
    plt.xlim(-0.5, jam.file_metadata.duration)
    fig.set_size_inches(6,3)
    if save_path:
        plt.savefig(save_path)

def visualize_jams_pt(jam, save_path=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    # fig = plt.figure()
    annos_pt = jam.search(namespace='pitch_contour')
    # plot pitch
    for string_tran in annos_pt:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        df = string_tran.to_dataframe()
        pitch_s = df.value.apply(
            lambda x: librosa.hz_to_midi(float(x['frequency'])))
        pitch_s.name = 'pitch'
        df = pd.concat([df, pitch_s], axis=1)
        plt.scatter(df.time, df.pitch, s=0.1, color=style_dict[s],
                        label=string_dict[s])

        s += 1

    # plot Beat
    anno_b = jam.search(namespace='beat_position')[0]
    handle_list.append(mlines.Line2D([], [], color='k',
                                         label='downbeat'))
    handle_list.append(mlines.Line2D([], [], color='k', linestyle='dotted',
                                     label='beat'))
    for b in anno_b.data:
        t = b.time
        plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
        if int(b.value['position']) == 1:
            plt.axvline(t, linestyle='-', color='k', alpha=0.8)


    # plt.xlabel('Time (sec)')
    plt.ylabel('Pitch Contour (midi note number)')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.title(jam.file_metadata.title)
    plt.xlim(-0.06, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)

def visualize_jams_onset(jam, save_path=None, low=None, high=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    # fig = plt.figure()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            
            start_time = note[0]
            if low and start_time < low:
                continue
            if high and start_time > high:
                continue
            plt.vlines(start_time,s, s+2,style_dict[s], label=string_dict[s])
        s += 1
    
    plt.xlabel('sec')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.ylabel('String Number')

    if not low:
        low = -0.1
    if not high:
        high = jam.file_metadata.duration
    plt.xlim(low, high)
    # fig.set_size_inches(jam.file_metadata.duration / 2.5, 6)
#    plt.title('Onsets of Individual Strings for excerpt of 00_Rock2-142-D_comp')
    if save_path:
        plt.savefig(save_path)


def tablaturize_jams(jam, save_path=None):
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e'}
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    s = 0

    handle_list = []

    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
        
    plt.figure(figsize=(20,6))

    for string_tran in annos:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            fret = int(round(midi_note - str_midi_dict[s]))
            plt.scatter(start_time, s+1, marker="${}$".format(fret), color =
            style_dict[s])
        s += 1

    # plot Beat
    anno_b = jam.search(namespace='beat_position')[0]
    for b in anno_b.data:
        t = b.time
        plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
        if int(b.value['position']) == 1:
            plt.axvline(t, linestyle='-', color='k', alpha=0.8)

    handle_list.append(mlines.Line2D([], [], color='k',
                                     label='downbeat'))
    handle_list.append(mlines.Line2D([], [], color='k', linestyle='dotted',
                                     label='beat'))
    plt.xlabel('Time (sec)')
    plt.ylabel('String Number')
    # plt.title(jam.file_metadata.title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               handles=handle_list, ncol=8)
    plt.xlim(-0.5, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)

def visualize_chords(jam, save_path=None):

    chord_ann = jam.search(namespace='chord')[1]


    for chord in chord_ann.data:
        t = chord.time


    # for string_tran in annos:
    #     for note in string_tran:
    #         start_time = note[0]
    #         midi_note = note[2]
    #         fret = int(round(midi_note - str_midi_dict[s]))
    #         plt.scatter(start_time, s+1, marker="${}$".format(fret), color =
    #         style_dict[s])
    #     s += 1
    #
    # # plot Beat
    # anno_b = jam.search(namespace='beat_position')[0]
    # for b in anno_b.data:
    #     t = b.time
    #     plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
    #     if int(b.value['position']) == 1:
    #         plt.axvline(t, linestyle='-', color='k', alpha=0.8)


    plt.xlabel('Time (sec)')
    plt.ylabel('String Number')
    # plt.title(jam.file_metadata.title)
    plt.xlim(-0.5, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)

def add_annotations_to_barline(ax, annotations, beat_annotations,
                               ygrow_ratio=1, label_xoffset=0, label_yoffset=0):
    """
    Add annotation values at each barline of a given pair of axes.
    
    Display at the top of the given Matplotlib Axes the values of a JAMS annotation as text
    labels at each barline specified by a JAMS beat_position annotation array. If there are
    multiple annotations per bar, only the first one will be displayed and if an annotation
    lasts multiple bars, its label will be repeated at every bar line.
    
    Keyword arguments:
    ax -- Matplotlib Axes
    annotations -- the JAMS AnnotationArray whose values to display
    beat_annotations -- the JAMS beat_position AnnotationArray specifying the barlines
    ygrow_ratio -- amount to increase the y-axis in order to accomodate the text fields,
                   specified as the ratio of the current y-range
    label_xoffset -- amount to offset the text labels in order to avoid overlap with barlines,
                     specified in the units of the x-axis
    label_yoffset -- amount to offset the text labels in order to avoid overlap with barlines,
                     specified in the units of the y-axis
    
    Note: the optimal values for ygrow_ratio, label_xoffset and label_yoffset are plot-dependent,
          so will have to be determined experimentally
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max+(y_max-y_min)*ygrow_ratio) # make room for label
    for beat in beat_annotations:
        if beat.value['position'] == 1 and beat.time >= x_min and beat.time < x_max:
            ax.text(beat.time+label_xoffset, y_max-label_yoffset,
                    annotations.to_samples([beat.time+0.001])[0][0],
                    fontdict={'backgroundcolor': ax.get_facecolor()})

def add_annotations(ax, annotations,
                    ygrow_ratio=0, label_xoffset=0, label_yoffset=0):
    """
        Add annotation values to a given pair of axes.
        
        Display at the top of the given Matplotlib Axes the values of a JAMS annotation as text
        labels.
        
        Keyword arguments:
        ax -- Matplotlib Axes
        annotations -- the JAMS AnnotationArray whose values to display
        ygrow_ratio -- amount to increase the y-axis in order to accomodate the text fields,
        specified as the ratio of the current y-range
        label_xoffset -- amount to offset the text labels in order to avoid overlap with barlines,
        specified in the units of the x-axis
        label_yoffset -- amount to offset the text labels in order to avoid overlap with barlines,
        specified in the units of the y-axis
        
        Note: the optimal values for ygrow_ratio, label_xoffset and label_yoffset are plot-dependent,
        so will have to be determined experimentally
        """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max+(y_max-y_min)*ygrow_ratio) # make room for label
    for segment in annotations:
        if segment.time >= x_min and segment.time < x_max:
            ax.text(segment.time+label_xoffset, y_max-label_yoffset,
                    annotations.to_samples([segment.time+0.001])[0][0],
                    fontdict={'backgroundcolor': ax.get_facecolor()})
            
            
            

#added mine
def visualize(jam, sr=22050, hop_length=512, min_midi=36, max_midi=88, save_path=None):
    """
    Visualizes note annotations from a JAMS file as a time vs. MIDI pitch plot and returns a piano roll.
    The intensity (velocity) is also normalized to range [0, 1] and added to the piano roll.

    Parameters:
    - jam (jams.JAMS): Loaded JAMS object.
    - sr (int): Sample rate (default = 22050).
    - hop_length (int): Hop size for discretization (default = 512).
    - min_midi (int): Minimum MIDI pitch to include in the piano roll (default = 36, C2).
    - max_midi (int): Maximum MIDI pitch to include (default = 88, C6).
    - save_path (str, optional): Path to save the figure (default = None).

    Returns:
    - piano_roll (numpy.ndarray): (num_pitches, num_time_bins) matrix with intensity values.
    - time_axis (numpy.ndarray): Time in seconds corresponding to each time bin.
    - midi_pitches (numpy.ndarray): Array of included MIDI pitches.
    """
    style_dict = {0: 'r', 1: 'y', 2: 'b', 3: '#FF7F50', 4: 'g', 5: '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e'}

    fig, ax = plt.subplots(figsize=(8, 4))
    handle_list = []

    # Search for MIDI note annotations
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')

    if not annos:
        print("No note annotations found in the JAMS file.")
        return None, None, None

    # Collect note data
    notes = []
    intensities = []  # List to hold intensity (velocity) values
    for s, string_tran in enumerate(annos):
        color = style_dict.get(s % len(style_dict), 'k')  # Use modulo to avoid index errors
        label = string_dict.get(s % len(string_dict), f"Track {s}")

        handle_list.append(mlines.Line2D([], [], color=color, label=label))

        for note in string_tran.data:
            start_time = note.time
            duration = note.duration
            midi_note = int(round(note.value))  # MIDI pitch as integer
            intensity = note.velocity if hasattr(note, 'velocity') else 1.0  # Default to 1 if no velocity info

            notes.append((start_time, duration, midi_note))
            intensities.append(intensity)

            # Plot each note as a horizontal line
            ax.plot([start_time, start_time + duration], [midi_note, midi_note],
                    color=color, label=label)

    # Normalize intensity values to range [0, 1]
    if intensities:
        min_intensity = min(intensities)
        max_intensity = max(intensities)
        intensity_range = max_intensity - min_intensity
        intensities = [(i - min_intensity) / intensity_range if intensity_range > 0 else 1.0 for i in intensities]
    else:
        intensities = [1.0] * len(notes)  # Default to 1 if no intensity data is found

    # Set labels and title
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(getattr(jam.file_metadata, "title", "MIDI Note Visualization"))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)

    # Set time limits
    duration = getattr(jam.file_metadata, "duration", None)
    if duration:
        ax.set_xlim(-0.5, duration)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    # Convert notes to piano roll
    if not notes:
        return None, None, None

    notes = np.array(notes)  # Convert list to numpy array
    frame_duration = hop_length / sr  # Time per frame in seconds
    max_time = np.max(notes[:, 0] + notes[:, 1])  # Last note's end time
    num_time_bins = int(np.ceil(max_time / frame_duration))  # Total time bins

    num_pitches = max_midi - min_midi + 1
    piano_roll = np.zeros((num_pitches, num_time_bins), dtype=np.float32)

    # Populate piano roll with normalized intensity values
    for i, (onset, duration, pitch) in enumerate(notes):
        if pitch < min_midi or pitch > max_midi:  # Ignore notes out of range
            continue

        start_idx = int(round(onset / frame_duration))
        end_idx = int(round((onset + duration) / frame_duration))
        pitch_idx = int(pitch - min_midi)  # Convert MIDI pitch to row index

        # Clip indices to stay within array bounds
        start_idx = max(0, min(start_idx, num_time_bins - 1))
        end_idx = max(start_idx, min(end_idx, num_time_bins))  # Ensure end_idx is not before start_idx

        # Use normalized intensity for the piano roll
        piano_roll[pitch_idx, start_idx:end_idx] = np.maximum(piano_roll[pitch_idx, start_idx:end_idx], intensities[i])


    time_axis = np.arange(num_time_bins) * frame_duration
    midi_pitches = np.arange(min_midi, max_midi + 1)

    return piano_roll, time_axis, midi_pitches