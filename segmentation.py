import json

import mne.io
import numpy as np
import os
import pyxdf
import scipy.io
import matplotlib
matplotlib.use('Qt5Agg')

import offline_analysis

def load_xdf(config_file, xdf_file_path, feedback):
    """Loads a xdf file and extracts the eeg and marker stream

    Parameters
    ----------

    feedback: `str`
        Feedback is on (runs > 1) or off (run = 1)

    Returns
    -------
    stream1: `dict`
        The eeg stream.
    stream2: `dict`
        The marker stream.
    stream3: `dict`
        The erds stream.
    stream4: `dict`
        The lda stream.
    """
    with open(config_file) as json_file:
        config = json.load(json_file)

    streams, fileheader = pyxdf.load_xdf(xdf_file_path)
    streams_info = []

    for stream in streams:
        streams_info.append(stream['info']['name'][0])

    streams_info = np.array(streams_info)

    lsl_config = config['general-settings']['lsl-streams']

    eeg_pos = np.where(streams_info == lsl_config['eeg']['name'])[0][0]
    marker_pos = np.where(streams_info == lsl_config['marker']['name'])[0][0]

    if feedback == "off":
        return streams[eeg_pos], streams[marker_pos]
    elif feedback == "on":
        erds_pos = np.where(streams_info == lsl_config['fb-erds']['name'])[0][0]
        lda_pos = np.where(streams_info == lsl_config['fb-lda']['name'])[0][0]
        return streams[eeg_pos], streams[marker_pos], streams[erds_pos], streams[lda_pos]
    else:
        print("ERROR: Undefined feedback state.")

def add_marker(stream, marker_stream):
    time = stream['time_stamps']
    marker_series = np.array(marker_stream['time_series'])
    marker_times = (marker_stream['time_stamps'])[np.nonzero(marker_series)[0]]
    marker_positions = np.zeros((np.shape(time)[0], 1), dtype='int')

    conditions = marker_series
    conditions[np.where(conditions == 'Reference')] = 120
    conditions[np.where(conditions == 'Start_of_Trial_l')] = 121
    conditions[np.where(conditions == 'Start_of_Trial_r')] = 122
    conditions[np.where(conditions == 'Cue')] = 130
    conditions[np.where(conditions == 'Feedback')] = 140
    conditions[np.where(conditions == 'End_of_Trial')] = 150
    conditions[np.where(conditions == 'Session_Start')] = 180
    conditions[np.where(conditions == 'Session_End')] = 190


    for t, m in zip(marker_times, marker_series):
        pos = find_nearest_index(time, t)
        try:
            marker_positions[pos] = m
        except: # only adds integers of conditions
            pass

    print("debug")
    return np.append(marker_positions, stream['time_series'], axis=1)

def find_nearest_index(array, value):
    """Finds the position of the value which is nearest to the input value in the array.

    Parameters
    ----------
    array: `ndarray`
        The array of time stamps.
    value: `float`
        The (nearest) value to find in the array.

    Returns
    -------
    idx: `int`
        The index of the (nearest) value in the array.
    """

    idx = np.searchsorted(array, value, side="right")
    if idx == len(array):
        return idx - 1
    else:
        return idx

def delete_breaks(eeg_with_marker):
    indices = np.nonzero(eeg_with_marker[:,0])
    indices = list(indices[0])
    marker = []
    ix_start_l = []
    ix_start_r = []
    ix_end = []
    for i in indices:
        m = eeg_with_marker[i,0]
        marker.append(m)
        if m == 121:
            ix_start_l.append(i)
        if m == 122:
            ix_start_r.append(i)
        if m == 150:
            ix_end.append(i)

    ix_start = ix_start_l + ix_start_r
    ix_start = sorted(ix_start)

    #remove period before first task and after last task
    eeg = eeg_with_marker[indices[0]:indices[-1], 1:33].T

    ch_names = ["Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1",
                "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2"]
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names, sfreq=500, ch_types='eeg')
    info.set_montage(montage)

    raw_eeg = mne.io.RawArray(eeg[:, ix_start[0]:(ix_start[0] + 4275)], info)

    for t in range(len(ix_start)-1):
        raw_eeg.append(mne.io.RawArray(eeg[:, ix_start[t+1]:(ix_start[t+1] + 4275)], info))

    return raw_eeg