import json

import mne.io
import numpy as np
import os
import pyxdf
import scipy.io
import matplotlib
matplotlib.use('Qt5Agg')

from mne.datasets import eegbci
from mne.preprocessing import ICA
import mne_bids_pipeline

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

def event_marker(stream, marker_stream, sample_rate=500):
    time = stream['time_stamps']
    marker_series = np.array(marker_stream['time_series'])
    marker_series = marker_series.tolist()
    marker_times = (marker_stream['time_stamps'])[np.nonzero(marker_series)[0]]
    event_index = []
    event_duration = []
    marker_int = []

    for i, event in enumerate(marker_series):
        event_str = marker_series[i][0]
        nearest_index = find_nearest_index(time, marker_times[i])
        event_index.append(nearest_index)
        if event_str == 'Reference':
            event_duration.append(3 * sample_rate)
            marker_int.append(120)
        elif event_str == 'Start_of_Trial_l':
            event_duration.append(0)
            marker_int.append(121)
        elif event_str == 'Start_of_Trial_r':
            event_duration.append(0)
            marker_int.append(122)
        elif event_str == 'Cue':
            event_duration.append(1.25 * sample_rate)
            marker_int.append(130)
        elif event_str == 'Feedback':
            event_duration.append(7 * sample_rate)
            marker_int.append(140)
        elif event_str == 'End_of_Trial':
            event_duration.append(0)
            marker_int.append(150)
        elif event_str == 'Session_Start':
            event_duration.append(0)
            marker_int.append(180)
        elif event_str == 'Session_End':
            event_duration.append(0)
            marker_int.append(190)
    event_details = np.array([event_index, event_duration, marker_int]).T
    return event_details

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

def save_to_mat(file_path, identifier, data):
    """Saves data as .mat file. If the file already exists data is appended to it.

    Parameters
    ----------
    file_path: `str`
        Path of the .mat file.
    identifier: `str`
        How the data is called in the .mat file.
    data: `ndarray`
        The data to be saved.
    """

    # if there is already a messung.mat file new data is appended to it
    if os.path.isfile(file_path):
        print('INFO new data is appended to ', file_path)
        data_old = scipy.io.loadmat(file_path)[identifier]
        data = np.append(data_old, data, axis=0)

    scipy.io.savemat(file_path, {identifier: data})

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

def gorella(raw_eeg):
    eegbci.standardize(raw_eeg)

    # raw_eeg.plot_sensors(kind='3d', show=True, block=True)
    #raw_eeg.plot(scalings='auto', title='Raw')  # +/- 200 µV scale (1 V = 1000000 µV)

    raw_eeg.plot_psd(tmin=0, tmax=60, fmin=2, fmax=80, average=True, spatial_colors=True, show=True)
    raw_eeg.compute_psd(tmin=0, tmax=60, fmin=2, fmax=80)  # .plot()
    # raw_eeg.compute_psd().plot()
    raw_eeg.notch_filter(50)
    raw_eeg.filter(l_freq=1.0, h_freq=50.0)
    raw_eeg.resample(120, npad='auto')
    raw_eeg.compute_psd(tmin=0, tmax=60, fmin=2, fmax=60)  # .plot()
    raw_eeg.plot(scalings='auto', title='Filtered')

    ica = ICA(n_components=15, method='fastica')
    ica.fit(raw_eeg)
    ica.plot_components()

    #raw_eeg.plot(n_channels=32, scalings='auto')

    ica.plot_properties(raw_eeg, picks=0)
    ica.plot_properties(raw_eeg, picks=1)
    ica.plot_overlay(raw_eeg, exclude=[0,1])
    ica.exclude = [0]
    ica.apply(raw_eeg)

    #raw_eeg.plot(n_channels=32, scalings='auto', title='ICA applied')

if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'

    # ----- Define Subject and Session ----
    subject_list = ['S20']
    session_list = [0, 1, 2]
    run_list = [1, 2, 3, 4, 5]

    subject_id = subject_list[0]
    session_id = session_list[1]
    run_id = run_list[2]

    subject_data_path = data_path + subject_id + '-ses' + str(session_id) + '/'
    all_xdf_files = []
    all_config_files = []
    for r in range(1, 6): #max. 5 runs
        for file in os.listdir(subject_data_path):
            if file.startswith(subject_id + '_run' + str(r)):
                if file.endswith('.xdf'):
                    all_xdf_files.append(file)
            if file.startswith('CONFIG_' + subject_id + '_run' + str(r)):
                if file.endswith('.json'):
                    all_config_files.append(file)
    current_xdf = [i for i in all_xdf_files if i.startswith(subject_id + '_run' + str(run_id))]
    current_xdf = subject_data_path + current_xdf[0]
    current_config = [i for i in all_config_files if i.startswith('CONFIG_' + subject_id + '_run' + str(run_id))]
    current_config = subject_data_path + current_config[0]

    if run_id == 1:
        eeg, marker = load_xdf(current_config, current_xdf, feedback="off")
    else:
        eeg, marker, erds, lda = load_xdf(current_config, current_xdf, feedback="on")

    eeg_with_marker = add_marker(eeg, marker)




    dir_out = subject_data_path + 'preparedData/'
    """
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    mat_file = dir_out + 'eeg_run' + str(run_id) + '.mat'
    if not os.path.exists(mat_file):
        save_to_mat(mat_file, 'eeg', eeg_with_marker)
    if os.path.exists(mat_file):
        print(mat_file + ' already exists.')

    raw_file = dir_out + 'raw_eeg_breakless_run' + str(run_id) + '.fif'
    raw_eeg = delete_breaks(eeg_with_marker)
    #raw_eeg = mne.io.read_raw_fif(raw_file)
    #fig = raw_eeg.plot(n_channels=32, show_scrollbars=True, block=True, show_options=True, scalings='auto')#, events=marker)
    """
    """
    if not os.path.exists(raw_file):
        save_to_mat(raw_file, 'eeg', eeg_with_marker)
    if os.path.exists(raw_file):
        print(raw_file + ' already exists.')
    """

    # create and export events
    event_marker = event_marker(eeg, marker)
    #new_event_marker = np.vstack((event_marker[:, 0].astype(int), event_marker[:, 1].astype(int), event_marker[:, 2].astype(int))).T
    event_file = dir_out + 'events_run' + str(run_id) + '_eve.fif'
    csv_file = dir_out + 'events_run' + str(run_id) + '_eve.eve'
    #mne.write_events(event_file, event_marker, overwrite=True, fmt='%.2f')
    np.savetxt(csv_file, event_marker, delimiter=',', fmt='%d')
    new_events = mne.read_events(csv_file)

    print("test")


    #gorella(raw_eeg)








