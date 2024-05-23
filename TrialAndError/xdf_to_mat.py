"""
Imports a xdf file and exports it to a .mat file.
"""

import json
import numpy as np
import os
import pyxdf
import scipy.io


def load_xdf(config, xdf_file_path, feedback):
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

def add_class_labels(stream, marker_stream):
    """Adds another row to the stream. It includes the class labels at the cue positions.

    Parameters
    ----------
    stream: `dict`
        The LSL stream which should be appended.
    marker_stream: `dict`
        Marker stream containing info about the cue times and class labels.

    Returns
    -------
    result: `ndarray`
        Data of stream with class labels in the first row.
    """

    time = stream['time_stamps']
    marker_series = np.array(marker_stream['time_series'])
    cue_times = (marker_stream['time_stamps'])[np.where(marker_series == 'Cue')[0]]

    conditions = marker_series[np.where(np.char.find(marker_series[:, 0], 'Start_of_Trial') == 0)[0]]
    conditions[np.where(conditions == 'Start_of_Trial_l')] = 121
    conditions[np.where(conditions == 'Start_of_Trial_r')] = 122

    # Versuch, nicht nur cue l+r, sondern auch die anderen Events mit aufzulisten
    '''
    event_types = ['Session_Start', 'Reference', 'Cue', 'Feedback', 'End_of_Trial', 'Session_End']
    events = []
    event_times = []
    for e in event_types:
        events.append(list(marker_series[np.where(np.char.find(marker_series[:, 0], e) == 0)[0]]))
        event_times.append(list((marker_stream['time_stamps'])[np.where(marker_series == e)[0]]))
    '''
    cue_positions = np.zeros((np.shape(time)[0], 1), dtype=int)
    for t, c in zip(cue_times, conditions):
        pos = find_nearest_index(time, t)
        cue_positions[pos] = c

    return np.append(cue_positions, stream['time_series'], axis=1)


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


'''
def xdf_to_mat(config):

    # ------------- Subject specific variables -------------
    motor_mode = config['gui-input-settings']['motor-mode']
    dimension = config['gui-input-settings']['dimension-mode']
    subject_id = config['gui-input-settings']['subject-id']
    session = str(config['gui-input-settings']['n-session'])
    run = str(config['gui-input-settings']['n-run'])
    # ------------------------------------------------------

    cwd = os.getcwd()
    root_dir = cwd + '/SubjectData/'


    file_name = subject_id + '_run' + run + '_' + motor_mode + '_' + dimension
    xdf_file_path = root_dir + subject_id + '-ses' + session + '/' + file_name + '.xdf'
    mat_file_path = root_dir + subject_id + '-ses' + session + '/' + file_name + '.mat'

    if os.path.exists(mat_file_path):
        print(".mat file for this configuration already exists.")

    else:
        if not os.path.exists(xdf_file_path):
            print(".xdf file for this configuration does not exist.")
            os.makedirs(root_dir + subject_id + '-ses' + session + '/')

        # Extract the eeg and marker lsl stream from the xdf file
        stream_eeg, stream_marker = load_xdf(xdf_file_path, config['general-settings']['lsl-streams'])

        # Add a row to the eeg data for the class labels
        eeg_and_label = add_class_labels(stream_eeg, stream_marker)

        save_to_mat(mat_file_path, 'data', eeg_and_label)
        print('.mat created')
'''
'''
def xdf_to_mat_new(config, path):

    # ------------- Subject specific variables -------------
    motor_mode = config['gui-input-settings']['motor-mode']
    dimension = config['gui-input-settings']['dimension-mode']
    subject_id = config['gui-input-settings']['subject-id']
    session = str(config['gui-input-settings']['n-session'])
    run = str(config['gui-input-settings']['n-run'])
    # ------------------------------------------------------

    file_name = subject_id + '_run' + run + '_' + motor_mode + '_' + dimension
    xdf_file_path = path + subject_id + '-ses' + session + '/' + file_name + '.xdf'
    mat_file_path = path + subject_id + '-ses' + session + '/' + file_name + '.mat'

    if os.path.exists(mat_file_path):
        print(mat_file_path + " already exists.")

    else:
        if not os.path.exists(xdf_file_path):
            print(xdf_file_path + " does not exist.")
            os.makedirs(path + subject_id + '-ses' + session + '/')

        try:
            # Extract the eeg and marker lsl stream from the xdf file
            stream_eeg, stream_marker = load_xdf(xdf_file_path, config['general-settings']['lsl-streams'])

             # Add a row to the eeg data for the class labels
            eeg_and_label = add_class_labels(stream_eeg, stream_marker)

            save_to_mat(mat_file_path, 'data', eeg_and_label)
            print(mat_file_path + ' created.')
        except:
            # Sometimes .xdf file has an invalid footer and cannot be handled the same way as other .xdf files
            print('ERROR: Could not save ' + mat_file_path)
'''
def xdf_to_mat_file(config, path):

    # ------------- Subject specific variables -------------
    motor_mode = config['gui-input-settings']['motor-mode']
    dimension = config['gui-input-settings']['dimension-mode']
    subject_id = config['gui-input-settings']['subject-id']
    session = str(config['gui-input-settings']['n-session'])
    run = str(config['gui-input-settings']['n-run'])
    feedback = config['gui-input-settings']['fb-mode']
    # ------------------------------------------------------

    out_dir = path + 'dataset'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    xdf_file_name = subject_id + '_run' + str(run) + '_' + motor_mode + '_' + dimension + '.xdf'
    xdf_file_path = path + xdf_file_name

    # Read XDF and save as mat
    if feedback == "on":
        stream_eeg, stream_marker, stream_erds, stream_lda = load_xdf(config, xdf_file_path, feedback)

        erds_with_labels = add_class_labels(stream_erds, stream_marker)
        xdf_file_erds = out_dir + '/erds' + '_run' + run + "_" + motor_mode + '.mat'
        if not os.path.exists(xdf_file_erds):
            save_to_mat(xdf_file_erds, 'erds', erds_with_labels)
        if os.path.exists(xdf_file_erds):
            print(xdf_file_erds + ' already exists.')

        lda_with_labels = add_class_labels(stream_lda, stream_marker)
        xdf_file_lda = out_dir + '/lda' + '_run' + run + "_" + motor_mode + '.mat'
        if not os.path.exists(xdf_file_lda):
            save_to_mat(xdf_file_lda, 'lda', lda_with_labels)
        if os.path.exists(xdf_file_lda):
            print(xdf_file_lda + ' already exists.')
    else:
        stream_eeg, stream_marker = load_xdf(config, xdf_file_path, feedback)

    eeg_with_labels = add_class_labels(stream_eeg, stream_marker)
    xdf_file_eeg = out_dir + '/eeg' + '_run' + run + "_" + motor_mode + '.mat'
    if not os.path.exists(xdf_file_eeg):
        save_to_mat(xdf_file_eeg, 'eeg', eeg_with_labels)
    if os.path.exists(xdf_file_eeg):
        print(xdf_file_eeg + ' already exists.')

