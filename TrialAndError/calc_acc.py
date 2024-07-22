
import pyxdf
import scipy.io
import json
import numpy as np
import os
import scipy.io


def load_xdf(config, xdf_file_path, feedback):
    streams, fileheader = pyxdf.load_xdf(xdf_file_path)
    streams_info = []

    for stream in streams:
        streams_info.append(stream['info']['name'][0])

    streams_info = np.array(streams_info)

    lsl_config = config['general-settings']['lsl-streams']

    eeg_pos = np.where(streams_info == lsl_config['eeg']['name'])[0][0]
    marker_pos = np.where(streams_info == lsl_config['marker']['name'])[0][0]


    if not feedback:
        return streams[eeg_pos], streams[marker_pos]
    elif feedback:
        erds_pos = np.where(streams_info == lsl_config['fb-erds']['name'])[0][0]
        lda_pos = np.where(streams_info == lsl_config['fb-lda']['name'])[0][0]
        return streams[eeg_pos], streams[marker_pos], streams[erds_pos], streams[lda_pos]
def add_class_labels(stream, stream_marker):
    time = stream['time_stamps']
    marker_series = np.array(stream_marker['time_series'])
    cue_times = (stream_marker['time_stamps'])[np.where(marker_series == 'Cue')[0]]

    conditions = marker_series[np.where(np.char.find(marker_series[:, 0], 'Start_of_Trial') == 0)[0]]
    conditions[np.where(conditions == 'Start_of_Trial_l')] = 121
    conditions[np.where(conditions == 'Start_of_Trial_r')] = 122

    cue_positions = np.zeros((np.shape(time)[0], 1), dtype=int)
    for t, c in zip(cue_times, conditions):
        pos = find_nearest_index(time, t)
        cue_positions[pos] = c

    return np.append(cue_positions, stream['time_series'], axis=1)

def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="right")
    if idx == len(array):
        return idx - 1
    else:
        return idx

def save_to_mat(file_path, identifier, data):
    if os.path.isfile(file_path):
        print(file_path, ' append data')
        data_old = scipy.io.loadmat(file_path)[identifier]
        data = np.append(data_old, data, axis=0)

    scipy.io.savemat(file_path, {identifier: data})

def extract_epochs(data, n_samples):
    data_labels = data[0, :]
    data = data[1:, :]
    indexes = np.where((data_labels == 121) | (data_labels == 122))[0]

    epochs = []
    n_trials = len(indexes)
    for i in range(n_trials):
        idx1 = indexes[i]
        idx2 = idx1 + n_samples
        if i < n_trials-1 and idx2 > indexes[i+1]:
            idx2 = indexes[i+1]
        epochs.append(data[:, idx1:idx2])

    return epochs, np.array(data_labels[indexes] - 121, dtype=int)

def compute_accuracy(data, class_labels):

    acc_per_trial = []

    for epoch, cl in zip(data, class_labels):
        acc_per_trial.append(np.sum(epoch[0, :] == cl) / len(epoch[0, :]))

    return np.mean(acc_per_trial)

def calculate_accuracy(current_config_file, root_dir):

    with open(current_config_file) as json_file:
        config = json.load(json_file)

    sample_rate = config['eeg-settings']['sample-rate']
    duration_task = config['general-settings']['timing']['duration-task']
    n_samples_task = int(np.floor(sample_rate * duration_task))

    subject_id = config['gui-input-settings']['subject-id']
    n_session = str(config['gui-input-settings']['n-session'])
    run = str(config['gui-input-settings']['n-run'])
    motor_mode = str(config['gui-input-settings']['motor-mode'])
    visualization = str(config['gui-input-settings']['dimension-mode'])

    # for modality in modalities:
    #if int(run) == 1:
        # print("1st run only for training of classifier - no accuracy can be computed.")
    if not int(run) == 1:
        file = root_dir + subject_id + '-ses' + str(n_session) + '/online_streams_mat/' + 'lda' + '_run' + run + "_" + motor_mode + '.mat'
        data_lda = scipy.io.loadmat(file)['lda'].T
        data_lda, labels = extract_epochs(data_lda, n_samples_task)
        accuracy = compute_accuracy(data_lda, labels)
        accuracy = round(accuracy, 4)
        #print(visualization + ' ' + motor_mode + ' Run ' + run + ': accuracy in %: ', accuracy * 100)
        print('run ' + run + ': ', accuracy * 100)

        return accuracy
"""
if __name__ == "__main__":
    cwd = os.getcwd()
    # Read BCI Configuration
    config_file = cwd + '/bci-config.json'
    with open(config_file) as json_file:
        config = json.load(json_file)

    subject_id = config['gui-input-settings']['subject-id']
    n_session = config['gui-input-settings']['n-session']

    root_dir = cwd + '/SubjectData/'

    # convert all .xdf files which have corresponding .json file to .mat
    all_config_files = glob.glob(root_dir + subject_id + '-ses' + str(n_session) + '/*.json')
    for current_config_file in all_config_files:
        xdf_to_mat_file(current_config_file, cwd)
        #xdf_to_mat.xdf_to_mat(current_config_file)

        # for all available .json files: calculate accuracy
    for current_config_file in all_config_files:
        calculate_accuracy(current_config_file)
"""
