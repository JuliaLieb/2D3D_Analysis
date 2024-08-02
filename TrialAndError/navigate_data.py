
import json
import os
import glob

import xdf_to_mat


def save_xdf_to_mat(data_path, subject_list, session_list):
    """Saves all xdf files in the data_path to mat files.

    Parameters:
    'data_path': `str` The path to the data.
    'subject_list': `list` The list of subjects.
    'session_list': `list` The list of sessions.
    """
    
    for current_subject in subject_list:
        for current_session in session_list:
            subject_directory = data_path + current_subject + '-ses' + str(current_session) + '/'
            all_config_files = glob.glob(subject_directory + '*.json')

            for current_config_file in all_config_files:
                with open(current_config_file) as json_file:
                    config = json.load(json_file)
                #xdf_to_mat.xdf_to_mat_new(config, data_path)
                xdf_to_mat.xdf_to_mat_file(config, subject_directory)

def get_all_mat_xdf_file_names(data_path, subject_id, session_id):
    """Saves all xdf files in the data_path to mat files.

    Parameters:
    'data_path': `str` The path to the data.
    'subject_list': `list` The list of subjects.
    'session_list': `list` The list of sessions.

    Returns:
    'all_mat_files': `list` The list of all mat files.
    'all_xdf_files': `list` The list of all xdf files.
    """

    subject_directory = data_path + subject_id + '-ses' + str(session_id) + '/'
    mat_directory = subject_directory + 'dataset/'
    all_mat_files = []
    all_xdf_files = []
    all_config_files = []

    for r in range(1, 6): #max. 5 runs
        for file in os.listdir(subject_directory):
            if file.startswith(subject_id + '_run' + str(r)):
                if file.endswith('.mat'):
                    all_mat_files.append(file)
                if file.endswith('.xdf'):
                    all_xdf_files.append(file)
            if file.startswith('CONFIG_' + subject_id + '_run' + str(r)):
                if file.endswith('.json'):
                    all_config_files.append(file)
        for file in os.listdir(mat_directory):
            if file.startswith('eeg_run' + str(r)):
                all_mat_files.append(file)

    return all_mat_files, all_xdf_files, all_config_files

def get_specific_run(data_path, subject_id, session_id, run_id):
    """Saves all xdf files in the data_path to mat files.

    Parameters:
    'data_path': `str` The path to the data.
    'subject_list': `list` The list of subjects.
    'session_list': `list` The list of sessions.
    'run_id': `int` The run id.

    Returns:
    'current_mat': 'str' The current mat file.
    'current_xdf': 'str' The current xdf file.
    """
    all_mat_files, all_xdf_files, all_config_files = get_all_mat_xdf_file_names(data_path, subject_id, session_id)
    current_mat = [i for i in all_mat_files if i.startswith('eeg_run' + str(run_id))]
    current_xdf = [i for i in all_xdf_files if i.startswith(subject_id + '_run' + str(run_id))]
    current_config = [i for i in all_config_files if i.startswith('CONFIG_' + subject_id + '_run' + str(run_id))]

    return current_mat[0], current_xdf[0], current_config[0]




