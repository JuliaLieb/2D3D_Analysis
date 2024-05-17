import json
import os

import numpy as np

import navigate_data
import EEG_analysis
import ERDS_analysis
import analysis
from offline_analysis import EEG_Signal
from signal_reading import Input_Data
import segmentation
from mne.datasets import eegbci
import calc_values
import mne
import create_epochs
import matplotlib.pyplot as plt


def find_config_files(path):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files


def save_epochs_per_run(config_files, path, signal, run=True):
    if run:
        for config in config_files:
            input_data = Input_Data(config, path)
            input_data.run_raw(signal=signal)
    else:
        return

def combine_epochs(n_runs, path, sig='raw'):
    epochs_left = []
    epochs_right = []
    for run in range(n_runs):
        filename_l = path + '/epochs/l_run' + str(run + 1) + '-' + sig + '-epo.fif'
        filename_r = path + '/epochs/r_run' + str(run + 1) + '-' + sig + '-epo.fif'
        if os.path.exists(filename_l):
            epochs_left.append(mne.read_epochs(filename_l, preload=True))
        if os.path.exists(filename_r):
            epochs_right.append(mne.read_epochs(filename_r, preload=True))
    epochs_left = mne.concatenate_epochs(epochs_left)
    epochs_right = mne.concatenate_epochs(epochs_right)

    return epochs_left, epochs_right

def calculate_erds_per_roi(n_runs, subject_data_path, plot_epochs=False):
    epochs_left, epochs_right = combine_epochs(n_runs, subject_data_path, sig='raw')
    if plot_epochs:
        epochs_left.compute_psd().plot()
        epochs_right.compute_psd().plot()

    # calculate erds per ROI
    erds_l_per_roi = calc_values.run(epochs_left)
    erds_r_per_roi = calc_values.run(epochs_right)

    return np.column_stack((erds_l_per_roi, erds_r_per_roi))


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'

    # ----- Define Subject and Session ----
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                  'S16', 'S17', 'S20']
    subject_list = ['S20']
    session_list = [0, 1, 2]

    #subject_id = subject_list[17]
    subject_id = subject_list[0] # =subject ID -1
    session_id = session_list[2]

    create_epoch_files = True
    create_epoch_files = False
    calculate_results = True
    #calculate_results = False
    sig = ['raw', 'alpha', 'beta'][0]
    # -------------------------------------

    subject_data_path = data_path + subject_id + '-ses' + str(session_id) + '/'
    config_files = find_config_files(subject_data_path)

    # preprocess, create and save epochs for each run
    save_epochs_per_run(config_files, subject_data_path, signal=sig, run=False)

    erds = calculate_erds_per_roi(len(config_files), subject_data_path, plot_epochs=False)


