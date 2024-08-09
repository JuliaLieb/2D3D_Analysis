# ----------------------------------------------------------------------------------------------------------------------
# Iterate over subjcects, sessions and runs
# ----------------------------------------------------------------------------------------------------------------------
import json
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
import mne
import os
import scipy
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.signal import iirfilter, sosfiltfilt, iirdesign, sosfilt_zi, sosfilt, butter, lfilter
from scipy import signal
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import winsound
from B_preprocessing import Measurement_Data
import C_ERDS


def find_config_files(path, subject_id):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files

def find_epoch_files(path):
    epoch_files_raw = []
    epoch_files_alpha = []
    epoch_files_beta = []
    for file in os.listdir(path):
        if file.endswith('-alpha-epo.fif'):
            epoch_files_alpha.append(path + file)
        elif file.endswith('-beta-epo.fif'):
            epoch_files_beta.append(path + file)
        elif file.endswith('-epo.fif'):
            epoch_files_raw.append(path + file)
    return epoch_files_raw, epoch_files_alpha, epoch_files_beta

def find_raw_files(path, suffix):
    raw_files = []
    for file in os.listdir(path):
        if file.endswith(suffix + '-raw.fif'):
            raw_files.append(path + file)
    return raw_files


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    results_path = cwd + '/Results/'
    interim_path = cwd + '/InterimResults/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")


    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # subjects and sessions
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                    'S16', 'S17']

    # study conditions
    mon_me = [0] * len(subject_list)
    mon_mi = [2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2] #S1-17
    vr_mi = [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1] #S1-17

    ### ----- DATA FOR TESTING ----- ###
    ''' # Subjects 14-17
    subject_list = ['S14', 'S15', 'S16', 'S17']
    mon_mi = [1, 2, 1, 2]
    vr_mi = [2, 1, 2, 1]
    '''
    #''' # Subject 14
    subject_list = ['S14']
    mon_mi = [1]
    vr_mi = [2]
    #'''
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    roi = ["F3", "F4", "C3", "C4", "P3", "P4"]
    task = ['MI left', 'MI right']
    #freq = 'alpha'
    #freq_band=[8, 12]
    freq = 'beta'
    freq_band = [16, 24]


# %% create raw, preproc, and epochs (raw, alpha, beta) for all subjects
    #"""
    for subj_ix, subj in enumerate(subject_list):
        interim_subj_path = interim_path + subj + '/'
        if not os.path.exists(interim_subj_path):
            os.makedirs(interim_subj_path)

        for ses_key, ses_values in conditions.items():
            ses_ix = ses_values[subj_ix]
            subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
            config_files = find_config_files(subject_data_path, subject_id=subj)
            for cur_config in config_files:
                print(f'\n\n---------------Current path: {cur_config}--------------- \n')
                try:
                    data = Measurement_Data(cur_config, subject_data_path)
                    data.run_preprocessing(plt=False)
                except Exception as e:
                    print(f'Error processing subject {subj}, session {ses_ix}, '
                          f'file {cur_config} . Exception: {e}')
                    continue

    #"""


    # %% iterate to combine epochs per subject and plot ERDS maps
    #"""
    for subj_ix, subj in enumerate(subject_list):
        interim_subj_path = interim_path + subj + '/'
        epochs_VR = []
        epochs_Mon = []
        epochs_Con = []
        for ses_key, ses_values in conditions.items():
            ses_ix = ses_values[subj_ix]
            subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
            epoch_files_raw, epoch_files_alpha, epoch_files_beta = find_epoch_files(interim_subj_path)
            for cur_epoch in epoch_files_raw:
                #print(f"\n \n --------------- Session {ses_key} Run {run} ---------------")
                epoch = mne.read_epochs(cur_epoch, preload=True)
                if cur_epoch.startswith(interim_subj_path + 'sesVR'):
                    epochs_VR.append(epoch)
                if cur_epoch.startswith(interim_subj_path + 'sesMonitor'):
                    epochs_Mon.append(epoch)
                if cur_epoch.startswith(interim_subj_path + 'sesControl'):
                    epochs_Con.append(epoch)

            combined_epochs_VR = mne.concatenate_epochs(epochs_VR)
            combined_epochs_Mon = mne.concatenate_epochs(epochs_Mon)
            combined_epochs_Con = mne.concatenate_epochs(epochs_Con)
            print(f'Number of epochs: Control - {len(combined_epochs_Con)}, Monitor - {len(combined_epochs_Mon)}, '
                  f'VR - {len(combined_epochs_VR)}')
            # '''
            C_ERDS.plot_erds_maps(combined_epochs_VR, picks=roi, t_min=0, t_max=11.25, path=interim_subj_path+'plots/',
                                  session='VR', cluster_mode=True)
            C_ERDS.plot_erds_maps(combined_epochs_Mon, picks=roi, t_min=0, t_max=11.25, path=interim_subj_path+'plots/',
                                  session='Monitor', cluster_mode=True)
            C_ERDS.plot_erds_maps(combined_epochs_Con, picks=roi, t_min=0, t_max=11.25, path=interim_subj_path+'plots/',
                                  session='Control', cluster_mode=True)

        # '''
        #compute_erds(combined_epochs_VR, roi, fs=500, t_min=0, f_min=0, f_max=50, path="C:/2D3D_Analysis/Results/")
    #"""

    winsound.Beep(750, 1000)

