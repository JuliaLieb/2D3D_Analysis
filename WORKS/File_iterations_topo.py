# ----------------------------------------------------------------------------------------------------------------------
# Read data of preprocessed .fif file and process to results
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


def find_config_files(path, subject_id):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files

def find_epoch_files(path, suffix):
    epoch_files = []
    for file in os.listdir(path):
        if file.endswith(suffix + '-epo.fif'):
            epoch_files.append(path + file)
    return epoch_files

def create_epochs(raw, tmin=0, tmax=11.25, reject_criteria={"eeg": 0.0002}):
    """
    Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
    the raw data according to the events
    :param visualize_epochs: boolean variable to select if generate epochs plots or not
    :param rois: boolean variable to select if visualize results according to the rois or for each channel
    :param set_annotations: boolean variable, if it's necessary to set the annotations or not
    """

    # set the annotations on the current raw and extract the correspondent events
    raw.set_annotations(raw.annotations)
    events, event_mapping = mne.events_from_annotations(raw)

    # generation of the epochs according to the events
    epochs = mne.Epochs(signal, events, preload=True, baseline=(tmin, 0),
                             reject=reject_criteria, tmin=tmin, tmax=tmax)  # event_id=self.event_mapping,


    return epochs

def run_epoch_to_evoked(file_path):

    epochs = mne.read_epochs(file_path, preload=True)

    evoked_left = epochs['1'].average()
    # evoked_left.plot_topomap(ch_type="eeg")
    evoked_right = epochs['2'].average()
    # evoked_right.plot_topomap(ch_type="eeg")

    return evoked_left, evoked_right


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/../Data/'
    results_path = cwd + '/../Results/'
    interim_path = cwd + '/../InterimResults/'
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
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    roi = ["F3", "F4", "C3", "C4", "P3", "P4"]
    task = ['MI left', 'MI right']
    freq = 'alpha'
    #freq = 'beta'

    ### ----- DATA FOR TESTING ----- ###
    '''
    subject_list = ['S14']
    mon_mi = [2]
    vr_mi = [1]
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    '''

    # %% iterate to calculate and save preprocessed, epochs and evoked results
    # Iterations over all files to calculate evoked results

    for subj_ix, subj in enumerate(subject_list):
        interim_subj_path = interim_path + subj + '/'
        if not os.path.exists(interim_subj_path):
            os.makedirs(interim_subj_path)
        for ses_key, ses_values in conditions.items():
            ses_ix = ses_values[subj_ix]
            subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
            epoch_files = find_epoch_files(interim_subj_path, freq)
            all_evoked_left_VR = []
            all_evoked_right_VR = []
            all_evoked_left_Mon = []
            all_evoked_right_Mon = []
            all_evoked_left_Con = []
            all_evoked_right_Con = []
            for run, cur_epoch in enumerate(epoch_files):
                print(f'\n\n---------------Current path: {interim_subj_path}--------------- \n')
                try:
                    if cur_epoch.startswith(interim_subj_path + 'sesVR'):
                        evoked_left, evoked_right = run_epoch_to_evoked(cur_epoch)
                        all_evoked_left_VR.append(evoked_left)
                        all_evoked_right_VR.append(evoked_right)
                    if cur_epoch.startswith(interim_subj_path + 'sesMonitor'):
                        evoked_left, evoked_right = run_epoch_to_evoked(cur_epoch)
                        all_evoked_left_Mon.append(evoked_left)
                        all_evoked_right_Mon.append(evoked_right)
                    if cur_epoch.startswith(interim_subj_path + 'sesControl'):
                        evoked_left, evoked_right = run_epoch_to_evoked(cur_epoch)
                        all_evoked_left_Con.append(evoked_left)
                        all_evoked_right_Con.append(evoked_right)
                except Exception as e:
                    print(f'Error processing subject {subj}, session {ses_ix}, '
                          f'file {cur_epoch} . Exception: {e}')
                    continue
            ev_left_VR = mne.combine_evoked(all_evoked_left_VR, 'nave')
            ev_right_VR = mne.combine_evoked(all_evoked_right_VR, 'nave')
            ev_left_VR.save(results_path + subj + '-sesVR_' + freq + '-left-ave.fif', overwrite=True)
            ev_right_VR.save(results_path + subj + '-sesVR_' + freq + '-right-ave.fif', overwrite=True)
            ev_left_Mon = mne.combine_evoked(all_evoked_left_Mon, 'nave')
            ev_right_Mon = mne.combine_evoked(all_evoked_right_Mon, 'nave')
            ev_left_Mon.save(results_path + subj + '-sesMonitor_' + freq + '-left-ave.fif', overwrite=True)
            ev_right_Mon.save(results_path + subj + '-sesMonitor_' + freq + '-right-ave.fif', overwrite=True)
            ev_left_Con = mne.combine_evoked(all_evoked_left_Con, 'nave')
            ev_right_Con = mne.combine_evoked(all_evoked_right_Con, 'nave')
            ev_left_Con.save(results_path + subj + '-sesControl_' + freq + '-left-ave.fif', overwrite=True)
            ev_right_Con.save(results_path + subj + '-sesControl_' + freq + '-right-ave.fif', overwrite=True)

            #fig_l = ev_left.plot_topomap(ch_type="eeg")
            #fig_l.savefig(subject_data_path + 'plots/left_topo_ses' + ses_key + '.png', format='png')
            #fig_r = ev_right.plot_topomap(ch_type="eeg")
            #fig_r.savefig(subject_data_path + 'plots/right_topo_ses' + ses_key + '.png', format='png')
    # """

