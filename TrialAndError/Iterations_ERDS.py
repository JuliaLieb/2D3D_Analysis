# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime
import winsound
from signal_preprocessing import Measurement_Data
import signal_preprocessing
import mne
from matplotlib.colors import TwoSlopeNorm


def find_config_files(path, subject_id):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files

# %% general conditions
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
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    roi = ["F3", "F4", "C3", "C4", "P3", "P4"]
    task = ['MI left', 'MI right']
    freq = 'alpha'
    #freq_band = [8,12]
    #freq = 'beta'
    #freq_band = [16, 24]

    ### ----- DATA FOR TESTING ----- ###
    #'''
    subject_list = ['S14']
    mon_mi = [2]
    vr_mi = [1]
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    #'''

    #"""
    # %% iterate to calculate and save preprocessed, epochs and evoked results
    # Iterations over all files to calculate evoked results
    for subj_ix, subj in enumerate(subject_list):
        interim_subj_path = interim_path + subj + '/'
        if not os.path.exists(interim_subj_path):
            os.makedirs(interim_subj_path)

        for ses_key, ses_values in conditions.items():
            ses_ix = ses_values[subj_ix]
            subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
            config_files = find_config_files(subject_data_path, subject_id=subj)
            all_evoked_left = []
            all_evoked_right = []
            for cur_config in config_files:
                print(f'\n\n---------------Current path: {cur_config}--------------- \n')
                try:
                    data = Measurement_Data(cur_config, subject_data_path)
                    evoked_left, evoked_right = data.run_preprocessing_to_epoch(ses_key, freq_band=True)
                    all_evoked_left.append(evoked_left)
                    all_evoked_right.append(evoked_right)
                    print("Debug")
                except Exception as e:
                    print(f'Error processing subject {subj}, session {ses_ix}, '
                          f'file {cur_config} . Exception: {e}')
                    continue
            ev_left = mne.combine_evoked(all_evoked_left, 'nave')
            ev_right = mne.combine_evoked(all_evoked_right, 'nave')
            ev_left.save(results_path + subj + '-ses' + ses_key + '_' + freq + '-left-ave.fif', overwrite=True)
            ev_right.save(results_path + subj + '-ses' + ses_key + '_' + freq + '-right-ave.fif', overwrite=True)

            #fig_l = ev_left.plot_topomap(ch_type="eeg")
            #fig_l.savefig(subject_data_path + 'plots/left_topo_ses' + ses_key + '_' + freq + '.png', format='png')
            #plt.close(fig_l)

            #fig_r = ev_right.plot_topomap(ch_type="eeg")
            #fig_r.savefig(subject_data_path + 'plots/right_topo_ses' + ses_key + '_' + freq + '.png', format='png')
            #plt.close(fig_r)
            winsound.Beep(600, 500)
    #"""

# %% combine evoked for conditions
    #"""
    all_evoked_left_VR = []
    all_evoked_right_VR = []
    all_evoked_left_Mon = []
    all_evoked_right_Mon = []
    all_evoked_left_Con = []
    all_evoked_right_Con = []

    # Iterations over all files to calculate evoked results
    for subj_ix, subj in enumerate(subject_list):
        interim_subj_path = interim_path + subj + '/'
        for ses_key, ses_values in conditions.items():
            ses_ix = ses_values[subj_ix]
            subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
            ev_left = mne.read_evokeds(results_path + subj + '-ses' + ses_key + '_' + freq + '-left-ave.fif')[0]
            ev_right = mne.read_evokeds(results_path + subj + '-ses' + ses_key + '_' + freq + '-right-ave.fif')[0]
            if ses_key == 'VR':
                all_evoked_left_VR.append(ev_left)
                all_evoked_right_VR.append(ev_right)
            elif ses_key == 'Monitor':
                all_evoked_left_Mon.append(ev_left)
                all_evoked_right_Mon.append(ev_right)
            elif ses_key == 'Control':
                all_evoked_left_Con.append(ev_left)
                all_evoked_right_Con.append(ev_right)
    evoked_left_VR = mne.combine_evoked(all_evoked_left_VR, weights='nave')
    evoked_right_VR = mne.combine_evoked(all_evoked_right_VR, weights='nave')
    evoked_left_Mon = mne.combine_evoked(all_evoked_left_Mon, weights='nave')
    evoked_right_Mon = mne.combine_evoked(all_evoked_right_Mon, weights='nave')
    evoked_left_Con = mne.combine_evoked(all_evoked_left_Con, weights='nave')
    evoked_right_Con = mne.combine_evoked(all_evoked_right_Con, weights='nave')

    #fig_VR_l = evoked_left_VR.plot_topomap(ch_type="eeg")
    #fig_VR_l = evoked_left_VR.plot_topomap(ch_type="eeg", average=4.6, times=[6.6, 8.9]) # zwei bilder pro fb
    fig_VR_l = evoked_left_VR.plot_topomap(ch_type="eeg", average=7, times=[7.75], size=2, sphere='eeglab', cmap='RdBu_r', contours=10)
    fig_VR_l.savefig(results_path + timestamp + '_topo_VR_left_' + freq + '.png', format='png')
    #fig_VR_r = evoked_right_VR.plot_topomap(ch_type="eeg")
    fig_VR_r = evoked_right_VR.plot_topomap(ch_type="eeg", average=7, times=[7.75], size=2, sphere='eeglab', cmap='RdBu_r', contours=10)
    fig_VR_r.savefig(results_path + timestamp + '_topo_VR_right_' + freq + '.png', format='png')

    #fig_Mon_l = evoked_left_Mon.plot_topomap(ch_type="eeg")
    fig_Mon_l = evoked_left_Mon.plot_topomap(ch_type="eeg", average=7, times=[7.75], size=2, sphere='eeglab', cmap='RdBu_r', contours=10)
    fig_Mon_l.savefig(results_path + timestamp + '_topo_Mon_left_' + freq + '.png', format='png')
    #fig_Mon_r = evoked_right_Mon.plot_topomap(ch_type="eeg")
    fig_Mon_r = evoked_right_Mon.plot_topomap(ch_type="eeg", average=7, times=[7.75], size=2, sphere='eeglab', cmap='RdBu_r', contours=10)
    fig_Mon_r.savefig(results_path + timestamp + '_topo_Mon_right_' + freq + '.png', format='png')

    #fig_Con_l = evoked_left_Con.plot_topomap(ch_type="eeg")
    fig_Con_l = evoked_left_Con.plot_topomap(ch_type="eeg", average=7, times=[7.75], size=2, sphere='eeglab', cmap='RdBu_r', contours=10)
    fig_Con_l.savefig(results_path + timestamp + '_topo_Con_left_' + freq + '.png', format='png')
    #fig_Con_r = evoked_right_Con.plot_topomap(ch_type="eeg")
    fig_Con_r = evoked_right_Con.plot_topomap(ch_type="eeg", average=7, times=[7.75], size=2, sphere='eeglab', cmap='RdBu_r', contours=10)
    fig_Con_r.savefig(results_path + timestamp + '_topo_Con_right_' + freq + '.png', format='png')
    #"""

    #winsound.Beep(750, 1000)
    #winsound.Beep(550, 1000)