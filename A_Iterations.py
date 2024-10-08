# ----------------------------------------------------------------------------------------------------------------------
# Iterate over subjcects, sessions and runs
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import mne
import os
import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime
from B_preprocessing import Measurement_Data
import C_ERDS, D_statistics

######## WHAT TO DO ########
preproc = False  # create raw, preproc, and epochs (raw, alpha, beta) for all subjects
erds_per_subj = True  # iterate to combine epochs plot ERDS maps per subject
topo_per_condition = False  # combine epochs of all subjects to evoked for conditions - plot topo map
calc_avg_erds_values = False  # iterate to combine epochs per condition and calculate avg ERDS
save_stat_erds = False  # sort and save data for statistical analysis from ERDS values
plt_inter_intra = False  # plot inter- and intra-individual ERDS values
calc_spearman = False  # calculate and plot spearman correlation matrix
plot_subject_survey = False  # plot results of participant survey
############################


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
    ''' # some subjects
    subject_list = ['S1', 'S2', 'S3', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17']
    mon_mi = [2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2]
    vr_mi = [1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1]
    '''
    ''' # single subject
    subject_list = ['S1']
    mon_mi = [2]
    vr_mi = [1]
    '''
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    roi = ["F3", "F4", "C3", "C4", "P3", "P4"]
    task = ['left', 'right']
    freq = 'alpha'
    freq_band=[8, 13]
    #freq = 'beta'
    #freq_band = [16, 24]


    # %% create raw, preproc, and epochs (raw, alpha, beta) for all subjects
    # prerequisite: config files and xdf files per run in subject_data_path
    # results: saved files (-original-raw.fif, -preproc-raw.fif, -epo.fif, -alpha-epo.fif, -beta-epo.fif)
    if preproc:
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


    # %% iterate to combine epochs plot ERDS maps per subject
    # prerequisite: epoch files in interim_subj_path
    # results: ERDS maps at interim_subj_path+plots/
    if erds_per_subj:
        epochs_counter = []
        for subj_ix, subj in enumerate(subject_list):
            print(f"\n \n --------------- Subject {subj} ---------------")
            interim_subj_path = interim_path + subj + '/'
            epochs_VR = []
            epochs_Mon = []
            epochs_Con = []
            epoch_files_raw, epoch_files_alpha, epoch_files_beta = find_epoch_files(interim_subj_path)
            for cur_epoch in epoch_files_raw:
                print(f"\n \n --------------- Epochs from {cur_epoch} ---------------")
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
            epochs_str = [subj_ix+1, len(combined_epochs_Con), len(combined_epochs_Mon), len(combined_epochs_VR)]
            epochs_counter.append(epochs_str)
            #'''
            C_ERDS.plot_erds_maps(combined_epochs_VR, picks=roi, t_min=0, t_max=11.25, path=interim_subj_path+'plots/',
                                  session='3D', subject=subj, cluster_mode=True)
            C_ERDS.plot_erds_maps(combined_epochs_Mon, picks=roi, t_min=0, t_max=11.25, path=interim_subj_path+'plots/',
                                  session='2D', subject=subj, cluster_mode=True)
            C_ERDS.plot_erds_maps(combined_epochs_Con, picks=roi, t_min=0, t_max=11.25, path=interim_subj_path+'plots/',
                                  session='Control', subject=subj, cluster_mode=True)
            #'''
        epochs_counter = np.array(epochs_counter)
        np.savetxt(interim_path + 'epochs_counter.txt', epochs_counter, delimiter=',', fmt='%d')

    # %% combine epochs to evoked for conditions - plot topo map
    # prerequisite: epoch files in interim_subj_path
    # results: topo plots at interim_path
    if topo_per_condition:
        all_evoked_left_VR = []
        all_evoked_right_VR = []
        all_evoked_left_Mon = []
        all_evoked_right_Mon = []
        all_evoked_left_Con = []
        all_evoked_right_Con = []

        # Iterations over all files to calculate evoked results
        for subj_ix, subj in enumerate(subject_list):
            print(f"\n \n --------------- Subject {subj} ---------------")
            interim_subj_path = interim_path + subj + '/'
            epoch_files_raw, epoch_files_alpha, epoch_files_beta = find_epoch_files(interim_subj_path)
            if freq == 'alpha':
                epochs = epoch_files_alpha
            elif freq == 'beta':
                epochs = epoch_files_beta
            else:
                epochs = epoch_files_raw
            for cur_epoch in epochs:
                print(f"\n \n --------------- Epochs from {cur_epoch} ---------------")
                epoch = mne.read_epochs(cur_epoch, preload=True)
                evoked_left = epoch['left'].average()
                evoked_right = epoch['right'].average()
                if cur_epoch.startswith(interim_subj_path + 'sesVR'):
                    all_evoked_left_VR.append(evoked_left)
                    all_evoked_right_VR.append(evoked_right)
                if cur_epoch.startswith(interim_subj_path + 'sesMonitor'):
                    all_evoked_left_Mon.append(evoked_left)
                    all_evoked_right_Mon.append(evoked_right)
                if cur_epoch.startswith(interim_subj_path + 'sesControl'):
                    all_evoked_left_Con.append(evoked_left)
                    all_evoked_right_Con.append(evoked_right)

        evoked_left_VR = mne.combine_evoked(all_evoked_left_VR, weights='nave')
        evoked_right_VR = mne.combine_evoked(all_evoked_right_VR, weights='nave')
        evoked_left_Mon = mne.combine_evoked(all_evoked_left_Mon, weights='nave')
        evoked_right_Mon = mne.combine_evoked(all_evoked_right_Mon, weights='nave')
        evoked_left_Con = mne.combine_evoked(all_evoked_left_Con, weights='nave')
        evoked_right_Con = mne.combine_evoked(all_evoked_right_Con, weights='nave')

        C_ERDS.plot_erds_topo(evoked_left_VR, evoked_right_VR, freq, freq_band, (1.5, 3), [4.25, 6],
                              interim_path, 'VR')
        C_ERDS.plot_erds_topo(evoked_left_Mon, evoked_right_Mon, freq, freq_band, (1.5, 3), [4.25, 6],
                              interim_path, 'Monitor')
        C_ERDS.plot_erds_topo(evoked_left_Con, evoked_right_Con, freq, freq_band, (1.5, 3), [4.25, 6],
                              interim_path, 'Control')


    # %% iterate to combine epochs per condition and calculate avg ERDS
    # prerequisite: epoch files in interim_subj_path
    # results:  file per condition (VR, Monitor) for erds per subject and ROI
    #           file per condition (ME, MI) for erds per subject averaged oder ROI
    if calc_avg_erds_values:
        avg_erds_VR_l = np.zeros((len(subject_list), len(roi)))
        avg_erds_VR_r = np.zeros((len(subject_list), len(roi)))
        avg_erds_Mon_l = np.zeros((len(subject_list), len(roi)))
        avg_erds_Mon_r = np.zeros((len(subject_list), len(roi)))

        avg_erds_ME_l = np.zeros((len(subject_list), 1))
        avg_erds_ME_r = np.zeros((len(subject_list), 1))
        avg_erds_MI_l = np.zeros((len(subject_list), 1))
        avg_erds_MI_r = np.zeros((len(subject_list), 1))
        for subj_ix, subj in enumerate(subject_list):
            print(f"\n \n --------------- Subject {subj} ---------------")
            interim_subj_path = interim_path + subj + '/'
            epochs_VR = []
            epochs_Mon = []
            epochs_Con = []
            epoch_files_raw, epoch_files_alpha, epoch_files_beta = find_epoch_files(interim_subj_path)
            if freq == 'alpha':
                epochs = epoch_files_alpha
            elif freq == 'beta':
                epochs = epoch_files_beta
            else:
                epochs = epoch_files_raw
            for cur_epoch in epochs:
                print(f"\n \n --------------- Epochs from {cur_epoch} ---------------")
                epoch = mne.read_epochs(cur_epoch, preload=True)
                if cur_epoch.startswith(interim_subj_path + 'sesVR'):
                    epochs_VR.append(epoch)
                if cur_epoch.startswith(interim_subj_path + 'sesMonitor'):
                    epochs_Mon.append(epoch)
                if cur_epoch.startswith(interim_subj_path + 'sesControl'):
                    epochs_Con.append(epoch)

            combined_epochs_VR = mne.concatenate_epochs(epochs_VR)
            combined_epochs_Mon = mne.concatenate_epochs(epochs_Mon)
            combined_epochs_ME = mne.concatenate_epochs(epochs_Con)
            combined_epochs_MI = mne.concatenate_epochs(epochs_Mon + epochs_VR)

            # per ROI
            avg_erds_VR_l[subj_ix, :], avg_erds_VR_r[subj_ix, :] = C_ERDS.calc_avg_erds_per_subj(combined_epochs_VR,
                                                                                                picks=roi, start_time=4.25,
                                                                                                end_time=11.25,
                                                                                                freq=freq_band)
            avg_erds_Mon_l[subj_ix, :], avg_erds_Mon_r[subj_ix, :] = C_ERDS.calc_avg_erds_per_subj(combined_epochs_Mon,
                                                                                                  picks=roi, start_time=4.25,
                                                                                                  end_time=11.25,
                                                                                                  freq=freq_band)
            # averaged ROIs
            avg_erds_ME_l[subj_ix, :], avg_erds_ME_r[subj_ix, :] = C_ERDS.calc_avg_erds_per_subj(combined_epochs_ME,
                                                                                                picks=roi, start_time=4.25,
                                                                                                end_time=11.25,
                                                                                                freq=freq_band,
                                                                                                avg_rois=True)
            avg_erds_MI_l[subj_ix, :], avg_erds_MI_r[subj_ix, :] = C_ERDS.calc_avg_erds_per_subj(combined_epochs_MI,
                                                                                                picks=roi, start_time=4.25,
                                                                                                end_time=11.25,
                                                                                                freq=freq_band,
                                                                                                avg_rois=True)
        # save per ROI
        np.save(interim_path + f'magnitudes_erds_VR_l_{freq}.npy', avg_erds_VR_l)
        np.save(interim_path + f'magnitudes_erds_VR_r_{freq}.npy', avg_erds_VR_r)
        np.save(interim_path + f'magnitudes_erds_Mon_l_{freq}.npy', avg_erds_Mon_l)
        np.save(interim_path + f'magnitudes_erds_Mon_r_{freq}.npy', avg_erds_Mon_r)
        # save averaged ROIs
        np.save(interim_path + f'magnitudes_erds_ME_l_{freq}.npy', avg_erds_ME_l)
        np.save(interim_path + f'magnitudes_erds_ME_r_{freq}.npy', avg_erds_ME_r)
        np.save(interim_path + f'magnitudes_erds_MI_l_{freq}.npy', avg_erds_MI_l)
        np.save(interim_path + f'magnitudes_erds_MI_r_{freq}.npy', avg_erds_MI_r)


    # %% sort and save data for statistical analysis from ERDS values
    # prerequisite: file per condition (VR, Monitor) for erds per subject and ROI and
    #               file per condition (ME, MI) for erds per subject averaged oder ROI
    # results: txt file with data for statistical analysis incl header
    if save_stat_erds:
        #''' # Calculate Results for statistiscal analysis
        # load per ROI
        avg_erds_VR_l = np.load(interim_path + f'magnitudes_erds_VR_l_{freq}.npy')
        avg_erds_VR_r = np.load(interim_path + f'magnitudes_erds_VR_r_{freq}.npy')
        avg_erds_Mon_l = np.load(interim_path + f'magnitudes_erds_Mon_l_{freq}.npy')
        avg_erds_Mon_r = np.load(interim_path + f'magnitudes_erds_Mon_r_{freq}.npy')
        # load averaged ROIs
        avg_erds_ME_l = np.load(interim_path + f'magnitudes_erds_ME_l_{freq}.npy')
        avg_erds_ME_r = np.load(interim_path + f'magnitudes_erds_ME_r_{freq}.npy')
        avg_erds_MI_l = np.load(interim_path + f'magnitudes_erds_MI_l_{freq}.npy')
        avg_erds_MI_r = np.load(interim_path + f'magnitudes_erds_MI_r_{freq}.npy')

        cond = ['2D', '3D'] # results for ANOVA 2x2x6 (2D/2D, MI L/MI R, 6 ROIs)
        header_VTR = []
        for ses in cond:
            for tsk in task:
                for ch in roi:
                    descr = ses + ' MI ' + tsk + ' ' + ch
                    header_VTR.append(descr)

        result_erds_VTR = np.concatenate((avg_erds_Mon_l, avg_erds_Mon_r, avg_erds_VR_l, avg_erds_VR_r), axis=1)*100
        results_file = interim_path + 'ERDS_results_' + freq + '.txt'
        header_str = ','.join(header_VTR)
        with open(results_file, 'w') as file:
            file.write(header_str + "\n")
        with open(results_file, 'a') as file:
            np.savetxt(file, result_erds_VTR, delimiter=',')

        cond = ['ME', 'MI']  # results for t-test for paired samples (MI vs. ME, l vs. r) without ROIs
        header_MI_ME = []
        for ses in cond:
            for tsk in task:
                descr = ses + ' ' + tsk
                header_MI_ME.append(descr)

        result_erds_MI_ME = np.concatenate((avg_erds_ME_l, avg_erds_ME_r, avg_erds_MI_l, avg_erds_MI_r), axis=1)*100
        results_file = interim_path + 'ERDS_results_MI_ME_' + freq + '.txt'
        header_str = ','.join(header_MI_ME)
        with open(results_file, 'w') as file:
            file.write(header_str + "\n")
        with open(results_file, 'a') as file:
            np.savetxt(file, result_erds_MI_ME, delimiter=',')

        D_statistics.plot_box_MI_ME(result_erds_MI_ME, header_MI_ME, freq, path=interim_path)
        D_statistics.plot_EMM(result_erds_VTR, header_VTR, freq, path=interim_path)

    # %% plot inter- and intra-individual ERDS values
    # prerequisite: file per condition (VR, Monitor) and frequency (alpha, beta) for erds per subject and ROI
    # results: plot of inter- and intra-individual ERDS values at interim_path

    if plt_inter_intra:

        # load per ROI
        avg_erds_VR_l_a = np.load(interim_path + f'magnitudes_erds_VR_l_alpha.npy')
        avg_erds_VR_r_a = np.load(interim_path + f'magnitudes_erds_VR_r_alpha.npy')
        avg_erds_Mon_l_a = np.load(interim_path + f'magnitudes_erds_Mon_l_alpha.npy')
        avg_erds_Mon_r_a = np.load(interim_path + f'magnitudes_erds_Mon_r_alpha.npy')
        avg_erds_VR_l_b = np.load(interim_path + f'magnitudes_erds_VR_l_beta.npy')
        avg_erds_VR_r_b = np.load(interim_path + f'magnitudes_erds_VR_r_beta.npy')
        avg_erds_Mon_l_b = np.load(interim_path + f'magnitudes_erds_Mon_l_beta.npy')
        avg_erds_Mon_r_b = np.load(interim_path + f'magnitudes_erds_Mon_r_beta.npy')

        list_avg_erds = [[avg_erds_Mon_l_a, avg_erds_Mon_l_b], [avg_erds_Mon_r_a, avg_erds_Mon_r_b],
                         [avg_erds_VR_l_a, avg_erds_VR_l_b], [avg_erds_VR_r_a, avg_erds_VR_r_b]]
        sessions = ['2D', '2D', '3D', '3D']
        cls = ['left', 'right', 'left', 'right']
        frequencies = ['alpha', 'beta']
        C_ERDS.plot_inter_intra_erds_subplot(list_avg_erds, roi, sessions, cls, frequencies, path=interim_path)


    # %% calculate and plot spearman correlation matrix
    # prerequisite: file per condition (VR, Monitor) for erds per subject and ROI
    # results: plot of distance matrix (1-rho, ranked, scaled to 0-1)
    if calc_spearman:
        # load per ROI
        avg_erds_VR_l = np.load(interim_path + f'magnitudes_erds_VR_l_{freq}.npy')
        avg_erds_VR_r = np.load(interim_path + f'magnitudes_erds_VR_r_{freq}.npy')
        avg_erds_Mon_l = np.load(interim_path + f'magnitudes_erds_Mon_l_{freq}.npy')
        avg_erds_Mon_r = np.load(interim_path + f'magnitudes_erds_Mon_r_{freq}.npy')

        list_avg_erds = [avg_erds_Mon_l, avg_erds_Mon_r, avg_erds_VR_l, avg_erds_VR_r]
        sessions = ['2D', '3D']

        """C_ERDS.run_spearman(avg_erds_VR_l, subject_list, 'VR left hand', freq, interim_path)
        C_ERDS.run_spearman(avg_erds_VR_r, subject_list, 'VR right hand', freq, interim_path)
        C_ERDS.run_spearman(avg_erds_Mon_l, subject_list, 'Monitor left hand', freq, interim_path)
        C_ERDS.run_spearman(avg_erds_Mon_r, subject_list, 'Monitor right hand', freq, interim_path)"""

        C_ERDS.run_spearman_subplot(list_avg_erds, subject_list, sessions, task, freq, path=interim_path)

    # %% plot results of
    # prerequisite: file per condition (VR, Monitor) for erds per subject and ROI
    # results: plot of distance matrix (1-rho, ranked, scaled to 0-1)
    # plot results of participant survey
    if plot_subject_survey:
        monitor = [7, 5, 8]
        vr = [10, 12, 9]
        D_statistics.plot_participant_survey(monitor, vr, path=interim_path)
    #winsound.Beep(750, 1000)

