import os
import numpy as np
from signal_reading import Input_Data
import ERDS_calculation
import mne
from colorama import init, Fore
import xdf_to_mat
import glob
import json
from calc_acc import calculate_accuracy
import sys
from datetime import datetime
import plt_compare


def find_config_files(path, subject_id):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files

def save_epochs_per_run(config_files, path):
    for config in config_files:
        input_data = Input_Data(config, path)
        input_data.run_raw() # raw data
        #input_data.run_preprocessing()  # preprocessed data

def combine_epochs(n_runs, path, task='r', sig='raw'):
    epochs = []
    for run in range(n_runs):
        filename = path + 'epochs/' + task + '_run' + str(run + 1) + '-' + sig + '-epo.fif'
        if os.path.exists(filename):
            epochs.append(mne.read_epochs(filename, preload=True))
        else:
            print(f'File not found: {filename}')
    epochs = mne.concatenate_epochs(epochs)

    return epochs

def calculate_erds_per_roi(n_runs, subject_data_path, task='r', sig='raw', plot_epochs=False):  #all runs
    epochs = combine_epochs(n_runs, subject_data_path, task=task, sig=sig)
    if plot_epochs:
        epochs.compute_psd().plot()

    roi_dict = {
        "frontal left": ["F3", "F7"],
        "frontal right": ["F4", "F8"],
        "central left": ["FC1", "FC5", "C3", "T7"],
        "central right": ["FC2", "FC6", "C4", "T8"],
        "parietal left": ["CP5", "CP1", "P3", "P7"],
        "parietal right": ["CP6", "CP2", "P4", "P8"]}  # ERDS mode = average

    # calculate erds per ROI
    erds_per_roi = ERDS_calculation.mean_erds_per_roi(epochs, roi_dict)

    return erds_per_roi

def calculate_erds_per_roi_per_run(path, run, task='r', sig='raw'):  #only one run - comparable to online results if average
    filename = path + '/epochs/' + task + '_run' + str(run) + '-' + sig + '-epo.fif'
    epochs = mne.read_epochs(filename, preload=True)

    roi_dict = {
        "frontal left": ["F3", "F7"],
        "frontal right": ["F4", "F8"],
        "central left": ["FC1", "FC5", "C3", "T7"],
        "central right": ["FC2", "FC6", "C4", "T8"],
        "parietal left": ["CP5", "CP1", "P3", "P7"],
        "parietal right": ["CP6", "CP2", "P4", "P8"]}  # ERDS mode = average

    # calculate erds per ROI
    erds_per_roi = ERDS_calculation.mean_erds_per_roi(epochs, roi_dict)

    return erds_per_roi

def calculate_erds_per_ch(path, run, task, sig='raw'): #only one run - comparable to online results if single
    filename = path + '/epochs/' + task + '_run' + str(run) + '-' + sig + '-epo.fif'
    epochs = mne.read_epochs(filename, preload=True)

    roi_dict = {
        "ROI1": ["F3"],
        "ROI2": ["F4"],
        "ROI3": ["C3"],
        "ROI4": ["C4"],
        "ROI5": ["P3"],
        "ROI6": ["P4"]} # ERDS mode = single

    erds_per_ch = ERDS_calculation.mean_erds_per_roi(epochs, roi_dict)

    return erds_per_ch

def load_current_file(subj, ses_ix, run):
    subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
    config_file_path = find_config_files(subject_data_path, subject_id=subj)[run-1]

    xdf_files = []
    for file in os.listdir(subject_data_path):
        if file.startswith(subj + '_run'):
            if file.endswith('.xdf'):
                xdf_files.append(subject_data_path + file)
    xdf_file_path = xdf_files[run-1]

    return subject_data_path, config_file_path, xdf_file_path

def load_curent_epoch(run, freq_band='raw'):
    epoch_path = subject_data_path + 'epochs/'
    epoch_files_l, epoch_files_r = None, None
    for file in os.listdir(epoch_path):
        if file.startswith('l_run' + str(run) + '-' + freq_band):
            if file.endswith('-epo.fif'):
                epoch_files_l = epoch_path + file
    for file in os.listdir(epoch_path):
        if file.startswith('r_run' + str(run) + '-' + freq_band):
            if file.endswith('-epo.fif'):
                epoch_files_r = epoch_path + file
    if epoch_files_l is None or epoch_files_r is None:
        return None
    else:
        return epoch_files_l, epoch_files_r

def get_timefilename(name, path, format):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    time_filename = f"{path}{name}_{timestamp}.{format}"
    return time_filename

if __name__ == "__main__":
    init()
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    results_path = cwd + '/Results/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # ----- Define Subject and Session ----
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                  'S16', 'S17']
    session_list = [0, 1, 2]

    ### ----- DATA FOR TESTING ----- ###
    subject_list = ['S15']
    subj = subject_list[0]
    ses_ix = session_list[1]
    run = 3

    mon_me = [0] * len(subject_list)
    mon_mi = [2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2]
    vr_mi = [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1]
    conditions = {'mon_me': mon_me, 'mon_mi': mon_mi, 'vr_mi': vr_mi}
    roi = ["frontal-left", "frontal-right", "central-left", "central-right", "parietal-left", "parietal-right"]

    #freq_band = ['alpha', 'beta']
    freq_band = ['raw']
    task = ['l', 'r']
    # -------------------------------------
    #######################################
    # -------------------------------------
    create_epoch_files = False  # raw or preprocessed eeg data cut into epochs and saved
    create_mean_erds_results = False  # calculate results for statistical analysis
    save_xdf_to_mat = False  # saves EEG, ERDS and LDA as .mat files
    plot_online_ERDS = True  # plots results of oline calculated ERDS
    calc_offline_online_ERDS = True  # reproduce online calculated results of ERDS
    plot_ERDS_maps = False  # plots ERDS maps from EEG
    calc_acc = False  # computes average of online calculated accuracy per run
    # -------------------------------------
    #######################################
    # -------------------------------------

    if create_epoch_files:
        '''
        calls run per config file and saves as epochs in .fif-file. 
        '''
        for subj in subject_list:
            for ses in session_list:
                try:
                    subject_data_path = data_path + subj + '-ses' + str(ses) + '/'
                    print(f'\n\n\n\n ---------------Current path: {subject_data_path}--------------- \n')
                    config_files = find_config_files(subject_data_path, subject_id=subj)
                    save_epochs_per_run(config_files, subject_data_path)
                except Exception as e:
                    print(f'Error processing subject {subj} {ses}. Exception: {e}')
                    continue

    if create_mean_erds_results:
        '''
            opens .fif-files with epochs, combines epochs from all runs per session and calculates mean erds per roi.
        '''


        total_combinations = len(freq_band) * len(session_list) * len(task) * len(roi)
        results_array = np.empty((0, total_combinations))
        #description_array = np.empty((0, total_combinations))
        header = []

        for sig in freq_band:
            for ses_ix, ses in enumerate(list(conditions.keys())):
                for tsk in task:
                    for r in roi:
                        descr = ses + ' ' + tsk + ' ' + sig + ' ' + r
                        header.append(descr)
        #results_array = np.vstack((results_array, header))

        for subj_ix, subj in enumerate(subject_list):
            result_row = []
            #description_row = []
            for sig in freq_band:
                for ses_ix, ses in enumerate(list(conditions.keys())):
                    subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
                    print(f'\n\n\n\n ---------------Current path: {subject_data_path}--------------- \n')
                    config_files = find_config_files(subject_data_path, subject_id=subj)
                    for tsk in task:
                        try:
                            erds = calculate_erds_per_roi(len(config_files), subject_data_path,task=tsk, sig=sig, plot_epochs=False)
                            #description = [subj + ' ' + ses + ' ' + tsk + ' ' + sig] * 6
                            #print(description)
                            result_row.extend(erds)
                            #description_row.extend(description)
                        except Exception as e:
                            print(f'Error processing subject {subj} {ses} {tsk} {sig}. Exception: {e}')
                            continue
            try:
                results_array = np.vstack((results_array, result_row))
                #description_array = np.vstack((description_array, description_row))
            except Exception as e:
                print(f'Error processing subject {subj}. Exception: {e}')
                continue

        filename = results_path + 'results_erds_test_squared.txt'
        header_str = ','.join(header)
        with open(filename, 'w') as file:
            file.write(header_str + "\n")
        with open(filename, 'a') as file:
            np.savetxt(file, results_array, delimiter=',')

        #np.savetxt(results_path + 'results_erds.txt', results_array, delimiter=',')  #, fmt='%.6f')
        #np.savetxt(results_path + 'description_results.txt', description_array, delimiter=',', fmt='%s')

    if save_xdf_to_mat:
        '''
           reads .xdf-file with streams and saves eeg, erds and lda stream to .mat-file.
        '''
        for current_subject in subject_list:
            try:
                for current_session in session_list:
                    subject_directory = data_path + current_subject + '-ses' + str(current_session) + '/'
                    all_config_files = glob.glob(subject_directory + '*.json')

                    for current_config_file in all_config_files:
                        with open(current_config_file) as json_file:
                            config = json.load(json_file)
                        xdf_to_mat.xdf_to_mat_file(config, subject_directory)
            except Exception as e:
                print(f'Error processing subject {current_subject}. Exception: {e}')
                continue

    if plot_online_ERDS:
        '''
            calls .mat-files of erds and plots results of online calculated ERDS values
        '''
        subject_data_path, config_file_path, xdf_file_path = load_current_file(subj, ses_ix, run)
        signal = Input_Data(config_file_path, subject_data_path)
        online_erds_l, online_erds_r = ERDS_calculation.erds_values_plot_preparation(subject_data_path, signal)
        file_l = get_timefilename(subj + '_ses' + str(ses_ix) + '_run' + str(run) + '_online_ERDS_left', results_path, format='csv')
        file_r = get_timefilename(subj + '_ses' + str(ses_ix) + '_run' + str(run) + '_online_ERDS_right', results_path, format='csv')
        np.savetxt(file_l, online_erds_l, delimiter=',')
        np.savetxt(file_r, online_erds_r, delimiter=',')

        print("Online ERDS plots done.")

    if calc_offline_online_ERDS:
        '''
            calculates the same results for ERDS from eeg data which have been online calculated
        '''
        subject_data_path, config_file_path, xdf_file_path = load_current_file(subj, ses_ix, run)
        #epochs = combine_epochs(len(config_files), subject_data_path, task=task[1], sig=signal[0])
        file_l, file_r = load_curent_epoch(run, freq_band[0])
        epochs_l = mne.read_epochs(file_l, preload=True) # 5626 samples per epoch
        epochs_r = mne.read_epochs(file_r, preload=True)
        roi_dict_single = {
            "ROI1": ["F3"],
            "ROI2": ["F4"],
            "ROI3": ["C3"],
            "ROI4": ["C4"],
            "ROI5": ["P3"],
            "ROI6": ["P4"]}  # ERDS mode = single
        offline_erds_mean_l, offline_erds_trial_l= ERDS_calculation.erds_per_roi(epochs_l, roi_dict_single)
        offline_erds_mean_r, offline_erds_trial_r = ERDS_calculation.erds_per_roi(epochs_r, roi_dict_single)
        file_l = get_timefilename(subj + '_ses' + str(ses_ix) + '_run' + str(run) + '_offline_ERDS_left', results_path,
                                  format='csv')
        file_r = get_timefilename(subj + '_ses' + str(ses_ix) + '_run' + str(run) + '_offline_ERDS_right', results_path,
                                  format='csv')
        np.savetxt(file_l, offline_erds_trial_l, delimiter=',')
        np.savetxt(file_r, offline_erds_trial_r, delimiter=',')
        print("done")

    if plot_ERDS_maps:
        '''
            plots ERDS maps calculated from EEG.mat
        '''
        subj = subject_list[0]
        ses_ix = session_list[0]
        subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
        config_file_path = find_config_files(subject_data_path, subject_id=subj)[1]
        data_from_mat = ERDS_calculation.Data_from_Mat(config_file_path, subject_data_path)
        ERDS_calculation.plot_erds_maps(data_from_mat, picks=['C3', 'C4'], show_epochs=True, show_erds=False, non_clustering=True, clustering=True)

    if calc_acc:
        filename = get_timefilename('acc_output.txt', results_path)

        with open(filename, 'w') as f:
            sys.stdout = f  # Redirect the standard output to the file
            print(f'Writing {filename}')
            header = ['Ses0 run1', 'Ses0 run2', 'Ses0 run3', 'Ses1 run1', 'Ses1 run2', 'Ses1 run3', 'Ses1 run4', 'Ses2 run1',
                      'Ses2 run2', 'Ses2 run3', 'Ses2 run4', 'Ses2 run5']
            accuracy_array = np.zeros((len(subject_list), len(header)))
            for subj_ix, subj in enumerate(subject_list):
                for ses in range(3):
                    i = 0
                    subject_data_path = data_path + subj + '-ses' + str(ses) + '/'
                    print(f'\n\n ---------------Current path: {subject_data_path}---------------')
                    config_files = find_config_files(subject_data_path, subject_id=subj)
                    for run, cur_config_file in enumerate(config_files):
                        try:
                            accuracy = calculate_accuracy(cur_config_file, data_path)
                            accuracy_array[subj_ix][ses+i] = accuracy
                            i+=1
                        except Exception as e:
                            print(f'Error processing subject {subj} session {ses}. Exception: {e}')
                            i+=1
                            continue
            #print(accuracy_array)
            '''
            filename = results_path + 'results_accuracy.csv'
            header_str = ','.join(header)
            with open(filename, 'w') as file:
                file.write(header_str + "\n")
            with open(filename, 'a') as file:
                np.savetxt(file, accuracy_array, delimiter=',')
            '''

            sys.stdout = sys.__stdout__  # Set the standard output back to the console
            print("Accuracy saved in 'acc_output.txt'.")

    if plot_online_ERDS and calc_offline_online_ERDS:
        plt_compare.plt_compar_on_off(results_path, subj, ses_ix, run)

    print("All tasks completed!")
