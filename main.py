import os
import numpy as np
from signal_reading import Input_Data
import ERDS_calculation
import mne
from colorama import init, Fore
import xdf_to_mat
import glob
import json


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
        input_data.run_raw()

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

    subject_list = ['S20']
    session_list = [1]

    #subject_id = subject_list[17]
    #subject_id = subject_list[0] # =subject ID -1

    #session_id = session_list[2]

    signal = ['alpha', 'beta']  #'raw'
    task = ['l', 'r']
    create_epoch_files = False # else: calculate results
    create_mean_erds_results = False
    save_xdf_to_mat = False
    plot_online_ERDS = False
    calc_offline_online_ERDS = True
    plot_ERDS_maps = False
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
    elif create_mean_erds_results:
        '''
            opens .fif-files with epochs, combines epochs from all runs per session and calculates mean erds per roi.
        '''
        mon_me = [0]*len(subject_list)
        mon_mi = [2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2]
        vr_mi = [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1]
        conditions = {'mon_me': mon_me, 'mon_mi': mon_mi, 'vr_mi': vr_mi}
        roi = ["frontal-left", "frontal-right", "central-left", "central-right", "parietal-left", "parietal-right"]

        total_combinations = len(signal) * len(session_list) * len(task) * len(roi)
        results_array = np.empty((0, total_combinations))
        #description_array = np.empty((0, total_combinations))
        header = []

        for sig in signal:
            for ses_ix, ses in enumerate(list(conditions.keys())):
                for tsk in task:
                    for r in roi:
                        descr = ses + ' ' + tsk + ' ' + sig + ' ' + r
                        header.append(descr)
        #results_array = np.vstack((results_array, header))

        for subj_ix, subj in enumerate(subject_list):
            result_row = []
            #description_row = []
            for sig in signal:
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

    elif save_xdf_to_mat:
        '''
           reads .xdf-file with streams and saves eeg, erds and lda stream to .mat-file.
        '''
        for current_subject in subject_list:
            for current_session in session_list:
                subject_directory = data_path + current_subject + '-ses' + str(current_session) + '/'
                all_config_files = glob.glob(subject_directory + '*.json')

                for current_config_file in all_config_files:
                    with open(current_config_file) as json_file:
                        config = json.load(json_file)
                    xdf_to_mat.xdf_to_mat_file(config, subject_directory)

    elif plot_online_ERDS:
        '''
            calls .mat-files of erds and plots results of online calculated ERDS values
        '''
        subj = subject_list[0]
        ses_ix = session_list[0]
        subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
        config_file_path = find_config_files(subject_data_path, subject_id=subj)[1]
        signal = Input_Data(config_file_path, subject_data_path)
        online_erds_l, online_erds_r = ERDS_calculation.erds_values_plot_preparation(subject_data_path, signal)
        print("Online ERDS plots done.")

    elif calc_offline_online_ERDS:
        '''
            calculates the same results for ERDS from eeg data which have been online calculated
        '''
        subj = subject_list[0]
        ses_ix = session_list[0]
        subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
        #config_files = find_config_files(subject_data_path, subject_id=subj)
        #config_file_path = config_files[1]
        #epochs = combine_epochs(len(config_files), subject_data_path, task=task[1], sig=signal[0])
        file= 'C:\\2D3D_Analysis\\Data\\S20-ses1\\epochs\\l_run2-raw-epo.fif'
        epochs = mne.read_epochs(file, preload=True)
        roi_dict_single = {
            "ROI1": ["F3"],
            "ROI2": ["F4"],
            "ROI3": ["C3"],
            "ROI4": ["C4"],
            "ROI5": ["P3"],
            "ROI6": ["P4"]}  # ERDS mode = single
        offline_erds_mean, offline_erds_trial= ERDS_calculation.erds_per_roi(epochs, roi_dict_single)
        filename = "C:\\2D3D_Analysis\\Results\\S20-ses1-run2-left-ERDS-calculation"
        np.savetxt(filename, offline_erds_trial, delimiter=',')
        print(offline_erds_mean)
        print(offline_erds_trial)
        print("done")


    elif plot_ERDS_maps:
        '''
            plots ERDS maps calculated from EEG.mat
        '''
        subj = subject_list[0]
        ses_ix = session_list[0]
        subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
        config_file_path = find_config_files(subject_data_path, subject_id=subj)[1]
        data_from_mat = ERDS_calculation.Data_from_Mat(config_file_path, subject_data_path)
        ERDS_calculation.plot_erds_maps(data_from_mat, picks=['C3', 'C4'], show_epochs=True, show_erds=False, non_clustering=True, clustering=True)

    print("Task completed!")
