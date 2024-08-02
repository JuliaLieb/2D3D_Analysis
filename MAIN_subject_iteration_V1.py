# ----------------------------------------------------------------------------------------------------------------------
# Manage iterations of subjects in list to
#  - preprocess EEG data and save as files
#  - calculate ERDS results and save it in a file
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime
import winsound
from SUB_one_subject_erds_offline import compute_offline_erds_per_run

def find_config_files(path, subject_id):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files


if __name__ == "__main__":
    # %%
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    results_path = cwd + '/Results/'
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
    conditions = {'mon_me': mon_me, 'mon_mi': mon_mi, 'vr_mi': vr_mi}
    roi = ["frontal-left", "frontal-right", "central-left", "central-right", "parietal-left", "parietal-right"]
    task = ['l', 'r']
    #freq_band = ['mu']
    freq_band = ['alpha', 'beta']

    ### ----- DATA FOR TESTING ----- ###
    '''
    subject_list = ['S14']
    mon_me = [0] * len(subject_list)
    mon_mi = [2]
    vr_mi = [1]
    conditions = {'mon_me': mon_me, 'mon_mi': mon_mi, 'vr_mi': vr_mi}
    '''

    # create description header for all conditions
    total_combinations = len(freq_band) * len(conditions) * len(task) * len(roi)
    results_array = np.empty((0, total_combinations))
    header = []
    for freq in freq_band:
        for ses_ix, ses in enumerate(list(conditions.keys())):
            for tsk in task:
                for r in roi:
                    descr = ses + ' ' + tsk + ' ' + freq + ' ' + r
                    header.append(descr)

    # Iterations over all files to calculate erds results
    for subj_ix, subj in enumerate(subject_list):
        result_row = []
        for freq in freq_band:
            for ses_key, ses_values in conditions.items():
                ses_ix = ses_values[subj_ix]
                subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
                config_files = find_config_files(subject_data_path, subject_id=subj)
                results_session_l = []
                results_session_r = []
                for cur_config in config_files:
                    print(f'\n\n---------------Current path: {cur_config}--------------- \n')

                    # Preprocess and save file
                    """
                    try:
                        input_data = Input_Data(cur_config, subject_data_path)
                        input_data.run_preprocessing_to_fif()  # preprocessing raw data
                    except Exception as e:
                        print(f'Error processing subject {subj}, frequency band {freq}, session {ses_ix}, '
                              f'file {cur_config} . Exception: {e}')
                        continue
                    """

                #""" # Calculate ERDS Results
                    try:
                        #erds_off_l, erds_off_r = compute_offline_erds_per_run(subject_data_path, cur_config)
                        # optional with preprocessed .fif file
                        erds_off_l, erds_off_r = compute_offline_erds_per_run(subject_data_path, cur_config,
                                                                              freq_band=freq, preproc=True)
                    except Exception as e:
                        print(f'Error processing subject {subj}, frequency band {freq}, session {ses_ix}, '
                              f'file {cur_config} . Exception: {e}')
                        erds_off_l = np.zeros(6)
                        erds_off_r = np.zeros(6)
                        continue
                    results_session_l.append(erds_off_l)
                    results_session_r.append(erds_off_r)
                results_session_l = np.array(results_session_l)
                results_session_r = np.array(results_session_r)
                results_session_avg_l = np.mean(results_session_l, axis=0)
                results_session_avg_r = np.mean(results_session_r, axis=0)
                result_row.extend(results_session_avg_l)
                result_row.extend(results_session_avg_r)
        try:
            results_array = np.vstack((results_array, result_row))
        except Exception as e:
            print(f'Error processing subject {subj}. Exception: {e}')
            continue

    results_file = results_path + timestamp + '_results_erds_acceptable.txt'
    header_str = ','.join(header)
    with open(results_file, 'w') as file:
        file.write(header_str + "\n")
    with open(results_file, 'a') as file:
        np.savetxt(file, results_array, delimiter=',')
        
#"""
    winsound.Beep(1000, 3000)
