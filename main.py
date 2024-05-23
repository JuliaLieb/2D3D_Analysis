import os
import numpy as np
from signal_reading import Input_Data
import calc_values
import mne
from colorama import init, Fore


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
        filename = path + '/epochs/' + task + '_run' + str(run + 1) + '-' + sig + '-epo.fif'
        if os.path.exists(filename):
            epochs.append(mne.read_epochs(filename, preload=True))
        else:
            print(f'File not found: {filename}')
    epochs = mne.concatenate_epochs(epochs)

    return epochs

def calculate_erds_per_roi(n_runs, subject_data_path, task='r', sig='raw', plot_epochs=False):
    epochs= combine_epochs(n_runs, subject_data_path, task=task, sig=sig)
    if plot_epochs:
        epochs.compute_psd().plot()

    # calculate erds per ROI
    erds_per_roi = calc_values.run(epochs)

    return erds_per_roi


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
    #subject_list = ['S6', 'S7', 'S8', 'S9', 'S10']


    #subject_list = ['S20']
    session_list = [0, 1, 2]

    #subject_id = subject_list[17]
    #subject_id = subject_list[0] # =subject ID -1

    #session_id = session_list[2]

    signal = ['alpha', 'beta']  #'raw'
    task = ['l', 'r']
    create_epoch_files = False # else: calculate results
    # -------------------------------------




    if create_epoch_files:
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
    else:
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

        filename = results_path + 'results_erds.txt'
        header_str = ','.join(header)
        with open(filename, 'w') as file:
            file.write(header_str + "\n")
        with open(filename, 'a') as file:
            np.savetxt(file, results_array, delimiter=',')

        #np.savetxt(results_path + 'results_erds.txt', results_array, delimiter=',')  #, fmt='%.6f')
        #np.savetxt(results_path + 'description_results.txt', description_array, delimiter=',', fmt='%s')

    print("Task completed!")
