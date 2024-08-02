import os
from sig_reading import Input_Data
import mne
import xdf_to_mat
import glob
import json
import SUB_erds_plotting


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
        input_data.run_preprocessing_to_epoch()  # preprocessed data

def signal_reading_preprocesiing_saving(config_files, path):
    for config in config_files:
        input_data = Input_Data(config, path)
        input_data.run_preprocessing_to_fif()  # preprocessing raw data

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


if __name__ == "__main__":

    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    results_path = cwd + '/Results/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # ----- Define Subject and Session ----
    subject_list = ['S14']
    session_list = [0, 1, 2]
    subj = subject_list[0]
    ses_ix = session_list[2]

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
    save_xdf_to_mat = False  # saves EEG, ERDS and LDA as .mat files
    plot_ERDS_maps = True  # plots ERDS maps from EEG
    preproc_to_fif = False

    # -------------------------------------
    #######################################
    # -------------------------------------

    if create_epoch_files:
        '''
        calls run per config file and saves as epochs in .fif-file. 
        '''
        for subj in subject_list:
            for ses in session_list:
            #es = ses_ix
                try:
                    subject_data_path = data_path + subj + '-ses' + str(ses) + '/'
                    print(f'\n\n\n\n ---------------Current path: {subject_data_path}--------------- \n')
                    config_files = find_config_files(subject_data_path, subject_id=subj)
                    save_epochs_per_run(config_files, subject_data_path)
                except Exception as e:
                    print(f'Error processing subject {subj} {ses}. Exception: {e}')
                    continue


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




    if plot_ERDS_maps:
        '''
            plots ERDS maps calculated from EEG.mat
        '''
        subject_data_path, config_file_path, xdf_file_path = load_current_file(subj, ses_ix, run=1)
        data_from_mat = SUB_erds_plotting.EEG_Data(config_file_path, subject_data_path, xdf_file_path)
        SUB_erds_plotting.plot_erds_maps(data_from_mat, picks=['C3', 'Cz', 'C4'], show_epochs=False,
                                         show_erds=True, cluster_mode=True, preproc_data=False, tfr_mode=True)


    if preproc_to_fif:
        subject_data_path, config_file_path, xdf_file_path = load_current_file(subj, ses_ix, run=1)
        signal_reading_preprocesiing_saving([config_file_path], subject_data_path)


    print("All tasks completed!")
