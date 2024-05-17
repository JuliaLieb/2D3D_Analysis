import json
import os

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

if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'

    # ----- Define Subject and Session ----
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                  'S16', 'S17', 'S20']
    subject_list = ['S20']
    session_list = [0, 1, 2]
    run_list = [1, 2, 3, 4, 5]

    #subject_id = subject_list[17]
    subject_id = subject_list[0] # =subject ID -1
    session_id = session_list[1]
    run_id = run_list[2]

    # -------------------------------------

    subject_data_path = data_path + subject_id + '-ses' + str(session_id) + '/'
    for file in os.listdir(subject_data_path):
        if file.startswith('CONFIG_' + subject_id + '_run' + str(run_id)):
            if file.endswith('.json'):
                config_file = subject_data_path + file
            else:
                print("No config file found.")
                break

    input_data = Input_Data(config_file, subject_data_path)
    epochs_l, epochs_r = input_data.run_raw()
    #epochs_l = mne.read_epochs('C:/2D3D_Analysis/Data/S20-ses1/datasetepochs_l_run3.fif', preload=True)
    #epochs_r = mne.read_epochs('C:/2D3D_Analysis/Data/S20-ses1/datasetepochs_r_run3.fif', preload=True)
    erds_l = calc_values.run(epochs_l)
    erds_r = calc_values.run(epochs_r)
    print(erds_l, erds_r)



    '''
    #Save all .xdf files to .mat files
    mat_transformation = False
    if mat_transformation:
        navigate_data.save_xdf_to_mat(data_path, subject_list, session_list)
    '''

    #'''
    #Get all .mat and .xdf file names - not necessary if already done.
    #all_mat_files, all_xdf_files = navigate_data.get_all_mat_xdf_file_names(data_path, subject_id, session_id)
    #print(all_mat_files, all_xdf_files)

    #Get specific run
    '''
    current_run_mat, current_run_xdf, current_config = navigate_data.get_specific_run(data_path, subject_id, session_id, run_id)
    current_run_mat_path = subject_data_path + 'dataset/' + current_run_mat
    current_run_xdf_path = subject_data_path + current_run_xdf
    current_config_path = subject_data_path + current_config
    #print(current_run_mat, current_run_xdf, current_config)


    eeg = EEG_Signal(current_config_path, subject_data_path)
    '''

    # zum durch-iterieren geplant - TBA
    # all_config_files = glob.glob(subject_directory + '*.json')
    # for current_config_file in all_config_files:

    #eeg.plot_raw_eeg(scaled=True, max_trial=2)  # funktioniert - plot von jedem einzelnen Trial
    #eeg.plot_erds_maps(picks=['C3', 'C4'], show_erds=True, show_epochs=False, clustering=False) # funktioniert
    #eeg.show_lda_plots() # funktioniert nicht korrekt (prozentsatz vs anzeige - oft falsche klasse)
    #eeg.show_erds_mean() # funktioniert auch wenn ich nicht weiß wofür
    #mean_acc = eeg.compute_accuracy() # funktioniert
    #print(subject_id + ' session ' + str(session_id) + ' run ' + str(run_id) + ': acc = ' + str(mean_acc))

    #eeg.preprocessing() # etwas wird geplottet (ica mäßig)

    #eeg.xdf_to_fif(subject_data_path)
    #eeg.raw_epochs()



