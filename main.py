import json
import os

import navigate_data
import EEG_analysis
import ERDS_analysis
import analysis
from offline_analysis import EEG_Signal

if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'

    # ----- Define Subject and Session ----
    #subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
    #              'S16', 'S17', 'S20']
    subject_list = ['S20']
    session_list = [0, 1, 2]
    run_list = [1, 2, 3, 4, 5]

    #subject_id = subject_list[17]
    subject_id = subject_list[0]
    session_id = session_list[1]
    run_id = run_list[1]
    # -------------------------------------

    subject_data_path = data_path + subject_id + '-ses' + str(session_id) + '/'

    #Save .xdf files to .mat files
    #navigate_data.save_xdf_to_mat(data_path, subject_list, session_list)

    #'''
    #Get all .mat and .xdf file names
    #all_mat_files, all_xdf_files = navigate_data.get_all_mat_xdf_file_names(data_path, subject_id, session_id)
    #print(all_mat_files, all_xdf_files)

    #Get specific run
    current_run_mat, current_run_xdf, current_config = navigate_data.get_specific_run(data_path, subject_id, session_id, run_id)
    current_run_mat_path = subject_data_path + 'dataset/' + current_run_mat
    current_run_xdf_path = subject_data_path + current_run_xdf
    current_config_path = subject_data_path + current_config
    print(current_run_mat, current_run_xdf, current_config)

    #Get something from eeg_analysis - TBA
    #if run_id == 1:
        #stream_eeg, stream_marker = analysis.load_xdf(current_config, current_run_xdf)
    #else:
        #stream_eeg, stream_marker, stream_erds, stream_lda = analysis.load_xdf(current_config, current_run_xdf)
    #'''
    eeg = EEG_Signal(current_config_path, subject_data_path)
    # all_config_files = glob.glob(subject_directory + '*.json')
    # for current_config_file in all_config_files:

    eeg.analyze_eeg()


    print("Works!")

