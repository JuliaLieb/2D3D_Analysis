import json
import os

import navigate_data

if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/SubjectDataSorted/'

    # ----- Define Subject and Session ----
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S14', 'S15',
                  'S16', 'S17']
    session_list = [0, 1, 2]
    run_list = [1, 2, 3, 4, 5]

    subject_id = subject_list[14]
    session_id = session_list[1]
    # -------------------------------------

    subject_data_path = data_path + subject_id + '-ses' + str(session_id) + '/'

    #Save .xdf files to .mat files
    navigate_data.save_xdf_to_mat(data_path, subject_list, session_list)

    #Get all .mat and .xdf file names
    all_mat_files, all_xdf_files = navigate_data.get_all_mat_xdf_file_names(data_path, subject_id, session_id)
    print(all_mat_files, all_xdf_files)

    #Get specific run
    run_id = 1
    current_run_mat, current_run_xdf = navigate_data.get_specific_run(data_path, subject_id, session_id, run_id)
    print(current_run_mat, current_run_xdf)

