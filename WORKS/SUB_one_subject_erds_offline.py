import numpy as np
import mne
import json
import matplotlib
import pyxdf

matplotlib.use('Qt5Agg')

import SUB_erds_management, SUB_trial_management, SUB_filtering



def compute_offline_erds_per_run(subject_data_path, config_file_path, freq_band='mu', preproc=False):

    # ==============================================================================
    # Load files: infos and data
    # ==============================================================================

    # CONFIG
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    subject_id = config['gui-input-settings']['subject-id']
    n_session = config['gui-input-settings']['n-session']
    n_run = config['gui-input-settings']['n-run']
    motor_mode = config['gui-input-settings']['motor-mode']
    erds_mode = config['feedback-model-settings']['erds']['mode']
    dimension = config['gui-input-settings']['dimension-mode']
    feedback = config['gui-input-settings']['fb-mode']

    lsl_config = config['general-settings']['lsl-streams']
    sample_rate = config['eeg-settings']['sample-rate']
    duration_ref = config['general-settings']['timing']['duration-ref']
    duration_cue = config['general-settings']['timing']['duration-cue']
    duration_fb = config['general-settings']['timing']['duration-task']
    duration_task = duration_cue + duration_fb
    n_ref = int(np.floor(sample_rate * duration_ref))
    n_cue = int(np.floor(sample_rate * duration_cue))
    n_fb = int(np.floor(sample_rate * duration_fb))
    n_samples_task = int(np.floor(sample_rate * duration_task))
    n_samples_trial = n_ref + n_samples_task

    # Channels & ROIs
    channel_dict = config['eeg-settings']['channels']
    enabled_ch_names = [name for name, settings in channel_dict.items() if settings['enabled']]
    enabled_ch = np.subtract([settings['id'] for name, settings in channel_dict.items() if settings['enabled']], 1) # Python starts at 0

    roi_ch_nr = config['feedback-model-settings']['erds']['single-mode-channels']
    n_roi = len(roi_ch_nr)
    roi_dict = {settings['id']: name for name, settings in channel_dict.items()}
    roi_ch_names = [roi_dict[id_] for id_ in roi_ch_nr]
    roi_enabled_ix = [enabled_ch_names.index(ch) for ch in roi_ch_names if ch in enabled_ch_names]

    # XDF
    xdf_file_path = subject_data_path + subject_id + '_run' + str(n_run) + '_' + motor_mode + '_' + dimension + '.xdf'
    streams, fileheader = pyxdf.load_xdf(xdf_file_path)
    stream_names = []

    for stream in streams:
        stream_names.append(stream['info']['name'][0])

    streams_info = np.array(stream_names)

    # gets 'BrainVision RDA Data' stream - EEG data
    eeg_pos = np.where(streams_info == lsl_config['eeg']['name'])[0][0]
    eeg = streams[eeg_pos]
    eeg_signal = eeg['time_series'][:, :32]
    eeg_signal = eeg_signal * 1e-6
    # get the instants
    eeg_instants = eeg['time_stamps']
    time_zero = eeg_instants[0]
    eeg_instants = eeg_instants - time_zero


    # gets marker stream
    marker_pos = np.where(streams_info == lsl_config['marker']['name'])[0][0]
    marker = streams[marker_pos]
    marker_ids = marker['time_series']
    marker_instants = marker['time_stamps']-time_zero
    marker_dict = {
        'Reference': 10,
        'Start_of_Trial_l': 1,
        'Start_of_Trial_r': 2,
        'Cue': 20,
        'Feedback': 30,
        'End_of_Trial': 5, # equal to break
        'Session_Start': 4,
        'Session_End': 3}
    # Manage markers
    if n_run == 1:
        marker_interpol = SUB_trial_management.interpolate_markers_first_run(marker_ids, marker_dict, marker_instants,
                                                                             eeg_instants)
    elif n_run > 1:
        marker_interpol = SUB_trial_management.interpolate_markers(marker_ids, marker_dict, marker_instants,
                                                                   eeg_instants)
    n_trials = marker_ids.count(['Start_of_Trial_l']) + marker_ids.count(['Start_of_Trial_r'])
    # find timespans when left/ right times with FEEDBACK
    fb_times = SUB_trial_management.find_marker_times(n_trials, marker_dict['Feedback'],
                                                      marker_interpol, eeg_instants)
    # find timespans when left/ right times with REFERENCE
    ref_times = SUB_trial_management.find_marker_times(n_trials, marker_dict['Reference'], marker_interpol,
                                                       eeg_instants)

    # ==============================================================================
    # Select and Filter EEG
    # ==============================================================================

    if preproc:
        try:
            preproc_file_path = subject_data_path + 'preproc_raw/' + '/run' + str(n_run) + '_preproc-raw.fif'
            signal_preproc = mne.io.read_raw_fif(preproc_file_path, preload=True)
            eeg_preproc = signal_preproc.get_data().T
            eeg_raw = eeg_preproc[:, enabled_ch]  # use preprocessed EEG
            print('Using preprocessed EEG data.')
        except Exception as e:
            print(f'Error processing subject {subject_id}, preprocessed file {preproc_file_path}. Continue processing raw signal.'
                  f' Exception: {e}')
            eeg_raw = eeg_signal[:, enabled_ch]
    else:
        eeg_raw = eeg_signal[:, enabled_ch]  # use raw EEG form .xdf file

    s_rate_half = sample_rate/2
    if freq_band == 'mu':
        fpass_erds = [freq / s_rate_half for freq in [9, 11]] # Mu-frequency-band wie online
        fstop_erds = [freq / s_rate_half for freq in [7.5, 12]]
    elif freq_band == 'alpha':
        fpass_erds = [freq / s_rate_half for freq in [8, 12]]
        fstop_erds = [freq / s_rate_half for freq in [6, 14]]
    elif freq_band == 'beta':
        fpass_erds = [freq / s_rate_half for freq in [16, 24]]
        fstop_erds = [freq / s_rate_half for freq in [14, 26]]

    bp_erds = SUB_filtering.Bandpass(order=12, fstop=fstop_erds, fpass=fpass_erds, n=eeg_raw.shape[1])


    # Filter EEG with Status = Start, Ref, FB (Marker = 4, 11, 12, 31, 32)  - like ONLINE!
    eeg_filt = SUB_filtering.filter_per_status(eeg_raw, eeg_instants, bp_erds, marker_interpol,
                                               status=[0, 4, 11, 12, 31, 32])

    # ==============================================================================
    # ERDS  from OFFLINE calculated results
    # ==============================================================================

    # eeg data for reference period of every task # left
    data_ref, data_ref_mean = SUB_erds_management.get_data_ref(eeg_filt, eeg_instants, ref_times, n_ref)

    # eeg data for feedback period of every task and erds per roi # left
    data_a, erds_offline_ch, erds_offline_roi = SUB_erds_management.get_data_a_calc_erds(eeg_filt, eeg_instants,
                                                                                         fb_times, n_fb, data_ref_mean,
                                                                                         roi_enabled_ix)

    # calculate the values for ANOVA
    erds_off_l, erds_off_r = SUB_erds_management.calc_avg_erds_per_class(erds_offline_roi, fb_times)

    return erds_off_l, erds_off_r
