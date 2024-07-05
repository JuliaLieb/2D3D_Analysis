import os
import sys

import numpy as np
from signal_reading import Input_Data
import ERDS_calculation
import mne
import json
import signal_reading
import matplotlib.pyplot as plt
import matplotlib
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from matplotlib.colors import TwoSlopeNorm
import offline_analysis
import ERDS_calculation
import main
import pyxdf
from matplotlib.colors import TwoSlopeNorm
matplotlib.use('Qt5Agg')
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt, sosfilt
import bandpass
from datetime import datetime
from scipy import signal

import SUB_plot_management, SUB_lda_management, SUB_erds_management, SUB_trial_management, SUB_filtering



if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    result_path = cwd + '/Results/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    config_file_path = "C:/2D3D_Analysis/Data/S14-ses0/CONFIG_S14_run2_ME_2D.json"
    xdf_file_path = "C:/2D3D_Analysis/Data/S14-ses0/S14_run2_ME_2D.xdf"
    preproc_file_path = "C:/2D3D_Analysis/Data/S14-ses0/preproc_raw/run2-_preproc-raw.fif"

    ###### Load files: infos and data ------------------------------------------------------------------------

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
    # get the sampling frequencies
    eeg_fs = int(float(eeg['info']['nominal_srate'][0]))
    effective_sample_frequency = float(eeg['info']['effective_srate'])

    # gets 'BrainVision RDA Markers' stream
    orn_pos = np.where(streams_info == 'BrainVision RDA Markers')[0][0]
    orn = streams[orn_pos]
    orn_signal = orn['time_series']
    orn_instants = orn['time_stamps']-time_zero

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
    marker_interpol = SUB_trial_management.interpolate_markers(marker_ids, marker_dict, marker_instants, eeg_instants)

    # ERDS and LDA recordings
    erds_pos = np.where(streams_info == lsl_config['fb-erds']['name'])[0][0]
    lda_pos = np.where(streams_info == lsl_config['fb-lda']['name'])[0][0]
    erds = streams[erds_pos]
    lda = streams[lda_pos]
    erds_time = erds['time_stamps']-time_zero
    erds_values = erds['time_series']
    lda_time = lda['time_stamps']-time_zero
    lda_values = lda['time_series']


    # ==============================================================================
    # ERDS from ONLINE calculated results
    # ==============================================================================

    # find timespans when left/ right times with FEEDBACK
    fb_times = SUB_trial_management.find_marker_times(marker_ids.count(['Start_of_Trial_l']) +
                                                      marker_ids.count(['Start_of_Trial_r']), marker_dict['Feedback'],
                                                      marker_interpol, eeg_instants)

    # find timespans when left/ right times with REFERENCE
    ref_times = SUB_trial_management.find_marker_times(marker_ids.count(['Start_of_Trial_l']) +
                                                      marker_ids.count(['Start_of_Trial_r']), marker_dict['Reference'],
                                                      marker_interpol, eeg_instants)

    # get online calculated erds
    erds_online = SUB_erds_management.assess_erds_online(erds_values, erds_time, fb_times, n_fb, n_roi)

    # calculate mean ERDS per task and ROI
    avg_erds_online = np.zeros((len(fb_times), n_roi))
    for trial in range(erds_online.shape[0]):
        trial_data = erds_online[trial]
        for roi in range(n_roi):
            avg_erds_online[trial, roi] = np.mean(trial_data[:, roi+1])

    # plot online calculated ERDS
    #SUB_plot_management.plot_online_erds(erds_online, avg_erds_online, fb_times, roi_ch_names, config_file_path)
    #plt.savefig(result_path + timestamp + '_online_ERDS')
    #plt.show()


    # ==============================================================================
    # LDA from ONLINE calculated results
    # ==============================================================================

    # get online calculated lda
    lda_online = SUB_lda_management.assess_lda_online(lda_values, lda_time, fb_times, n_fb)

    # calculate mean LDA accuracy per task
    avg_lda_acc_online = []
    for trial in range(lda_online.shape[0]):
        avg_lda_acc_online.append(np.mean(lda_online[trial][:, 1]))

    # plot online calculated LDA accuracy
    #SUB_plot_management.plot_online_lda(lda_online, avg_lda_acc_online, fb_times, config_file_path)
    #plt.savefig(result_path + timestamp + '_online_LDA_acc')
    #plt.show()


    # ==============================================================================
    # Load preprocessed EEG
    # ==============================================================================
    '''
    if os.path.exists(preproc_file_path):
        signal_preproc = mne.io.read_raw_fif(preproc_file_path, preload=True)
        eeg_preproc = signal_preproc.get_data().T
    else:
        print(f'File not found: {preproc_file_path}')
    '''

    # ==============================================================================
    # Filter EEG
    # ==============================================================================

    eeg_raw = eeg_signal[:, enabled_ch]
    #eeg_raw = eeg_preproc[:, enabled_ch] # use preprocessed EEG

    s_rate_half = sample_rate/2
    fpass_erds = [freq / s_rate_half for freq in [9, 11]] # Mu-frequency-band
    fstop_erds = [freq / s_rate_half for freq in [7.5, 12]]
    bp_erds = SUB_filtering.Bandpass(order=12, fstop=fstop_erds, fpass=fpass_erds, n=eeg_raw.shape[1])

    '''
    #ganzes EEG vorfiltern - nicht gut
    eeg_raw = SUB_filtering.filter_complete_eeg(eeg_raw, eeg_instants, bp_erds)

    # nur Status = Start (Marker = 4) vorfiltern - bei data_ref und data_a samples dann zusätzlich filtern - nicht gut
    eeg_raw = SUB_filtering.filter_per_status(eeg_raw, eeg_instants, bp_erds, marker_interpol, status=[4])
    '''
    # Filter EEG with Status = Start, Ref, FB (Marker = 4, 11, 12, 31, 32)  - like ONLINE!
    eeg_raw = SUB_filtering.filter_per_status(eeg_raw, eeg_instants, bp_erds, marker_interpol,
                                              status=[4, 11, 12, 31, 32])


    # ==============================================================================
    # ERDS  from OFFLINE calculated results
    # ==============================================================================

    # eeg data for reference period of every task # left
    data_ref, data_ref_mean = SUB_erds_management.get_data_ref(eeg_raw, eeg_instants, ref_times, n_ref)

    # eeg data for feedback period of every task and erds per roi # left
    data_a, erds_offline_ch, erds_offline_roi = SUB_erds_management.get_data_a_calc_erds(eeg_raw, eeg_instants,
                                                                                         fb_times, n_fb, data_ref_mean,
                                                                                         roi_enabled_ix)

    # calculate mean ERDS per task and ROI
    avg_erds_offline = np.zeros((len(fb_times), n_roi))
    for trial in range(erds_offline_roi.shape[0]):
        trial_data = erds_offline_roi[trial]
        for roi in range(6):
            avg_erds_offline[trial, roi] = np.mean(trial_data[:, roi+1])

    # plot signals for offline calculating ERDS
    #SUB_plot_management.plot_signals_for_eeg_calculation(data_ref, data_a, ref_times, fb_times, config_file_path)
    #plt.savefig(result_path + timestamp + '_R_A_signal')
    #plt.show()

    # plot offline calculated ERDS for ROI channels
    #SUB_plot_management.plot_offline_erds(erds_offline_ch, roi_enabled_ix, fb_times, config_file_path)
    #plt.savefig(result_path + timestamp + '_offline_erds')
    plt.show()



    # plot comparison online - offline ERDS
    ch_comp = ['C4']
    ch_ix = enabled_ch_names.index(ch_comp[0])
    ch_roi = roi_ch_names.index(ch_comp[0])
    trial = 0
    trial_cl = fb_times[trial, 2]
    #SUB_plot_management.plot_online_vs_offline_erds_per_trial(erds_online, erds_offline_ch, ch_ix, ch_roi, trial, trial_cl, config_file_path)
    #SUB_plot_management.plot_online_vs_offline_erds_all_trials(erds_online, erds_offline_ch, ch_ix, ch_roi, fb_times, config_file_path)
    #plt.show()
    #"""

    # plot comparison average online - offline ERDS
    SUB_plot_management.plot_online_vs_offline_avg_erds(avg_erds_online, avg_erds_offline, fb_times, config_file_path)
    plt.show()
