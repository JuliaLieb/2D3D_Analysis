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



def compute_erds_per_run(config_file_path, xdf_file_path, preproc_file_path=None):

    # ==============================================================================
    # Load files: infos and data
    # ==============================================================================

    # CONFIG
    with open(config_file_path) as json_file:
        config = json.load(json_file)

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
    n_trials = marker_ids.count(['Start_of_Trial_l']) + marker_ids.count(['Start_of_Trial_r'])
    # find timespans when left/ right times with FEEDBACK
    fb_times = SUB_trial_management.find_marker_times(n_trials, marker_dict['Feedback'],
                                                      marker_interpol, eeg_instants)
    # find timespans when left/ right times with REFERENCE
    ref_times = SUB_trial_management.find_marker_times(n_trials, marker_dict['Reference'], marker_interpol,
                                                       eeg_instants)

    # ERDS online recordings
    erds_pos = np.where(streams_info == lsl_config['fb-erds']['name'])[0][0]
    erds = streams[erds_pos]
    erds_time = erds['time_stamps']-time_zero
    erds_values = erds['time_series']


    # ==============================================================================
    # ERDS from ONLINE calculated results
    # ==============================================================================

    # get online calculated erds
    erds_online = SUB_erds_management.assess_erds_online(erds_values, erds_time, fb_times, n_fb, n_roi)

    # calculate the values for ANOVA
    erds_on_l, erds_on_r = SUB_erds_management.calc_avg_erds_per_class(erds_online, fb_times)

    return erds_on_l, erds_on_r
