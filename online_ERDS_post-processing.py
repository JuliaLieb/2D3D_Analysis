import os
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
from scipy.signal import butter, filtfilt, sosfiltfilt
import bandpass


def butter_bandpass_filter(data, fs, fpass, fstop, order=12):
    """
    Apply a Butterworth bandpass filter to raw EEG data.

    Parameters:
        data (array_like): Raw EEG data.
        fs (float): Sampling frequency of the data.
        fpass (tuple): Passband frequencies (lowcut, highcut).
        fstop (tuple): Stopband frequencies (lowcut, highcut).
        order (int): Order of the Butterworth filter.

    Returns:
        array_like: Filtered EEG data.
    """
    nyq = 0.5 * fs
    lowcut, highcut = fpass
    lowstop, highstop = fstop
    low = lowcut / nyq
    high = highcut / nyq
    stop_low = lowstop / nyq
    stop_high = highstop / nyq

    '''
    # Design the Butterworth filter
    b, a = butter(order, [low, high], btype='band', analog=False)

    # Initialize array to store filtered data
    filtered_data = np.zeros_like(data)

    # Apply the filter to each channel separately
    for i in range(data.shape[1]):
        # Apply the filter to the data using filtfilt for zero-phase filtering
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    '''
    # Design the Butterworth filter and convert to SOS format
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')

    # Initialize array to store filtered data
    filtered_data = np.zeros_like(data)

    # Apply the filter to the data using sosfiltfilt for zero-phase filtering
    for i in range(data.shape[1]):
        # Apply the filter to the data using filtfilt for zero-phase filtering
        filtered_data[:, i] = sosfiltfilt(sos, data[:, i])

    return filtered_data


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'

    config_file_path = "C:/2D3D_Analysis/Data/S20-ses1/CONFIG_S20_run2_MI_2D.json"
    xdf_file_path = "C:/2D3D_Analysis/Data/S20-ses1/S20_run2_MI_2D.xdf"

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
    duration_task = duration_cue + config['general-settings']['timing']['duration-task']
    n_ref = int(np.floor(sample_rate * duration_ref))
    n_cue = int(np.floor(sample_rate * duration_cue))
    n_samples_task = int(np.floor(sample_rate * duration_task))
    n_samples_trial = n_ref + n_samples_task

    # XDF
    streams, fileheader = pyxdf.load_xdf(xdf_file_path)
    stream_names = []

    for stream in streams:
        stream_names.append(stream['info']['name'][0])

    streams_info = np.array(stream_names)

    # ERDS and LDA recordings
    erds_pos = np.where(streams_info == lsl_config['fb-erds']['name'])[0][0]
    lda_pos = np.where(streams_info == lsl_config['fb-lda']['name'])[0][0]
    erds = streams[erds_pos]
    lda = streams[lda_pos]
    erds_time = erds['time_stamps']
    erds_values = erds['time_series']
    lda_time = lda['time_stamps']
    lda_values = lda['time_series']

    # gets 'BrainVision RDA Markers' stream
    orn_pos = np.where(streams_info == 'BrainVision RDA Markers')[0][0]
    orn = streams[orn_pos]
    orn_signal = orn['time_series']
    orn_instants = orn['time_stamps']

    # gets 'BrainVision RDA Data' stream - EEG data
    eeg_pos = np.where(streams_info == lsl_config['eeg']['name'])[0][0]
    eeg = streams[eeg_pos]
    eeg_signal = eeg['time_series'][:, :32]
    eeg_signal = eeg_signal * 1e-6
    # get the instants
    eeg_instants = eeg['time_stamps']
    # get the sampling frequencies
    eeg_fs = int(float(eeg['info']['nominal_srate'][0]))
    effective_sample_frequency = float(eeg['info']['effective_srate'])

    # gets marker stream
    marker_pos = np.where(streams_info == lsl_config['marker']['name'])[0][0]
    marker = streams[marker_pos]
    marker_ids = marker['time_series']
    marker_instants = marker['time_stamps']
    marker_dict = {
        'Reference': 10,
        'Start_of_Trial_l': 1,
        'Start_of_Trial_r': 2,
        'Cue': 20,
        'Feedback': 30,
        'End_of_Trial': 5,
        'Session_Start': 4,
        'Session_End': 3}

    erds_on = []
    eeg_time_round = np.round(eeg_instants, 4)
    erds_time_round = np.round(erds_time, 4)
    lda_time_round = np.round(lda_time, 4)
    marker_instants_round = np.round(marker_instants,4)

    marker_values = []
    task_class = 0
    for index, marker in enumerate(marker_ids):
        if marker == ['Start_of_Trial_l']:
            task_class = 1 # change class
            value = marker_dict[marker[0]]
            marker_values.append([value])
        elif marker == ['Start_of_Trial_r']:
            task_class = 2 # change class
            value = marker_dict[marker[0]]
            marker_values.append([value])
        elif marker == ['Reference'] or marker == ['Cue'] or marker == ['Feedback']: # apply class to ref, cue, fb
            value = marker_dict[marker[0]] + task_class
            marker_values.append([value])
        else:
            value = marker_dict[marker[0]]
            marker_values.append([value])

    marker_values = [[0]] + marker_values # add zero for data before paradigm is started

    i = 0
    marker_interpol = []
    for index, time in enumerate(eeg_time_round):
        if time < marker_instants_round[i]:
            marker_interpol.extend(marker_values[i])
        elif time >= marker_instants_round[i]:
            i += 1
            marker_interpol.extend(marker_values[i])

        if i > len(marker_values)-2:
            break

    while len(marker_interpol) < len(eeg_instants):
        marker_interpol.extend(marker_values[i])

    plt.plot(eeg_instants, eeg_signal[:, 1])
    plt.scatter(erds_time, erds_values[:, 0])
    plt.plot(eeg_instants, marker_interpol)
    plt.scatter(marker_instants, marker_values[1:])

    # find timespans when left/ right times with feedback

    fb_times_l = np.zeros((marker_ids.count(['Start_of_Trial_l']), 2))
    fb_times_r = np.zeros((marker_ids.count(['Start_of_Trial_r']), 2))
    i, j = 0, 0
    for index, marker in enumerate(marker_interpol):
        if marker == 31 and marker_interpol[index-1] != 31: #FB_l start
            fb_times_l[i, 0] = eeg_instants[index]
        elif marker != 31 and marker_interpol[index-1] == 31: #FB_l end
            fb_times_l[i, 1] = eeg_instants[index]
            i += 1
        if marker == 32 and marker_interpol[index-1] != 32: #FB_r start
            fb_times_r[j, 0] = eeg_instants[index]
        elif marker != 32 and marker_interpol[index-1] == 32: #FB_r end
            fb_times_r[j, 1] = eeg_instants[index]
            j += 1

    ref_times_l = np.zeros((marker_ids.count(['Start_of_Trial_l']), 2))
    ref_times_r = np.zeros((marker_ids.count(['Start_of_Trial_r']), 2))
    i, j = 0, 0
    for index, marker in enumerate(marker_interpol):
        if marker == 11 and marker_interpol[index-1] != 11: #ref_l start
            ref_times_l[i, 0] = eeg_instants[index]
        elif marker != 11 and marker_interpol[index-1] == 11: #ref_l end
            ref_times_l[i, 1] = eeg_instants[index]
            i += 1
        if marker == 12 and marker_interpol[index-1] != 12: #ref_r start
            ref_times_r[j, 0] = eeg_instants[index]
        elif marker != 12 and marker_interpol[index-1] == 12: #ref_r end
            ref_times_r[j, 1] = eeg_instants[index]
            j += 1

    print("Debug")

    # ==============================================================================
    # ERDS and LDA from ONLINE calculated results
    # ==============================================================================
    """
    # calculate ERDS per task

    mean_erds_l = np.zeros((len(fb_times_l), 6))
    for task in range(len(fb_times_l)): # alle linken tasks
        erds_per_task = [erds_values[i] for i in range(len(erds_time)) if
                         fb_times_l[task][0] <= erds_time[i] <= fb_times_l[task][
                             1]]  # finde alle erds werte die zwischen start und ende für diesen Trial liegen
        erds_roi = []
        for roi in range(6):  # iteration über rois
            for sample in range(len(erds_per_task)): # jeder wert wird angefügt
                erds_roi.append(erds_per_task[sample][roi])
            mean_erds_l[task][roi] = np.mean(erds_roi)

    mean_erds_r = np.zeros((len(fb_times_r), 6))
    for task in range(len(fb_times_r)):  # alle rechten tasks
        erds_per_task = [erds_values[i] for i in range(len(erds_time)) if
                         fb_times_r[task][0] <= erds_time[i] <= fb_times_r[task][
                             1]]  # finde alle erds werte die zwischen start und ende für diesen Trial liegen
        erds_roi = []
        for roi in range(6):  # iteration über rois
            for sample in range(len(erds_per_task)):  # jeder wert wird angefügt
                erds_roi.append(erds_per_task[sample][roi])
            mean_erds_r[task][roi] = np.mean(erds_roi)
            erds_roi = [] #clear for next roi

    print("Debug")
    '''
    for i in range(6): # zeigt mittleren erds wert pro task und pro ROI
        plt.plot(range(len(fb_times_l)), mean_erds_l[:, i])
        plt.plot(range(len(fb_times_r)), mean_erds_r[:, i])
    plt.show()
    '''
    # ------ bis hier hin bin ich momentan zufrieden! ------

    # LDA classification
    '''
    plt.plot(eeg_instants, eeg_signal[:, 8])
    plt.plot(eeg_instants, marker_interpol)
    plt.scatter(marker_instants, marker_values[1:])
    plt.scatter(lda_time, lda_values[:,0])
    '''
    mean_lda_l = np.zeros((len(fb_times_l)))
    for task in range(len(fb_times_l)):  # alle linken tasks
        lda_cur = []
        lda_per_task = [lda_values[i] for i in range(len(lda_time)) if
                         fb_times_l[task][0] <= lda_time[i] <= fb_times_l[task][
                             1]]  # finde alle lda werte die zwischen start und ende für diesen Trial liegen
        for sample in range(len(lda_per_task)):  # jeder wert wird angefügt
            lda_cur.append(lda_per_task[sample][1]) # only append lda distance, not calculated class label
        mean_lda_l[task] = np.mean(lda_cur)
        lda_cur = [] # clear for next task

    mean_lda_r = np.zeros((len(fb_times_r)))
    for task in range(len(fb_times_r)):  # alle linken tasks
        lda_cur = []
        lda_per_task = [lda_values[i] for i in range(len(lda_time)) if
                         fb_times_r[task][0] <= lda_time[i] <= fb_times_r[task][
                             1]]  # finde alle lda werte die zwischen start und ende für diesen Trial liegen
        for sample in range(len(lda_per_task)):  # jeder wert wird angefügt
            lda_cur.append(lda_per_task[sample][1]) # only append lda distance, not calculated class label
        mean_lda_r[task] = np.mean(lda_cur)
        lda_cur = [] # clear for next task
    '''
    plt.plot(range(len(fb_times_l)), mean_lda_l)
    plt.plot(range(len(fb_times_r)), mean_lda_r)
    plt.show()
    '''
    '''
    filename = cwd + '/Results/' + subject_id + '_ses' + str(n_session) + '_run' + str(n_run)
    filename_on_l = cwd + '/Results/' + subject_id + '_ses' + str(n_session) + '_run' + str(n_run)  + '_mean_erds_l.csv'
    filename_on_r = cwd + '/Results/' + subject_id + '_ses' + str(n_session) + '_run' + str(n_run)  + '_mean_erds_r.csv'
    np.savetxt(filename_on_l, mean_erds_l, delimiter=',')
    np.savetxt(filename_on_r, mean_erds_r, delimiter=',')
    np.savetxt(filename + '_mean_lda_l.csv', mean_lda_l, delimiter=',')
    np.savetxt(filename + '_mean_lda_r.csv', mean_lda_r, delimiter=',')
    '''
    """

    """
    # ERROR: CAN BE DISCHARTED - RESULTS ARE SHIT!
    # ==============================================================================
    # ERDS and LDA from OFFLINE calculated results
    # ==============================================================================
    # calculate mean ref per task

    eeg_roi_raw = eeg_signal[:,[2,29,7,24,13,19]] # roi channels F3, F4, C3, C4, P3, P4
    #eeg_roi_filt = butter_bandpass_filter(eeg_roi_raw, fs=sample_rate/2, fpass=(9,11) ,fstop=(7.5, 12))
    #eeg_roi = np.square(eeg_roi_filt)
    s_rate_half = sample_rate/2
    fpass = (9, 11)
    fstop = (7.5, 12)
    fstop_erds = [freq / s_rate_half for freq in fstop]
    fpass_erds = [freq / s_rate_half for freq in fpass]
    bp = bandpass.Bandpass(order=12, fstop=fstop_erds, fpass=fpass_erds, n=len(eeg_roi_raw[1]))
    eeg_roi_filt = bp.bandpass_filter(eeg_roi_raw)
    eeg_roi = np.square(eeg_roi_filt)
    '''
    mean_ref_l = np.zeros((len(ref_times_l), 6))
        for task in range(len(ref_times_l)): # all left tasks
        ref_per_task = [eeg_roi[i] for i in range(len(eeg_instants)) if
                         fb_times_l[task][0] <= eeg_instants[i] <= fb_times_l[task][
                             1]]  # find all eeg values lying between start and end for this task
        ref_roi = []
        for roi in range(6):  # iteration over rois
            for sample in range(len(ref_per_task)): # find every value
                ref_roi.append(ref_per_task[sample][roi])
            mean_ref_l[task][roi] = np.mean(ref_roi)

    act_l =[]
    for task in range(len(fb_times_l)): # all left tasks
        fb_per_task = [eeg_roi[i] for i in range(len(eeg_instants)) if
                         fb_times_l[task][0] <= eeg_instants[i] <= fb_times_l[task][
                             1]]  # find all eeg values lying between start and end for this task
        act_l.append(fb_per_task)
    '''
    # Compute mean reference EEG for each ROI
    mean_ref_l = []
    for task in range(len(ref_times_l)):
        start_time, end_time = fb_times_l[task]
        ref_per_task = [eeg_roi[i] for i in range(len(eeg_instants)) if start_time <= eeg_instants[i] <= end_time]
        ref_per_task = np.array(ref_per_task)
        mean_ref_l.append(np.mean(ref_per_task, axis=0))

    mean_ref_r = []
    for task in range(len(ref_times_r)):
        start_time, end_time = fb_times_r[task]
        ref_per_task = [eeg_roi[i] for i in range(len(eeg_instants)) if start_time <= eeg_instants[i] <= end_time]
        ref_per_task = np.array(ref_per_task)
        mean_ref_r.append(np.mean(ref_per_task, axis=0))

    # Extract EEG data for each left task
    act_l = []
    for task in range(len(fb_times_l)):
        start_time, end_time = fb_times_l[task]
        fb_per_task = [eeg_roi[i] for i in range(len(eeg_instants)) if start_time <= eeg_instants[i] <= end_time]
        act_l.append(fb_per_task)

    mean_erds_calc_l_ = np.zeros((len(fb_times_l),6))
    for task in range(len(fb_times_l)):
        erds_per_task = -(np.divide((mean_ref_l[task]-act_l[task]), mean_ref_l[task]))
        erds_roi=[]
        for roi in range(6):
            mean_erds_calc_l_[task][roi] = np.mean(erds_per_task[:,roi])

    # Extract EEG data for each right task
    act_r= []
    for task in range(len(fb_times_r)):
        start_time, end_time = fb_times_r[task]
        fb_per_task = [eeg_roi[i] for i in range(len(eeg_instants)) if start_time <= eeg_instants[i] <= end_time]
        act_r.append(fb_per_task)

    mean_erds_calc_r_ = np.zeros((len(fb_times_r), 6))
    for task in range(len(fb_times_r)):
        erds_per_task = -(np.divide((mean_ref_r[task] - act_r[task]), mean_ref_r[task]))
        erds_roi = []
        for roi in range(6):
            mean_erds_calc_r_[task][roi] = np.mean(erds_per_task[:, roi])

    filename_off_l = cwd + '/Results/' + subject_id + '_ses' + str(n_session) + '_run' + str(
        n_run) + 'mean_erds_calc_l_2.csv'
    filename_off_r = cwd + '/Results/' + subject_id + '_ses' + str(n_session) + '_run' + str(
        n_run) + 'mean_erds_calc_r_2.csv'
    np.savetxt(filename_off_l, mean_erds_calc_l_, delimiter=',')
    np.savetxt(filename_off_r, mean_erds_calc_r_, delimiter=',')

    print("Created bullshit!")
    """
