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
from scipy.signal import butter, filtfilt, sosfiltfilt, sosfilt
import bandpass
from datetime import datetime
from scipy import signal

class Bandpass:
    """Bandpass unit.

    Holds parameters and methods for bandpass filtering.

    Parameters
    ----------
    order: `int`
        The order of the filter.
    fpass: `list`
        Frequencies of the filter.
    n: `int`
        Number of eeg channels.

    Other Parameters
    ----------------
    sos: `ndarray`
        Second-order sections representation of the filter.
    zi0: `ndarray`
        Initial conditions for the filter delay.
    zi: `ndarray`
        Current filter delay values.
    """

    def __init__(self, order, fstop, fpass, n):
        self.order = order
        self.fstop = fstop
        self.fpass = fpass
        self.n = n
        self.sos = None
        self.zi0 = None
        self.zi = None

        self.__init_filter()

    def __init_filter(self):
        """Computes the second order sections of the filter and the initial conditions for the filter delay.
        """

        self.sos = signal.iirfilter(int(self.order / 2), self.fpass, btype='bandpass', ftype='butter',
                                    output='sos')

        zi = signal.sosfilt_zi(self.sos)

        if self.n > 1:
            zi = np.tile(zi, (self.n, 1, 1)).T
        self.zi0 = zi.reshape((np.shape(self.sos)[0], 2, self.n))
        self.zi = self.zi0

    def bandpass_filter(self, x):
        """Bandpass filters the input array.

        Parameters
        ----------
        x: `ndarray`
            Raw eeg data.

        Returns
        -------
        y: `int`
            Band passed data.
        """

        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi, axis=0)
        return y


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

def remove_zero_lines(array):
    # input: ndarray size (x,y,z)
    # mask to remove zero-lines
    array_clean = []
    for i in range(array.shape[0]):
        slice_ = array[i]
        mask = ~(np.all(slice_ == 0, axis=1))
        filtered_slice = slice_[mask]
        array_clean.append(filtered_slice)
    array_clean = np.array(array_clean, dtype=object)
    return array_clean


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    result_path = cwd + '/Results/'

    config_file_path = "C:/2D3D_Analysis/Data/S14-ses0/CONFIG_S14_run2_ME_2D.json"
    xdf_file_path = "C:/2D3D_Analysis/Data/S14-ses0/S14_run2_ME_2D.xdf"

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

    # ERDS and LDA recordings
    erds_pos = np.where(streams_info == lsl_config['fb-erds']['name'])[0][0]
    lda_pos = np.where(streams_info == lsl_config['fb-lda']['name'])[0][0]
    erds = streams[erds_pos]
    lda = streams[lda_pos]
    erds_time = erds['time_stamps']-time_zero
    erds_values = erds['time_series']
    lda_time = lda['time_stamps']-time_zero
    lda_values = lda['time_series']

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


    # find timespans when left/ right times with feedback
    
    fb_times_l = np.zeros((marker_ids.count(['Start_of_Trial_l']), 2))
    fb_times_r = np.zeros((marker_ids.count(['Start_of_Trial_r']), 2))
    i, j = 0, 0
    for index, marker in enumerate(marker_interpol):
        if marker == 31 and marker_interpol[index-1] != 31: #FB_l start
            fb_times_l[i, 0] = eeg_instants[index]
        elif marker == 31 and marker_interpol[index+1] != 31: #FB_l end
            fb_times_l[i, 1] = eeg_instants[index]
            i += 1
        if marker == 32 and marker_interpol[index-1] != 32: #FB_r start
            fb_times_r[j, 0] = eeg_instants[index]
        elif marker == 32 and marker_interpol[index+1] != 32: #FB_r end
            fb_times_r[j, 1] = eeg_instants[index]
            j += 1

    ref_times_l = np.zeros((marker_ids.count(['Start_of_Trial_l']), 2))
    ref_times_r = np.zeros((marker_ids.count(['Start_of_Trial_r']), 2))
    i, j = 0, 0
    for index, marker in enumerate(marker_interpol):
        if marker == 11 and marker_interpol[index-1] != 11: #ref_l start
            ref_times_l[i, 0] = eeg_instants[index]
        elif marker == 11 and marker_interpol[index+1] != 11: #ref_l end
            ref_times_l[i, 1] = eeg_instants[index]
            i += 1
        if marker == 12 and marker_interpol[index-1] != 12: #ref_r start
            ref_times_r[j, 0] = eeg_instants[index]
        elif marker == 12 and marker_interpol[index+1] != 12: #ref_r end
            ref_times_r[j, 1] = eeg_instants[index]
            j += 1

    #"""
    # ==============================================================================
    # ERDS from ONLINE calculated results
    # ==============================================================================
    erds_l_clean = np.zeros((len(fb_times_l[:,0]),3600,7)) # [trial][timepoint][ROI1][ROI2]...
    for trial in range(len(fb_times_l[:,0])):
        cnt = 0
        for index, t in enumerate(erds_time):
            if fb_times_l[trial][0] <= t <= fb_times_l[trial][1]:
                erds_l_clean[trial][cnt][0] = t
                erds_l_clean[trial][cnt][1:7] = erds_values[index]
                cnt += 1

    erds_r_clean = np.zeros((len(fb_times_r[:,0]),3600,7)) # [trial][timepoint][ROI1][ROI2]...
    for trial in range(len(fb_times_r[:,0])):
        cnt = 0
        for index, t in enumerate(erds_time):
            if fb_times_r[trial][0] <= t <= fb_times_r[trial][1]:
                erds_r_clean[trial][cnt][0] = t
                erds_r_clean[trial][cnt][1:7] = erds_values[index]
                cnt += 1

    # mark to remove zero-lines
    filtered_erds_l_clean = remove_zero_lines(erds_l_clean)
    filtered_erds_r_clean = remove_zero_lines(erds_r_clean)


    # calculate mean ERDS per task and ROI
    avg_erds_l = np.zeros((len(fb_times_l),6))
    avg_times_l = np.zeros(len(fb_times_l))
    for trial in range(filtered_erds_l_clean.shape[0]):
        trial_data = filtered_erds_l_clean[trial]
        for roi in range(6):
            avg_erds_l[trial,roi] = np.mean(trial_data[:, roi+1])
            avg_times_l[trial] = trial_data[-1, 0]

    avg_erds_r = np.zeros((len(fb_times_r),6))
    avg_times_r = np.zeros(len(fb_times_r))
    for trial in range(filtered_erds_r_clean.shape[0]):
        trial_data = filtered_erds_r_clean[trial]
        for roi in range(6):
            avg_erds_r[trial,roi] = np.mean(trial_data[:, roi+1])
            avg_times_r[trial] = trial_data[-1, 0]

    '''
    # plotting
    # plot online calculated ERDS
    roi_color = ['b', 'g', 'r', 'c', 'm', 'y']
    roi_legend = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    plt.figure(figsize=(10, 5))
    for trial in range(len(erds_l_clean)):
        trial_data = filtered_erds_l_clean[trial]
        time_l = trial_data[:, 0]
        for roi in range(6):
            value_l = trial_data[:, roi+1]
            if trial == 0:
                plt.plot(time_l, value_l, color=roi_color[roi], label=roi_legend[roi])
            else:
                plt.plot(time_l, value_l, color=roi_color[roi])
            plt.axvline(x=fb_times_l[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_l[trial,1], color='k', linestyle='dotted')
            plt.scatter(avg_times_l[trial], avg_erds_l[trial, roi], color=roi_color[roi], marker='<')

    for trial in range(len(erds_r_clean)):
        trial_data = filtered_erds_r_clean[trial]
        time_r = trial_data[:, 0]
        for roi in range(6):
            value_r = trial_data[:, roi+1]
            plt.plot(time_r, value_r, color=roi_color[roi])
            plt.axvline(x=fb_times_r[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_r[trial,1], color='k', linestyle='dotted')
            plt.scatter(avg_times_r[trial], avg_erds_r[trial, roi], color=roi_color[roi], marker='<')
    plt.title("Online calucated ERDS Values \n" + config_file_path)
    plt.legend(title='ROIs', loc='best', ncols=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(result_path + timestamp + '_online_ERDS')
    plt.show()
    '''
    #"""

    print("Debug")

    # ==============================================================================
    # LDA from ONLINE calculated results
    # ==============================================================================
    """
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

    #"""
    #
    # ==============================================================================
    # ERDS and LDA from OFFLINE calculated results
    # ==============================================================================
    # calculate mean ref per task
    '''
    eeg_raw = eeg_signal[:,[2,29,7,24,13,19]] # roi channels F3, F4, C3, C4, P3, P4

    #eeg_roi_filt = butter_bandpass_filter(eeg_roi_raw, fs=sample_rate/2, fpass=(9,11) ,fstop=(7.5, 12))
    #eeg_roi = np.square(eeg_roi_filt)
    nyquist = sample_rate/2
    fpass = (9, 11)
    fstop = (7.5, 12)
    low = 9/nyquist
    high = 11/nyquist

    sos = butter(12, [low, high], btype='bandpass', output='sos')
    eeg_filt = sosfilt(sos, eeg_raw)
    eeg_filt_roi =eeg_filt[:,[2,29,7,24,13,19]]
    eeg_roi = np.square(eeg_filt_roi)
    '''
    ############### Versuch wie Online #########################
    eeg_raw = eeg_signal[:,
              [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30]]  # enabled channels
    s_rate_half = sample_rate/2
    fpass_erds = [freq / s_rate_half for freq in[7.5, 12]]
    fstop_erds = [freq / s_rate_half for freq in[9, 11]]
    bp_erds = Bandpass(order=12, fstop=fstop_erds, fpass=fpass_erds, n=len(eeg_raw))


    '''
    fstop_erds = [freq / s_rate_half for freq in fstop]
    fpass_erds = [freq / s_rate_half for freq in fpass]
    bp = bandpass.Bandpass(order=12, fstop=fstop_erds, fpass=fpass_erds, n=len(eeg_roi_raw[1]))
    eeg_roi_filt = bp.bandpass_filter(eeg_roi_raw)
    eeg_roi = np.square(eeg_roi_filt)
    '''
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
    # compute ERDS for every sample in feedback period
    # LEFT
    eeg_roi = eeg_raw[:,[1, 21, 5, 17, 10, 12]]
    ref_samples_l = np.zeros((len(fb_times_l[:,0]),1600,7)) # [trial][timepoint][ROI1][ROI2]...
    act_samples_l = np.zeros((len(fb_times_l[:,0]),3600,7)) # [trial][timepoint][ROI1][ROI2]...
    avg_ref_l = np.zeros((len(fb_times_l[:,0]),6)) #[trial][ROI]
    for trial in range(len(fb_times_l[:,0])):
        #eeg_data = eeg_roi[:,3]
        cnt_ref = 0
        cnt_act = 0
        for index, t in enumerate(eeg_instants):
            if ref_times_l[trial][0] <= t <= ref_times_l[trial][1]:
                #ref_samples_l.append(eeg_data[index])
                ref_samples_l[trial][cnt_ref][0] = t # time
                ref_samples_l[trial][cnt_ref][1:7] = eeg_roi[index]
                cnt_ref += 1
            if fb_times_l[trial][0] <= t <= fb_times_l[trial][1]:
                act_samples_l[trial][cnt_act][0] = t # time
                act_samples_l[trial][cnt_act][1:7] = eeg_roi[index]
                cnt_act += 1
        for roi in range(6):
            avg_ref_l[trial][roi] = np.mean(ref_samples_l[trial][:,roi+1])
    # mark to remove zero-lines
    ref_samples_l_clean = remove_zero_lines(ref_samples_l)
    act_samples_l_clean = remove_zero_lines(act_samples_l)

    erds_value_l = np.zeros((len(fb_times_l[:,0]),3600,7))
    for trial in range(len(fb_times_l[:,0])):
        for index, sample in enumerate(act_samples_l_clean[trial]):
            erds_value_l[trial][index][0] = sample[0] #time
            for roi in range(6):
                erds_value_l[trial][index][roi+1] = -(avg_ref_l[trial][roi]-sample[roi+1])/avg_ref_l[trial][roi]
    erds_value_l_clean = remove_zero_lines(erds_value_l)

    # RIGHT
    ref_samples_r = np.zeros((len(fb_times_r[:,0]),1600,7)) # [trial][timepoint][ROI1][ROI2]...
    act_samples_r = np.zeros((len(fb_times_r[:,0]),3600,7)) # [trial][timepoint][ROI1][ROI2]...
    avg_ref_r = np.zeros((len(fb_times_r[:,0]),6)) #[trial][ROI]
    for trial in range(len(fb_times_r[:,0])):
        #eeg_data = eeg_roi[:,3]
        cnt_ref = 0
        cnt_act = 0
        for index, t in enumerate(eeg_instants):
            if ref_times_r[trial][0] <= t <= ref_times_r[trial][1]:
                #ref_samples_l.append(eeg_data[index])
                ref_samples_r[trial][cnt_ref][0] = t # time
                ref_samples_r[trial][cnt_ref][1:7] = eeg_roi[index]
                cnt_ref += 1
            if fb_times_r[trial][0] <= t <= fb_times_r[trial][1]:
                act_samples_r[trial][cnt_act][0] = t # time
                act_samples_r[trial][cnt_act][1:7] = eeg_roi[index]
                cnt_act += 1
        for roi in range(6):
            avg_ref_r[trial][roi] = np.mean(ref_samples_r[trial][:,roi+1])
    # mark to remove zero-lines
    ref_samples_r_clean = remove_zero_lines(ref_samples_r)
    act_samples_r_clean = remove_zero_lines(act_samples_r)

    erds_value_r = np.zeros((len(fb_times_r[:,0]),3600,7))
    for trial in range(len(fb_times_r[:,0])):
        for index, sample in enumerate(act_samples_r_clean[trial]):
            erds_value_r[trial][index][0] = sample[0] #time
            for roi in range(6):
                erds_value_r[trial][index][roi+1] = -(avg_ref_r[trial][roi]-sample[roi+1])/avg_ref_r[trial][roi]
    erds_value_r_clean = remove_zero_lines(erds_value_r)


    roi_color = ['b', 'g', 'r', 'c', 'm', 'y']
    roi_legend = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    '''
    # plot eeg spans for calculating ERDS
    plt.figure(figsize=(10, 5))
    for trial in range(len(ref_times_l)):
        trial_data = ref_samples_l_clean[trial]
        time_l = trial_data[:, 0]
        for roi in range(6):
            value_l = trial_data[:, roi + 1]
            if trial == 0:
                plt.plot(time_l, value_l, color=roi_color[roi], label=roi_legend[roi])
            else:
                plt.plot(time_l, value_l, color=roi_color[roi])
            plt.axvline(x=ref_times_l[trial, 0], color='k', linestyle='dotted')
            plt.axvline(x=ref_times_l[trial, 1], color='k', linestyle='dotted')
            # plt.scatter(avg_times_l[trial], avg_erds_l[trial, roi], color=roi_color[roi], marker='<')
    for trial in range(len(fb_times_l)):
        trial_data = act_samples_l_clean[trial]
        time_l = trial_data[:, 0]
        for roi in range(6):
            value_l = trial_data[:, roi + 1]
            if trial == 0:
                plt.plot(time_l, value_l, color=roi_color[roi], label=roi_legend[roi])
            else:
                plt.plot(time_l, value_l, color=roi_color[roi])
            plt.axvline(x=fb_times_l[trial, 0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_l[trial, 1], color='k', linestyle='dotted')
            # plt.scatter(avg_times_l[trial], avg_erds_l[trial, roi], color=roi_color[roi], marker='<')
    plt.title('EEG data in state = reference and state = feedback')
    plt.legend()
    #plt.show()
    '''

    '''
    # plot offline calculated ERDS
    plt.figure(figsize=(10, 5))
    for trial in range(len(fb_times_l)):
        trial_data = erds_value_l_clean[trial]
        time_l = trial_data[:, 0]
        for roi in range(6):
            value_l = trial_data[:, roi+1]
            if trial == 0:
                plt.plot(time_l, value_l, color=roi_color[roi], label=roi_legend[roi])
            else:
                plt.plot(time_l, value_l, color=roi_color[roi])
            plt.axvline(x=fb_times_l[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_l[trial,1], color='k', linestyle='dotted')
            #plt.scatter(avg_times_l[trial], avg_erds_l[trial, roi], color=roi_color[roi], marker='<')
    plt.title('ERDS in state = feedback')
    plt.legend()
    plt.show()
    '''

    # Compare online/offline ERDS
    plt.figure(figsize=(15, 5))
    #for trial in range(len(fb_times_l)):
    for trial in range(1):
        trial_data = erds_value_r_clean[trial]
        time_r = trial_data[:, 0]
        #for roi in range(1): # hier nur ein ROI!
        roi=4 # C4
        value_r = trial_data[:, roi] #+1]
        plt.plot(time_r, value_r, color='b', label='offline calculated ERDS')
        plt.axvline(x=fb_times_r[trial, 0], color='k', linestyle='dotted')
        plt.axvline(x=fb_times_r[trial, 1], color='k', linestyle='dotted')
        # plt.scatter(avg_times_l[trial], avg_erds_l[trial, roi], color=roi_color[roi], marker='<')
    #for trial in range(len(erds_l_clean)):
    for trial in range(1):
        trial_data = filtered_erds_r_clean[trial]
        time_r = trial_data[:, 0]
        #for roi in range(1):
        roi = 4 #C4
        value_r = trial_data[:, roi] #+1]
        plt.plot(time_r, value_r, color='r', label='online calculated ERDS')
        plt.axvline(x=fb_times_r[trial,0], color='k', linestyle='dotted')
        plt.axvline(x=fb_times_r[trial,1], color='k', linestyle='dotted')
        plt.scatter(avg_times_r[trial], avg_erds_r[trial, roi], color=roi_color[roi], marker='<')
    plt.title('ERDS in state = feedback')
    plt.legend()
    plt.show()



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
    '''
