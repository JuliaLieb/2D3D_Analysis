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

        self.sos = signal.iirfilter(int(self.order / 2), self.fpass, btype='bandpass', ftype='butter', output='sos')
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

    ###### Load files: infos and data

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

    ######## Manage markers
    marker_values = [] # marker as value - not str
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
    marker_interpol = [] # marker for every sample
    for index, time in enumerate(eeg_instants):
        if time < marker_instants[i]:
            marker_interpol.extend(marker_values[i])
        elif time >= marker_instants[i]:
            i += 1
            marker_interpol.extend(marker_values[i])

        if i > len(marker_values)-2:
            break

    while len(marker_interpol) < len(eeg_instants):
        marker_interpol.extend(marker_values[i])


    # find timespans when left/ right times with feedback
    # column[0]: start time, column[1] end time
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

    # find timespans when left/ right times for reference
    # column[0]: start time, column[1] end time
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
    erds_online_l = np.zeros((len(fb_times_l[:, 0]), 3600, 7))
    # column[0]: time, columns[1-7]: erds per ROI
    for trial in range(len(fb_times_l[:,0])):
        cnt = 0
        for index, t in enumerate(erds_time):
            if fb_times_l[trial][0] <= t <= fb_times_l[trial][1]:
                erds_online_l[trial][cnt][0] = t
                erds_online_l[trial][cnt][1:7] = erds_values[index]
                cnt += 1
    erds_online_l = remove_zero_lines(erds_online_l)  # mark to remove zero-lines

    erds_online_r = np.zeros((len(fb_times_r[:, 0]), 3600, 7))
    # column[0]: time, columns[1-7]: erds per ROI
    for trial in range(len(fb_times_r[:,0])):
        cnt = 0
        for index, t in enumerate(erds_time):
            if fb_times_r[trial][0] <= t <= fb_times_r[trial][1]:
                erds_online_r[trial][cnt][0] = t
                erds_online_r[trial][cnt][1:7] = erds_values[index]
                cnt += 1
    erds_online_r = remove_zero_lines(erds_online_r) # mark to remove zero-lines

    # calculate mean ERDS per task and ROI
    avg_erds_online_l = np.zeros((len(fb_times_l), 6))
    last_t_fb_on_l = np.zeros(len(fb_times_l))
    for trial in range(erds_online_l.shape[0]):
        trial_data = erds_online_l[trial]
        for roi in range(6):
            avg_erds_online_l[trial,roi] = np.mean(trial_data[:, roi + 1])
            last_t_fb_on_l[trial] = trial_data[-1, 0]

    avg_erds_online_r = np.zeros((len(fb_times_r), 6))
    last_t_fb_on_r = np.zeros(len(fb_times_r))
    for trial in range(erds_online_r.shape[0]):
        trial_data = erds_online_r[trial]
        for roi in range(6):
            avg_erds_online_r[trial,roi] = np.mean(trial_data[:, roi + 1])
            last_t_fb_on_r[trial] = trial_data[-1, 0]


    # plotting
    # plot online calculated ERDS with vertical lines for start and stop time of FB
    '''
    roi_color = ['b', 'g', 'r', 'c', 'm', 'y']
    roi_legend = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    plt.figure(figsize=(10, 5))
    for trial in range(len(erds_online_l)): # left
        trial_data = erds_online_l_[trial]
        time_l = trial_data[:, 0]
        for roi in range(6):
            value_l = trial_data[:, roi+1]
            if trial == 0:
                plt.plot(time_l, value_l, color=roi_color[roi], label=roi_legend[roi])
            else:
                plt.plot(time_l, value_l, color=roi_color[roi])
            plt.axvline(x=fb_times_l[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_l[trial,1], color='k', linestyle='dotted')
            plt.scatter(last_t_fb_on_l[trial], avg_erds_online_l[trial, roi], color=roi_color[roi], marker='<')
    for trial in range(len(erds_online_r)): # right
        trial_data = erds_online_r_[trial]
        time_r = trial_data[:, 0]
        for roi in range(6):
            value_r = trial_data[:, roi+1]
            plt.plot(time_r, value_r, color=roi_color[roi])
            plt.axvline(x=fb_times_r[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_r[trial,1], color='k', linestyle='dotted')
            plt.scatter(last_t_fb_on_r[trial], avg_erds_online_r[trial, roi], color=roi_color[roi], marker='<')
    plt.title("Online calucated ERDS Values \n" + config_file_path)
    plt.legend(title='ROIs', loc='best', ncols=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(result_path + timestamp + '_online_ERDS')
    #plt.show()
    '''
    #"""

    print("Debug")

    # ==============================================================================
    # LDA from ONLINE calculated results - not up to date!
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

    #
    enabled_ch = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30]
    enabled_ch_names = ['Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'P4', 'P8', 'CP6',
                       'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8']
    eeg_raw = eeg_signal[:, enabled_ch]
    ####### TESTING Data
    '''
    eeg_start = np.ones((2314, np.size(eeg_raw, axis=1))) # vor erster Ref
    eeg_ref = np.ones((1500, np.size(eeg_raw, axis=1)))
    eeg_cue = np.ones((625, np.size(eeg_raw, axis=1)))
    eeg_fb = np.ones((3500, np.size(eeg_raw, axis=1))) *2
    eeg_trial = np.concatenate((eeg_ref, eeg_cue, eeg_fb), axis=0)
    eeg_test = np.concatenate((eeg_start, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial,
                             eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial,
                             eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_trial, eeg_start, eeg_start))
    eeg_raw = eeg_test
    eeg_instants = eeg_instants[:len(eeg_raw)]
    '''
    #######################

    roi_ch = [1, 21, 5, 17, 10, 12]
    roi_ch_names = ['F4', 'F4', 'C3', 'C4', 'P3', 'P4']
    roi_color = ['b', 'g', 'r', 'c', 'm', 'y']

    s_rate_half = sample_rate/2
    fpass_erds = [freq / s_rate_half for freq in [9, 11]]
    fstop_erds = [freq / s_rate_half for freq in [7.5, 12]]
    bp_erds = Bandpass(order=12, fstop=fstop_erds, fpass=fpass_erds, n=eeg_raw.shape[1])

    # eeg data for reference period of every task # left
    data_ref_mean_l = np.zeros((len(ref_times_l), 1, eeg_raw.shape[1]))
    data_ref_mean_l[:] = np.nan
    data_time_ref_mean_l = []
    data_ref_l = np.zeros((len(ref_times_l), 1600, eeg_raw.shape[1]+1))
    for trial in range(len(ref_times_l)):
        cnt = 0
        data_ref = np.zeros((1, eeg_raw.shape[1]))  # first line only zeros
        for index, t in enumerate(eeg_instants):
            if ref_times_l[trial][0] <= t <= ref_times_l[trial][1]:
                sample = eeg_raw[index,:]
                sample_reshaped = sample[np.newaxis,:]
                sample_filt = bp_erds.bandpass_filter(sample_reshaped)
                #sample_filt = sample_reshaped
                ##########data_ref = np.append(data_ref, np.square(bp_erds.bandpass_filter(sample_reshaped)), axis=0) # apply filter!
                data_ref_l[trial, cnt, 0] = t
                data_ref_l[trial, cnt, 1:] = np.square(sample_filt)
                data_ref = np.append(data_ref, np.square(sample_filt), axis=0)
                cnt += 1
        data_ref_mean_l[trial] = np.mean(np.delete(data_ref, 0, 0), axis=0) # remove first line with only zeros
        data_time_ref_mean_l.append(t)
    data_ref_l=remove_zero_lines(data_ref_l)

    # eeg data for reference period of every task # right
    data_ref_mean_r = np.zeros((len(ref_times_r), 1, eeg_raw.shape[1]))
    data_ref_mean_r[:] = np.nan
    data_time_ref_mean_r = []
    data_ref_r = np.zeros((len(ref_times_l), 1600, eeg_raw.shape[1]+1))
    for trial in range(len(ref_times_r)):
        cnt = 0
        data_ref = np.zeros((1, eeg_raw.shape[1]))  # first line only zeros
        for index, t in enumerate(eeg_instants):
            if ref_times_r[trial][0] <= t <= ref_times_r[trial][1]:
                sample = eeg_raw[index, :]
                sample_reshaped = sample[np.newaxis, :]
                sample_filt = bp_erds.bandpass_filter(sample_reshaped)
                # sample_filt = sample_reshaped
                ##########data_ref = np.append(data_ref, np.square(bp_erds.bandpass_filter(sample_reshaped)), axis=0) # apply filter!
                data_ref_r[trial, cnt, 0] = t # row[0]: timepoint
                data_ref_r[trial, cnt, 1:] = np.square(sample_filt)
                data_ref = np.append(data_ref, np.square(sample_filt), axis=0)
                cnt += 1
        data_ref_mean_r[trial] = np.mean(np.delete(data_ref, 0, 0), axis=0)  # remove first line with only zeros
        data_time_ref_mean_r.append(t)
    data_ref_r = remove_zero_lines(data_ref_r)

    # eeg data for feedback period of every task and erds per roi # left
    data_a_l = np.zeros((len(ref_times_l), 3600, eeg_raw.shape[1]+1))
    erds_offline_ch_l = np.zeros((len(ref_times_l), 3600, eeg_raw.shape[1] + 1))
    for trial in range(len(fb_times_l)):
        cnt = 0
        for index, t in enumerate(eeg_instants):
            if fb_times_l[trial][0] <= t <= fb_times_l[trial][1]:
                sample = eeg_raw[index, :]
                sample_reshaped = sample[np.newaxis, :]
                erds_ref = data_ref_mean_l[trial]
                ##########erds_a = np.square(bp_erds.bandpass_filter(sample_reshaped)) # apply filter!
                ##########data_a_r = np.append(data_a, np.square(bp_erds.bandpass_filter(sample_reshaped)), axis=0)  # debug
                sample_filt = bp_erds.bandpass_filter(sample_reshaped)
                #sample_filt = sample_reshaped
                erds_a = np.square(sample_filt)
                data_a_l[trial, cnt, 0] = t # row[0]: timepoint
                data_a_l[trial,cnt,1:] = erds_a
                cur_erds = np.divide(-(-erds_ref - erds_a), erds_ref)
                erds_offline_ch_l[trial, cnt, 0] = t # row[0]: timepoint
                erds_offline_ch_l[trial, cnt, 1:] = cur_erds
                cnt+=1
    data_a_l = remove_zero_lines(data_a_l)
    erds_offline_ch_l = remove_zero_lines(erds_offline_ch_l)

    # eeg data for feedback period of every task and erds per roi # right
    data_a_r = np.zeros((len(ref_times_r), 3600, eeg_raw.shape[1]+1))
    erds_offline_ch_r = np.zeros((len(ref_times_r), 3600, eeg_raw.shape[1]+1))
    for trial in range(len(fb_times_r)):
        cnt = 0
        for index, t in enumerate(eeg_instants):
            if fb_times_r[trial][0] <= t <= fb_times_r[trial][1]:
                sample = eeg_raw[index, :]
                sample_reshaped = sample[np.newaxis, :]
                erds_ref = data_ref_mean_r[trial]
                ##########erds_a = np.square(bp_erds.bandpass_filter(sample_reshaped)) # apply filter!
                ##########data_a_r = np.append(data_a, np.square(bp_erds.bandpass_filter(sample_reshaped)), axis=0)  # debug
                sample_filt = bp_erds.bandpass_filter(sample_reshaped)
                # sample_filt = sample_reshaped
                erds_a = np.square(sample_filt)
                data_a_r[trial, cnt, 0] = t
                data_a_r[trial,cnt,1:] = erds_a
                cur_erds = np.divide(-(-erds_ref - erds_a), erds_ref)
                erds_offline_ch_r[trial, cnt, 0] = t  # row[0]: timepoint
                erds_offline_ch_r[trial, cnt, 1:] = cur_erds
                cnt+=1
    data_a_r = remove_zero_lines(data_a_r)
    erds_offline_ch_r = remove_zero_lines(erds_offline_ch_r)

    print('Debug ERDS calc finished')

    # plotting
    # plot eeg signals for calculating ERDS
    '''
    plt.figure(figsize=(10, 5))
    # LEFT hand trials
    for trial in range(len(data_ref_l)): # left ref
        trial_data = data_ref_l[trial]
        time = trial_data[:, 0]
        for ch in range(23):
            value = trial_data[:, ch+1]
            plt.plot(time, value)#, color=roi_color[roi], label=roi_legend[roi])
            plt.axvline(x=ref_times_l[trial,0], color='grey', linestyle='dotted')
            plt.axvline(x=ref_times_l[trial,1], color='grey', linestyle='dotted')
    for trial in range(len(data_a_l)): # left act
        trial_data = data_a_l[trial]
        time = trial_data[:, 0]
        for ch in range(23):
            value = trial_data[:, ch+1]
            plt.plot(time, value) #, color=roi_color[roi])
            plt.axvline(x=fb_times_l[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_l[trial,1], color='k', linestyle='dotted')
    # RIGHT hand trials
    for trial in range(len(data_ref_r)): # right ref
        trial_data = data_ref_r[trial]
        time = trial_data[:, 0]
        for ch in range(23):
            value = trial_data[:, ch+1]
            plt.plot(time, value)#, color=roi_color[roi], label=roi_legend[roi])
            plt.axvline(x=ref_times_r[trial,0], color='grey', linestyle='dotted')
            plt.axvline(x=ref_times_r[trial,1], color='grey', linestyle='dotted')
    for trial in range(len(data_a_r)): # right act
        trial_data = data_a_r[trial]
        time = trial_data[:, 0]
        for ch in range(23):
            value = trial_data[:, ch+1]
            plt.plot(time, value) #, color=roi_color[roi])
            plt.axvline(x=fb_times_r[trial,0], color='k', linestyle='dotted')
            plt.axvline(x=fb_times_r[trial,1], color='k', linestyle='dotted')
    plt.title("EEG signals for calculating ERDS Values \n" + config_file_path)
    plt.legend(title='CHs', loc='best', ncols=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(result_path + timestamp + '_R_A_signal')
    #plt.show()
    '''

    # plotting
    # plot offline calculated ERDS values
    '''
    plt.figure(figsize=(10, 5))
    # LEFT hand trials
    for trial in range(len(erds_offline_ch_l)):  # left
        trial_data = erds_offline_ch_l[trial]
        time = trial_data[:, 0]
        for ch_ix, ch in enumerate(enabled_ch_names):
            value = trial_data[:, ch_ix + 1]
            if ch in roi_ch_names:
                plt.plot(time, value) #, label=enabled_ch_names[ch_ix])
            plt.axvline(x=fb_times_l[trial, 0], color='grey', linestyle='dotted')
            plt.axvline(x=fb_times_l[trial, 1], color='grey', linestyle='dotted')
    # RIGHT hand trials
    for trial in range(len(erds_offline_ch_r)):  # left
        trial_data = erds_offline_ch_r[trial]
        time = trial_data[:, 0]
        for ch_ix, ch in enumerate(enabled_ch_names):
            value = trial_data[:, ch_ix + 1]
            if ch in roi_ch_names:
                plt.plot(time, value) #, label=enabled_ch_names[ch_ix])
            plt.axvline(x=fb_times_r[trial, 0], color='grey', linestyle='dotted')
            plt.axvline(x=fb_times_r[trial, 1], color='grey', linestyle='dotted')
    plt.title("Offline calculated ERDS values \n" + config_file_path)
    plt.legend(title='ROIs', loc='best', ncols=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(result_path + timestamp + '_offline_ERDS')
    #plt.show()
    '''

    # plot comparison online - offline ERDS
    trial = 1
    ch_comp = ['C4']

    plt.figure(figsize=(10, 5))
    on_erds = erds_online_r[trial]
    on_time = erds_online_r[trial][:, 0]
    off_erds = erds_offline_ch_r[trial]
    off_time = erds_offline_ch_r[trial][:, 0]
    for ch_ix, ch in enumerate(enabled_ch_names):
        if ch in ch_comp:
            #plt.plot(on_time, on_erds[:,ch_ix+1], label='online')
            plt.plot(off_time, off_erds[:, ch_ix + 1], label='offline')
    plt.plot(on_time, on_erds[:, 4], label='online')
    plt.legend()
    plt.title("Comparison online / offline calculated ERDS \n" + config_file_path)
    plt.show()

