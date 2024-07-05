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


def assess_erds_online(erds_values, erds_time, fb_times, n_fb, n_roi):
    erds_online = np.zeros((len(fb_times[:, 0]), n_fb+50, n_roi+1))
    for trial in range(len(fb_times[:,0])):
        cnt = 0
        for index, t in enumerate(erds_time):
            if fb_times[trial][0] <= t <= fb_times[trial][1]:
                erds_online[trial][cnt][0] = t # time
                erds_online[trial][cnt][1:7] = erds_values[index] # ERDS per ROI
                cnt += 1
    erds_online = remove_zero_lines(erds_online)  # mark to remove zero-lines

    return erds_online # weird shape because erds_online has not same length for every trial


def get_data_ref(eeg, eeg_instants, ref_times, n_ref):
    data_ref_mean = np.zeros((len(ref_times), 1, eeg.shape[1]))
    data_ref_mean[:] = np.nan
    data_ref = np.zeros((len(ref_times), n_ref + 50, eeg.shape[1] + 1))
    for trial in range(len(ref_times)):
        cnt = 0
        cur_ref = np.zeros((1, eeg.shape[1]))  # first line only zeros
        for index, t in enumerate(eeg_instants):
            if ref_times[trial][0] <= t <= ref_times[trial][1]:
                sample = eeg[index, :][np.newaxis, :]
                # sample = bp_erds.bandpass_filter(sample) # additional filtering: initialize and apply BP filter
                data_ref[trial, cnt, 0] = t  # time
                data_ref[trial, cnt, 1:] = np.square(sample)  # squared eeg sample
                cur_ref = np.append(cur_ref, np.square(sample), axis=0)
                cnt += 1
        data_ref_mean[trial] = np.mean(np.delete(cur_ref, 0, 0), axis=0)  # remove first line with only zeros
    data_ref = remove_zero_lines(data_ref)
    data_ref_mean = data_ref_mean.reshape(data_ref_mean.shape[0], data_ref_mean.shape[2])

    return data_ref, data_ref_mean


def get_data_a_calc_erds(eeg, eeg_instants, fb_times, n_fb, data_ref_mean, roi_enabled_ix):
    data_a = np.zeros((len(fb_times), n_fb + 50, eeg.shape[1] + 1))
    erds_offline_ch = np.zeros((len(fb_times), n_fb + 50, eeg.shape[1] + 1))
    for trial in range(len(fb_times)):
        cnt = 0
        for index, t in enumerate(eeg_instants):
            if fb_times[trial][0] <= t <= fb_times[trial][1]:
                sample = eeg[index, :][np.newaxis, :]
                # sample = bp_erds.bandpass_filter(sample) # additional filtering
                erds_ref = data_ref_mean[trial]
                erds_a = np.square(sample)
                data_a[trial, cnt, 0] = t  # time
                data_a[trial, cnt, 1:] = erds_a  # squared eeg sample
                cur_erds = np.divide(-(erds_ref - erds_a), erds_ref)
                erds_offline_ch[trial, cnt, 0] = t  # time
                erds_offline_ch[trial, cnt, 1:] = cur_erds  # erds value
                cnt += 1

    ch_to_select = list(np.add(roi_enabled_ix, 1))
    ch_to_select.insert(0, 0)
    erds_offline_roi = erds_offline_ch[:, :, ch_to_select]

    data_a = remove_zero_lines(data_a)
    erds_offline_ch = remove_zero_lines(erds_offline_ch)
    erds_offline_roi = remove_zero_lines(erds_offline_roi)

    return data_a, erds_offline_ch, erds_offline_roi

