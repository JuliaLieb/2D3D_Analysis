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


def assess_lda_online(lda_values, lda_time, fb_times, n_fb):
    lda_online = np.zeros((len(fb_times), n_fb+50, 3))
    for trial in range(len(fb_times)):
        cnt = 0
        for index, t in enumerate(lda_time):
            if fb_times[trial][0] <= t <= fb_times[trial][1]:
                sample = lda_values[index, :]
                lda_online[trial, cnt, 0] = t # time
                lda_online[trial, cnt, 1:] = sample[0] # accuracy
                lda_online[trial, cnt, 2:] = sample[1]  # class
                cnt+=1
    lda_online = remove_zero_lines(lda_online)

    return lda_online