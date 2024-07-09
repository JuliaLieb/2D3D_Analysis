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
import scipy.io
import SUB_trial_management


def assess_lda_online(lda_values, lda_time, fb_times, n_fb):
    """
    Assesses LDA (Linear Discriminant Analysis) accuracy values which where calculated online for specified
    feedback periods.

    Parameters:
    lda_values (ndarray): A 2D NumPy array of shape (n_times, 2) containing LDA values for different time points, where
                            the first column represents accuracy and the second column represents class.
    lda_time (ndarray): A 1D NumPy array containing the time points corresponding to the LDA values.
    fb_times (ndarray): A 2D NumPy array of shape (n_trials, 2) containing the start and end times for feedback periods
                        in each trial.
    n_fb (int): The number of feedback periods.

    Returns:
    ndarray: A list of 2D NumPy arrays, each of varying lengths corresponding to the trials, containing the time points
            and LDA values for the specified feedback periods.
    """
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
    lda_online = SUB_trial_management.remove_zero_lines(lda_online)

    return lda_online

"""
def extract_epochs(data, n_samples):
    data_labels = data[0, :]
    data = data[1:, :]
    indexes = np.where((data_labels == 121) | (data_labels == 122))[0]

    epochs = []
    n_trials = len(indexes)
    for i in range(n_trials):
        idx1 = indexes[i]
        idx2 = idx1 + n_samples
        if i < n_trials-1 and idx2 > indexes[i+1]:
            idx2 = indexes[i+1]
        epochs.append(data[:, idx1:idx2])

    return epochs, np.array(data_labels[indexes] - 121, dtype=int)

def calc_avg_acc(lda_data):
    data_lda = scipy.io.loadmat(file)['lda'].T
    data_lda, labels = extract_epochs(data_lda, n_samples_task)

    acc_per_trial = []

    for epoch, cl in zip(data_lda, labels):
        acc_per_trial.append(np.sum(epoch[0, :] == cl) / len(epoch[0, :]))


    accuracy = np.mean(acc_per_trial)
    accuracy = round(accuracy, 4)
"""

