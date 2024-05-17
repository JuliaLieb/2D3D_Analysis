# ----------------------------------------------------------------------------------------------------------------------
# Calculate values for statistical comparison
# ----------------------------------------------------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import os
import scipy.io
import glob
import sys
import matplotlib
matplotlib.use('Qt5Agg')



def run(epochs):
    # regions of interest
    roi_dict = {
        "frontal left": ["F3", "F7"],
        "frontal right": ["F4", "F8"],
        "central left": ["FC1", "FC5", "C3", "T7"],
        "central right": ["FC2", "FC6", "C4", "T8"],
        "parietal left": ["CP5", "CP1", "P3", "P7"],
        "parietal right": ["CP6", "CP2", "P4", "P8"]}

    sample_rate = 500
    time_r = [int(1.5*sample_rate), 3*sample_rate]
    time_a = [int(4.25*sample_rate), int(11.25*sample_rate)]

    erds_roi_list = []
    for roi in roi_dict.keys():
        erds_epoch = []
        epochs_roi = epochs.copy().pick(roi_dict[roi])
        for epoch in epochs_roi:
            r_period = epoch[:, time_r[0]:time_r[1]]
            r_mean = np.average(r_period)
            a_period = epoch[:, time_a[0]:time_a[1]]
            a_mean = np.average(a_period)
            erds_epoch.append(-(r_mean - a_mean) / r_mean)
        erds_roi_list.append(np.mean(erds_epoch))

    return erds_roi_list
