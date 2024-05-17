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
    erds = []
    sample_rate = 500
    time_r = [int(1.5*sample_rate), 3*sample_rate]
    time_a = [int(4.25*sample_rate), int(11.25*sample_rate)]

    for epoch in epochs:
        r_period = epoch[:, time_r[0]:time_r[1]]
        r_mean = np.average(r_period)
        a_period = epoch[:, time_a[0]:time_a[1]]
        a_mean = np.average(a_period)
        erds.append(-(r_mean - a_mean) / r_mean)

    return np.mean(erds)
