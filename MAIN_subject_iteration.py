# %%
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
import SUB_one_subject_erds_online, SUB_one_subject_erds_offline


if __name__ == "__main__":
    # %%
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    results_path = cwd + '/Results/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # subjects and sessions
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                    'S16', 'S17']
    session_list = [0, 1, 2]

    # study conditions
    mon_me = [0] * len(subject_list)
    mon_mi = [2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2]
    vr_mi = [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1]
    conditions = {'mon_me': mon_me, 'mon_mi': mon_mi, 'vr_mi': vr_mi}
    roi = ["frontal-left", "frontal-right", "central-left", "central-right", "parietal-left", "parietal-right"]
    task = ['l', 'r']
    freq_band = ['raw']  #['alpha', 'beta']

    erds_on_l, erds_on_r = SUB_one_subject_erds_online.compute_erds_per_run(config_file_path="C:/2D3D_Analysis/Data/S14-ses0/CONFIG_S14_run2_ME_2D.json",
                         xdf_file_path="C:/2D3D_Analysis/Data/S14-ses0/S14_run2_ME_2D.xdf",
                         preproc_file_path=None)

    print('ERDS online:')
    print(erds_on_l, erds_on_r)

    erds_off_l, erds_off_r = SUB_one_subject_erds_offline.compute_erds_per_run(config_file_path="C:/2D3D_Analysis/Data/S14-ses0/CONFIG_S14_run2_ME_2D.json",
                         xdf_file_path="C:/2D3D_Analysis/Data/S14-ses0/S14_run2_ME_2D.xdf",
                         preproc_file_path=None)

    print('ERDS offline:')
    print(erds_off_l, erds_off_r)
