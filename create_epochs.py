# ----------------------------------------------------------------------------------------------------------------------
# Creates epochs per session for prerocessed data
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
from mne.datasets import eegbci


def create_epochs(self, signal, visualize_epochs=True, rois=True, set_annotations=True):
    """
    Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
    the raw data according to the events
    :param visualize_epochs: boolean variable to select if generate epochs plots or not
    :param rois: boolean variable to select if visualize results according to the rois or for each channel
    :param set_annotations: boolean variable, if it's necessary to set the annotations or not
    """

    # set the annotations on the current raw and extract the correspondent events
    if set_annotations:
        signal.set_annotations(self.annotations)
    self.events, self.event_mapping = mne.events_from_annotations(signal)

    # Automatic rejection criteria for the epochs
    reject_criteria = self.epochs_reject_criteria

    # generation of the epochs according to the events
    self.epochs = mne.Epochs(signal, self.events, preload=True, baseline=(self.tmin, 0),
                             reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)  # event_id=self.event_mapping,

    # separate left and right
    index_r = np.where(self.events[:, 2] == 2)
    index_l = np.where(self.events[:, 2] == 1)
    self.events_r = self.events[index_r]
    self.events_l = self.events[index_l]
    self.epochs_r = mne.Epochs(signal, self.events_r, preload=True, baseline=(self.tmin, 0),
                               reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)
    self.epochs_l = mne.Epochs(signal, self.events_l, preload=True, baseline=(self.tmin, 0),
                               reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)

    if visualize_epochs:
        self.visualize_all_epochs(signal=False, rois=False)
        # self.visualize_l_r_epochs(signal=False, rois=False)

    self.epochs.save(self.dir_files + '/epochs_run' + str(self.n_run) + '-epo.fif', overwrite=True)
    self.epochs_l.save(self.dir_files + '/epochs_l_run' + str(self.n_run) + '-epo.fif', overwrite=True)
    self.epochs_r.save(self.dir_files + '/epochs_r_run' + str(self.n_run) + '-epo.fif', overwrite=True)

    return self.epochs_l, self.epochs_r

def visualize_all_epochs(self, signal=True, rois=True):
    """
    :param signal: boolean, if visualize the whole signal with triggers or not
    :param conditional_epoch: boolean, if visualize the epochs extracted from the events or the general mean epoch
    :param rois: boolean (only if conditional_epoch=True), if visualize the epochs according to the rois or not
    """
    #self.visualize_raw(signal=signal, psd=False, psd_topo=False)

    rois_names = list(self.roi_info)
    ch_picks = ['C3', 'C4']

    # generate the mean plot considering all the epochs conditions

        # generate the epochs plots according to the roi and save them
    if rois:
        images = self.epochs.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
        for idx, img in enumerate(images):
            img.savefig(self.dir_plots + '/' + rois_names[idx] + '.png')
            plt.close(img)

    # generate the epochs plots for each channel and save them
    else:
        images = self.epochs.plot_image(show=False, picks=ch_picks)
        for idx, img in enumerate(images):
            img.savefig(self.dir_plots + '/' + ch_picks[idx] + '.png')
            plt.close(img)

    plt.close('all')

def visualize_l_r_epochs(self, signal=True, rois=True):

    #self.visualize_raw(signal=signal, psd=False, psd_topo=False)
    rois_names = list(self.roi_info)
    ch_picks = ['C3', 'C4']

    # generate the mean plots according to the condition in the annotation value

    # generate the epochs plots according to the roi and save them
    if rois:
        # right
        images = self.epochs_r.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
        for idx, img in enumerate(images):
            img.savefig(self.dir_plots + '/right_' + rois_names[idx] + '.png')
            plt.close(img)
        # left
        images = self.epochs_l.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
        for idx, img in enumerate(images):
            img.savefig(self.dir_plots + '/left_' + rois_names[idx] + '.png')
            plt.close(img)

    # generate the epochs plots for each channel and save them
    else:
        #right
        images = self.epochs_r.plot_image(picks=ch_picks, show=False)
        for idx, img in enumerate(images):
            img.savefig(self.dir_plots + '/right_' + ch_picks[idx] + '.png')
            plt.close(img)
        #left
        images = self.epochs_l.plot_image(picks=ch_picks, show=False)
        for idx, img in enumerate(images):
            img.savefig(self.dir_plots + '/left_' + ch_picks[idx] + '.png')
            plt.close(img)


def run_epochs(signal):


    print('Epochs created!')