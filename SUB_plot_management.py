import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


roi_color = ['darkgrey', 'lightgrey', 'mediumorchid', 'mediumseagreen', 'wheat', 'tan']
roi_linestyle = ['dotted', 'dotted', 'solid', 'solid', 'dotted', 'dotted']
l_color_on = 'lightskyblue'
r_color_on = 'moccasin'
l_color_off = 'dodgerblue'
r_color_off = 'orange'


def plot_online_erds(erds_online, avg_erds_online, fb_times, roi_ch_names, config_file_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    lab_l = True
    lab_r = True
    for trial in range(len(erds_online)):
        trial_data = erds_online[trial]
        time = trial_data[:, 0]
        for roi in range(6):
            value = trial_data[:, roi + 1]
            if trial == 0:
                ax.plot(time, value, color=roi_color[roi], label=roi_ch_names[roi], linestyle=roi_linestyle[roi])
            else:
                ax.plot(time, value, color=roi_color[roi])
        if fb_times[trial, 2] == 1:  # Left
            if lab_l:
                ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=l_color_on, alpha=0.3, label='Left')
                lab_l = False
            else:
                ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=l_color_on, alpha=0.3)
        if fb_times[trial, 2] == 2:  # Right
            if lab_r:
                ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=r_color_on, alpha=0.3, label='Right')
                lab_r = False
            else:
                ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=r_color_on, alpha=0.3)
    plt.title("Online ERDS values \n" + config_file_path)
    plt.legend()

def plot_online_lda(lda_online, avg_lda_acc_online, fb_times, config_file_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    lab_l = True
    lab_r = True
    for trial in range(len(fb_times)):
        if fb_times[trial, 2] == 1:  # Left
            if lab_l:
                ax.scatter(lda_online[trial][:, 0], lda_online[trial][:, 1], color=l_color_on, marker='x', s=0.5, label='Left')
                ax.scatter(fb_times[trial, 1], avg_lda_acc_online[trial], color=l_color_on, marker='<', label='Avg. per trial left')
                lab_l = False
            else:
                ax.scatter(lda_online[trial][:, 0], lda_online[trial][:, 1], color=l_color_on, marker='x', s=0.5)
                ax.scatter(fb_times[trial, 1], avg_lda_acc_online[trial], color=l_color_on, marker='<')
            ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=l_color_on, alpha=0.3)
        if fb_times[trial, 2] == 2:  # Right
            if lab_r:
                ax.scatter(lda_online[trial][:, 0], lda_online[trial][:, 1], color=r_color_on, marker='x', s=0.5, label='Right')
                ax.scatter(fb_times[trial, 1], avg_lda_acc_online[trial], color=r_color_on, marker='<', label='Avg. per trial right')
                lab_r = False
            else:
                ax.scatter(lda_online[trial][:, 0], lda_online[trial][:, 1], color=r_color_on, marker='x', s=0.5)
                ax.scatter(fb_times[trial, 1], avg_lda_acc_online[trial], color=r_color_on, marker='<')
            ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=r_color_on, alpha=0.3)
    plt.plot(fb_times[:, 1], avg_lda_acc_online, color='forestgreen', label='Mean accuracy')
    plt.title("Online LDA accuracies \n" + config_file_path)
    plt.legend()

def plot_signals_for_eeg_calculation(data_ref, data_a, ref_times, fb_times, config_file_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for trial in range(data_ref.shape[0]):
        trial_data = data_ref[trial]
        time = trial_data[:, 0]
        for ch in range(1,trial_data.shape[1]):
            value = trial_data[:, ch]
            ax.plot(time, value)

    for trial in range(data_a.shape[0]):
        trial_data = data_a[trial]
        time = trial_data[:, 0]
        for ch in range(1,trial_data.shape[1]):
            value = trial_data[:, ch]
            ax.plot(time, value)
        if fb_times[trial, 2] == 1: # Left
            ax.axvspan(ref_times[trial, 0], ref_times[trial, 1], color=l_color_off, alpha=0.1)
            ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=l_color_off, alpha=0.3)
        if fb_times[trial, 2] == 2: # Right
            ax.axvspan(ref_times[trial, 0], ref_times[trial, 1], color=r_color_off, alpha=0.1)
            ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=r_color_off, alpha=0.3)

    plt.title("EEG signals for calculating ERDS Values \n" + config_file_path)

def plot_offline_erds(erds_offline, channels, fb_times, config_file_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for trial in range(erds_offline.shape[0]):
        trial_data = erds_offline[trial]
        time = trial_data[:, 0]
        for ch in range(erds_offline[trial].shape[1]-1):
            value = trial_data[:, ch + 1]
            if ch in channels:
                ax.plot(time, value) #, label=enabled_ch_names[ch_ix])

        if fb_times[trial, 2] == 1: # Left
            ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=l_color_off, alpha=0.3)
        if fb_times[trial, 2] == 2: # Right
            ax.axvspan(fb_times[trial, 0], fb_times[trial, 1], color=r_color_off, alpha=0.3)
    plt.title("Offline calculated ERDS values \n" + config_file_path)
    plt.legend(title='ROIs', loc='best', ncols=2)

def plot_online_vs_offline_erds_per_trial(erds_online, erds_offline, ch_ix, ch_roi, trial, trial_cl, config_file_path):
    plt.figure(figsize=(10, 5))
    if trial_cl == 1: # Left
        plt.plot(erds_offline[trial][:, 0], erds_offline[trial][:, ch_ix + 1], label='offline left', color=l_color_off)
        plt.plot(erds_online[trial][:, 0], erds_online[trial][:, ch_roi + 1], label='online left', color=l_color_on)
    if trial_cl == 2: # Right
        plt.plot(erds_offline[trial][:, 0], erds_offline[trial][:, ch_ix + 1], label='offline right', color=r_color_off)
        plt.plot(erds_online[trial][:, 0], erds_online[trial][:, ch_roi + 1], label='online right', color=r_color_on)
    plt.legend()
    plt.title(f"Comparison online / offline calculated ERDS for Trial {trial}:\n" + config_file_path)

def plot_online_vs_offline_erds_all_trials(erds_online, erds_offline, ch_ix, ch_roi, fb_times, config_file_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    lab_l = True
    lab_r = True
    for trial in range(erds_offline.shape[0]):
        if fb_times[trial, 2] == 1:  # Left
            if lab_l:
                ax.plot(erds_offline[trial][:, 0], erds_offline[trial][:, ch_ix + 1], label='offline left', color=l_color_off)
                ax.plot(erds_online[trial][:, 0], erds_online[trial][:, ch_roi + 1], label='online left', color=l_color_on)
                lab_l = False
            else:
                ax.plot(erds_offline[trial][:, 0], erds_offline[trial][:, ch_ix + 1], color=l_color_off)
                ax.plot(erds_online[trial][:, 0], erds_online[trial][:, ch_roi + 1], color=l_color_on)
        if fb_times[trial, 2] == 2:  # Right
            if lab_r:
                ax.plot(erds_offline[trial][:, 0], erds_offline[trial][:, ch_ix + 1], label='offline right', color=r_color_off)
                ax.plot(erds_online[trial][:, 0], erds_online[trial][:, ch_roi + 1], label='online right', color=r_color_on)
                lab_r = False
            else:
                ax.plot(erds_offline[trial][:, 0], erds_offline[trial][:, ch_ix + 1], color=r_color_off)
                ax.plot(erds_online[trial][:, 0], erds_online[trial][:, ch_roi + 1], color=r_color_on)
        plt.legend()
        plt.title(f"Comparison online / offline calculated ERDS:\n" + config_file_path)

def plot_online_vs_offline_avg_erds(avg_erds_online, avg_erds_offline, fb_times, config_file_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    plt.plot(avg_erds_online[:, 4], label='online left', color=l_color_on)
    plt.plot(avg_erds_offline[:, 4], label='offline left', color=r_color_off)

    plt.legend()
    plt.title("Comparison online / offline calculated average ERDS per trial:\n" + config_file_path)
