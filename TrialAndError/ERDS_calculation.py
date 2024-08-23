# ----------------------------------------------------------------------------------------------------------------------
# Calculate values for statistical comparison
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
matplotlib.use('Qt5Agg')
import glob
import scipy.io
import os
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import mne
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

class Data_from_Mat:
    def __init__(self, config_file_path, subject_directory):
        with open(config_file_path) as json_file:
            config = json.load(json_file)
        # CONFIG Infos
        self.subject_id = config['gui-input-settings']['subject-id']
        self.n_session = config['gui-input-settings']['n-session']
        self.n_run = config['gui-input-settings']['n-run']
        self.motor_mode = config['gui-input-settings']['motor-mode']
        self.dimension = config['gui-input-settings']['dimension-mode']

        self.sample_rate = config['eeg-settings']['sample-rate']
        self.duration_ref = config['general-settings']['timing']['duration-ref']
        self.duration_cue = config['general-settings']['timing']['duration-cue']
        self.duration_task = self.duration_cue + config['general-settings']['timing']['duration-task']

        # directories
        self.dir_plots = subject_directory + '/plots'
        if not os.path.exists(self.dir_plots):
            os.makedirs(self.dir_plots)
        self.dir_files = subject_directory + '/online_streams_mat'
        if not os.path.exists(self.dir_files):
            print(".mat files are not available to initialize EEG signal.")
            return

        # EEG / ERDS / LDA files
        self.eeg_name = self.dir_files + '/eeg_run' + str(self.n_run) + '_' + self.motor_mode + '.mat'
        self.data_eeg = scipy.io.loadmat(self.eeg_name)['eeg'].T
        self.eeg_scaled = self.data_eeg * 20e-3

        if self.n_run > 1:
            self.erds_name = self.dir_files + '/erds_run' + str(self.n_run) + '_' + self.motor_mode + '.mat'
            self.lda_name = self.dir_files + '/lda_run' + str(self.n_run) + '_' + self.motor_mode + '.mat'
            self.data_erds = scipy.io.loadmat(self.erds_name)['erds'].T
            self.data_lda = scipy.io.loadmat(self.lda_name)['lda'].T


        self.n_ref = int(np.floor(self.sample_rate * self.duration_ref))
        self.n_samples_task = int(np.floor(self.sample_rate * self.duration_task))
        self.n_samples_trial = self.n_ref + self.n_samples_task

        # class infos
        indexes_class_1 = np.where(self.data_eeg[0, :] == 121)[0]
        indexes_class_2 = np.where(self.data_eeg[0, :] == 122)[0]
        self.indexes_class_all = np.sort(np.append(indexes_class_1, indexes_class_2), axis=0)
        self.n_trials = len(self.indexes_class_all)
        self.class_labels = self.data_eeg[0, self.indexes_class_all] - 121
        self.event_dict = dict(left=0, right=1)
        self.events = np.column_stack(
            (self.indexes_class_all, np.zeros(len(self.indexes_class_all), dtype=int), np.array(self.class_labels, dtype=int)))

        # channels and bads
        self.ch_names = []
        self.bads = []
        for name in config['eeg-settings']['channels']:
            self.ch_names.append(name)
            if config['eeg-settings']['channels'][name]['enabled'] is False:
                self.bads.append(name)
        self.n_ch = len(self.ch_names)
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sample_rate, ch_types=['eeg'] * self.n_ch)
        # info['bads'] = bads
        # print(info)

        self.raw = mne.io.RawArray(self.data_eeg[1:self.n_ch + 1, :], self.info)
        self.epochs = []

        print('MAT file reading completed.')

def mean_erds_per_roi(epochs, roi_dict, task='r'):
    sample_rate = 500
    time_r = [int(0*sample_rate), 3*sample_rate]
    time_a = [int(3*sample_rate), int(11.25*sample_rate)]

    erds_roi_list = []
    for roi in roi_dict.keys():
        erds_epoch = []
        epochs_roi = epochs.copy().pick(roi_dict[roi])
        for epoch in epochs_roi:
            r_period = np.square(epoch[:, time_r[0]:time_r[1]])
            r_mean = np.average(r_period)
            a_period = np.square(epoch[:, time_a[0]:time_a[1]])
            a_mean = np.average(a_period)
            erds_epoch.append(-(r_mean - a_mean) / r_mean)
        erds_roi_list.append(np.mean(erds_epoch))

    return erds_roi_list

def erds_per_roi(epochs, roi_dict):
    sample_rate = 500
    time_r = [int(0*sample_rate)+1, 3*sample_rate+1] # 0-3s per epoch
    time_a = [int(5.25*sample_rate+1), int(11.25*sample_rate)+1]  # 4.25-11.25s per epoch

    erds_roi_mean = []
    erds_roi_trial = []
    for roi in roi_dict.keys():
        erds_epoch = []
        epochs_roi = epochs.copy().pick(roi_dict[roi])
        for epoch in epochs_roi:
            r_period = np.square(epoch[:, time_r[0]:time_r[1]])
            r_mean = np.average(r_period)
            a_period = np.square(epoch[:, time_a[0]:time_a[1]])
            a_mean = np.average(a_period)
            erds_epoch.append(-(r_mean - a_mean) / r_mean)
        erds_roi_mean.append(np.mean(erds_epoch))
        erds_roi_trial.append(erds_epoch)

    return erds_roi_mean, erds_roi_trial

def extract_epochs(data, n):
    cue_indexes = np.where(data[0, :] != 0)[0]
    cl_labels = np.array(data[0, cue_indexes]-121, dtype=int)
    data = data[1:, :]
    epochs = []
    min_length = n
    n_trials = len(cue_indexes)

    for i in range(n_trials):
        idx1 = cue_indexes[i]
        idx2 = idx1 + n
        if i < n_trials-1 and idx2 > cue_indexes[i+1]:
            idx2 = cue_indexes[i+1]
        epochs.append(data[:, idx1:idx2])
        min_length = np.min([min_length, idx2-idx1])

    data_nd = np.zeros((len(epochs), np.shape(data)[0], min_length))
    data_nd[:] = None

    # list to np array
    i = 0
    for e in epochs:
        data_nd[i, :, :] = e[:, 0:min_length]
        i += 1

    return data_nd, min_length, cl_labels

def plot_erds_values(data, title, name, s_rate, n_cue, erds_mode, cl, dir_plots, n_ref):

    def create_subplot(fig, ax, y, ext, subtitle, xtick_l):

        y = np.append(data_fill, y, axis=0)
        im = ax.imshow(y[np.newaxis, :], vmin=-1.5, vmax=1.5, cmap="RdBu", aspect="auto", extent=ext)
        ax.set_yticks([])
        ax.set_xlim(ext[0], ext[1])
        ax.set_xticklabels(xtick_l)
        # ax.axvline(n_ref, linewidth=1, color="black", linestyle=":")  #strichlierte linie für ref-zeit
        #fig.colorbar(im, ax=ax)  # colour bar für jeden einzelnen plot
        ax.set_title(subtitle)

    data_fill = np.zeros((n_cue,))
    x = np.arange(0, np.shape(data)[1] + n_cue, 1)
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2., 0, 1]

    figure, axs = plt.subplots(3, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [10, 10, 0.5]})
    step = s_rate/2
    xtick_labels = np.arange(-step, int(np.shape(x)[0]), step) / s_rate
    if len(xtick_labels) > 10:
        step = s_rate
        xtick_labels = np.arange(-step, int(np.shape(x)[0]), step) / s_rate

    """
    if len(xtick_labels) > 9:
        step = s_rate*2
        xtick_labels = np.arange(-step, int(np.shape(x)[0]), step) / s_rate
    """

    xtick_labels = np.array(xtick_labels, dtype=str)
    xtick_labels[-1] = xtick_labels[-1] + ' s'

    create_subplot(figure, axs[0, 0], data[0, :], extent, 'ROI1', xtick_labels)
    create_subplot(figure, axs[0, 1], data[1, :], extent, 'ROI2', xtick_labels)
    create_subplot(figure, axs[1, 0], data[2, :], extent, 'ROI3', xtick_labels)
    create_subplot(figure, axs[1, 1], data[3, :], extent, 'ROI4', xtick_labels)
    create_subplot(figure, axs[2, 0], data[4, :], extent, 'ROI5', xtick_labels)
    create_subplot(figure, axs[2, 1], data[5, :], extent, 'ROI6', xtick_labels)

    figure.colorbar(axs[0][1].images[-1], cax=axs[0][-1])
    figure.colorbar(axs[1][1].images[-1], cax=axs[1][-1])
    figure.colorbar(axs[2][1].images[-1], cax=axs[2][-1])

    figure.suptitle(title)
    plt.savefig('{}/online_{}_{}_{}.png'.format(dir_plots, name, erds_mode, cl), format='png')
    plt.show()

def erds_values_plot_preparation(subject_data_path, signal):
    my_files = glob.glob(subject_data_path + '/online_streams_mat/erds_run2*.mat')
    n_samples_task = signal.n_samples_task

    cnt = 0
    for f in my_files:
        name = f[f.find('erds'):-4]
        data_erds = scipy.io.loadmat(f)['erds'].T

        data_epoched, n_samples, labels = extract_epochs(data_erds, n_samples_task)
        data_1 = data_epoched[np.where(labels == 0)[0]]  # left
        data_2 = data_epoched[np.where(labels == 1)[0]]  # right

        #plt.scatter(range(0, len(data_1[0][3])), data_1[0][3])  # right, first epoch, roi3 = C4
        plt.plot(range(0, len(data_2[0][3])), data_2[0][3], 'g-') #right, first epoch, roi3 = C4
        plt.show()

        #plot_erds_values(np.mean(data_1, axis=0), 'ERDS - left hand', name, signal.sample_rate, signal.n_cue, signal.erds_mode, cl='l', dir_plots=signal.dir_plots, n_ref=signal.n_ref)
        #plot_erds_values(np.mean(data_2, axis=0), 'ERDS - right hand', name, signal.sample_rate, signal.n_cue, signal.erds_mode, cl='r', dir_plots=signal.dir_plots, n_ref=signal.n_ref)

    return np.mean(data_1, axis=2), np.mean(data_2, axis=2)  #returns erds values shown for class 1 (left) and class 2 (right) as online feedback fpr 6 ROIs

def calc_clustering(tfr_ev, ch, kwargs):
    # positive clusters
    _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
    # negative clusters
    _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

    # note that we keep clusters with p <= 0.05 from the combined clusters
    # of two independent tests; in this example, we do not correct for
    # these two comparisons
    c = np.stack(c1 + c2, axis=2)  # combined clusters
    p = np.concatenate((p1, p2))  # combined p-values
    mask = c[..., p <= 0.5].any(axis=-1)  # 0.0

    return c, p, mask

def plot_erds_maps(data_from_mat, picks, show_epochs=True, show_erds=True, cluster_mode=False,
                   preproc_data=None, tfr_mode=False):
    tmin = -data_from_mat.duration_ref
    tmax = data_from_mat.duration_task
    if preproc_data==None:
        raw_data = data_from_mat.raw
    else:
        raw_data = preproc_data

    epochs = mne.Epochs(raw_data, data_from_mat.events, data_from_mat.event_dict, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None,
                        preload=True)
    if show_epochs == True:
        epochs.plot(picks=picks, show_scrollbars=True, events=data_from_mat.events, event_id=data_from_mat.event_dict, block=False)
        plt.savefig('{}/epochs_{}_run{}_{}.png'.format(data_from_mat.dir_plots, data_from_mat.motor_mode, str(data_from_mat.n_run),
                                                             data_from_mat.dimension), format='png')

    freqs = np.arange(1, 30)
    vmin, vmax = -1, 1  # set min and max ERDS values in plot
    # baseline = [tmin, -0.5]  # baseline interval (in s)
    baseline = [tmin, 0]  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test

    tfr = epochs.compute_tfr(method="multitaper", picks=picks, freqs=freqs, n_cycles=freqs, use_fft=True,
                             return_itc=False, average=False, decim=2)
    tfr.crop(tmin, tmax).apply_baseline(baseline,
                                        mode="percent")  # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    tfr.crop(0, tmax)


    for event in data_from_mat.event_dict:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]})  # , 0.5
        axes = axes.flatten()
        for ch, ax in enumerate(axes[:-1]):  # for each channel  axes[:-1]
            if cluster_mode:
                # find clusters
                c, p, mask = calc_clustering(tfr_ev, ch, kwargs)

                # plot TFR (ERDS map with masking)
                tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                      colorbar=False, show=False, vlim=(-1.5, 1.5), mask=mask)  # , mask=mask,
            else:
                tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                      colorbar=False, show=False, vlim=(-1.5, 1.5))

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")

        fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle(f"ERDS - {event} hand {data_from_mat.motor_mode} run {data_from_mat.n_run} {data_from_mat.dimension}")
        fig.canvas.manager.set_window_title(event + " hand ERDS maps")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig('{}/erds_map_{}_run{}_{}_{}_{}{}_{}.png'.format(data_from_mat.dir_plots,
                                                                 data_from_mat.motor_mode, str(data_from_mat.n_run),
                                                                 data_from_mat.dimension, event, picks[0], picks[1], timestamp),
                    format='png')
    if show_erds == True:
        plt.show()

    if tfr_mode:
        df = tfr.to_data_frame(time_format=None)
        print(df.head())

        df = tfr.to_data_frame(time_format=None, long_format=True)

        # Map to frequency bands:
        freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
        df["band"] = pd.cut(
            df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
        )

        # Filter to retain only relevant frequency bands:
        freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
        df = df[df.band.isin(freq_bands_of_interest)]
        df["band"] = df["band"].cat.remove_unused_categories()

        # Order channels for plotting:
        df["channel"] = df["channel"].cat.reorder_categories(picks, ordered=True)

        g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
        g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
        axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
        g.map(plt.axhline, y=0, **axline_kw)
        g.map(plt.axvline, x=3, **axline_kw)
        g.set(ylim=(None, 1.5))
        g.set_axis_labels("Time (s)", "ERDS")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.add_legend(ncol=2, loc="lower center")
        g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
        plt.show()

        df_mean = (
            df.query("time > 0")
            .groupby(["condition", "epoch", "band", "channel"], observed=False)[["value"]]
            .mean()
            .reset_index()
        )

        g = sns.FacetGrid(
            df_mean, col="condition", col_order=["left", "right"], margin_titles=True
        )
        g = g.map(
            sns.violinplot,
            "channel",
            "value",
            "band",
            cut=0,
            palette="deep",
            order=picks,
            hue_order=freq_bands_of_interest,
            linewidth=0.5,
        ).add_legend(ncol=4, loc="lower center")

        g.map(plt.axhline, **axline_kw)
        g.set_axis_labels("", "ERDS")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
        plt.show()
