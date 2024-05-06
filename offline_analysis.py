# ----------------------------------------------------------------------------------------------------------------------
# Offline analysis of recorded eeg data
# ----------------------------------------------------------------------------------------------------------------------
import json
import numpy as np
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

class EEG_Signal:

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
        self.dir_files = subject_directory + '/dataset'
        if not os.path.exists(self.dir_files):
            print(".mat files are not available to initialize EEG signal.")
            return

        # EEG / ERDS / LDA files
        self.eeg_name = self.dir_files + '/eeg_run' + str(self.n_run) + '_' + self.motor_mode + '.mat'
        self.erds_name = self.dir_files + '/erds_run' + str(self.n_run) + '_' + self.motor_mode + '.mat'
        self.lda_name = self.dir_files + '/lda_run' + str(self.n_run) + '_' + self.motor_mode + '.mat'

        self.data_eeg = scipy.io.loadmat(self.eeg_name)['eeg'].T
        self.data_erds = scipy.io.loadmat(self.erds_name)['erds'].T
        self.data_lda = scipy.io.loadmat(self.lda_name)['lda'].T
        self.eeg_scaled = self.data_eeg * 20e-3

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
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sample_rate, ch_types=['misc'] * self.n_ch)
        # info['bads'] = bads
        # print(info)

        self.raw = mne.io.RawArray(self.data_eeg[1:self.n_ch + 1, :], self.info)
        self.epochs = []

        print('EEG signal initialization completed.')


    def extract_epochs(self, data, n_samples):
        '''data_labels = data[0, :]
        data = data[1:self.n_ch+1, :]
        indexes = np.where((data_labels == 121) | (data_labels == 122))[0]
        epochs = []

        for idx in indexes:
            epochs.append(data[:, idx:idx + n_samples_task])

        return data, data_labels
        '''
        data_labels = data[0, :]
        data = data[1:, :]
        indexes = np.where((data_labels == 121) | (data_labels == 122))[0]

        epochs = []
        n_trials = len(indexes)
        for i in range(n_trials):
            idx1 = indexes[i]
            idx2 = idx1 + n_samples
            if i < n_trials - 1 and idx2 > indexes[i + 1]:
                idx2 = indexes[i + 1]
            epochs.append(data[:, idx1:idx2])

        return epochs, np.array(data_labels[indexes] - 121, dtype=int)


    def show_lda_plots(self):
        time = [0, self.duration_cue + self.duration_task]
        step = (self.duration_cue + self.duration_task) / self.n_samples_task
        time_series = np.arange(start=time[0], stop=time[1], step=step)

        data_lda, labels = EEG_Signal.extract_epochs(self, self.data_lda, self.n_samples_task)
        class_label = np.zeros_like(time_series)
        trial_acc = EEG_Signal.compute_accuracy(self, list=True)
        i = 0
        #for epoch, cl in zip(data_lda, labels):
        for epoch in data_lda:
            accuracy = trial_acc[i]
            i += 1
            class_label[:] = epoch[0, 0] #- 120
            plt.clf()
            samples = np.shape(epoch[1, :])[0]
            plt.plot(time_series[0:samples], class_label[0:samples], 'k') # true label
            plt.plot(time_series[0:samples], epoch[1, 0:samples], 'b') #distance
            plt.plot(time_series[0:samples], epoch[0, 0:samples], 'r') # classified label
            plt.legend(['true label', 'distance', 'classified label'])
            plt.title(f'Trial {i} with mean accuracy = {accuracy}')
            #plt.savefig('{}/LDA_{}_{}_run{}_task{}_acc={:.2f}.png'.format(self.dir_plots, self.motor_mode,
            #                                                      self.dimension, str(self.n_run), str(i), trial_acc), format='png')
            plt.show()

        acc = EEG_Signal.compute_accuracy(self)
        print("Mean accuracy = {:.2f}".format(acc))

    def show_erds_mean(self):
        time = [0, self.duration_cue + self.duration_task]
        step = (self.duration_cue + self.duration_task) / self.n_samples_task

        data_erds, labels = EEG_Signal.extract_epochs(self, self.data_erds, self.n_samples_task)

        #find max. sized epoch
        max_length = 0
        max_index = 0
        for i, arr in enumerate(data_erds):
            if arr.shape[1] > max_length:
                max_length = arr.shape[1]
                max_index = i

        nr_roi, samples = np.shape(data_erds[max_index])
        data_nd = np.zeros((len(data_erds), nr_roi, samples))
        data_nd[:] = None

        # list to np array
        cnt = 0
        for epoch in data_erds:
            data_nd[cnt, :, 0:np.shape(epoch)[1]] = epoch
            cnt += 1

        #idx_1 = np.where(self.class_labels == 0)[0]
        #idx_2 = np.where(self.class_labels == 1)[0]
        idx_1 = np.where(labels == 0)[0]
        idx_2 = np.where(labels == 1)[0]

        # remove short trials (nan values)
        '''
        for val in np.unique(np.where(np.isnan(data_nd))[0]):
            idx_1 = np.delete(idx_1, np.argwhere(idx_1 == val))
            idx_2 = np.delete(idx_2, np.argwhere(idx_2 == val))
        '''

        # average erds values for all trials
        erds_mean_cl1 = np.mean(data_nd[idx_1], axis=0)
        erds_mean_cl2 = np.mean(data_nd[idx_2], axis=0)
        time_series = np.arange(start=time[0], stop=time[1], step=step)
        time_series = time_series[:len(erds_mean_cl1[1])]

        # new_samples = int(samples/3)
        # erds_mean_cl1 = signal.resample(erds_mean_cl1, new_samples, axis=1)
        # erds_mean_cl2 = signal.resample(erds_mean_cl2, new_samples, axis=1)
        # time_series = np.arange(start=time[0], stop=time[1], step=(config['general-settings']['timing']['duration-cue']+config['general-settings']['timing']['duration-task'])/new_samples)

        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(time_series, erds_mean_cl1[0, :])
        axs[0, 0].set_title('ROI 1')
        axs[0, 1].plot(time_series, erds_mean_cl1[1, :])
        axs[0, 1].set_title('ROI 2')
        axs[1, 0].plot(time_series, erds_mean_cl1[2, :])
        axs[1, 0].set_title('ROI 3')
        axs[1, 1].plot(time_series, erds_mean_cl1[3, :])
        axs[1, 1].set_title('ROI 4')
        axs[2, 0].plot(time_series, erds_mean_cl1[4, :])
        axs[2, 0].set_title('ROI 5')
        axs[2, 1].plot(time_series, erds_mean_cl1[5, :])
        axs[2, 1].set_title('ROI 6')
        fig.suptitle('Mean ERDS left hand')
        plt.savefig('{}/meanERDS_left_{}_{}_run{}.png'.format(self.dir_plots, self.motor_mode,
                                                           self.dimension, str(self.n_run)), format='png')
        #plt.show()

        fig2, axs2 = plt.subplots(3, 2)
        axs2[0, 0].plot(time_series, erds_mean_cl2[0, :])
        axs2[0, 0].set_title('ROI 1')
        axs2[0, 1].plot(time_series, erds_mean_cl2[1, :])
        axs2[0, 1].set_title('ROI 2')
        axs2[1, 0].plot(time_series, erds_mean_cl2[2, :])
        axs2[1, 0].set_title('ROI 3')
        axs2[1, 1].plot(time_series, erds_mean_cl2[3, :])
        axs2[1, 1].set_title('ROI 4')
        axs2[2, 0].plot(time_series, erds_mean_cl2[4, :])
        axs2[2, 0].set_title('ROI 5')
        axs2[2, 1].plot(time_series, erds_mean_cl2[5, :])
        axs2[2, 1].set_title('ROI 6')
        fig2.suptitle('Mean ERDS right hand')
        plt.savefig('{}/meanERDS_right_{}_{}_run{}.png'.format(self.dir_plots, self.motor_mode,
                                                              self.dimension, str(self.n_run)), format='png')
        #plt.show()


    def compute_accuracy(self, list=False):
        '''samples_accurate = 0
        samples_total = 0

        if median:
            for epoch, cl in zip(self.data_lda, self.class_labels):
                if np.median(epoch[0, :]) == cl:
                    samples_accurate += 1
                samples_total += 1

        else:
            for epoch, cl in zip(self.data_lda, self.class_labels):
                samples_accurate += len(np.where(epoch[0, :] == cl)[0])
                samples_total += len(epoch[0, :])


        return samples_accurate / samples_total
        '''

        if not self.n_run == 1:
            data_lda, labels = EEG_Signal.extract_epochs(self, self.data_lda, self.n_samples_task)
            acc_per_trial = []
            for epoch, cl in zip(data_lda, labels):
                acc_per_trial.append(np.sum(epoch[0, :] == cl) / len(epoch[0, :]))
            if list:
                return acc_per_trial
            else:
                return np.mean(acc_per_trial)*100


    #def plot_eeg(eeg, n_ch, info):
    #    for i in range(len(indexes_class_all)):
    #        raw = mne.io.RawArray(eeg[1:n_ch+1, indexes_class_all[i]-n_ref:indexes_class_all[i]+n_samples_trial], info)
    #        raw.plot()

    def plot_raw_eeg(self, scaled=False, max_trial=0):
        if max_trial == 0:
            max_trial = len(self.indexes_class_all)
        if scaled:
            eeg = self.eeg_scaled
        else:
            eeg = self.data_eeg
        for trial in range(max_trial):
            raw = mne.io.RawArray(
                eeg[1:self.n_ch + 1,
                self.indexes_class_all[trial] - self.n_ref:self.indexes_class_all[trial] + self.n_samples_trial], self.info)
            raw.plot(duration=14.25, n_channels=self.n_ch, show_scrollbars=True, block=True, show_options=True,
                     title="Trial %i" % (trial+1))
            plt.savefig('{}/raw_{}_{}_run{}_task{}.png'.format(self.dir_plots, self.motor_mode,
                                                               self.dimension, str(self.n_run), (trial+1)), format='png')

    def clustering(self, tfr_ev, ch, kwargs):
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

    def plot_erds_maps(self, picks, show_epochs, show_erds, clustering=False):
        tmin = -self.duration_ref
        tmax = self.duration_task

        epochs = mne.Epochs(self.raw, self.events, self.event_dict, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None,
                            preload=True)
        if show_epochs == True:
            epochs.plot(picks=picks, show_scrollbars=True, events=self.events, event_id=self.event_dict, block=False)
            plt.savefig('{}/epochs_{}_run{}_{}.png'.format(self.dir_plots, self.motor_mode, str(self.n_run),
                                                                 self.dimension), format='png')

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


        for event in self.event_dict:
            # select desired epochs for visualization
            tfr_ev = tfr[event]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 0.5]})  # , 0.5
            axes = axes.flatten()
            for ch, ax in enumerate(axes[:-1]):  # for each channel  axes[:-1]
                if clustering:
                    # find clusters
                    c, p, mask = EEG_Signal.clustering(self, tfr_ev, ch, kwargs)

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
            fig.suptitle(f"ERDS - {event} hand {self.motor_mode} run {self.n_run} {self.dimension}")
            fig.canvas.manager.set_window_title(event + " hand ERDS maps")

            if clustering:
                plt.savefig('{}/erds_cluster_{}_run{}_{}_{}_{}{}.png'.format(self.dir_plots, self.motor_mode, str(self.n_run),
                                                                     self.dimension, event, picks[0], picks[1]),
                            format='png')
            else:
                plt.savefig('{}/erds_{}_run{}_{}_{}_{}{}.png'.format(self.dir_plots, self.motor_mode, str(self.n_run), self.dimension, event, picks[0], picks[1]), format='png')
            if show_erds == True:
                plt.show()



