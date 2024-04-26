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

def extract_epochs(data):
    data_labels = data[0, :]
    data = data[1:, :]
    indexes = np.where((data_labels == 121) | (data_labels == 122))[0]

    epochs = []
    for idx in indexes:
        epochs.append(data[:, idx:idx + n_samples_task])

    return epochs


def show_lda_plots(data):
    time = [0, config['general-settings']['timing']['duration-cue'] + config['general-settings']['timing'][
        'duration-task']]
    step = (config['general-settings']['timing']['duration-cue'] + config['general-settings']['timing'][
        'duration-task']) / n_samples_task
    time_series = np.arange(start=time[0], stop=time[1], step=step)
    class_label = np.zeros_like(time_series)
    for epoch in data:
        class_label[:] = epoch[0, 0] - 120
        plt.clf()
        samples = np.shape(epoch[1, :])[0]
        plt.plot(time_series[0:samples], class_label[0:samples], 'k')
        plt.plot(time_series[0:samples], epoch[1, 0:samples] + 1, 'b')
        plt.show()


def show_erds_plots(data):
    time = [0, config['general-settings']['timing']['duration-cue'] + config['general-settings']['timing'][
        'duration-task']]
    step = (config['general-settings']['timing']['duration-cue'] + config['general-settings']['timing'][
        'duration-task']) / n_samples_task
    nr_roi, samples = np.shape(data[0])
    data_nd = np.zeros((len(data), nr_roi, samples))
    data_nd[:] = None

    # list to np array
    cnt = 0
    for epoch in data:
        data_nd[cnt, :, 0:np.shape(epoch)[1]] = epoch
        cnt += 1

    idx_1 = np.where(class_labels == 0)[0]
    idx_2 = np.where(class_labels == 1)[0]

    # remove short trials (nan values)
    for val in np.unique(np.where(np.isnan(data_nd))[0]):
        idx_1 = np.delete(idx_1, np.argwhere(idx_1 == val))
        idx_2 = np.delete(idx_2, np.argwhere(idx_2 == val))

    # average erds values for all trials
    erds_mean_cl1 = np.mean(data_nd[idx_1], axis=0)
    erds_mean_cl2 = np.mean(data_nd[idx_2], axis=0)
    time_series = np.arange(start=time[0], stop=time[1], step=step)

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
    fig.suptitle('Mean ERDS Class 1')
    plt.show()

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
    fig2.suptitle('Mean ERDS Class 2')
    plt.show()


def compute_accuracy(data, median=False):
    samples_accurate = 0
    samples_total = 0

    if median:
        for epoch, cl in zip(data, class_labels):
            if np.median(epoch[0, :]) == cl:
                samples_accurate += 1
            samples_total += 1

    else:
        for epoch, cl in zip(data, class_labels):
            samples_accurate += len(np.where(epoch[0, :] == cl)[0])
            samples_total += len(epoch[0, :])

    return samples_accurate / samples_total

#def plot_eeg(eeg, n_ch, info):
#    for i in range(len(indexes_class_all)):
#        raw = mne.io.RawArray(eeg[1:n_ch+1, indexes_class_all[i]-n_ref:indexes_class_all[i]+n_samples_trial], info)
#        raw.plot()


def analyze_eeg(eeg, config):
    ch_names = []
    bads = []
    for name in config['eeg-settings']['channels']:
        ch_names.append(name)
        if config['eeg-settings']['channels'][name]['enabled'] is False:
            bads.append(name)

    sample_rate = config['eeg-settings']['sample-rate']
    duration_ref = config['general-settings']['timing']['duration-ref']
    duration_cue = config['general-settings']['timing']['duration-cue']
    duration_task = duration_cue + config['general-settings']['timing']['duration-task']

    n_ref = int(np.floor(sample_rate * duration_ref))
    n_ch = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=['misc'] * n_ch)
    info['bads'] = bads
    print(info)

    eeg_scaled = eeg * 20e-3
    #raw = mne.io.RawArray(eeg[1:n_ch + 1, :], info)



    """
    event_dict = dict(left=0, right=1)
    # picks = ['C3', 'Cz', 'C4']
    # picks = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    # picks = ['F3', 'F4']
    picks = ['C3', 'C4']
    picked_channels = [raw.ch_names.index(ch) for ch in picks]
    #picks = ['P3', 'P4']

    eeg = extract_epochs(eeg)
    n_ep, n_ch, n_s = np.shape(eeg)
    events_array = np.column_stack((np.arange(0, n_ep*n_s, n_s), np.zeros(n_ep, dtype=int), np.array(class_labels, dtype=int)))
    eeg_epochs_array = mne.EpochsArray(eeg, info, tmin=-config['general-settings']['timing']['duration-ref'], events=events_array, event_id=event_dict)
    eeg_epochs_array.plot(picks=picks, show_scrollbars=False, events=events_array, event_id=event_dict, n_epochs=5)
    """

    indexes_class_1 = np.where(eeg[0, :] == 121)[0]
    indexes_class_2 = np.where(eeg[0, :] == 122)[0]
    indexes_class_all = np.sort(np.append(indexes_class_1, indexes_class_2), axis=0)
    class_labels = eeg[0, indexes_class_all] - 121

    # Versuch, nicht nur cue l+r, sondern auch die anderen Events mit aufzulisten
    '''
    indexes_class_all_together = []
    indexes_class_all_together.append(indexes_class_all)
    index_Session_Start = np.where(eeg[0, :] == 130)[0]
    indexes_class_all_together.append(index_Session_Start)
    index_Reference = np.where(eeg[0, :] == 131)[0]
    indexes_class_all_together.append(index_Reference)
    # index_Cue = np.where(eeg[0, :] == 132)[0]
    index_Feedback = np.where(eeg[0, :] == 133)[0]
    indexes_class_all_together.append(index_Feedback)
    index_End_of_Trial = np.where(eeg[0, :] == 134)[0]
    indexes_class_all_together.append(index_End_of_Trial)
    index_Session_End = np.where(eeg[0, :] == 135)[0]
    indexes_class_all_together.append(index_Session_End)
    indexes_class_all_together = np.sort(indexes_class_all_together, axis=0)
    class_labels_together = eeg[0, indexes_class_all_together] - 121
    '''


    n_samples_task = int(np.floor(sample_rate * duration_task))
    n_samples_trial = n_ref + n_samples_task
    """
    tmin = -duration_ref
    tmax = duration_task
    events = np.column_stack(
        (indexes_class_all, np.zeros(len(indexes_class_all), dtype=int), np.array(class_labels, dtype=int)))
    #events_together = np.column_stack(
    #    (indexes_class_all_together, np.zeros(len(indexes_class_all_together), dtype=int), np.array(class_labels_together, dtype=int)))
    epochs = mne.Epochs(raw, events, event_dict, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None, preload=True)
    # epochs.plot(picks=picks, show_scrollbars=True, events=events, event_id=event_dict)

    # freqs = np.arange(2, 31)  # frequencies from 2-30Hz
    freqs = np.arange(1, 30)
    vmin, vmax = -1, 1  # set min and max ERDS values in plot
    # baseline = [tmin, -0.5]  # baseline interval (in s)
    baseline = [tmin, 0]  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test

    tfr = tfr_multitaper(epochs, picks=picks, freqs=freqs, n_cycles=freqs, use_fft=True, return_itc=False, average=False, decim=2)
    tfr.crop(tmin, tmax).apply_baseline(baseline,mode="percent")  # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    tfr.crop(0, tmax)
    """


    #for i in range(len(indexes_class_all)):
    for i in range(3, 4):
        raw = mne.io.RawArray(eeg_scaled[1:n_ch+1, indexes_class_all[i]-n_ref:indexes_class_all[i]+n_samples_trial], info)
        raw.plot(duration=14.25, n_channels=n_ch, show_scrollbars=True, block=True, show_options=True, title="Trial %i" % i)

    '''
    for event in event_dict:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 0.5]}) # , 0.5
        axes = axes.flatten()
        for ch, ax in enumerate(axes[:-1]):  # for each channel  axes[:-1]
            # # positive clusters
            # _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
            # # negative clusters
            # _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)
            #
            # # note that we keep clusters with p <= 0.05 from the combined clusters
            # # of two independent tests; in this example, we do not correct for
            # # these two comparisons
            # c = np.stack(c1 + c2, axis=2)  # combined clusters
            # p = np.concatenate((p1, p2))  # combined p-values
            # mask = c[..., p <= 0.5].any(axis=-1)  # 0.05

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                  colorbar=False, show=False, vmin=-1.5, vmax=1.5)  #, mask=mask,
                                  # mask_style="mask")

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")

        fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle(f"ERDS - {event} hand {motor_mode} run {run}")

        plt.savefig('{}/erds_{}_{}_{}_{}{}.png'.format(dir_plots, motor_mode, str(run), event, picks[0], picks[1]), format='png')
        plt.show()
    '''
    return eeg


#if __name__ == "__main__":
def offline_analysis(config_file_path, subject_directory):

    with open(config_file_path) as json_file:
        config = json.load(json_file)

    subject_id = config['gui-input-settings']['subject-id']
    n_session = config['gui-input-settings']['n-session']
    n_run = config['gui-input-settings']['n-run']
    motor_mode = config['gui-input-settings']['motor-mode']
    dimension = config['gui-input-settings']['dimension-mode']

    sample_rate = config['eeg-settings']['sample-rate']
    duration_ref = config['general-settings']['timing']['duration-ref']
    duration_cue = config['general-settings']['timing']['duration-cue']
    n_ref = int(np.floor(sample_rate * duration_ref))


    #all_config_files = glob.glob(subject_directory + '*.json')
    #for current_config_file in all_config_files:
    #    break

    dir_plots = subject_directory + '/plots'
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)
    dir_files = subject_directory + '/dataset'
    if not os.path.exists(dir_files):
        print(".mat files are not available.")
        return

    #for run in range(2, 5):
    eeg_name = dir_files + '/eeg_run'+str(n_run) + '_' + motor_mode +'.mat'
    erds_name = dir_files + '/erds_run'+str(n_run) + '_' + motor_mode + '.mat'
    lda_name = dir_files + '/lda_run'+str(n_run) + '_' + motor_mode  +'.mat'
    #if not os.path.exists(eeg_name):
    #    continue
    # LOAD MAT FILES
    data_eeg = scipy.io.loadmat(eeg_name)['eeg'].T
    data_erds = scipy.io.loadmat(erds_name)['erds'].T
    data_lda = scipy.io.loadmat(lda_name)['lda'].T
    #
    # EXTRACT EPOCHS

    duration_task = duration_cue + config['general-settings']['timing']['duration-task']
    #if subject_id == 'S1' and motor_mode == 'ME' and (run == 1 or run == 2):
    #    duration_task = duration_cue + 3.75

    n_samples_task = int(np.floor(sample_rate * duration_task))
    n_samples_trial = n_ref + n_samples_task

    indexes_class_1 = np.where(data_eeg[0, :] == 121)[0]
    indexes_class_2 = np.where(data_eeg[0, :] == 122)[0]
    indexes_class_all = np.sort(np.append(indexes_class_1, indexes_class_2), axis=0)

    n_trials = len(indexes_class_all)
    class_labels = data_eeg[0, indexes_class_all] - 121

    # COMPUTE ERDS MAP FROM EEG SIGNAL
    print('Subject: ' + subject_id + ' Session: ' + str(n_session) + ' Run: ' + str(n_run))
    data_eeg = analyze_eeg(data_eeg, config)

