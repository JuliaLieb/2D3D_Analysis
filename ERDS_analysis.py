import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from more_itertools import locate

'''
Modified from source: https://github.com/scwlab/EEG-preprocessing
Giulia Pezzutti
'''

def compute_erds(epochs, rois, fs, t_min, f_min=0, f_max=50, path=None):
    """
    Function to compute ERDS maps for a set of epochs according to different regions of interest
    :param epochs: MNE epochs object containing the annotated epochs of interest with EEG channels
    :param rois: dict object containing a set of key-value pairs. The key must be the name of the region of
    interest, the value is a list containing the channels belonging to that ROI
    :param fs: sampling frequency of the EEG acquisition
    :param t_min: time instant of the epoch (with respect to the stimuli instant)
    :param f_min: minimum frequency for which the ERDS maps are visualized
    :param f_max: maximum frequency for which the ERDS maps are visualized
    :param path: path where to save the computed ERDS maps. If None, the maps are just shown.
    """

    # get EEG data
    signals = epochs.get_data()  # epochs x channels x instants

    # get the list of unique annotations from the data
    annotations = [annotation[0][2] for annotation in epochs.get_annotations_per_epoch()]
    conditions = list(set(annotations))

    # compute the spectrogram for the signals
    freq, time, spectrogram = scipy.signal.spectrogram(signals, fs=fs, nperseg=250, noverlap=225, scaling='spectrum')
    # epochs x channels x frequencies x time windows

    # adjust time interval and consider just frequencies of interest
    time = time + t_min
    spectrogram = np.squeeze(spectrogram[:, :, np.argwhere(np.logical_and(f_min < freq, freq < f_max)), :])
    freq = np.squeeze(freq[np.argwhere(np.logical_and(f_min < freq, freq < f_max))])

    # compute the vectors for visualization axis (it's necessary to have axis of length greater than the data for a
    # complete visualization)
    last_time = time[-1] + (time[-1] - time[-2])
    x_axis = np.insert(time, -1, last_time)
    last_freq = freq[-1] + (freq[-1] - freq[-2])
    y_axis = np.insert(freq, -1, last_freq)

    # extract channel indexes for the rois
    rois_numbers = {}
    for roi in rois.keys():
        rois_numbers[roi] = np.array([epochs.ch_names.index(i) for i in rois[roi]])

    # for each region of interest
    for roi in rois_numbers.keys():

        # extract the channels of interest
        numbers = rois_numbers[roi]
        roi_epochs = spectrogram[:, numbers, :, :]

        # compute reference power
        reference = np.squeeze(roi_epochs[:, :, :, np.argwhere(time < 0)])
        power_reference = np.mean(reference, axis=-1)

        # compute ERDS maps according to the definition: for each epoch and for each channel, compute (A-R)/R for each
        # time instant and each frequency, where A is the current power and R the reference power
        for idx_epoch, epoch in enumerate(roi_epochs):
            for idx_channel, channel in enumerate(epoch):
                reference = power_reference[idx_epoch, idx_channel][:, np.newaxis]
                roi_epochs[idx_epoch, idx_channel] = (channel - reference)/reference

        # for each different condition found in the epochs
        for condition in conditions:

            # extract only the epochs of interest
            roi_condition_epochs = roi_epochs[list(locate(annotations, lambda x: x == condition))]

            # mean over extracted epochs and channels
            roi_condition_epochs = np.mean(roi_condition_epochs, axis=0)
            roi_condition_epochs = np.mean(roi_condition_epochs, axis=0)

            # plot the ERDS map
            roi_condition_epochs = np.array(roi_condition_epochs)
            z_min, z_max = -np.abs(roi_condition_epochs).max(), np.abs(roi_condition_epochs).max()
            fig, ax = plt.subplots()
            p = ax.pcolor(x_axis, y_axis, roi_condition_epochs, cmap='RdBu', snap=True, vmin=z_min, vmax=z_max)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(condition + ' ' + roi)
            ax.axvline(0, color='k')
            fig.colorbar(p, ax=ax)
            if path is not None:
                fig.savefig(path + '/' + condition + '_' + roi + '_erds.png')
            else:
                plt.show()
            plt.close()


def compute_erds_numpy(signals, annotations, channels_list, rois, fs, t_min, f_min=0, f_max=50, path=None):
    """
    Function to compute ERDS maps for a set of epochs according to different regions of interest
    :param signals: Numpy array containing epochs data - shape: epochs x channels x instants
    :param annotations: numpy array containing labels of the epochs - shape: epochs
    :param channels_list: list of channels name - shape: channels
    :param rois: dict object containing a set of key-value pairs. The key must be the name of the region of
    interest, the value is a list containing the channels belonging to that ROI
    :param fs: sampling frequency of the EEG acquisition
    :param t_min: time instant of the epoch (with respect to the stimuli instant)
    :param f_min: minimum frequency for which the ERDS maps are visualized
    :param f_max: maximum frequency for which the ERDS maps are visualized
    :param path: path where to save the computed ERDS maps. If None, the maps are just shown.
    """

    # get the list of unique annotations from the data
    conditions = list(set(annotations))

    # compute the spectrogram for the signals
    freq, time, spectrogram = scipy.signal.spectrogram(signals, fs=fs, nperseg=250, noverlap=225, scaling='spectrum')
    # epochs x channels x frequencies x time windows

    # adjust time interval and consider just frequencies of interest
    time = time + t_min
    spectrogram = np.squeeze(spectrogram[:, :, np.argwhere(np.logical_and(f_min < freq, freq < f_max)), :])
    freq = np.squeeze(freq[np.argwhere(np.logical_and(f_min < freq, freq < f_max))])

    # compute the vectors for visualization axis (it's necessary to have axis of length greater than the data for a
    # complete visualization)
    last_time = time[-1] + (time[-1] - time[-2])
    x_axis = np.insert(time, -1, last_time)
    last_freq = freq[-1] + (freq[-1] - freq[-2])
    y_axis = np.insert(freq, -1, last_freq)

    # extract channel indexes for the rois
    rois_numbers = {}
    for roi in rois.keys():
        rois_numbers[roi] = np.array([channels_list.index(i) for i in rois[roi]])

    # for each region of interest
    for roi in rois_numbers.keys():

        # extract the channels of interest
        numbers = rois_numbers[roi]
        roi_epochs = spectrogram[:, numbers, :, :]

        # compute reference power
        reference = np.squeeze(roi_epochs[:, :, :, np.argwhere(time < 0)])
        power_reference = np.mean(reference, axis=-1)

        # compute ERDS maps according to the definition: for each epoch and for each channel, compute (A-R)/R for each
        # time instant and each frequency, where A is the current power and R the reference power
        for idx_epoch, epoch in enumerate(roi_epochs):
            for idx_channel, channel in enumerate(epoch):
                reference = power_reference[idx_epoch, idx_channel][:, np.newaxis]
                roi_epochs[idx_epoch, idx_channel] = (channel - reference)/reference

        # for each different condition found in the epochs
        for condition in conditions:

            # extract only the epochs of interest
            roi_condition_epochs = roi_epochs[list(locate(annotations, lambda x: x == condition))]

            # mean over extracted epochs and channels
            roi_condition_epochs = np.mean(roi_condition_epochs, axis=0)
            roi_condition_epochs = np.mean(roi_condition_epochs, axis=0)

            # plot the ERDS map
            roi_condition_epochs = np.array(roi_condition_epochs)
            z_min, z_max = -np.abs(roi_condition_epochs).max(), np.abs(roi_condition_epochs).max()
            fig, ax = plt.subplots()
            p = ax.pcolor(x_axis, y_axis, roi_condition_epochs, cmap='RdBu', snap=True, vmin=z_min, vmax=z_max)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(condition + ' ' + roi)
            ax.axvline(0, color='k')
            fig.colorbar(p, ax=ax)
            if path is not None:
                fig.savefig(path + '/' + condition + '_' + roi + '_erds.png')
            else:
                plt.show()
            plt.close()
