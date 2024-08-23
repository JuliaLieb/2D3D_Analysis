import numpy as np
from scipy import signal


class Bandpass:
    """Bandpass unit.

    Holds parameters and methods for bandpass filtering.

    Parameters
    ----------
    order: `int`
        The order of the filter.
    fpass: `list`
        Frequencies of the filter.
    n: `int`
        Number of eeg channels.

    Other Parameters
    ----------------
    sos: `ndarray`
        Second-order sections representation of the filter.
    zi0: `ndarray`
        Initial conditions for the filter delay.
    zi: `ndarray`
        Current filter delay values.
    """

    def __init__(self, order, fstop, fpass, n):
        self.order = order
        self.fstop = fstop
        self.fpass = fpass
        self.n = n
        self.sos = None
        self.zi0 = None
        self.zi = None

        self.__init_filter()

    def __init_filter(self):
        """Computes the second order sections of the filter and the initial conditions for the filter delay.
        """

        self.sos = signal.iirfilter(int(self.order / 2), self.fpass, btype='bandpass', ftype='butter', output='sos')
        zi = signal.sosfilt_zi(self.sos)

        if self.n > 1:
            zi = np.tile(zi, (self.n, 1, 1)).T
        self.zi0 = zi.reshape((np.shape(self.sos)[0], 2, self.n))
        self.zi = self.zi0

    def bandpass_filter(self, x):
        """Bandpass filters the input array.

        Parameters
        ----------
        x: `ndarray`
            Raw eeg data.

        Returns
        -------
        y: `int`
            Band passed data.
        """

        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi, axis=0)
        return y

def filter_complete_eeg(eeg, eeg_instants, bp_filter):
    """
    Applies a bandpass filter to each sample of EEG data at specified instants.

    Parameters:
    eeg (ndarray): A 2D NumPy array of shape (n_samples, n_channels) containing EEG data, where each row represents a
                    sample and each column represents a channel.
    eeg_instants (ndarray): A 1D array or list containing the instants (time indices) at which the EEG samples
                            are taken.
    bp_filter (object): An object with a method `bandpass_filter` that takes a 2D NumPy array and returns a filtered
                        2D NumPy array of the same shape.

    Returns:
    ndarray: A 2D NumPy array of shape (n_samples, n_channels) containing the bandpass-filtered EEG data.
    """
    eeg_filt_all = np.zeros(eeg.shape)
    for index, t in enumerate(eeg_instants):
        sample = eeg[index, :][np.newaxis, :]
        eeg_filt_all[index,:] = bp_filter.bandpass_filter(sample)

    return eeg_filt_all

def filter_per_status(eeg, eeg_instants, bp_filter, marker_interpol, status):
    """
    Applies a bandpass filter to each sample of EEG data at specified instants only if the sample's marker status
    matches specified statuses.

    Parameters:
    eeg (ndarray): A 2D NumPy array of shape (n_samples, n_channels) containing EEG data, where each row represents
                    a sample and each column represents a channel.
    eeg_instants (ndarray): A 1D array or list containing the instants (time indices) at which the EEG samples
                            are taken.
    bp_filter (object): An object of class Bandpass with a method `bandpass_filter` that takes a 2D NumPy array and
                        returns a filtered 2D NumPy array of the same shape.
    marker_interpol (ndarray): A 1D array or list containing markers corresponding to each sample in the EEG data.
    status (set or list): A set or list of statuses for which the bandpass filter should be applied.

    Returns:
    ndarray: A 2D NumPy array of shape (n_samples, n_channels) containing the EEG data, with the bandpass filter
                applied to samples with specified statuses.
    """
    eeg_filt_stat = np.zeros(eeg.shape)
    for index, t in enumerate(eeg_instants):
        sample = eeg[index, :][np.newaxis, :]
        if marker_interpol[index] in status:  # filter only status = Start, Ref, Feedback
            eeg_filt_stat[index, :] = bp_filter.bandpass_filter(sample)
        else:
            eeg_filt_stat[index, :] = sample

    return eeg_filt_stat