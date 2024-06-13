from scipy import signal
import numpy as np


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

        self.sos = signal.iirfilter(int(self.order / 2), self.fpass, btype='bandpass', ftype='butter',
                                    output='sos')

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