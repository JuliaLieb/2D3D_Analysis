import numpy as np
from scipy.signal import iirfilter, sosfiltfilt, iirdesign, sosfilt_zi, sosfilt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

data = np.random.uniform(-0.01, 0.01, size=(32, 50))

f_pass1 = 9.0  # Lower passband frequency
f_stop1 = 7.5  # Lower stopband frequency
f_pass2 = 11.0  # Upper passband frequency
f_stop2 = 12.0  # Upper stopband frequency
n = 12
nyquist = 500 / 2.0
wp = [f_pass1 / nyquist, f_pass2 / nyquist]
ws = [f_stop1 / nyquist, f_stop2 / nyquist]

filt_data = np.zeros_like(data)
for sample in range(data.shape[1]):
    sample_data = data[:, sample]
    sos = iirfilter(int(n / 2), wp, btype='bandpass', ftype='butter', output='sos')
    zi = sosfilt_zi(sos)
    filt_sample, _ = sosfilt(sos, sample_data, zi=zi, axis=0)
    filt_data[:,sample] = filt_sample

'''
filt_data2 = np.zeros_like(data)
for sample in range(data.shape[1]):
    sample_data = data[:, sample]
    sos = iirfilter(N=n, Wn=wp, rs=60, btype='band', analog=False, ftype='butter', output='sos')
    filt_sample = sosfilt(sos, sample_data, axis=1)
    filt_data2[:,sample] = filt_sample
'''

plt.plot(filt_data[0,:], label='filtered data')
plt.plot(data[0,:], label='data')
plt.legend()
plt.show()



