import mne
import matplotlib
matplotlib.use('Qt5Agg')
from mne.datasets import eegbci
from mne.preprocessing import ICA
import mne_bids_pipeline

from offline_analysis import EEG_Signal

def gorella(current_config_path, subject_data_path):
    # source: https://g0rella.github.io/gorella_mwn/preprocessing_eeg.html

    eeg = EEG_Signal(current_config_path, subject_data_path)
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=eeg.ch_names, sfreq=eeg.sample_rate, ch_types='eeg')
    info.set_montage(montage)

    raw = mne.io.RawArray(eeg.data_eeg[1:eeg.n_ch + 1, :], info)

    eegbci.standardize(raw)

    #raw.plot_sensors(kind='3d', show=True, block=True)
    #raw.plot(scalings='auto')  # +/- 200 µV scale (1 V = 1000000 µV)

    #raw.plot_psd(tmin=0, tmax=60, fmin=2, fmax=80, average=True, spatial_colors=False, show=True)
    raw.compute_psd(tmin=0, tmax=60, fmin=2, fmax=80)#.plot()
    #raw.compute_psd().plot()
    raw.notch_filter(50)
    raw.filter(l_freq=1.0, h_freq=50.0)
    raw.resample(120, npad='auto')
    raw.compute_psd(tmin=0, tmax=60, fmin=2, fmax=60)#.plot()
    #raw.plot(scalings='auto')

    ica = ICA(n_components=15, method='fastica')
    ica.fit(raw)
    ica.plot_components()

    #raw.plot(n_channels=32, scalings='auto')

    ica.plot_properties(raw, picks=0)
    ica.plot_properties(raw, picks=9)
    ica.plot_overlay(raw, exclude=[0])
    ica.exclude = [0]
    ica.apply(raw)

    #raw.plot(n_channels=32, scalings='auto')


if __name__ == "__main__":
    current_config_path = "/Data/S20-ses0/CONFIG_S20_run1_ME_2D.json"
    subject_data_path = "/Data/S20-ses0/"

    eeg = EEG_Signal(current_config_path, subject_data_path)

    #gorella(current_config_path, subject_data_path)
    mne_bids_pipeline


    print("iwas")
