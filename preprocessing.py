import numpy as np
import mne

from offline_analysis import EEG_Signal



if __name__ == "__main__":
    current_config_path = "C:/2D3D_Analysis/Data/S1-ses0/CONFIG_S1_run1_ME_2D.json"
    subject_data_path = "C:/2D3D_Analysis/Data/S1-ses0/"

    eeg = EEG_Signal(current_config_path, subject_data_path)
    raw = eeg.raw

    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    #ica = mne.preprocessing.ICA(n_components=0.99999, method='fastica', random_state=97)
    ica.fit(raw)
    ica.exclude = [1, 2]
    ica.plot_properties(raw, picks=ica.exclude)
