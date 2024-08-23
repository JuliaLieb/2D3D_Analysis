# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime
import mne
from matplotlib.colors import TwoSlopeNorm


def find_config_files(path, subject_id):
    config_files = []
    for file in os.listdir(path):
        if file.startswith('CONFIG_' + subject_id):
            if file.endswith('.json'):
                config_files.append(path + file)
    return config_files

# %% general conditions
if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/../Data/'
    results_path = cwd + '/../Results/'
    interim_path = cwd + '/../InterimResults/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # subjects and sessions
    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                    'S16', 'S17']

    # study conditions
    mon_me = [0] * len(subject_list)
    mon_mi = [2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2] #S1-17
    vr_mi = [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1] #S1-17
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    roi = ["F3", "F4", "C3", "C4", "P3", "P4"]
    task = ['MI left', 'MI right']
    freq = 'alpha'
    freq_band = [8,12]
    #freq = 'beta'
    #freq_band = [16, 24]

    ### ----- DATA FOR TESTING ----- ###
    '''
    subject_list = ['S14']
    mon_mi = [2]
    vr_mi = [1]
    conditions = {'Control': mon_me, 'Monitor': mon_mi, 'VR': vr_mi}
    '''

# %% combine evoked for conditions
    #"""
    all_evoked_left_VR = []
    all_evoked_right_VR = []
    all_evoked_left_Mon = []
    all_evoked_right_Mon = []
    all_evoked_left_Con = []
    all_evoked_right_Con = []

    # Iterations over all files to calculate evoked results
    for subj_ix, subj in enumerate(subject_list):
        interim_subj_path = interim_path + subj + '/'
        for ses_key, ses_values in conditions.items():
            ses_ix = ses_values[subj_ix]
            subject_data_path = data_path + subj + '-ses' + str(ses_ix) + '/'
            ev_left = mne.read_evokeds(results_path + subj + '-ses' + ses_key + '_' + freq + '-left-ave.fif')[0]
            ev_right = mne.read_evokeds(results_path + subj + '-ses' + ses_key + '_' + freq + '-right-ave.fif')[0]
            if ses_key == 'VR':
                all_evoked_left_VR.append(ev_left)
                all_evoked_right_VR.append(ev_right)
            elif ses_key == 'Monitor':
                all_evoked_left_Mon.append(ev_left)
                all_evoked_right_Mon.append(ev_right)
            elif ses_key == 'Control':
                all_evoked_left_Con.append(ev_left)
                all_evoked_right_Con.append(ev_right)
    evoked_left_VR = mne.combine_evoked(all_evoked_left_VR, weights='nave')
    evoked_right_VR = mne.combine_evoked(all_evoked_right_VR, weights='nave')
    evoked_left_Mon = mne.combine_evoked(all_evoked_left_Mon, weights='nave')
    evoked_right_Mon = mne.combine_evoked(all_evoked_right_Mon, weights='nave')
    evoked_left_Con = mne.combine_evoked(all_evoked_left_Con, weights='nave')
    evoked_right_Con = mne.combine_evoked(all_evoked_right_Con, weights='nave')
    print('debug')


    freqs = np.linspace(freq_band[0], freq_band[1], 10)  # Frequenzen von bis
    baseline = (1.5, 3)
    vmin, vmax = -1, 1.5
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS

    power_left_VR = evoked_left_VR.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    power_left_VR.apply_baseline(baseline, mode='percent')
    power_right_VR = evoked_right_VR.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    power_right_VR.apply_baseline(baseline, mode='percent')
    power_left_Mon = evoked_left_Mon.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    power_left_Mon.apply_baseline(baseline, mode='percent')
    power_right_Mon = evoked_right_Mon.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    power_right_Mon.apply_baseline(baseline, mode='percent')
    power_left_Con = evoked_left_Con.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    power_left_Con.apply_baseline(baseline, mode='percent')
    power_right_Con = evoked_right_Con.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    power_right_Con.apply_baseline(baseline, mode='percent')

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    power_left_VR.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, axes=axes[0, 0], show=False, cmap="RdBu", vlim=(vmin, vmax), cnorm=cnorm, contours=10)
    axes[0, 0].set_title('Left Hand - VR')
    power_right_VR.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, axes=axes[0, 1], show=False, cmap="RdBu", vlim=(vmin, vmax), cnorm=cnorm, contours=10)
    axes[0, 1].set_title('Right Hand - VR')
    power_left_Mon.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, axes=axes[1, 0], show=False, cmap="RdBu", vlim=(vmin, vmax), cnorm=cnorm, contours=10)
    axes[1, 0].set_title('Left Hand - Monitor')
    power_right_Mon.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, axes=axes[1, 1], show=False, cmap="RdBu", vlim=(vmin, vmax), cnorm=cnorm, contours=10)
    axes[1, 1].set_title('Right Hand - Monitor')
    power_left_Con.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, axes=axes[2, 0], show=False, cmap="RdBu", vlim=(vmin, vmax), cnorm=cnorm, contours=10)
    axes[2, 0].set_title('Left Hand - Control')
    power_right_Con.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, axes=axes[2, 1], show=False, cmap="RdBu", vlim=(vmin, vmax), cnorm=cnorm, contours=10)
    axes[2, 1].set_title('Right Hand - Control')
    if freq == 'alpha':
        fig.suptitle('Alpha Frequency Band (8-12Hz)')
    else:
        fig.suptitle('Beta Frequency Band (16-24 Hz)')
    plt.tight_layout()
    plt.show()

    #winsound.Beep(750, 1000)
    #winsound.Beep(550, 1000)