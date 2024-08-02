# ----------------------------------------------------------------------------------------------------------------------
# ERDS maps
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import mne
import json
import matplotlib.pyplot as plt
import matplotlib
import pyxdf
matplotlib.use('Qt5Agg')
from datetime import datetime
from WORKS import SUB_trial_management
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import pandas as pd
import seaborn as sns


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

def plot_erds_maps(data, picks, events, event_dict, t_min, t_max, show_epochs=True, show_erds=True, cluster_mode=False,
                   preproc_data=None, tfr_mode=False):


    epochs = mne.Epochs(data, events, event_dict, t_min - 0.5, t_max + 0.5, picks=picks, baseline=None,
                        preload=True)
    if show_epochs == True:
        epochs.plot(picks=picks, show_scrollbars=True, events=events, event_id=event_dict, block=False)
        #lt.savefig('{}/epochs_{}_run{}_{}.png'.format(data_from_mat.dir_plots, data_from_mat.motor_mode, str(data_from_mat.n_run),
        #                                                     data_from_mat.dimension), format='png')

    freqs = np.arange(1, 30)
    vmin, vmax = -1, 1  # set min and max ERDS values in plot
    # baseline = [tmin, -0.5]  # baseline interval (in s)
    baseline = [t_min, 0]  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test

    tfr = epochs.compute_tfr(method="multitaper", picks=picks, freqs=freqs, n_cycles=freqs, use_fft=True,
                             return_itc=False, average=False, decim=2)
    tfr.crop(t_min, t_max).apply_baseline(baseline,
                                        mode="percent")  # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    tfr.crop(0, t_max)


    for event in event_dict:
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
        fig.suptitle(f"ERDS - {event} hand")  #{data_from_mat.motor_mode} run {data_from_mat.n_run} {data_from_mat.dimension}")
        fig.canvas.manager.set_window_title(event + " hand ERDS maps")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        #plt.savefig('{}/erds_map_{}_run{}_{}_{}_{}{}_{}.png'.format(data_from_mat.dir_plots,
        #                                                         data_from_mat.motor_mode, str(data_from_mat.n_run),
        #                                                         data_from_mat.dimension, event, picks[0], picks[1], timestamp),
        #            format='png')
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

if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    result_path = cwd + '/Results/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    sub_path = "C:/2D3D_Analysis/Data/S14-ses2/"
    config_file_path = sub_path + "CONFIG_S14_run3_MI_3D.json"
    xdf_file_path = sub_path + "S14_run3_MI_3D.xdf"
    #preproc_file_path = sub_path + "preproc_raw/run3_preproc-raw.fif"


    # ==============================================================================
    # Load files: infos and data
    # ==============================================================================

    # CONFIG
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    subject_id = config['gui-input-settings']['subject-id']
    n_session = config['gui-input-settings']['n-session']
    n_run = config['gui-input-settings']['n-run']
    motor_mode = config['gui-input-settings']['motor-mode']
    erds_mode = config['feedback-model-settings']['erds']['mode']
    dimension = config['gui-input-settings']['dimension-mode']
    feedback = config['gui-input-settings']['fb-mode']

    lsl_config = config['general-settings']['lsl-streams']
    sample_rate = config['eeg-settings']['sample-rate']
    duration_ref = config['general-settings']['timing']['duration-ref']
    duration_cue = config['general-settings']['timing']['duration-cue']
    duration_fb = config['general-settings']['timing']['duration-task']
    duration_task = duration_cue + duration_fb
    n_ref = int(np.floor(sample_rate * duration_ref))
    n_cue = int(np.floor(sample_rate * duration_cue))
    n_fb = int(np.floor(sample_rate * duration_fb))
    n_samples_task = int(np.floor(sample_rate * duration_task))
    n_samples_trial = n_ref + n_samples_task

    # Channels & ROIs
    channel_dict = config['eeg-settings']['channels']
    enabled_ch_names = [name for name, settings in channel_dict.items() if settings['enabled']]
    enabled_ch = np.subtract([settings['id'] for name, settings in channel_dict.items() if settings['enabled']], 1) # Python starts at 0

    roi_ch_nr = config['feedback-model-settings']['erds']['single-mode-channels']
    n_roi = len(roi_ch_nr)
    roi_dict = {settings['id']: name for name, settings in channel_dict.items()}
    roi_ch_names = [roi_dict[id_] for id_ in roi_ch_nr]
    roi_enabled_ix = [enabled_ch_names.index(ch) for ch in roi_ch_names if ch in enabled_ch_names]

    # XDF
    streams, fileheader = pyxdf.load_xdf(xdf_file_path)
    stream_names = []

    for stream in streams:
        stream_names.append(stream['info']['name'][0])

    streams_info = np.array(stream_names)

    # gets 'BrainVision RDA Data' stream - EEG data
    eeg_pos = np.where(streams_info == lsl_config['eeg']['name'])[0][0]
    eeg = streams[eeg_pos]
    eeg_signal = eeg['time_series'][:, :32]
    eeg_signal = eeg_signal * 1e-6
    # get the instants
    eeg_instants = eeg['time_stamps']
    time_zero = eeg_instants[0]
    eeg_instants = eeg_instants - time_zero
    # get the sampling frequencies
    eeg_fs = int(float(eeg['info']['nominal_srate'][0]))
    effective_sample_frequency = float(eeg['info']['effective_srate'])
    eeg_raw = eeg_signal[:, enabled_ch]

    # gets 'BrainVision RDA Markers' stream
    orn_pos = np.where(streams_info == 'BrainVision RDA Markers')[0][0]
    orn = streams[orn_pos]
    orn_signal = orn['time_series']
    orn_instants = orn['time_stamps']-time_zero

    # gets marker stream
    marker_pos = np.where(streams_info == lsl_config['marker']['name'])[0][0]
    marker = streams[marker_pos]
    marker_ids = marker['time_series']
    marker_instants = marker['time_stamps']-time_zero
    marker_dict = {
        'Reference': 10,
        'Start_of_Trial_l': 1,
        'Start_of_Trial_r': 2,
        'Cue': 20,
        'Feedback': 30,
        'End_of_Trial': 5, # equal to break
        'Session_Start': 4,
        'Session_End': 3}
    # Manage markers
    marker_interpol = SUB_trial_management.interpolate_markers(marker_ids, marker_dict, marker_instants, eeg_instants)
    n_trials = marker_ids.count(['Start_of_Trial_l']) + marker_ids.count(['Start_of_Trial_r'])


    ###################################################
    # ERDS map plotting
    ###################################################
    plot_erds_maps()