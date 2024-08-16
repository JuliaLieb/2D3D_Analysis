# ----------------------------------------------------------------------------------------------------------------------
# ERDS processing
# ----------------------------------------------------------------------------------------------------------------------
import json
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
import mne
import os
import scipy
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.signal import iirfilter, sosfiltfilt, iirdesign, sosfilt_zi, sosfilt, butter, lfilter
from scipy import signal
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.time_frequency import tfr_multitaper
import winsound
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap


def calc_clustering(tfr_ev, ch, kwargs):
    # positive clusters
    _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
    # negative clusters
    _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

    c = np.stack(c1 + c2, axis=2)  # combined clusters
    p = np.concatenate((p1, p2))  # combined p-values
    mask = c[..., p <= 0.05].any(axis=-1)  # mind the p-value!

    return c, p, mask


def plot_erds_maps(epochs, picks, t_min, t_max, path, session, show_erds=False, cluster_mode=False, tfr_mode=False):
    freqs = np.arange(1, 30)
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    # baseline = [tmin, -0.5]  # baseline interval (in s)
    baseline = [1.5, 3]  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test

    tfr = epochs.compute_tfr(method="multitaper", picks=picks, freqs=freqs, n_cycles=freqs, use_fft=True,
                             return_itc=False, average=False, decim=2)
    tfr.crop(t_min, t_max).apply_baseline(baseline,
                                        mode="percent")  # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    tfr.crop(0, t_max)


    for event in epochs.event_id:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        num_channels = len(picks)
        num_cols = 2
        num_rows = int(np.ceil(num_channels/num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4),
                                 gridspec_kw={'height_ratios': [1]*num_rows, 'hspace': 0.3})
        axes = axes.flatten()

        for ch in range(num_channels):
            ax = axes[ch]
            print(f"\n \n --------------- Channel {ch+1} ---------------")

            if cluster_mode:
                # find clusters
                c, p, mask = calc_clustering(tfr_ev, ch, kwargs)

                # plot TFR (ERDS map with masking)
                tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                      colorbar=False, show=False, mask=mask, mask_style="both")  #mask_style="both", no contours ="mask"
            else:
                tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                      colorbar=False, show=False)  # , vlim=(-1.5, 1.5)

            ax.set_title(tfr_ev.ch_names[ch], fontsize=10)
            ax.axvline(3, linewidth=1, color="black", linestyle=":")  # event after 3 seconds
            if ch in [1, 3, 5]:
                ax.set_ylabel("")
                ax.set_yticklabels("")

        # Remove any extra axes that may not be used
        for idx in range(num_channels, len(axes)):
            fig.delaxes(axes[idx])

        fig.colorbar(axes[0].images[-1], ax=axes, orientation='horizontal', fraction=0.025, pad=0.08)
        #fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle(f"{session} - {event} Hand")  #{data_from_mat.motor_mode} run {data_from_mat.n_run} {data_from_mat.dimension}")
        fig.canvas.manager.set_window_title(event + " hand ERDS maps")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig('{}/erds_map_ses{}_{}_hand_{}.png'.format(path, session, event, timestamp),
                    format='png')
        plt.close(fig)
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


def calc_avg_erds_per_subj(epochs, picks, t_min, t_max, freq, avg_rois=False):
    freqs = np.arange(freq[0], freq[1]+1)
    start_time = 4.25
    end_time = 11.25

    baseline = [1.5, 3]  # baseline interval (in s)

    tfr = epochs.compute_tfr(method="multitaper", picks=picks, freqs=freqs, n_cycles=freqs/2, use_fft=True,
                             return_itc=False, average=False, decim=2)
    tfr.crop(t_min, t_max).apply_baseline(baseline,
                                        mode="percent")  # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    tfr.crop(0, t_max)

    # Find indices corresponding to the time range
    times = tfr.times
    start_idx = np.argmin(np.abs(times - start_time))
    end_idx = np.argmin(np.abs(times - end_time))

    tfr_ev_l = tfr['left']
    selected_data_l = tfr_ev_l.data[:, :, :, start_idx:end_idx]
    # Average over the selected time interval and all frequencies
    avg_over_time_l = np.mean(selected_data_l, axis=-1)
    avg_over_freq_and_time_l = np.mean(avg_over_time_l, axis=2)
    final_avg_l = np.mean(avg_over_freq_and_time_l, axis=0)

    tfr_ev_r = tfr['right']
    selected_data_r = tfr_ev_r.data[:, :, :, start_idx:end_idx]
    # Average over the selected time interval and all frequencies
    avg_over_time_r = np.mean(selected_data_r, axis=-1)
    avg_over_freq_and_time_r = np.mean(avg_over_time_r, axis=2)
    final_avg_r = np.mean(avg_over_freq_and_time_r, axis=0)

    if avg_rois:
        roi_avg_l = np.mean(final_avg_l, axis=0)
        roi_avg_r = np.mean(final_avg_r, axis=0)
        return roi_avg_l, roi_avg_r
    else:
        return final_avg_l, final_avg_r

def plot_inter_intra_erds(avg_erds, roi, session, cl, freq, path):
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_erds, cmap="RdYlBu", aspect='auto', norm=cnorm)  # cmap="viridis"
    plt.colorbar(label='ERD/S (%)')
    plt.title(f'{session} {cl}')
    plt.xlabel(f'{freq} frequency band')
    plt.ylabel('Subjects')

    plt.xticks(np.arange(6), roi)
    plt.yticks(np.arange(17), [f'Sub{i + 1}' for i in range(17)])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig('{}/erds_magnitudes_ses{}_{}_hand_{}_{}.png'.format(path, session, cl, freq, timestamp),
                format='png')


def plot_erds_topo(evoked, freq, freq_band, baseline, path, session, cl):
    freqs = np.linspace(freq_band[0], freq_band[1], 10)  # Frequenzen von bis
    vmin, vmax = -1, 1.5
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS

    tfr = evoked.compute_tfr(method="multitaper", freqs=freqs, n_cycles=freqs/2, use_fft=True)
    tfr.apply_baseline(baseline, mode='percent')

    fig = tfr.plot_topomap(ch_type="eeg", tmin=4.25, tmax=11.25, show=False, cmap="RdBu", vlim=(vmin, vmax),
                           cnorm=cnorm, contours=10, size=4)
    fig.suptitle(f"{session} - {cl} hand / {freq} ({freq_band[0]} - {freq_band[1]} Hz)", fontsize=10)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fig.savefig('{}/erds_topo_ses{}_{}_hand_{}_{}.png'.format(path, session, cl, freq, timestamp),
                format='png')


def run_spearman(data, subjects, title, freq, path=None):
    '''
    Runs spearman's correlation between subjects.'
    Args:
        data: ndarray size (len(subjects), 6 rois)
        subjects: list of str

    Returns: saved plot

    '''

    # Calculate Spearman correlation matrix and pairwise distance
    corr_matrix = np.zeros((len(subjects), len(subjects)))
    distance_matrix = np.zeros((len(subjects), len(subjects)))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:  # No need to correlate a subject with itself
                corr, _ = spearmanr(data[i], data[j])
                corr_matrix[i, j] = corr
                distance_matrix[i, j] = 1 - corr
            else:
                corr_matrix[i, j] = 1  # Self-correlation is always 1
                distance_matrix[i, j] = 0  # Distance to self is 0

    # Mask to only rank lower half of matrix - minimize errors due to ranking
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    masked_distance_matrix = np.where(mask, np.nan, distance_matrix)

    # Flatten the distance matrix for ranking
    flat_distances = masked_distance_matrix.flatten()

    # Rank the distances
    ranked_distances = np.argsort(np.argsort(flat_distances))

    # Reshape to the original distance matrix shape
    ranked_matrix = ranked_distances.reshape(distance_matrix.shape)

    # Scale the ranked distances between 0 and 1
    scaler = MinMaxScaler()
    scaled_ranked_matrix_half = scaler.fit_transform(ranked_matrix)

    # Mirror lower half to upper half of matrix
    results = np.zeros((len(subjects), len(subjects)))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i >= j:
                results[i, j] = scaled_ranked_matrix_half[i, j]
            elif i < j:
                results[i, j] = scaled_ranked_matrix_half[j, i]

    # Create DataFrames for better visualization
    scaled_distance_df = pd.DataFrame(results, index=subjects, columns=subjects)

    # Plotting the scaled distance heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(scaled_distance_df, annot=False, fmt=".2f", cmap="RdYlBu", linewidths=0.5, square=True,
                cbar_kws={"shrink": .75})
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig('{}/spearman_{}_{}_{}.png'.format(path, title, freq, timestamp),
                    format='png')
    else:
        plt.show()
