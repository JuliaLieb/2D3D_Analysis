# ----------------------------------------------------------------------------------------------------------------------
# Read data of config and xdf signal
# save original_raw, preproc_raw, epochs, evoked
# ----------------------------------------------------------------------------------------------------------------------
import json
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
import mne
import os
import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test

class Measurement_Data:
    def __init__(self, config_file_path, subject_directory):
        with open(config_file_path) as json_file:
            self.config = json.load(json_file)

        # CONFIG Infos
        self.subject_id = self.config['gui-input-settings']['subject-id']
        self.n_session = self.config['gui-input-settings']['n-session']
        self.n_run = self.config['gui-input-settings']['n-run']
        self.motor_mode = self.config['gui-input-settings']['motor-mode']
        self.erds_mode = self.config['feedback-model-settings']['erds']['mode']
        self.dimension = self.config['gui-input-settings']['dimension-mode']
        self.feedback = self.config['gui-input-settings']['fb-mode']
        if self.motor_mode == "MI":
            if self.dimension == "2D":
                self.ses_key = "Monitor"
            if self.dimension == "3D":
                self.ses_key = "VR"
        elif self.motor_mode == "ME":
            self.ses_key = "Control"

        self.lsl_config = self.config['general-settings']['lsl-streams']
        self.sample_rate = self.config['eeg-settings']['sample-rate']
        self.duration_ref = self.config['general-settings']['timing']['duration-ref']
        self.duration_cue = self.config['general-settings']['timing']['duration-cue']
        self.duration_task = self.duration_cue + self.config['general-settings']['timing']['duration-task']
        self.n_ref = int(np.floor(self.sample_rate * self.duration_ref))
        self.n_cue = int(np.floor(self.sample_rate * self.duration_cue))
        self.n_samples_task = int(np.floor(self.sample_rate * self.duration_task))
        self.n_samples_trial = self.n_ref + self.n_samples_task

        # other settings
        self.lp_freq = 0.5  #9  # low pass frequency
        self.hp_freq = 30  #11    # high pass frequency
        self.n_freq = 50  # Notch filter frequency

        # initializations
        self.raw, self.info = None, None
        self.raw_filt = None
        self.eeg_stream, self.eeg_data, self.eeg_signal, self.eeg_instants, self.eeg_fs, self.length = None, None, None, None, None, None
        self.marker_stream, self.marker_ids, self.marker_instants = None, None, None

        # channels and bads
        self.ch_names = []
        self.bads = []
        self.ch_types = []
        for name in self.config['eeg-settings']['channels']:
            self.ch_types.append('eeg')
            if name == 'FP2':
                self.ch_names.append('Fp2')  # solve problem with MNE and BrainProduct incompatibility
            else:
                self.ch_names.append(name)
            if self.config['eeg-settings']['channels'][name]['enabled'] is False:
                if name == 'FP2':
                    self.bads.append('Fp2')
                else:
                    self.bads.append(name)
        self.n_ch = len(self.ch_names)
        # info['bads'] = bads
        # print(info)

        # regions of interest
        self.roi_info = {
            "frontal left": ["F3", "F7"],
            "frontal right": ["F4", "F8"],
            "central left": ["FC1", "FC5", "C3", "T7"],
            "central right": ["FC2", "FC6", "C4", "T8"],
            "parietal left": ["CP5", "CP1", "P3", "P7"],
            "parietal right": ["CP6", "CP2", "P4", "P8"]}
        self.rois_numbers = {}

        # directories
        self.dir_subj_interim = subject_directory + '/../../InterimResults/' + self.subject_id + '/'
        self.dir_plots = self.dir_subj_interim + 'plots/'
        if not os.path.exists(self.dir_plots):
            os.makedirs(self.dir_plots)

        # .xdf file
        self.xdf_file = subject_directory + self.subject_id + '_run' + str(self.n_run) + '_' + self.motor_mode + '_' + self.dimension + '.xdf'
        self.load_xdf()
        self.eeg_signal = self.eeg_signal * 1e-6

        #class info
        self.event_dict = dict(left=0, right=1)
        self.event_id = {'left': 1, 'right': 2}
        self.eeg_data = self.add_class_labels()
        indexes_class_1 = np.where(self.eeg_data[0, :] == 121)[0]
        indexes_class_2 = np.where(self.eeg_data[0, :] == 122)[0]
        self.ix_class_all = np.sort(np.append(indexes_class_1, indexes_class_2), axis=0)
        self.cl_labels = self.eeg_data[0, self.ix_class_all] - 121
        self.evnt = np.column_stack(
            (self.ix_class_all, np.zeros(len(self.ix_class_all), dtype=int),
             np.array(self.cl_labels, dtype=int)))


        # epochs & annotations & events & frequency bands
        self.bad_epochs = ['Reference', 'Cue', 'Feedback', 'Session_Start', 'Session_End', 'End_of_Trial' ]
        self.epochs_reject_criteria = {"eeg": 0.0002}  # original 0.0002, 0.0005 & None probiert
        self.annotations = None
        self.events, self.event_mapping, self.epochs, self.epochs_alpha, self.epochs_beta = None, None, None, None, None
        self.events_r, self.events_l, self.epochs_r, self.epochs_l = None, None, None, None
        self.tmin = 0
        self.tmax = 11.25
        self.baseline = [1.5, 3]
        self.evoked = {}
        self.evoked_left, self.evoked_right = None, None
        self.alpha_band, self.beta_band = None, None

    def load_xdf(self):
        """Loads a xdf file and extracts the eeg and marker stream, also erds and lda data
        Parameters
        ----------
        self

        Returns
        -------
        stream1: `dict`
            The eeg stream.
        stream2: `dict`
            The marker stream.
        stream3: `dict`
            The erds stream.
        stream4: `dict`
            The lda stream.
        """
        streams, fileheader = pyxdf.load_xdf(self.xdf_file)
        stream_names = []

        for stream in streams:
            stream_names.append(stream['info']['name'][0])

        streams_info = np.array(stream_names)

        ''' # Read results from online ERDS calculation and LDA
        if self.feedback == "on":
            erds_pos = np.where(streams_info == self.lsl_config['fb-erds']['name'])[0][0]
            lda_pos = np.where(streams_info == self.lsl_config['fb-lda']['name'])[0][0]
            erds = streams[erds_pos]
            lda = streams[lda_pos]
        '''
        # gets 'BrainVision RDA Markers' stream
        if self.subject_id != 'S10': # Missing stream for S10
            orn_pos = np.where(streams_info == 'BrainVision RDA Markers')[0][0]
            orn = streams[orn_pos]
            orn = streams[orn_pos]
            orn_signal = orn['time_series']
            orn_instants = orn['time_stamps']

        # gets 'BrainVision RDA Data' stream
        eeg_pos = np.where(streams_info == self.lsl_config['eeg']['name'])[0][0]
        self.eeg_stream = streams[eeg_pos]
        self.eeg_signal = self.eeg_stream['time_series'][:, :32]
        #
        # get the instants
        self.eeg_instants = self.eeg_stream['time_stamps']
        # get the sampling frequencies
        self.eeg_fs = int(float(self.eeg_stream['info']['nominal_srate'][0]))
        effective_sample_frequency = float(self.eeg_stream['info']['effective_srate'])

        # gets marker stream
        marker_pos = np.where(streams_info == self.lsl_config['marker']['name'])[0][0]
        self.marker_stream = streams[marker_pos]
        self.marker_ids = self.marker_stream['time_series']
        self.marker_instants = self.marker_stream['time_stamps']

        # cast to arrays
        #self.eeg_instants = np.array(self.eeg_instants)
        #self.eeg_signal = np.asmatrix(self.eeg_signal)

        # check lost-samples problem
        if self.subject_id != 'S10': # Missing stream for S10
            if len(orn_signal) != 0:
                print('\n\nATTENTION: some samples have been lost during the acquisition!!\n\n')
                self.fix_lost_samples(orn_signal, orn_instants, effective_sample_frequency)

        # remove signal mean
        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

        # get the length of the acquisition
        self.length = self.eeg_instants.shape[0]

    def add_class_labels(self):
        """Adds another row to the stream. It includes the class labels at the cue positions.

        Parameters
        ----------
        stream: `dict`
            The LSL stream which should be appended.
        marker_stream: `dict`
            Marker stream containing info about the cue times and class labels.

        Returns
        -------
        result: `ndarray`
            Data of stream with class labels in the first row.
        """

        time = self.eeg_stream['time_stamps']
        marker_series = np.array(self.marker_stream['time_series'])
        cue_times = (self.marker_stream['time_stamps'])[np.where(marker_series == 'Cue')[0]]

        conditions = marker_series[np.where(np.char.find(marker_series[:, 0], 'Start_of_Trial') == 0)[0]]
        conditions[np.where(conditions == 'Start_of_Trial_l')] = 121
        conditions[np.where(conditions == 'Start_of_Trial_r')] = 122

        cue_positions = np.zeros((np.shape(time)[0], 1), dtype=int)
        for t, c in zip(cue_times, conditions):
            pos = self.find_nearest_index(time, t)
            cue_positions[pos] = c

        return (np.append(cue_positions, self.eeg_stream['time_series'], axis=1)).T

    def find_nearest_index(self, array, value):
        """Finds the position of the value which is nearest to the input value in the array.

        Parameters
        ----------
        array: `ndarray`
            The array of time stamps.
        value: `float`
            The (nearest) value to find in the array.

        Returns
        -------
        idx: `int`
            The index of the (nearest) value in the array.
        """

        idx = np.searchsorted(array, value, side="right")
        if idx == len(array):
            return idx - 1
        else:
            return idx

    def fix_lost_samples(self, orn_signal, orn_instants, effective_sample_frequency):

        print('BrainVision RDA Markers: ', orn_signal)
        print('BrainVision RDA Markers instants: ', orn_instants)
        print('\nNominal srate: ', self.eeg_fs)
        print('Effective srate: ', effective_sample_frequency)

        print('Total number of samples: ', len(self.eeg_instants))
        final_count = len(self.eeg_signal)
        for lost in orn_signal:
            final_count += int(lost[0].split(': ')[1])
        print('Number of samples with lost samples integration: ', final_count)

        total_time = len(self.eeg_instants) / effective_sample_frequency
        real_number_samples = total_time * self.eeg_fs
        print('Number of samples with real sampling frequency: ', real_number_samples)

        # print(self.eeg_instants)

        differences = np.diff(self.eeg_instants)
        differences = (differences - (1 / self.eeg_fs)) * self.eeg_fs
        # differences = np.round(differences, 4)
        print('Unique differences in instants: ', np.unique(differences))
        print('Sum of diff ', np.sum(differences))
        # plt.plot(differences)
        # plt.ylim([1, 2])
        # plt.show()

        new_marker_signal = self.marker_instants

        for idx, lost_instant in enumerate(orn_instants):
            x = np.where(self.marker_instants < lost_instant)[0][-1]

            missing_samples = int(orn_signal[idx][0].split(': ')[1])
            additional_time = missing_samples / self.eeg_fs

            new_marker_signal[(x + 1):] = np.array(new_marker_signal[(x + 1):]) + additional_time

    def create_raw(self):
        """
        Creation of MNE raw instance from the data, setting the general information and the relative montage.
        Create also the dictionary for the regions of interest according to the current data file
        """

        # create info and RAW variables with MNE for the data
        self.info = mne.create_info(self.ch_names, self.eeg_fs, self.ch_types)
        self.raw = mne.io.RawArray(self.eeg_signal.T, self.info) #, first_samp=self.eeg_instants[0])
        self.marker_instants = self.marker_instants - self.eeg_instants[0]  # have same time instants for eeg and marker

        # set montage setting according to the input
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(standard_montage)

        if len(self.bads) > 0:
            self.raw.info['bads'] = self.bads
            self.raw.interpolate_bads(reset_bads=True)

        # channel numbers associated to each roi
        for roi in self.roi_info.keys():
            self.rois_numbers[roi] = np.array([self.raw.ch_names.index(i) for i in self.roi_info[roi]])

    def raw_spatial_filtering(self, signal):
        """
        Resetting the reference in raw data according to the spatial filtering type in the input dict
        """

        mne.set_eeg_reference(signal, ref_channels='average', copy=False)

    def raw_time_filtering(self, signal):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        """
        # apply band-pass filter
        if not (self.lp_freq is None and self.hp_freq is None):
            signal.filter(l_freq=self.lp_freq, h_freq=self.hp_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1,
                            verbose=40, method='fir')

        # apply notch filter
        if self.n_freq is not None:
            signal.notch_filter(freqs=self.n_freq, verbose=40)

    def visualize_raw(self, signal=True, psd=False, psd_topo=False):
        """
        Visualization of the plots that could be generated with MNE according to a scaling property
        :param signal: boolean, if the signal plot should be generated
        :param psd: boolean, if the psd plot should be generated
        :param psd_topo: boolean, if the topographic psd plot should be generated
        """

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)

        if signal:
            mne.viz.plot_raw(self.raw, scalings=viz_scaling, duration=10, show_first_samp=True)
        if psd:
            self.raw.plot_psd()
        if psd_topo:
            self.raw.compute_psd().plot_topo()
        plt.show()

    def raw_ica(self, signal):
        n_components = len(self.ch_names)

        eeg_raw = signal.copy()
        eeg_raw = eeg_raw.pick_types(eeg=True)

        ica = mne.preprocessing.ICA(n_components=0.99999, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        #ica.plot_sources(eeg_raw)
        #ica.plot_components()
        # ica.plot_properties(eeg_raw)

        reconst_raw = self.raw_filt.copy()
        ica.apply(reconst_raw)

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)
        #reconst_raw.plot(scalings=viz_scaling)
        #reconst_raw.plot_psd()
        return reconst_raw

    def create_annotations(self, full=True):
        """
        Annotations creation according to MNE definition. Annotations are extracted from markers stream data (onset,
        duration and description)
        :param full: annotations can be made of just one word or more than one. In 'full' case the whole annotation is
        considered, otherwise only the second word is kept
        :return:
        """

        # generation of the events according to the definition
        triggers = {'onsets': [], 'duration': [], 'description': []}

        # read every trigger in the stream
        for idx, marker_data in enumerate(self.marker_ids):

            # annotations to be rejected
            if marker_data[0] in self.bad_epochs:
                continue

            # extract triggers information
            triggers['onsets'].append(self.marker_instants[idx])
            if marker_data[0] == 'Start_of_Trial_l':
                triggers['duration'].append(11.25)
            elif marker_data[0] == 'Start_of_Trial_r':
                triggers['duration'].append(11.25)
            else:
                triggers['duration'].append(int(0))

            # according to 'full' parameter, extract the correct annotation description
            if not full:
                condition = marker_data[0].split('/')[-1]
            else:
                condition = marker_data[0]
            if condition == 'edges': condition = 'canny'
            triggers['description'].append(condition)

        # define MNE annotations
        self.annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'])

    def create_epochs(self, signal):
        """
        Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
        the raw data according to the events
        :param signal: Measurement_Data variable (all frequency bands, alpha or beta)
        :param visualize_epochs: boolean variable to select if generate epochs plots or not
        :param set_annotations: boolean variable, if it's necessary to set the annotations or not
        """

        # set the annotations on the current raw and extract the correspondent events
        signal.set_annotations(self.annotations)
        self.events, self.event_mapping = mne.events_from_annotations(signal)

        # generation of the epochs according to the events
        epochs = mne.Epochs(signal, self.events, preload=True, baseline=self.baseline,
                            reject=self.epochs_reject_criteria, tmin=self.tmin, tmax=self.tmax, event_id=self.event_id)
        # event_id=self.event_mapping, baseline=(self.tmin, 0)

        return epochs

    def get_frequency_band(self, freqency_band):
        frequency_band=self.raw_filt.copy()
        frequency_band.filter(freqency_band[0], freqency_band[1], fir_design='firwin')
        return frequency_band

    def visualize_all_epochs(self, signal=True, rois=True):
        """
        :param signal: boolean, if visualize the whole signal with triggers or not
        :param conditional_epoch: boolean, if visualize the epochs extracted from the events or the general mean epoch
        :param rois: boolean (only if conditional_epoch=True), if visualize the epochs according to the rois or not
        """
        #self.visualize_raw(signal=signal, psd=False, psd_topo=False)

        rois_names = list(self.roi_info)
        ch_picks = ['C3', 'C4']

        # generate the mean plot considering all the epochs conditions

            # generate the epochs plots according to the roi and save them
        if rois:
            images = self.epochs.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots  + rois_names[idx] + '.png')
                plt.close(img)

        # generate the epochs plots for each channel and save them
        else:
            images = self.epochs.plot_image(show=False, picks=ch_picks)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots + ch_picks[idx] + '.png')
                plt.close(img)

        plt.close('all')

    # EVOKED
    def create_evoked(self, rois=True):
        """
        Function to define the evoked variables starting from the epochs. The evoked will be considered separately for
        each condition present in the annotation and for each ROI (otherwise, in general for the whole dataset)
        :param rois: boolean variable to select if visualize results according to the rois or just the  general results
        """
        # for each condition
        for condition, key in self.event_mapping.items():

            # get only the epochs of interest
            condition_epochs = self.epochs[key]
            #condition_epochs = self.events_l

            if rois:
                # for each roi of interest
                for roi in sorted(self.rois_numbers.keys()):
                    # extract only the channels of interest
                    condition_roi_epoch = condition_epochs.copy()
                    condition_roi_epoch = condition_roi_epoch.pick(self.rois_numbers[roi])

                    # average for each epoch and for each channel
                    condition_roi_epoch = condition_roi_epoch.average()
                    condition_roi_epoch = mne.channels.combine_channels(condition_roi_epoch, groups={'mean': list(range(len(self.rois_numbers[roi])))})

                    # define the label for the current evoked and save it
                    label = condition + '/' + roi
                    self.evoked[label] = condition_roi_epoch

            else:

                # average for each epoch and for each channel
                condition_epochs = condition_epochs.average()
                condition_epochs = mne.channels.combine_channels(condition_epochs, groups={'mean': list(
                    range(len(self.epochs.ch_names)))})

                # save the current evoked
                self.evoked['mean'] = condition_epochs

    def visualize_evoked(self):
        """
        Function to plot the computed evoked for each condition and for each region of interest
        """

        # get minimum and maximum value of the mean signals
        min_value, max_value = np.inf, -np.inf
        for label in self.evoked.keys():
            data = self.evoked[label].get_data()[0]
            min_value = min(np.min(data), min_value)
            max_value = max(np.max(data), max_value)

        # path for images saving
        #Path(self.dir_files + '/epochs/').mkdir(parents=True, exist_ok=True)

        number_conditions = len(list(self.event_mapping.keys()))
        #path = self.dir_files + '/epochs/conditions.png'
        fig, axs = plt.subplots(int(np.ceil(number_conditions/2)), 2, figsize=(25.6, 19.2))

        for i, ax in enumerate(fig.axes):

            if i >= number_conditions:
                break

            condition = list(self.event_mapping.keys())[i]

            # extract the roi from the key name of the dictionary containing the evoked
            correct_labels = [s for s in self.evoked.keys() if condition + '/' in s]
            correct_short_labels = [s.split('/')[1] for s in correct_labels]

            # correctly plot all evoked
            for idx, label in enumerate(correct_labels):
                ax.plot(self.evoked[label].times * 1000, self.evoked[label].get_data()[0],
                        label=correct_short_labels[idx])

            # draw ERP vertical lines to see the peak of interest
            #for erp in self.input_info['erp']:
            #    ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(condition)
            ax.legend()

        #plt.savefig(path)
        #plt.close()

        #path = self.dir_files + '/epochs/rois.png'
        number_rois = len(list(self.rois_numbers.keys()))
        fig, axs = plt.subplots(int(np.ceil(number_rois/2)), 2, figsize=(25.6, 19.2))

        for i, ax in enumerate(fig.axes):

            if i >= number_rois:
                break

            roi = list(self.rois_numbers.keys())[i]

            # extract the condition from the key name of the dictionary containing the evoked
            correct_labels = [s for s in self.evoked.keys() if '/' + roi in s]
            correct_short_labels = [s.split('/')[0] for s in correct_labels]

            # correctly plot all evoked
            for idx, label in enumerate(correct_labels):
                ax.plot(self.evoked[label].times * 1000, self.evoked[label].get_data()[0],
                        label=correct_short_labels[idx])

            # draw ERP vertical lines to see the peak of interest
            #for erp in self.input_info['erp']:
            #    ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(roi)
            ax.legend()

        #plt.savefig(path)
        #plt.close()


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

    def plot_erds_maps(self, epochs, picks=['C3', 'C4'], show_epochs=False, show_erds=False, cluster_mode=False):

        if show_epochs == True:
            epochs.plot(picks=picks, show_scrollbars=True, events=self.evnt,
                        event_id=self.event_dict, block=False)

        freqs = np.arange(1, 30)
        vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
        # baseline = [tmin, -0.5]  # baseline interval (in s)
        # baseline interval (in s)  #[self.tmin, 0]
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS
        kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                      buffer_size=None, out_type='mask')  # for cluster test

        tfr = epochs.compute_tfr(method="multitaper", picks=picks, freqs=freqs, n_cycles=freqs, use_fft=True,
                                 return_itc=False, average=False, decim=2)
        tfr.crop(self.tmin, self.tmax).apply_baseline(self.baseline, mode="percent")
        tfr.crop(0, self.tmax)

        for event in self.event_dict:
            # select desired epochs for visualization
            tfr_ev = tfr[event]
            num_channels = len(picks)
            num_cols = 2
            num_rows = int(np.ceil(num_channels / num_cols))
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4),
                                 gridspec_kw={'height_ratios': [1]*num_rows, 'hspace': 0.3})
            axes = axes.flatten()
            for ch in range(num_channels):
                ax = axes[ch]
                if cluster_mode:
                    # find clusters
                    c, p, mask = self.calc_clustering(tfr_ev, ch, kwargs)

                    # plot TFR (ERDS map with masking)
                    tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                          colorbar=False, show=False, vlim=(-1.5, 1.5), mask=mask)  # , mask=mask,
                else:
                    tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                          colorbar=False, show=False, vlim=(-1.5, 1.5))

                ax.set_title(tfr_ev.ch_names[ch], fontsize=10)
                ax.axvline(3, linewidth=1, color="black", linestyle=":")  # event
                if ch != 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels("")

            fig.colorbar(axes[0].images[-1], ax=axes, orientation='horizontal', fraction=0.025, pad=0.08)
            fig.suptitle(
                f"{event} hand {self.ses_key} run {self.n_run}")
            fig.canvas.manager.set_window_title(event + " hand ERDS maps")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            plt.savefig('{}erds_map_{}_run{}_{}_{}.png'.format(self.dir_plots,
                                                                        self.ses_key,
                                                                        str(self.n_run), event, timestamp),
                        format='png')
        if show_erds == True:
            plt.show()

    def run_preprocessing(self, plt=False):
        file_name = self.dir_subj_interim + 'ses' + self.ses_key + '_run' + str(self.n_run)
        self.create_raw()
        self.raw.save(file_name + '-original-raw.fif', overwrite=True)
        self.raw_filt = self.raw.copy()

        self.raw_spatial_filtering(self.raw_filt)
        self.raw_time_filtering(self.raw_filt)
        self.raw_filt = self.raw_ica(self.raw_filt)

        self.create_annotations()
        self.raw.set_annotations(self.annotations)
        self.raw_filt.set_annotations(self.annotations)

        self.raw_filt.save(file_name + '-preproc-raw.fif', overwrite=True)

        # all frequencies
        self.epochs = self.create_epochs(self.raw_filt)
        self.epochs.save(file_name + '-epo.fif', overwrite=True)
        #self.evoked_left = self.epochs['1'].average()
        #self.evoked_left.save(file_name + '-LEFT-ave.fif', overwrite=True)
        #self.evoked_right = self.epochs['2'].average()
        #self.evoked_right.save(file_name + '-RIGHT-ave.fif', overwrite=True)

        # alpha band
        self.alpha_band = self.get_frequency_band([8, 13])
        self.epochs_alpha = self.create_epochs(self.alpha_band)
        self.epochs_alpha.save(file_name + '-alpha-epo.fif', overwrite=True)
        #evoked_left_alpha = self.epochs['1'].average()
        #evoked_left_alpha.save(file_name + '-LEFT-alpha-ave.fif', overwrite=True)
        #evoked_right_alpha = self.epochs['2'].average()
        #evoked_right_alpha.save(file_name + '-RIGHT-alpha-ave.fif', overwrite=True)

        # beta band
        self.beta_band = self.get_frequency_band([16, 24])
        self.epochs_beta = self.create_epochs(self.beta_band)
        self.epochs_beta.save(file_name + '-beta-epo.fif', overwrite=True)
        #evoked_left_beta = self.epochs['1'].average()
        #evoked_left_beta.save(file_name + '-LEFT-beta-ave.fif', overwrite=True)
        #evoked_right_beta = self.epochs['2'].average()
        #evoked_right_beta.save(file_name + '-RIGHT-beta-ave.fif', overwrite=True)

        if plt:
            self.plot_erds_maps(self.epochs)

