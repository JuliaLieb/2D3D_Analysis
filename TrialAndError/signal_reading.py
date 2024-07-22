# ----------------------------------------------------------------------------------------------------------------------
# Read data of config and xdf signal
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

class Input_Data:
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
        self.dir_plots = subject_directory + 'plots'
        if not os.path.exists(self.dir_plots):
            os.makedirs(self.dir_plots)
        #self.dir_files = subject_directory + '/dataset'
        #if not os.path.exists(self.dir_files):
        #    print(".mat files are not available to initialize EEG signal.")
        #    return
        self.dir_epochs = subject_directory + 'epochs'
        if not os.path.exists(self.dir_epochs):
            os.makedirs(self.dir_epochs)

        self.dir_preproc = subject_directory + 'preproc_raw'
        if not os.path.exists(self.dir_preproc):
            os.makedirs(self.dir_preproc)

        # .xdf file
        self.xdf_file = subject_directory + self.subject_id + '_run' + str(self.n_run) + '_' + self.motor_mode + '_' + self.dimension + '.xdf'
        self.load_xdf()
        self.eeg_signal = self.eeg_signal * 1e-6

        #class info
        self.event_dict = dict(left=0, right=1)
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
        self.epochs_reject_criteria = None  #{"eeg": 0.0002}  # original 0.0002, 0.0005 & None probiert
        self.annotations = None
        self.events, self.event_mapping, self.epochs = None, None, None
        self.events_r, self.events_l, self.epochs_r, self.epochs_l = None, None, None, None
        self.tmin = 0
        self.tmax = 11.25
        self.evoked = {}
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
        #self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0) # wird bei spatial filtering gemacht

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
        self.marker_instants = self.marker_instants - self.eeg_instants[0]  # have same time instants for eeg and marker #Todo check if true!!!
        #self.raw.set_meas_date(self.eeg_instants[0])

        # set montage setting according to the input
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(standard_montage)

        if len(self.bads) > 0:
            self.raw.info['bads'] = self.bads
            self.raw.interpolate_bads(reset_bads=True)

        # channel numbers associated to each roi
        for roi in self.roi_info.keys():
            self.rois_numbers[roi] = np.array([self.raw.ch_names.index(i) for i in self.roi_info[roi]])

    def raw_spatial_filtering(self):
        """
        Resetting the reference in raw data according to the spatial filtering type in the input dict
        """

        mne.set_eeg_reference(self.raw_filt, ref_channels='average', copy=False)

    def raw_time_filtering(self):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        """
        # apply band-pass filter
        if not (self.lp_freq is None and self.hp_freq is None):
            self.raw_filt.filter(l_freq=self.lp_freq, h_freq=self.hp_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1,
                            verbose=40, method='fir')

        # apply notch filter
        if self.n_freq is not None:
            self.raw_filt.notch_filter(freqs=self.n_freq, verbose=40)

    def raw_time_filtering_options(self):
        # Define the frequency specifications
        f_pass1 = 9.0  # Lower passband frequency
        f_stop1 = 7.5  # Lower stopband frequency
        f_pass2 = 11.0  # Upper passband frequency
        f_stop2 = 12.0  # Upper stopband frequency
        n = 12

        # Normalize frequencies by Nyquist frequency (half the sampling rate)
        nyquist = self.sample_rate / 2.0
        wp = [f_pass1 / nyquist, f_pass2 / nyquist]
        ws = [f_stop1 / nyquist, f_stop2 / nyquist]

        # Get data from raw
        data = self.raw_filt.get_data()

        opt1, opt2, opt3 = False, False, True

        if opt1:
            # Option 1
            # Filterentwurf: Übergangsbänder und Passband-Ripple sowie Stopband-Dämpfung spezifizieren
            gpass = 1  # Maximaler Verlust im Passband (dB)
            gstop = 40  # Minimale Dämpfung im Stopband (dB)
            sos = iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop, ftype='butter', output='sos')
            filt_data = sosfiltfilt(sos, data, axis=1)

        elif opt2:
            # Option 2
            sos = iirfilter(N=n, Wn=wp, rs=60, btype='band', analog=False, ftype='butter', output='sos')
            filt_data = sosfilt(sos, data, axis=1)

        elif opt3:
            # Option 3
            sos = iirfilter(int(n / 2), wp, btype='bandpass', ftype='butter', output='sos')
            zi = sosfilt_zi(sos)
            filt_data, _ = sosfilt(sos, data, zi=zi, axis=0)

        else:
            print("no filter option chosen - ERROR")

        # Write filtered data to raw
        self.raw_filt._data = filt_data

    def filter_by_sample(self):
        # Define the frequency specifications
        f_pass1 = 9.0  # Lower passband frequency
        f_stop1 = 7.5  # Lower stopband frequency
        f_pass2 = 11.0  # Upper passband frequency
        f_stop2 = 12.0  # Upper stopband frequency
        n = 12

        # Normalize frequencies by Nyquist frequency (half the sampling rate)
        nyquist = self.sample_rate / 2.0
        wp = [f_pass1 / nyquist, f_pass2 / nyquist]
        ws = [f_stop1 / nyquist, f_stop2 / nyquist]

        # Get data from raw
        data = self.raw_filt.get_data()

        '''
        # -----------like opt 3
        filt_data = np.zeros_like(data) 
        for sample in range(data.shape[1]):
            sample_data = data[:, sample]
            sos = iirfilter(int(n / 2), wp, btype='bandpass', ftype='butter', output='sos')
            zi = sosfilt_zi(sos)
            filt_sample, _ = sosfilt(sos, sample_data, zi=zi, axis=0)
            filt_data[:, sample] = filt_sample
        '''
        '''
        # -----------like opt 3 # funktioniert nicht!
        filt_data = np.zeros_like(data)
        for sample in range(data.shape[1]):
            sample_data = data[:, sample]
            sos = iirdesign(wp=wp, ws=ws, gpass=1, gstop=40, ftype='butter', output='sos')
            filt_sample = sosfiltfilt(sos, sample_data)
            filt_data[:, sample] = filt_sample
        '''

        # -----------nächster Versuch!
        # Define your filter parameters
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y

        filt_data = np.zeros_like(data)
        for i in range(data.shape[1]):  # Loop over each sample
            filt_data[:, i] = butter_bandpass_filter(data[:, i], lowcut=f_pass1, highcut=f_pass2, fs=self.sample_rate, order=n)

        # Write filtered data to raw
        self.raw_filt._data = filt_data

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

    def raw_ica(self):
        n_components = len(self.ch_names)

        eeg_raw = self.raw_filt.copy()
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
        self.raw_filt = reconst_raw

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

    def create_epochs(self, sig='raw', visualize_epochs=True, rois=True, set_annotations=True):
        """
        Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
        the raw data according to the events
        :param visualize_epochs: boolean variable to select if generate epochs plots or not
        :param rois: boolean variable to select if visualize results according to the rois or for each channel
        :param set_annotations: boolean variable, if it's necessary to set the annotations or not
        """
        if sig == 'raw':
            #signal = self.raw
            signal = self.raw_filt
        elif sig == 'alpha':
            signal = self.alpha_band
        elif sig == 'beta':
            signal = self.beta_band
        else:
            print('Select correct signal type raw, alpha or beta.')

        # set the annotations on the current raw and extract the correspondent events
        if set_annotations:
            signal.set_annotations(self.annotations)
        self.events, self.event_mapping = mne.events_from_annotations(signal)

        # Automatic rejection criteria for the epochs
        reject_criteria = self.epochs_reject_criteria

        # generation of the epochs according to the events
        self.epochs = mne.Epochs(signal, self.events, preload=True, baseline=(self.tmin, 0),
                                 reject=reject_criteria, tmin=self.tmin, tmax=self.tmax) # event_id=self.event_mapping,

        # separate left and right
        index_r = np.where(self.events[:, 2] == 2)
        index_l = np.where(self.events[:, 2] == 1)
        self.events_r = self.events[index_r]
        self.events_l = self.events[index_l]
        self.epochs_r = mne.Epochs(signal, self.events_r, preload=True, baseline=(self.tmin, 0),
                                 reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)
        self.epochs_l = mne.Epochs(signal, self.events_l, preload=True, baseline=(self.tmin, 0),
                                 reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)


        if visualize_epochs:
            self.visualize_all_epochs(signal=False, rois=False)
            #self.visualize_l_r_epochs(signal=False, rois=False)

        self.epochs.save(self.dir_epochs + '/run' + str(self.n_run) + '-' + sig + '-epo.fif', overwrite=True)
        self.epochs_l.save(self.dir_epochs + '/l_run' + str(self.n_run) + '-' + sig + '-epo.fif', overwrite=True)
        self.epochs_r.save(self.dir_epochs + '/r_run' + str(self.n_run) + '-' + sig + '-epo.fif', overwrite=True)

        return self.epochs_l, self.epochs_r

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
                img.savefig(self.dir_plots + '/' + rois_names[idx] + '.png')
                plt.close(img)

        # generate the epochs plots for each channel and save them
        else:
            images = self.epochs.plot_image(show=False, picks=ch_picks)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots + '/' + ch_picks[idx] + '.png')
                plt.close(img)

        plt.close('all')

    def visualize_l_r_epochs(self, signal=True, rois=True):

        #self.visualize_raw(signal=signal, psd=False, psd_topo=False)
        rois_names = list(self.roi_info)
        ch_picks = ['C3', 'C4']

        # generate the mean plots according to the condition in the annotation value

        # generate the epochs plots according to the roi and save them
        if rois:
            # right
            images = self.epochs_r.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots + '/right_' + rois_names[idx] + '.png')
                plt.close(img)
            # left
            images = self.epochs_l.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots + '/left_' + rois_names[idx] + '.png')
                plt.close(img)

        # generate the epochs plots for each channel and save them
        else:
            #right
            images = self.epochs_r.plot_image(picks=ch_picks, show=False)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots + '/right_' + ch_picks[idx] + '.png')
                plt.close(img)
            #left
            images = self.epochs_l.plot_image(picks=ch_picks, show=False)
            for idx, img in enumerate(images):
                img.savefig(self.dir_plots + '/left_' + ch_picks[idx] + '.png')
                plt.close(img)

    # EVOKED
    '''
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
        Path(self.dir_files + '/epochs/').mkdir(parents=True, exist_ok=True)

        number_conditions = len(list(self.event_mapping.keys()))
        path = self.dir_files + '/epochs/conditions.png'
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

        plt.savefig(path)
        plt.close()

        path = self.dir_files + '/epochs/rois.png'
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

        plt.savefig(path)
        plt.close()
    '''

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

    def plot_erds_maps(self, picks=['C3', 'Cz', 'C4'], show_epochs=False, show_erds=True, cluster_mode=False,
                       preproc_data=False, tfr_mode=False):

        if preproc_data == False:
            raw_data = self.raw.copy()
        else:
            raw_data = self.raw_filt.copy()

        epochs = mne.Epochs(raw_data, self.evnt, self.event_dict, self.tmin - 0.5, self.tmax + 0.5,
                            picks=picks, baseline=None,
                            preload=True)
        if show_epochs == True:
            epochs.plot(picks=picks, show_scrollbars=True, events=self.evnt,
                        event_id=self.event_dict, block=False)
            plt.savefig('{}/epochs_{}_run{}_{}.png'.format(self.dir_plots, self.motor_mode,
                                                           str(self.n_run),
                                                           self.dimension), format='png')

        freqs = np.arange(1, 30)
        vmin, vmax = -1, 1  # set min and max ERDS values in plot
        # baseline = [tmin, -0.5]  # baseline interval (in s)
        baseline = [self.tmin, 0]  # baseline interval (in s)
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center, and max ERDS
        kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                      buffer_size=None, out_type='mask')  # for cluster test

        tfr = epochs.compute_tfr(method="multitaper", picks=picks, freqs=freqs, n_cycles=freqs, use_fft=True,
                                 return_itc=False, average=False, decim=2)
        tfr.crop(self.tmin, self.tmax).apply_baseline(baseline,
                                                      mode="percent")  # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
        tfr.crop(0, self.tmax)

        for event in self.event_dict:
            # select desired epochs for visualization
            tfr_ev = tfr[event]
            fig, axes = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]})  # , 0.5
            axes = axes.flatten()
            for ch, ax in enumerate(axes[:-1]):  # for each channel  axes[:-1]
                if cluster_mode:
                    # find clusters
                    c, p, mask = self.calc_clustering(tfr_ev, ch, kwargs)

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
            fig.suptitle(
                f"ERDS - {event} hand {self.motor_mode} run {self.n_run} {self.dimension}")
            fig.canvas.manager.set_window_title(event + " hand ERDS maps")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            plt.savefig('{}/erds_map_{}_run{}_{}_{}_{}{}_{}.png'.format(self.dir_plots,
                                                                        self.motor_mode,
                                                                        str(self.n_run),
                                                                        self.dimension, event, picks[0],
                                                                        picks[1], timestamp),
                        format='png')
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
            freq_bands_of_interest = ["alpha", "beta"]
            df = df[df.band.isin(freq_bands_of_interest)]
            df["band"] = df["band"].cat.remove_unused_categories()

            # Order channels for plotting:
            df["channel"] = df["channel"].cat.reorder_categories(picks, ordered=True)

            g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
            g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
            axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
            g.map(plt.axhline, y=0, **axline_kw)
            g.map(plt.axvline, x=3, **axline_kw)
            g.set(ylim=(None, None))
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

    def run_preprocessing_to_epoch(self):
        self.create_raw()
        self.raw_filt = self.raw.copy()

        #self.visualize_raw()

        self.raw_spatial_filtering()

        self.raw_time_filtering()

        #self.visualize_raw()

        self.raw_ica()

        self.create_annotations()
        self.raw.set_annotations(self.annotations)
        self.raw_filt.set_annotations(self.annotations)

        #self.visualize_raw()
        self.raw_filt.save(self.dir_epochs + '/run' + str(self.n_run) + '-raw.fif', overwrite=True)

        alpha_freq = [8, 12]
        self.alpha_band = self.get_frequency_band(alpha_freq)
        beta_freq = [16, 24]
        self.beta_band = self.get_frequency_band(beta_freq)

        self.create_epochs(sig='raw', visualize_epochs=False)
        self.create_epochs(sig='alpha', visualize_epochs=False)
        self.create_epochs(sig='beta', visualize_epochs=False)
        #self.create_evoked()
        #self.visualize_evoked()

        #return epochs_l, epochs_r

    def run_raw(self):
        self.create_raw()
        self.raw_filt = self.raw.copy()
        #self.raw_spatial_filtering()

        self.raw_time_filtering()
        #self.raw_time_filtering_options()
        #self.filter_by_sample()

        self.create_annotations()
        self.raw.set_annotations(self.annotations)
        self.raw_filt.set_annotations(self.annotations)
        self.create_epochs(sig='raw', visualize_epochs=False)

    def preprocess_raw(self):
        self.create_raw()
        self.raw_filt = self.raw.copy()
        self.raw_spatial_filtering()
        self.raw_time_filtering()
        self.raw_ica()

        #self.raw_filt.save(self.dir_preproc + '/run' + str(self.n_run) + '-' + '_preproc-raw.fif', overwrite=True)

        #self.raw.plot_psd()
        #self.raw_filt.plot_psd()
        #plt.show()

        self.plot_erds_maps(preproc_data=False, tfr_mode=True)
        self.plot_erds_maps(preproc_data=True, tfr_mode=True)

    def run_preprocessing_to_fif(self):
        self.create_raw()
        self.raw_filt = self.raw.copy()
        self.raw_spatial_filtering()
        self.raw_time_filtering()
        self.raw_ica()

        preproc_filename = self.dir_preproc + '/run' + str(self.n_run) + '_preproc-raw.fif'
        self.raw_filt.save(preproc_filename, overwrite=True)
        print(f"Preprocessed raw data saved to {self.dir_preproc}/run{str(self.n_run)}")
        return preproc_filename








