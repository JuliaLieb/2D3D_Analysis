# ----------------------------------------------------------------------------------------------------------------------
# Read data of config and xdf signal
# ----------------------------------------------------------------------------------------------------------------------
import json
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import os
import scipy.io
import glob
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from mne.datasets import eegbci

class Input_Data:
    def __init__(self, config_file_path, subject_directory):
        with open(config_file_path) as json_file:
            self.config = json.load(json_file)

        # CONFIG Infos
        self.subject_id = self.config['gui-input-settings']['subject-id']
        self.n_session = self.config['gui-input-settings']['n-session']
        self.n_run = self.config['gui-input-settings']['n-run']
        self.motor_mode = self.config['gui-input-settings']['motor-mode']
        self.dimension = self.config['gui-input-settings']['dimension-mode']
        self.feedback = self.config['gui-input-settings']['fb-mode']

        self.lsl_config = self.config['general-settings']['lsl-streams']
        self.sample_rate = self.config['eeg-settings']['sample-rate']
        self.duration_ref = self.config['general-settings']['timing']['duration-ref']
        self.duration_cue = self.config['general-settings']['timing']['duration-cue']
        self.duration_task = self.duration_cue + self.config['general-settings']['timing']['duration-task']
        self.n_ref = int(np.floor(self.sample_rate * self.duration_ref))
        self.n_samples_task = int(np.floor(self.sample_rate * self.duration_task))
        self.n_samples_trial = self.n_ref + self.n_samples_task

        # other settings
        self.lp_freq = 1  # low pass frequency
        self.hp_freq = 60  # high pass frequency
        self.n_freq = 50  # Notch filter frequency

        # initializations
        self.raw, self.info = None, None
        self.eeg_signal, self.eeg_instants, self.eeg_fs, self.length = None, None, None, None
        self.marker_ids, self.marker_instants = None, None

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
        self.dir_plots = subject_directory + '/plots'
        if not os.path.exists(self.dir_plots):
            os.makedirs(self.dir_plots)
        self.dir_files = subject_directory + '/dataset'
        if not os.path.exists(self.dir_files):
            print(".mat files are not available to initialize EEG signal.")
            return
        # .xdf file
        self.xdf_file = subject_directory + self.subject_id + '_run' + str(self.n_run) + '_' + self.motor_mode + '_' + self.dimension + '.xdf'
        self.load_xdf()

        # epochs & annotations & events
        self.bad_epochs = ['Reference', 'Cue', 'Feedback', 'Session_Start', 'Session_End', 'End_of_Trial' ]
        self.epochs_reject_criteria = {"eeg": 0.0005}  # original 0.0002
        self.annotations = None
        self.events, self.event_mapping, self.epochs = None, None, None
        self.events_r, self.events_l, self.epochs_r, self.epochs_l = None, None, None, None
        self.tmin = 0
        self.tmax = 11.25

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
        orn_pos = np.where(streams_info == 'BrainVision RDA Markers')[0][0]
        orn = streams[orn_pos]
        orn_signal = orn['time_series']
        orn_instants = orn['time_stamps']

        # gets 'BrainVision RDA Data' stream
        eeg_pos = np.where(streams_info == self.lsl_config['eeg']['name'])[0][0]
        eeg = streams[eeg_pos]
        self.eeg_signal = eeg['time_series'][:, :32]
        self.eeg_signal = self.eeg_signal * 1e-6
        # get the instants
        self.eeg_instants = eeg['time_stamps']
        # get the sampling frequencies
        self.eeg_fs = int(float(eeg['info']['nominal_srate'][0]))
        effective_sample_frequency = float(eeg['info']['effective_srate'])

        # gets marker stream
        marker_pos = np.where(streams_info == self.lsl_config['marker']['name'])[0][0]
        marker = streams[marker_pos]
        self.marker_ids = marker['time_series']
        self.marker_instants = marker['time_stamps']


        # cast to arrays
        #self.eeg_instants = np.array(self.eeg_instants)
        #self.eeg_signal = np.asmatrix(self.eeg_signal)

        # check lost-samples problem
        if len(orn_signal) != 0:
            print('\n\nATTENTION: some samples have been lost during the acquisition!!\n\n')
            self.fix_lost_samples(orn_signal, orn_instants, effective_sample_frequency)

        # remove signal mean
        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

        # get the length of the acquisition
        self.length = self.eeg_instants.shape[0]

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

        mne.set_eeg_reference(self.raw, ref_channels='average', copy=False)

    def raw_time_filtering(self):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        """
        # apply band-pass filter
        if not (self.lp_freq is None and self.hp_freq is None):
            self.raw.filter(l_freq=self.lp_freq, h_freq=self.hp_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1, verbose=40)

        # apply notch filter
        if self.n_freq is not None:
            self.raw.notch_filter(freqs=self.n_freq, verbose=40)

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

    def raw_ica_remove_eog(self):

        n_components = len(self.ch_names)

        eeg_raw = self.raw.copy()
        eeg_raw = eeg_raw.pick_types(eeg=True)

        ica = mne.preprocessing.ICA(n_components=0.99999, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        ica.plot_sources(eeg_raw)
        ica.plot_components()
        # ica.plot_properties(eeg_raw)

        reconst_raw = self.raw.copy()
        ica.apply(reconst_raw)

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)
        reconst_raw.plot(scalings=viz_scaling)
        reconst_raw.plot_psd()
        self.raw = reconst_raw

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

    def create_epochs(self, visualize_epochs=True, rois=True, set_annotations=True):
        """
        Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
        the raw data according to the events
        :param visualize_epochs: boolean variable to select if generate epochs plots or not
        :param rois: boolean variable to select if visualize results according to the rois or for each channel
        :param set_annotations: boolean variable, if it's necessary to set the annotations or not
        """

        # set the annotations on the current raw and extract the correspondent events
        if set_annotations:
            self.raw.set_annotations(self.annotations)
        self.events, self.event_mapping = mne.events_from_annotations(self.raw)

        # Automatic rejection criteria for the epochs
        reject_criteria = self.epochs_reject_criteria

        # generation of the epochs according to the events
        self.epochs = mne.Epochs(self.raw, self.events, preload=True, baseline=(self.tmin, 0),
                                 reject=reject_criteria, tmin=self.tmin, tmax=self.tmax) # event_id=self.event_mapping,

        # separate left and right
        index_r = np.where(self.events[:, 2] == 2)
        index_l = np.where(self.events[:, 2] == 1)
        self.events_r = self.events[index_r]
        self.events_l = self.events[index_l]
        self.epochs_r = mne.Epochs(self.raw, self.events_r, preload=True, baseline=(self.tmin, 0),
                                 reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)
        self.epochs_l = mne.Epochs(self.raw, self.events_l, preload=True, baseline=(self.tmin, 0),
                                 reject=reject_criteria, tmin=self.tmin, tmax=self.tmax)


        if visualize_epochs:
            #self.visualize_all_epochs(signal=False, rois=False)
            self.visualize_l_r_epochs(signal=False, rois=False)

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
            '''
            for key, condition in self.event_mapping.items():
                images = self.epochs[condition].plot_image(combine='mean')
                #images = self.epochs[key].plot()
                for idx, img in enumerate(images):
                    img.savefig(
                        self.dir_plots + '/' + condition + '_' + self.ch_names[idx] + '.png')
                    plt.close(img)
            '''


    def run_raw(self):
        self.create_raw()

        #self.visualize_raw()

        #self.raw_spatial_filtering()

        #self.raw_time_filtering()

        #self.visualize_raw()

        #self.raw_ica_remove_eog()

        self.create_annotations()
        self.raw.set_annotations(self.annotations)

        #self.visualize_raw()

        self.create_epochs()




        print('Debug')






