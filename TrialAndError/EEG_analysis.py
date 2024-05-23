import os
import pathlib
import pickle
from pathlib import Path

import mne
import numpy as np
import pyxdf
from matplotlib import pyplot as plt
from ERDS_analysis import compute_erds

'''
Modified from source: https://github.com/scwlab/EEG-preprocessing
Giulia Pezzutti
'''

class EEGAnalysis:
    """
    Class implemented for the EEG preprocessing and visualization from the reading of .xdf files. This class is
    intended with the use of BrainVision products, LSL and Psychopy.
    """

    def __init__(self, path, dict_info):
        """
        Constructor of the class: it initializes all the necessary variables, loads the xdf file with all the
        correspondent information
        :param path: path to .xdf file containing the data. The filepath must be with following structure:
        *folder*/subj_*sub-code*_block*num*.xdf (e.g. "/data/subj_001_block1.xdf")
        :param dict_info: dict containing the main information about the processing. It must contain the following keys:
        streams, montage, filtering, spatial_filtering, samples_remove, t_min, t_max, full_annotation,
        epochs_reject_criteria, rois, bad_epoch_names, erp, erds. See the documentation for further info about the dict
        """

        # initialize variables
        self.data_path = path
        self.file_info = {}
        self.eeg_signal, self.eeg_instants, self.eeg_fs, self.length = None, None, None, None
        self.marker_ids, self.marker_instants = None, None
        self.channels_names, self.channels_types, self.evoked_rois = {}, {}, {}
        self.info, self.raw, self.bad_channels = None, None, None
        self.events, self.event_mapping, self.epochs, self.annotations = None, None, None, None
        self.evoked = {}
        self.rois_numbers = {}

        # extract info from the path
        self.get_info_from_path()
        # extract info from the dict
        self.input_info = dict_info
        self.t_min = self.input_info['t_min']  # start of each epoch
        self.t_max = self.input_info['t_max']  # end of each epoch
        # load xdf file in raw variable
        self.load_xdf()

    '''def get_info_from_path(self): 
        """
        Getting main information from file path regarding subject, folder and output folder according to
        LSLRecorder standard
        """

        # get name of the original file
        base = os.path.basename(self.data_path)
        file_name = os.path.splitext(base)[0]

        # main folder in which data is contained
        base = os.path.abspath(self.data_path)
        folder = os.path.dirname(base).split('data/')[0]
        folder = folder.replace('\\', '/')

        project_folder = str(pathlib.Path(__file__).parent.parent.absolute())

        '''
    '''
        # extraction of subject, session and run indexes
        if self.input_info['lsl-version'] == '1.12':
            subject = (file_name.split('subj_')[1]).split('_block')[0]
        elif self.input_info['lsl-version'] == '1.16':
            subject = (file_name.split('sub-')[1]).split('_ses')[0]
        else:
            subject = ''
        '''
    '''
        subject = 'S8'

        # output folder according to the standard
        output_folder = str(pathlib.Path(__file__).parent.parent.absolute()) + '/images/sub-' + subject
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.file_info = {'input_folder': folder, 'file_name': file_name, 'subject': subject,
                          'output_images_folder': output_folder, 'project_folder': project_folder}

        '''

    def load_xdf(self):
        """
        Load of .xdf file from the filepath given in input to the constructor. The function automatically divides the
        different streams in the file and extract their main information
        """

        stream_names = self.input_info['streams']

        # data loading
        dat = pyxdf.load_xdf(self.data_path)[0]

        orn_signal, orn_instants = [], []
        effective_sample_frequency = None

        # data iteration to extract the main information
        for i in range(len(dat)):
            stream_name = dat[i]['info']['name'][0]

            # gets 'BrainVision RDA Markers' stream
            if stream_name == stream_names['EEGMarkers']:
                orn_signal = dat[i]['time_series']
                orn_instants = dat[i]['time_stamps']

            # gets 'BrainVision RDA Data' stream
            if stream_name == stream_names['EEGData']:
                # get the signal
                self.eeg_signal = dat[i]['time_series'][:, :32]
                self.eeg_signal = self.eeg_signal * 1e-6
                # get the instants
                self.eeg_instants = dat[i]['time_stamps']
                # get the sampling frequencies
                self.eeg_fs = int(float(dat[i]['info']['nominal_srate'][0]))
                effective_sample_frequency = float(dat[i]['info']['effective_srate'])
                # load the channels from the data
                self.load_channels(dat[i]['info']['desc'][0]['channels'][0]['channel'])

            # gets 'PsychoPy' stream
            if stream_name == stream_names['Triggers']:
                self.marker_ids = dat[i]['time_series']
                self.marker_instants = dat[i]['time_stamps']

        # cast to arrays
        self.eeg_instants = np.array(self.eeg_instants)
        self.eeg_signal = np.asmatrix(self.eeg_signal)

        # check lost-samples problem
        if len(orn_signal) != 0:
            print('\n\nATTENTION: some samples have been lost during the acquisition!!\n\n')
            self.fix_lost_samples(orn_signal, orn_instants, effective_sample_frequency)

        # get the length of the acquisition
        self.length = self.eeg_instants.shape[0]

        # remove samples at the beginning and at the end
        samples_to_be_removed = self.input_info['samples_remove']
        if samples_to_be_removed > 0:
            self.eeg_signal = self.eeg_signal[samples_to_be_removed:self.length - samples_to_be_removed]
            self.eeg_instants = self.eeg_instants[samples_to_be_removed:self.length - samples_to_be_removed]

        # reference all the markers instant to the eeg instants (since some samples at the beginning of the
        # recording have been removed)
        self.marker_instants -= self.eeg_instants[0]
        self.marker_instants = self.marker_instants[self.marker_instants >= 0]

        # remove signal mean
        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

    def load_channels(self, dict_channels):
        """
        Upload channels name from a xdf file
        """

        # x = data[0][0]['info']['desc'][0]["channels"][0]['channel']
        # to obtain the default-dict list of the channels from the original file (data, not dat!!)

        # cycle over the info of the channels
        for idx, info in enumerate(dict_channels):

            if info['label'][0].find('dir') != -1 or info['label'][0] == 'MkIdx':
                continue

            # get channel name
            self.channels_names[idx] = info['label'][0]

            # solve problem with MNE and BrainProduct incompatibility
            if self.channels_names[idx] == 'FP2':
                self.channels_names[idx] = 'Fp2'

            # get channel type
            self.channels_types[idx] = 'eog' if info['label'][0].find('EOG') != -1 else 'eeg'

        if self.file_info['subject'] in self.input_info['bad_channels'].keys():
            self.bad_channels = self.input_info['bad_channels'][self.file_info['subject']]
        else:
            self.bad_channels = []

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
        self.info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))
        self.raw = mne.io.RawArray(self.eeg_signal.T, self.info, first_samp=self.eeg_instants[0])

        # set montage setting according to the input
        standard_montage = mne.channels.make_standard_montage(self.input_info['montage'])
        self.raw.set_montage(standard_montage)

        if len(self.bad_channels) > 0:
            self.raw.info['bads'] = self.bad_channels
            self.raw.interpolate_bads(reset_bads=True)

        rois = self.input_info['rois']

        # channel numbers associated to each roi
        for roi in rois.keys():
            self.rois_numbers[roi] = np.array([self.raw.ch_names.index(i) for i in rois[roi]])

    def visualize_raw(self, signal=True, psd=True, psd_topo=True):
        """
        Visualization of the plots that could be generated with MNE according to a scaling property
        :param signal: boolean, if the signal plot should be generated
        :param psd: boolean, if the psd plot should be generated
        :param psd_topo: boolean, if the topographic psd plot should be generated
        """

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)

        if signal:
            mne.viz.plot_raw(self.raw, scalings=viz_scaling, duration=50, show_first_samp=True)
        if psd:
            self.raw.plot_psd()
            plt.close()
        if psd_topo:
            pass

    def raw_spatial_filtering(self):
        """
        Resetting the reference in raw data according to the spatial filtering type in the input dict
        """

        mne.set_eeg_reference(self.raw, ref_channels=self.input_info['spatial_filtering'], copy=False)

    def raw_time_filtering(self):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        """

        # extract the frequencies for the filtering
        l_freq = self.input_info['filtering']['low']
        h_freq = self.input_info['filtering']['high']
        n_freq = self.input_info['filtering']['notch']

        # apply band-pass filter
        if not (l_freq is None and h_freq is None):
            self.raw.filter(l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1, verbose=40)

        # apply notch filter
        if n_freq is not None:
            self.raw.notch_filter(freqs=n_freq, verbose=40)

    def raw_ica_remove_eog(self):

        n_components = list(self.channels_types.values()).count('eeg')

        eeg_raw = self.raw.copy()
        eeg_raw = eeg_raw.pick_types(eeg=True)

        ica = mne.preprocessing.ICA(n_components=0.99999, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        ica.plot_sources(eeg_raw)
        ica.plot_components()
        # ica.plot_properties(eeg_raw)

        # # find which ICs match the EOG pattern
        # eog_indices, eog_scores = ica.find_bads_eog(self.raw, h_freq=5, threshold=3)
        # print(eog_indices)
        # ica.exclude = ica.exclude + eog_indices

        # # barplot of ICA component "EOG match" scores
        # ica.plot_scores(eog_scores)
        # # plot diagnostics
        # # ica.plot_properties(self.raw, picks=eog_indices)
        # # plot ICs applied to raw data, with EOG matches highlighted
        # ica.plot_sources(eeg_raw)

        reconst_raw = self.raw.copy()
        ica.apply(reconst_raw)

        # viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)
        # reconst_raw.plot(scalings=viz_scaling)
        # reconst_raw.plot_psd()
        self.raw = reconst_raw

    def create_annotations(self, full=False):
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
            if marker_data[0] in self.input_info['bad_epoch_names']:
                continue

            # extract triggers information
            triggers['onsets'].append(self.marker_instants[idx])
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
        reject_criteria = self.input_info['epochs_reject_criteria']

        # generation of the epochs according to the events
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_mapping, preload=True,
                                 baseline=(self.t_min, 0), reject=reject_criteria, tmin=self.t_min, tmax=self.t_max)

        if visualize_epochs:
            self.visualize_epochs(signal=False, conditional_epoch=True, rois=rois)

    def visualize_epochs(self, signal=True, conditional_epoch=True, rois=True):
        """
        :param signal: boolean, if visualize the whole signal with triggers or not
        :param conditional_epoch: boolean, if visualize the epochs extracted from the events or the general mean epoch
        :param rois: boolean (only if conditional_epoch=True), if visualize the epochs according to the rois or not
        """

        self.visualize_raw(signal=signal, psd=False, psd_topo=False)

        rois_names = list(self.rois_numbers.keys())

        # generate the mean plots according to the condition in the annotation value
        if conditional_epoch:

            # generate the epochs plots according to the roi and save them
            if rois:
                for condition in self.event_mapping.keys():
                    images = self.epochs[condition].plot_image(combine='mean', group_by=self.rois_numbers, show=False)
                    for idx, img in enumerate(images):
                        img.savefig(
                            self.file_info['output_images_folder'] + '/' + condition + '_' + rois_names[idx] + '.png')
                        plt.close(img)

            # generate the epochs plots for each channel and save them
            else:
                for condition in self.event_mapping.keys():
                    images = self.epochs[condition].plot_image(show=False)
                    for idx, img in enumerate(images):
                        img.savefig(
                            self.file_info['output_images_folder'] + '/' + condition + '_' + self.channels_names[
                                idx] + '.png')
                        plt.close(img)

        # generate the mean plot considering all the epochs conditions
        else:

            # generate the epochs plots according to the roi and save them
            if rois:
                images = self.epochs.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
                for idx, img in enumerate(images):
                    img.savefig(self.file_info['output_images_folder'] + '/' + rois_names[idx] + '.png')
                    plt.close(img)

            # generate the epochs plots for each channel and save them
            else:
                images = self.epochs.plot_image(show=False)
                for idx, img in enumerate(images):
                    img.savefig(self.file_info['output_images_folder'] + '/' + self.channels_names[idx] + '.png')
                    plt.close(img)

        plt.close('all')

    def create_evoked(self, rois=True):
        """
        Function to define the evoked variables starting from the epochs. The evoked will be considered separately for
        each condition present in the annotation and for each ROI (otherwise, in general for the whole dataset)
        :param rois: boolean variable to select if visualize results according to the rois or just the  general results
        """

        # for each condition
        for condition in self.event_mapping.keys():

            # get only the epochs of interest
            condition_epochs = self.epochs[condition]

            if rois:
                # for each roi of interest
                for roi in sorted(self.rois_numbers.keys()):
                    # extract only the channels of interest
                    condition_roi_epoch = condition_epochs.copy()
                    condition_roi_epoch = condition_roi_epoch.pick(self.rois_numbers[roi])

                    # average for each epoch and for each channel
                    condition_roi_epoch = condition_roi_epoch.average()
                    condition_roi_epoch = mne.channels.combine_channels(condition_roi_epoch, groups={
                        'mean': list(range(len(self.rois_numbers[roi])))})

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
        Path(self.file_info['output_images_folder'] + '/epochs/').mkdir(parents=True, exist_ok=True)

        number_conditions = len(list(self.event_mapping.keys()))
        path = self.file_info['output_images_folder'] + '/epochs/conditions.png'
        fig, axs = plt.subplots(int(np.ceil(number_conditions / 2)), 2, figsize=(25.6, 19.2))

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
            for erp in self.input_info['erp']:
                ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(condition)
            ax.legend()

        plt.savefig(path)
        plt.close()

        path = self.file_info['output_images_folder'] + '/epochs/rois.png'
        number_rois = len(list(self.rois_numbers.keys()))
        fig, axs = plt.subplots(int(np.ceil(number_rois / 2)), 2, figsize=(25.6, 19.2))

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
            for erp in self.input_info['erp']:
                ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(roi)
            ax.legend()

        plt.savefig(path)
        plt.close()

    def get_peak(self, t_min, t_max, peak, mean=True, channels=None):
        """
        Function to extract the peaks' amplitude from the epochs separately for each condition found and returns them
        or the mean value
        :param t_min: lower bound of the time window in which the algorithm should look for the peak
        :param t_max: upper bound of the time window in which the algorithm should look for the peak
        :param peak: +1 for a positive peak, -1 for a negative peak
        :param mean: boolean value, if the return value should be the mean value or the list of amplitudes
        :param channels: list of channels name to be investigated. If None, all the channels are considered
        :return: if mean=True, mean amplitude value; otherwise list of detected peaks' amplitude and list of the
        correspondent annotations
        """

        if channels is None:
            channels = self.raw.ch_names
        peaks = {}
        annotations = {}

        # extraction of the data of interest and of the correspondent annotations
        epochs_interest = self.epochs.copy()
        epochs_interest = epochs_interest.pick_channels(channels)
        labels = np.squeeze(np.array(epochs_interest.get_annotations_per_epoch())[:, :, 2])

        # get the unique conditions of interest
        if len(list(self.event_mapping.keys())[0].split('/')) > 1:
            conditions_interest = [ann.split('/')[1] for ann in self.event_mapping.keys()]
        else:
            conditions_interest = self.event_mapping.keys()
        conditions_interest = list(set(conditions_interest))

        # for each condition of interest
        for condition in conditions_interest:

            # get the correspondent epochs and crop the signal in the time interval for the peak searching
            condition_roi_epoch = epochs_interest[condition]
            data = condition_roi_epoch.crop(tmin=t_min, tmax=t_max).get_data()

            # if necessary, get the annotation correspondent at each epoch
            condition_labels = []
            if not mean:
                condition_labels = [label for label in labels if '/' + condition in label]

            peak_condition, latency_condition, annotation_condition = [], [], []

            # for each epoch
            for idx, epoch in enumerate(data):

                # extract the mean signal between channels
                signal = np.array(epoch).mean(axis=0)

                # find location and amplitude of the peak of interest
                peak_loc, peak_mag = mne.preprocessing.peak_finder(signal, thresh=(max(signal) - min(signal)) / 50,
                                                                   extrema=peak, verbose=False)
                peak_mag = peak_mag * 1e6

                # reject peaks too close to the beginning or to the end of the window
                if len(peak_loc) > 1 and peak_loc[0] == 0:
                    peak_loc = peak_loc[1:]
                    peak_mag = peak_mag[1:]
                if len(peak_loc) > 1 and peak_loc[-1] == (len(signal) - 1):
                    peak_loc = peak_loc[:-1]
                    peak_mag = peak_mag[:-1]

                # select peak according to the minimum or maximum one and convert the location from number of sample
                # (inside the window) to time instant inside the epoch
                if peak == -1:
                    peak_loc = peak_loc[np.argmin(peak_mag)] / self.eeg_fs + t_min
                    peak_mag = np.min(peak_mag)
                if peak == +1:
                    peak_loc = peak_loc[np.argmax(peak_mag)] / self.eeg_fs + t_min
                    peak_mag = np.max(peak_mag)

                # save the values found
                peak_condition.append(peak_mag)
                latency_condition.append(peak_loc)

                # in the not-mean case, it's necessary to save the correct labelling
                if not mean:
                    annotation_condition.append(condition_labels[idx].split('/')[0])

            # compute output values or arrays for each condition
            if mean:
                peaks[condition] = np.mean(np.array(peak_condition))
            else:
                peaks[condition] = np.array(peak_condition)
                annotations[condition] = annotation_condition

        if not mean:
            return peaks, annotations

        return peaks

    def save_pickle(self):
        """
        Function to save epochs, labels and main information into pickle files. The first two are saved as numpy arrays,
        the last one is saved as dictionary
        """

        epochs = np.array(self.epochs.get_data())
        labels = [annotation[0][2] for annotation in self.epochs.get_annotations_per_epoch()]
        info = {'fs': self.eeg_fs, 'channels': self.epochs.ch_names, 'tmin': self.t_min, 'tmax': self.t_max}

        Path(self.file_info['project_folder'] + 'data/pickle/').mkdir(parents=True, exist_ok=True)

        with open(self.file_info['project_folder'] + '/data/pickle/' + self.file_info['subject'] + '_data.pkl',
                  'wb') as f:
            pickle.dump(epochs, f)
        with open(self.file_info['project_folder'] + '/data/pickle/' + self.file_info['subject'] + '_labels.pkl',
                  'wb') as f:
            pickle.dump(labels, f)
        with open(self.file_info['project_folder'] + '/data/pickle/' + self.file_info['subject'] + '_info.pkl',
                  'wb') as f:
            pickle.dump(info, f)

        print('Pickle files correctly saved')

    def run_raw_epochs(self, visualize_raw=False, save_images=True, ica_analysis=False, create_evoked=True,
                       save_pickle=True):
        """
        Function to run all the methods previously reported. Attention: ICA is for now not used.
        :param visualize_raw: boolean, if raw signals should be visualized or not
        :param save_images: boolean, if epoch plots should be saved or not (note: they are never visualized)
        :param ica_analysis: boolean, if ICA analysis should be performed
        :param create_evoked: boolean, if Evoked computation is necessary
        :param save_pickle: boolean, if the pickles with data, label and info should be saved
        """

        self.create_raw()

        if visualize_raw:
            self.visualize_raw()

        if self.input_info['spatial_filtering'] is not None:
            self.raw_spatial_filtering()

        if self.input_info['filtering'] is not None:
            self.raw_time_filtering()

        if visualize_raw:
            self.visualize_raw()

        if ica_analysis:
            self.raw_ica_remove_eog()

        self.create_annotations(full=self.input_info['full_annotation'])
        self.create_epochs(visualize_epochs=save_images)
        if save_images:
            compute_erds(epochs=self.epochs, rois=self.input_info['rois'], fs=self.eeg_fs, t_min=self.t_min,
                         f_min=self.input_info['erds'][0], f_max=self.input_info['erds'][1],
                         path=self.file_info['output_images_folder'])
        if create_evoked:
            self.create_evoked()
            if save_images:
                self.visualize_evoked()
        if save_pickle:
            self.save_pickle()

    def run_raw(self, visualize_raw=False, ica_analysis=False):
        """
        Function to run the analysis part regarding the RAW generation, filtering and annotation (it can be useful to
        generate new raw files concatenating multiple raw, e.g. when the acquisition is in more files)
        :param visualize_raw: boolean, if raw signals should be visualized or not
        :param ica_analysis: boolean, if ICA analysis should be performed
        """

        self.create_raw()

        if visualize_raw:
            self.visualize_raw()

        if self.input_info['spatial_filtering'] is not None:
            self.raw_spatial_filtering()

        if self.input_info['filtering'] is not None:
            self.raw_time_filtering()

        if visualize_raw:
            self.visualize_raw()

        if ica_analysis:
            self.raw_ica_remove_eog()

        self.create_annotations(full=self.input_info['full_annotation'])
        self.raw.set_annotations(self.annotations)

    def run_combine_raw_epochs(self, visualize_raw=False, save_images=True, ica_analysis=False,
                               create_evoked=True, save_pickle=True, new_raws=None):
        """
        Function to combine different raw data and to create the correspondent new epochs (it can be useful when the
        acquisition is in more files)
        :param visualize_raw: boolean, if raw signals should be visualized or not
        :param save_images: boolean, if epoch plots should be saved or not (note: they are never visualized)
        :param ica_analysis: boolean, if ICA analysis should be performed
        :param create_evoked: boolean, if Evoked computation is necessary
        :param save_pickle: boolean, if the pickles with data, label and info should be saved
        :param new_raws: list of raws files to be concatenated after the current raw variable
        :return:
        """

        if not isinstance(new_raws, list):
            new_raws = []

        self.create_raw()

        if visualize_raw:
            self.visualize_raw()

        if self.input_info['spatial_filtering'] is not None:
            self.raw_spatial_filtering()

        if self.input_info['filtering'] is not None:
            self.raw_time_filtering()

        self.create_annotations(full=self.input_info['full_annotation'])
        self.raw.set_annotations(self.annotations)

        new_raws.insert(0, self.raw)
        self.raw = mne.concatenate_raws(new_raws)

        if visualize_raw:
            self.visualize_raw()

        if ica_analysis:
            self.raw_ica_remove_eog()

        self.create_epochs(visualize_epochs=save_images, set_annotations=False)
        if save_images:
            compute_erds(epochs=self.epochs, rois=self.input_info['rois'], fs=self.eeg_fs, t_min=self.t_min,
                         f_min=self.input_info['erds'][0], f_max=self.input_info['erds'][1],
                         path=self.file_info['output_images_folder'])
        if create_evoked:
            self.create_evoked()
            if save_images:
                self.visualize_evoked()
        if save_pickle:
            self.save_pickle()

    def __getattr__(self, name):
        return 'EEGAnalysis does not have `{}` attribute.'.format(str(name))
