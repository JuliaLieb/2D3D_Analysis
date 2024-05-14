import os.path as op
import os
import shutil

import mne
from mne.datasets import eegbci

from mne_bids import write_raw_bids, BIDSPath, print_dir_tree
from mne_bids.stats import count_events

# source: https://mne.tools/mne-bids/stable/auto_examples/convert_eeg_to_bids.html

subject = 1
run = 2
eegbci.load_data(subject=subject, runs=run, update_path=True)

#mne_data_dir = mne.get_config("MNE_DATASETS_EEGBCI_PATH")
#data_dir = op.join(mne_data_dir, "MNE-eegbci-data")
src_data_dir = os.getcwd() + "/Data/"
bids_dir = os.getcwd() + "/Bidsdata/"

#print_dir_tree(src_data_dir)
#print(bids_dir)

edf_path = eegbci.load_data(subject=subject, runs=run)[0]
raw = mne.io.read_raw_edf(edf_path, preload=False)
raw.info["line_freq"] = 50

print('test')
'''
# Get the electrode coordinates
testing_data = mne.datasets.testing.data_path()
captrak_path = op.join(testing_data, "montage", "captrak_coords.bvct")
montage = mne.channels.read_dig_captrak(captrak_path)

# Rename the montage channel names only for this example, because as said
# before, coordinate and EEG data were not actually collected together
# Do *not* do this for your own data.
montage.rename_channels(dict(zip(montage.ch_names, raw.ch_names)))

# "attach" the electrode coordinates to the `raw` object
# Note that this only works for some channel types (EEG/sEEG/ECoG/DBS/fNIRS)
raw.set_montage(montage)

# show the electrode positions
raw.plot_sensors()

# zero padding to account for >100 subjects in this dataset
subject_id = "001"

# define a task name and a directory where to save the data to
task = "RestEyesClosed"
bids_root = op.join(mne_data_dir, "eegmmidb_bids_eeg_example")

if op.exists(bids_root):
    shutil.rmtree(bids_root)

print(raw.annotations)

bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)
write_raw_bids(raw, bids_path, overwrite=True)
print_dir_tree(bids_root)
'''