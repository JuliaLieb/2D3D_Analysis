import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')


def remove_zero_lines(array):
    """
    Removes rows from each 2D slice of a 3D NumPy array where all elements in the row are zero. Number of non-zero rows
    does not have to be consistent.

    Parameters:
    array (ndarray): A 3D NumPy array of shape (x, y, z) where x is the number of 2D slices, y is the number of rows in
                    each slice, and z is the number of columns in each row.

    Returns:
    ndarray: A list of 2D NumPy arrays (dtype=object), where each 2D array corresponds to a slice from the original
            input array with rows containing all zeros removed.
    """

    array_clean = []
    for i in range(array.shape[0]):
        slice_ = array[i]
        mask = ~(np.all(slice_ == 0, axis=1))
        filtered_slice = slice_[mask]
        array_clean.append(filtered_slice)
    array_clean = np.array(array_clean, dtype=object)
    return array_clean

def interpolate_markers(marker_ids, marker_dict, marker_instants, eeg_instants):
    """
    Interpolates marker values across EEG samples based on given marker information for the first run,
    were no 'Feedback' is available.

    This function assigns marker values to each EEG sample instant by interpolating
    markers based on the timing of the markers and the EEG sample instants. The
    interpolation follows specific rules for assigning classes to markers and
    applying them to EEG samples.

    Parameters:
    marker_ids (list of lists): A list where each sublist contains a single string
                                representing the marker type (e.g., ['Start_of_Trial_l'],
                                ['Start_of_Trial_r'], ['Reference'], ['Cue'], ['Feedback']).
    marker_dict (dict): A dictionary mapping marker types (strings) to integer values.
    marker_instants (list of floats): A list of time instants (in seconds) when each
                                      marker occurs.
    eeg_instants (list of floats): A list of time instants (in seconds) corresponding to
                                   each EEG sample.

    Returns:
    list of ints: A list of interpolated marker values for each EEG sample instant.
    """

    marker_values = [0]  # Initialize with zero for data before the paradigm starts.
    task_class = 0

    # Generate the marker values based on the given rules
    for marker in marker_ids:
        marker_type = marker[0]
        if marker_type == 'Start_of_Trial_l':
            task_class = 1
        elif marker_type == 'Start_of_Trial_r':
            task_class = 2
        if marker_type in ['Reference', 'Cue', 'Feedback']:
            value = marker_dict[marker_type] + task_class
        else:
            value = marker_dict[marker_type]
        marker_values.append(value)

    # Interpolate marker values for each EEG instant
    marker_interpol = []
    marker_idx = 0
    num_markers = len(marker_instants)

    for time in eeg_instants:
        while marker_idx < num_markers and time >= marker_instants[marker_idx]:
            marker_idx += 1
        marker_interpol.append(marker_values[marker_idx])

    return marker_interpol


def interpolate_markers_first_run(marker_ids, marker_dict, marker_instants, eeg_instants):
    """
    Interpolates marker values across EEG samples based on given marker information.

    This function assigns marker values to each EEG sample instant by interpolating
    markers based on the timing of the markers and the EEG sample instants. The
    interpolation follows specific rules for assigning classes to markers and
    applying them to EEG samples.

    Parameters:
    marker_ids (list of lists): A list where each sublist contains a single string
                                representing the marker type (e.g., ['Start_of_Trial_l'],
                                ['Start_of_Trial_r'], ['Reference'], ['Cue'], ['Feedback']).
    marker_dict (dict): A dictionary mapping marker types (strings) to integer values.
    marker_instants (list of floats): A list of time instants (in seconds) when each
                                      marker occurs.
    eeg_instants (list of floats): A list of time instants (in seconds) corresponding to
                                   each EEG sample.

    Returns:
    list of ints: A list of interpolated marker values for each EEG sample instant.
    """

    marker_values = []
    task_class = 0

    # Generate the marker values based on the given rules
    for marker in marker_ids:
        marker_type = marker[0]
        if marker_type == 'Start_of_Trial_l':
            task_class = 1
        elif marker_type == 'Start_of_Trial_r':
            task_class = 2
        if marker_type in ['Reference', 'Cue', 'Feedback']:
            value = marker_dict[marker_type] + task_class
        else:
            value = marker_dict[marker_type]
        marker_values.append(value)

    # add reference id in marker_values and time in marker_instants
    marker_values_with_fb = []
    marker_instants_with_fb = []
    for index, marker_val in enumerate(marker_values):
        marker_values_with_fb.append(marker_val)
        marker_instants_with_fb.append(marker_instants[index])
        if marker_val == 21: ## cue left
            marker_values_with_fb.append(31)
            marker_instants_with_fb.append(marker_instants[index]+1.25)
        elif marker_val == 22:  # cue right
            marker_values_with_fb.append(32)
            marker_instants_with_fb.append(marker_instants[index]+1.25)

    marker_values_with_fb.insert(0,0)  # Initialize with zero for data before the paradigm starts.

    # Interpolate marker values for each EEG instant
    marker_interpol = []
    marker_idx = 0
    num_markers = len(marker_instants_with_fb)

    for time in eeg_instants:
        while marker_idx < num_markers and time >= marker_instants_with_fb[marker_idx]:
            marker_idx += 1
        marker_interpol.append(marker_values_with_fb[marker_idx])

    return marker_interpol

def find_marker_times(marker_amout, marker_value, marker_interpol, eeg_instants):
    """
    Finds the start and end times of markers within EEG data and assigns classes to them.
    Can be used for Reference and Feedback markers.

    Parameters:
    marker_amount (int): The total number of markers to find.
    marker_value (int): The base value of the marker.
    marker_interpol (list of ints): Interpolated marker values for each EEG sample instant.
    eeg_instants (list of floats): Time instants for each EEG sample.

    Returns:
    ndarray: A 2D array where each row contains the start time, end time, and class
                                (1 for left, 2 for right) of each marker.
    """

    marker_times = np.zeros((marker_amout, 3))
    k = 0
    for index, marker in enumerate(marker_interpol):
        if marker == marker_value+1 and marker_interpol[index-1] != marker_value+1: #Left start
            marker_times[k, 0] = eeg_instants[index]
            marker_times[k, 2] = 1 # class = Left
        elif marker == marker_value+1 and marker_interpol[index+1] != marker_value+1: #Left end
            marker_times[k, 1] = eeg_instants[index]
            k +=1
        if marker == marker_value+2 and marker_interpol[index-1] != marker_value+2: #right start
            marker_times[k, 0] = eeg_instants[index]
            marker_times[k, 2] = 2  # class = Right
        elif marker == marker_value+2 and marker_interpol[index+1] != marker_value+2: #right end
            marker_times[k, 1] = eeg_instants[index]
            k += 1

    return marker_times


