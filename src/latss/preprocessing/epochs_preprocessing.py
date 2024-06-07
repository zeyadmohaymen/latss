from sklearn.base import BaseEstimator, TransformerMixin
from mne.io import Raw, BaseRaw
from mne import Epochs, EpochsArray, events_from_annotations
import numpy as np
from latss.utils.utils import validate_input_type
from typing import Union
import logging
logger = logging.getLogger('epochs-preprocessing')

    
class Cropper(BaseEstimator, TransformerMixin):
    """
    A transformer class for cropping epochs.

    Parameters:
    -----------
    tmin : float, optional
        The start time of the cropping window in seconds. Default is 0.5.
    length : float, optional
        The final length of the epoch after cropping. Default is 3.

    Methods:
    --------
    fit(X, y=None)
        Does nothing. Exists for compatibility.

    transform(epochs: Epochs)
        Crop the input epochs using the specified time window.

    Returns:
    --------
    cropped_epochs : Epochs
        The cropped epochs.
    """

    def __init__(self, tmin=0.5, length=3):
        self.tmin = tmin
        self.length = length

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        tmax = self.tmin + self.length
        logger.info(f"Cropped epochs from {self.tmin}s to {tmax}s")
        return epochs.copy().crop(self.tmin, self.tmax)
    
class EventsEqualizer(BaseEstimator, TransformerMixin):
    """
    A transformer class to equalize event counts in epochs data.

    Parameters:
    -----------
    None

    Returns:
    --------
    epochs : Epochs
        The input epochs data with equalized event counts.
    """
    def __init__(self, method='truncate'):
        self.method = method
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        eq_epochs = epochs.copy().equalize_event_counts(method=self.method)[0]
        print(eq_epochs)
        return eq_epochs
    
class Resampler(BaseEstimator, TransformerMixin):
    """
    A class for resampling epochs data.

    Parameters:
    -----------
    sfreq : int, optional
        The desired sampling frequency for resampling the epochs data.

    Methods:
    --------
    transform(epochs)
        Resample the input epochs data.

    Returns:
    --------
    resampled_epochs : Epochs
        The resampled epochs data.
    """

    def __init__(self, sfreq):
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        return epochs.copy().resample(self.sfreq)
    
class IntraEpochSegmentation(BaseEstimator, TransformerMixin):
    """
    A transformer class for segmenting epochs into smaller windows.

    Parameters:
    -----------
    window_size : float, optional
        The size of each window in seconds. Default is 1 second.
    overlap : float, optional
        The overlap between consecutive windows as a fraction of the window size.
        Default is 0.5 (50% overlap).

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(epochs)
        Transform the epochs by segmenting them into smaller windows.

    Returns:
    --------
    new_epochs : mne.Epochs
        The segmented epochs object.

    """

    def __init__(self, window_size=1, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Union[Epochs, dict]):
        """
        Transform the epochs with intra-epoch segmentation using a sliding window.

        Parameters:
        -----------
        epochs : mne.Epochs | mne.io.Raw
            The epochs or raw object to be segmented.

        Returns:
        --------
        new_data : ndarray
            The segmented data.
        new_events : ndarray
            The segmented events.

        """
        # get input type 
        data_type = validate_input_type(epochs, [Epochs, Raw, EpochsArray, BaseRaw])

        # Copy the epochs object to avoid modifying the original data
        epochs = epochs.copy()

        data = None
        events = None
        if data_type == 'Epochs':
            data = epochs.get_data(copy=False)
            events = epochs.events
        else:
            data = epochs.get_data()
            data = data[np.newaxis, ...]
            events, _ = events_from_annotations(epochs)
        
        # Get the sampling frequency of the epochs data
        sfreq = epochs.info['sfreq']
        
        # Check if the window size is greater than the epoch length
        epoch_length = epochs.tmax - epochs.tmin if data_type == 'Epochs' else epochs.n_times / sfreq
        if self.window_size >= epoch_length:
            return data, events
        
        # Calculate the number of samples in each window
        n_samples = int(self.window_size * sfreq)
        
        # Calculate the number of samples to step between windows
        step_samples = int(n_samples * (1 - self.overlap))
    
        # Get the shape of the data
        n_epochs, n_channels, n_times = data.shape
        
        # Calculate the number of windows per epoch
        n_windows = (n_times - n_samples) // step_samples + 1
        
        # Initialize lists to store the new data and events
        new_data = []
        new_events = []
        
        # Iterate over each epoch
        for epoch_idx in range(n_epochs):
            # Iterate over each window
            for window_idx in range(n_windows):
                # Calculate the start and stop indices of the window
                start = window_idx * step_samples
                stop = start + n_samples
                
                # Append the data and events of the window to the respective lists
                new_data.append(data[epoch_idx, :, start:stop])

                # Check if events is not empty
                if len(events) > 0:
                    new_events.append([events[epoch_idx, 0] + start, 0, events[epoch_idx, 2]])
        
        # Convert the lists to numpy arrays
        new_data = np.array(new_data)
        new_events = np.array(new_events)
        
        # Create a new epochs object with the segmented data and events
        # new_epochs = reconstruct_epochs(epochs, new_data, new_events)

        logger.info(f"Segmented epochs into windows of {self.window_size}s with {self.overlap * 100}% overlap")
        
        # Return the new epochs object
        return new_data, new_events
    
class EpochsDecoder(BaseEstimator, TransformerMixin):
    """
    A transformer class for extracting data from epochs.

    Parameters:
    -----------
    None

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(epochs: Epochs)
        Extract the data from the epochs.

    Returns:
    --------
    data : ndarray
        The extracted data and events from the epochs.
    """

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs):

        if isinstance(epochs, Epochs) or isinstance(epochs, EpochsArray):
            # print("Extracted Data from Epochs")
            return epochs.get_data(copy=False), epochs.events
        
        elif isinstance(epochs, tuple):
            if isinstance(epochs[0], np.ndarray) and isinstance(epochs[1], np.ndarray):
                return epochs
            
        else:
            raise ValueError("Invalid input type. Must be mne.Epochs or (np.ndarray, np.ndarray).")
