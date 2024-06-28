from mne import pick_types, Epochs, events_from_annotations
from mne.io import Raw
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.base import BaseEstimator, TransformerMixin

class FilterRaw(BaseEstimator, TransformerMixin):
    """
    A transformer class for applying bandpass filter to raw data.

    Parameters:
    -----------
    l_freq : float, optional
        The lower frequency of the bandpass filter. Default is 8.0 Hz.
    h_freq : float, optional
        The higher frequency of the bandpass filter. Default is 30.0 Hz.

    Methods:
    transform(raw)
        Transform the raw data by applying bandpass filter.

    Returns:
    --------
    new_raw : mne.io.Raw
        The filtered raw data.
    """

    def __init__(self, l_freq=8.0, h_freq=30.0):
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        # print("Applying bandpass filter to raw data...")
        new_raw = raw.copy().filter(self.l_freq, self.h_freq, fir_design='firwin')
        return new_raw
    
class RemoveArtifacts(BaseEstimator, TransformerMixin):
    """
    A transformer class for automated removal of artifacts from raw data using ICA.

    Methods:
    --------
    transform(raw)
        Transform the raw data by removing artifacts.

    Returns:
    --------
    new_raw : mne.io.Raw
        The raw data with artifacts removed.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components


    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        """
        Transform the raw data by removing artifacts.

        Parameters:
        -----------
        raw : mne.io.Raw
            The raw data to be cleaned.

        Returns:
        --------
        new_raw : mne.io.Raw
            The raw data with artifacts removed.
        """
        # Apply common average reference
        raw = raw.copy().set_eeg_reference('average')

        # Filter from 1-100 Hz
        sfreq = raw.info['sfreq']
        hfreq = 100 if sfreq > 200 else None
        raw = raw.filter(1, hfreq, fir_design='firwin')

        # Create ICA object
        ica = ICA(n_components=self.n_components, random_state=97, max_iter="auto", method="infomax", fit_params=dict(extended=True)) 
        ica.fit(raw)

        # Extract labels
        ica_labels = label_components(raw, ica, method='iclabel')
        labels = ica_labels['labels']

        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        ]

        # Reconstruct clean raw data
        new_raw = ica.apply(raw, exclude=exclude_idx)
        return new_raw
    
class SelectChannels(BaseEstimator, TransformerMixin):
    """
    A transformer class for selecting specific channels from raw data.

    Parameters:
    -----------
    channels : list of str
        A list of channel names to include in the raw data.

    Methods:
    --------
    transform(raw)
        Transform the raw data by selecting specific channels.

    Returns:
    --------
    new_raw : mne.io.Raw
        The raw data with specific channels selected.
    """

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        selected_channels = pick_types(raw.info, include=self.channels, exclude="bads")
        new_raw = raw.copy().pick_channels(selected_channels)
        return new_raw
    
class Epochify(BaseEstimator, TransformerMixin):
    """
    A transformer class for segmenting raw data into epochs.

    Parameters:
    -----------
    event_ids : dict
        A dictionary mapping event names to event IDs.
    tmin : float, optional
        The start time of each epoch in seconds. Default is -1.
    length : float, optional
        The desired length of each epoch after tmin in seconds. Default is 3.
    baseline : tuple of float, optional
        The time interval to use for baseline correction. Default is (-0.5, 0).

    Methods:
    --------
    transform(raw)
        Transform the raw data by segmenting it into epochs.

    Returns:
    --------
    epochs : mne.Epochs
        The segmented epochs.
    """

    def __init__(self, extract_events=True, event_id={'0': 0, '1': 1}, channels=None, tmin=0.5, length=3, baseline=(-0.5, 0)):
        self.extract_events = extract_events
        self.event_id = event_id
        self.channels = channels
        self.tmin = tmin
        self.length = length
        self.baseline = baseline

    def _select_channels(self, raw):
        if self.channels:
            selected_channels = pick_types(raw.info, include=self.channels, exclude="bads")
            return selected_channels
        return None
    
    def _crop_to_tmin(self, epochs: Epochs):
        return epochs.crop(tmin=self.tmin, tmax=self.tmin + self.length, include_tmax=False)

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        """
        Transform the raw data by segmenting it into epochs.
        Baseline correction is applied to each epoch.

        Parameters:
        -----------
        raw : mne.io.Raw
            The raw data to be segmented.

        Returns:
        --------
        epochs : mne.Epochs
            The segmented epochs.
        """
        # Select specified channels
        selected_channels = self._select_channels(raw)

        events = None
        if self.extract_events:
            events, _ = events_from_annotations(raw, event_id=self.event_id)

        # Segment raw data into epochs while applying baseline correction
        if self.baseline:
            tmin = self.tmin if self.tmin < self.baseline[0] else self.baseline[0]
        else:
            tmin = self.tmin
        tmax = self.tmin + self.length
        epochs = Epochs(raw, events=events, event_id=self.event_id, tmin=tmin, tmax=tmax, picks=selected_channels, baseline=self.baseline, preload=True)

        # Crop epochs to specified time window
        epochs = self._crop_to_tmin(epochs)

        return epochs