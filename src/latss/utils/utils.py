import mne
import numpy as np

def validate_raw(raw: mne.io.Raw, sfreq, epoch_length, window_size):
    """
    Validates that raw data matches the model requirements.

    Parameters:
    - raw: Raw data to be validated.

    Raises:
    - ValueError: If the epoch length of the raw data is invalid.
    """
    data = raw.get_data()
    exp_length = sfreq * epoch_length if window_size is None else sfreq * window_size

    if data.shape[-1] != exp_length:
        raise ValueError(f"Invalid epoch length: Expected {exp_length}, got {data.shape[-1]}")
    
def validate_epochs(epochs: mne.Epochs, sfreq, length, window_size):
    """
    Validates the epochs match the model requirements.

    Parameters:
    - epochs: Epochs to be validated.

    Raises:
    - ValueError: If the sampling frequency or epoch length of the epochs is invalid.
    """
    epochs_sfreq = epochs.info['sfreq']
    if epochs_sfreq != sfreq:
        raise ValueError(f"Invalid sampling frequency: Expected {sfreq}, got {epochs_sfreq}")
    
    epoch_length = epochs.tmax - epochs.tmin
    exp_length = sfreq * length if window_size is None else sfreq * window_size
    if epoch_length != exp_length:
        raise ValueError(f"Invalid epoch length: Expected {exp_length}, got {epoch_length}")
    
def validate_dict(data_dict: dict, sfreq, length, window_size):
    """
    Validates the EEG data is in the correct format and matches the model requirements.

    Parameters:
    - data_dict: Data dictionary to be validated.
    - sfreq: Sampling frequency of the data.
    - length: Expected epoch length.
    - window_size: Inta-epoch window size.

    Raises:
    - KeyError: If the data dictionary is missing required keys.
    - ValueError: If the data dictionary does not match the model requirements.
    """
    required_keys = ["data", "events"]
    if not all([k in data_dict for k in required_keys]):
        raise KeyError(f"Missing keys: {required_keys}")
    
    data = data_dict["data"]
    events = data_dict["events"]

    if not isinstance(data, np.ndarray) or not isinstance(events, np.ndarray):
        raise ValueError("Invalid data type: Expected numpy arrays")
    
    if data.ndim != 3:
        raise ValueError("Invalid data shape: Expected 3D array (n_epochs, n_channels, n_samples)")
    
    if data.shape[0] != events.shape[0]:
        raise ValueError(f"Data and events shape mismatch. Data: {data.shape[0]}, Events: {events.shape[0]}")
    
    exp_length = sfreq * length if window_size is None else sfreq * window_size
    if data.shape[-1] != exp_length:
        raise ValueError(f"Invalid epoch length: Expected {exp_length}, got {data.shape[-1]}")
