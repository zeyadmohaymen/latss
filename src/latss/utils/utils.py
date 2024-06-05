import mne

def validate_raw(raw: mne.io.Raw, sfreq, epoch_length, window_size):
    """
    Validates data has the correct epoch length.

    Parameters:
    - raw: Raw data to be validated.

    Raises:
    - ValueError: If the epoch length of the raw data is invalid.
    """
    data = raw.get_data()
    exp_length = sfreq * epoch_length if window_size is None else sfreq * window_size

    if data.shape[-1] != exp_length:
        raise ValueError(f"Invalid epoch length: {data.shape[-1] / sfreq} seconds")
    
def validate_epochs(epochs: mne.Epochs, sfreq, length, window_size):
    """
    Validates the epochs have the correct sampling frequency and epoch length.

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