import os
from importlib.resources import files
from os import path as op
from pathlib import Path
import urllib.request
from mne import concatenate_epochs
from mne import read_epochs


def _download_file(url, dest_path):
    """
    Downloads a file from a given URL to a destination path.
    
    Args:
        url (str): The URL to download the file from.
        dest_path (str): The local path where the file will be saved.
    """
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded {url} to {dest_path}")
    else:
        print(f"File {dest_path} already exists. Skipping download.")

    return dest_path

def load_data(datasets, destination_dir=None):
    """
    Downloads necessary data files into the specified directory if they do not already exist.
    
    Args:
        datasets (dict): A dictionary containing the filenames as keys and the corresponding FIF download URLs as values. Filenames must end in '-epo.fif'. Must be in the format {'filename-epo.fif': 'url'}.
        destination_dir (str): The directory where the data files will be saved. Defaults to '~/latss/datasets'.
    
    Returns:
        list: A list of concatenated epochs.
    """
    # Set a default destination directory if none is provided
    if destination_dir is None:
        destination_dir = op.join(op.expanduser("~"), "latss", "datasets")

    
    # Create the target directory if it does not exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    epochs = []
    # Download each file and read the epochs
    for filename, url in datasets.items():
        dest_path = os.path.join(destination_dir, filename)
        fname = _download_file(url, dest_path)
        epochs.append(read_epochs(fname))

    # Concatenate the epochs and return them
    return concatenate_epochs(epochs)

def unpack_fifs(fnames: list):
    """
    Unpacks a list of .fif files into a single mne.Epochs object.

    Args:
        fnames (list): A list of .fif file paths. File names must be in the format 'filename-epo.fif'.

    Returns:
        mne.Epochs: A single mne.Epochs object containing the epochs from all the input files.
    """
    epochs = []
    for fname in fnames:
        epochs.append(read_epochs(fname))
    return concatenate_epochs(epochs)

