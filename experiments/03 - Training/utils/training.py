"""
This script provides functions to train and save a Random Forest model for artifact detection.
"""

import numpy as np
from mne_features.feature_extraction import extract_features
import h5py
from tqdm import tqdm


def calculate_features(feature_file: str, data_file: str, features: list[str], sfreq: float = 250.) -> None:
    """
    Calculates the specified list of MNE features for the specified data and stores them in the specified HDF5 file.
    If the features have already been calculated, they will not be calculated again.

    :param feature_file: File path to the hdf5 file containing the features.
    :type feature_file: str
    :param data_file: File path to the hdf5 file containing the data.
    :type data_file: str
    :param features: List of the features from `mne-features` that will be calculated.
    :type features: list[str]
    :param sfreq: The sampling frequency of the EEG data. Defaults to 250.
    :type sfreq: float
    :return: None
    """
    with h5py.File(data_file) as df, h5py.File(feature_file) as ff:
        for session in tqdm(df, desc="Calculating features for sessions"):
            if session not in ff:
                ff.create_group(session)
                for channel in tqdm(df[session], desc=f"Calculating features for channels in session {session}"):
                    if channel not in ff[session]:
                        df[session].create_group(channel)
                    missing_features = [feat for feat in features if not feat in ff[session][channel]]
                    feature_matrix = extract_features(
                        # Add new axis to data because it is required for MNE
                        df[session][channel]['data_windows'][:, np.newaxis, :],
                        sfreq,
                        missing_features
                    )
                    for i, feature in tqdm(
                        enumerate(missing_features),
                        desc=f"Calculating features for channel {channel} in session {session}"
                    ):
                        ff[session][channel].create_dataset(
                            feature,
                            feature_matrix[:][i]
                        )
