"""
This script provides functions to train and save a Random Forest model for artifact detection.
"""

from mne_features.univariate import get_univariate_funcs
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
    # Get all the functions as a dict, because feature extraction module does not work
    univariate_functions = get_univariate_funcs(sfreq)

    with h5py.File(data_file, 'r') as df, h5py.File(feature_file, 'a') as ff:
        for session in tqdm(df, desc="Calculating features for sessions"):
            if session not in ff:
                ff.create_group(session)
            for channel in df[session]:
                if channel not in ff[session]:
                    ff[session].create_group(channel)
                missing_features = [feat for feat in features if feat not in ff[session][channel]]
                for feature in missing_features:
                    feature_values = univariate_functions[feature](df[session][channel]['data_windows'][:])
                    # Add if statement in case  univariate function returns more than one number for each window
                    if len(feature_values) > len(df[session][channel]['data_windows'][:]):
                        data_window_length = len(df[session][channel]['data_windows'][:])
                        length_coeff = len(feature_values) // data_window_length
                        feature_values = feature_values.reshape(data_window_length, length_coeff)
                    ff[session][channel].create_dataset(
                        feature,
                        data=feature_values
                    )


def delete_features(hdf5_path: str, features: list[str]) -> None:
    """
    Deletes the specified features from all sessions and channels in the specified HDF5 file.

    :param hdf5_path: Path to the HDF5 file.
    :type hdf5_path: str
    :param features: The features to be removed.
    :type features: list[str]
    :return: None
    """
    with h5py.File(hdf5_path, 'a') as f:
        for session in tqdm(f, desc="Deleting features in sessions"):
            for channel in f[session]:
                for feature in features:
                    if feature in f[session][channel]:
                        del f[session][channel][feature]
