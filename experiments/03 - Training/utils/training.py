"""
This script provides functions to train and save a Random Forest model for artifact detection.
"""

from mne_features.univariate import get_univariate_funcs
import h5py
from tqdm import tqdm
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


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
            ff.require_group(session)
            for channel in df[session]:
                ff[session].require_group(channel)
                missing_features = [feat for feat in features if feat not in ff[session][channel]]
                for feature in missing_features:
                    feature_values = univariate_functions[feature](df[session][channel]['data_windows'][:])
                    # Add if statement in case univariate function returns more than one number for each window
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


def get_features_and_labels(hdf5_path: str, features: list[str] | None, set: str,
                            data_split_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the specified features and the labels from the specified HDF5 file. The patients to be considered
    can be specified with the `data_split_file` and the `set` parameter. It is mandatory that all the specified features
    have already been calculated.

    :param hdf5_path: Path to the HDF5 file containing the features.
    :type hdf5_path: str
    :param features: The features to be returned. If `None` then all available features will be used.
    :type features: list[str] | None
    :param set: The set to be considered. Should be one of `train`, `val` and `test`.
    :type set: str
    :param data_split_file: The path to the ymal file containing the data split.
    :type data_split_file: str
    :return: Tuple consisting of the feature matrix and the label array
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    with open(data_split_file) as f:
        data_split = yaml.safe_load(f)
    relevant_patients = data_split[set]

    feature_matrix_list = []
    label_vector_list = []

    with h5py.File(hdf5_path) as f:
        for session in tqdm(f, desc="Extracting features and labels for sessions"):
            if session.split('_')[0] not in relevant_patients:
                continue
            for channel in f[session]:
                label_vector_list.append(f[session][channel]['labels'][:])
                feature_vector_list = []
                # If features is None, add all the calculated features to the list
                if not features:
                    features = [feat for feat in f[session][channel].keys() if not feat == 'labels']
                for feature in features:
                    feature_vector_list.append(f[session][channel][feature][:])
                feature_matrix_list.append(np.column_stack(feature_vector_list))

    return np.vstack(feature_matrix_list), np.concatenate(label_vector_list)


def train_random_forest(clf: RandomForestClassifier, hdf5_path: str, features: list[str],
                        data_split_file: str) -> RandomForestClassifier:
    """
    Trains a random forest classifier based on the specified features provided in the specified HDF5 file and
    returns it.

    :param clf: A `RandomForestClassifier` instance that is to be trained.
    :type clf: RandomForestClassifier
    :param hdf5_path: The path to the HDF5 file containing the features and labels.
    :type hdf5_path: str
    :param features: The features to be considered.
    :type features: list[str]
    :param data_split_file: The path to the ymal file containing the data split.
    :type data_split_file: str
    :return: The trained `RandomForestClassifier` instance.
    :rtype: RandomForestClassifier
    """
    X_train, y_train = get_features_and_labels(hdf5_path, features, 'train', data_split_file)
    return clf.fit(X_train, y_train)


def predict_random_forest(clf: RandomForestClassifier, hdf5_path: str, features: list[str],
                          data_split_file: str, set: str = 'val') -> tuple[np.ndarray, np.ndarray]:
    """
    Uses the provieded random forest to predict labels for the specified features provided in the specified HDF5 file
    and returns the predictions. It is mandatory that the specified features equal those used to train the model.

    :param clf: A `RandomForestClassifier` instance that is to be evaluated.
    :type clf: RandomForestClassifier
    :param hdf5_path: The path to the HDF5 file containing the features and labels.
    :type hdf5_path: str
    :param features: The features to be considered.
    :type features: list[str]
    :param data_split_file: The path to the ymal file containing the data split.
    :type data_split_file: str
    :param set: The set to use for prediction. Should be either `train`, `val` or `test`. `test` should only be used
            for final evaluation. Defaults to `val`.
    :type set: str
    :return: The actual and predicted labels.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    X, y = get_features_and_labels(hdf5_path, features, set, data_split_file)
    return y, clf.predict(X)
