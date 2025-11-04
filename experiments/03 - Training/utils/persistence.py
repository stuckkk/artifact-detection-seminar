"""
This script provides functions to store labels and features in HDF5 files.
"""

from typing import Iterator
import h5py
import numpy as np


def save_session_labels(hdf5_path: str, session: str, label_mapping: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Saves the provided data windows and their corresponding labels into an HDF5 file under the specified session.
    The data windows will be stored in a dataset named `data_windows` and the labels in a dataset named `labels` for
    each channel. These groups will be created under the specified session group in the HDF5 file.

    :param hdf5_path: Path to the HDF5 file where the labels will be stored.
    :type hdf5_path: str
    :param session: The session name under which the data will be stored in the HDF5 file.
    :type session: str
    :param label_mapping: A dictionary mapping each channel to a tuple containing the data windows stored as a 2D
            NumPy Array and their corresponding labels.
    :type label_mapping: dict[str, tuple[np.ndarray, np.ndarray]]
    :return: None
    """
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        session_group = hdf5_file.require_group(session)

        for channel, (data, labels) in label_mapping.items():
            channel_group = session_group.require_group(channel)

            if 'data' in channel_group:
                del channel_group['data_windows']
            if 'labels' in channel_group:
                del channel_group['labels']

            channel_group.create_dataset('data_windows', data=data, compression="gzip", shuffle=True, chunks=True)
            channel_group.create_dataset('labels', data=labels, compression="gzip", shuffle=True, chunks=True)
