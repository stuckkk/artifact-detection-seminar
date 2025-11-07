"""
This script provides functions to store labels and features in HDF5 files.
"""

import h5py
import numpy as np
from tqdm import tqdm


def save_set_labels(hdf5_path: str, label_mapping: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Saves the provided data windows and their corresponding labels into an hdf5 file under the corresponding set group.
    The data windows will be stored in a dataset named `data_windows` and the labels in a dataset named `labels` for
    each channel. These datasets will be created under the set group in the HDF5 file.

    :param hdf5_path: Path to the HDF5 file where the labels will be stored.
    :type hdf5_path: str
    :param label_mapping: A dictionary mapping each set to a tuple containing the data windows stored as a 2D
            NumPy Array and their corresponding labels.
    :type label_mapping: dict[str, tuple[np.ndarray, np.ndarray]]
    :return: None
    """
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        for set, (data, labels) in tqdm(label_mapping.items(), desc="Saving labels"):
            set_group = hdf5_file.require_group(set)

            if 'data' in set_group:
                del set_group['data_windows']
            if 'labels' in set_group:
                del set_group['labels']

            set_group.create_dataset('data_windows', data=data, compression="gzip", shuffle=True, chunks=True)
            set_group.create_dataset('labels', data=labels, compression="gzip", shuffle=True, chunks=True)


def copy_labels(src_path: str, dest_path: str) -> None:
    """
    Copies all the labels from the specified source HDF5 file to the specified destination HDF5 file. The source file's
    relative structure is maintained. It is expected taht the source HDF5 file has groups for each session and subgroups
    for each channel.

    :param src_path: Path to the source HDF5 file.
    :type hdf5_path: str
    :param src_path: Path to the destination HDF5 file.
    :type hdf5_path: str
    :return: None
    """
    with h5py.File(src_path, 'r') as sf, h5py.File(dest_path, 'a') as df:
        for session in tqdm(sf, desc="Copying labels in sessions"):
            if session not in df:
                df.create_group(session)
            for channel in sf[session]:
                if channel not in df[session]:
                    df[session].create_group(channel)
                df[session][channel].create_dataset('labels', data=sf[session][channel]['labels'][:])
