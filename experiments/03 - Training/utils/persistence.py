"""
This script provides functions to store labels and features in HDF5 files.
"""

import h5py
import numpy as np
from tqdm import tqdm


def save_session_labels(hdf5_path: str, session: str,
                        label_mapping: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Saves the provided data windows and their corresponding labels into an HDF5 file under the corresponding session
    group. The data windows will be stored in a dataset named `data_windows` and the labels in a dataset named `labels`
    for each channel. These datasets will be created under the set group in the HDF5 file.

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

        for channel, (data, labels) in tqdm(label_mapping.items(), desc="Saving labels"):
            channel_group = session_group.require_group(channel)

            if 'data' in channel_group:
                del channel_group['data_windows']
            if 'labels' in channel_group:
                del channel_group['labels']

            channel_group.create_dataset('data_windows', data=data, compression="gzip", shuffle=True, chunks=True)
            channel_group.create_dataset('labels', data=labels, compression="gzip", shuffle=True, chunks=True)


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
