"""
This script provides functions to resample and apply high-pass and low-pass filters to EDF files.
"""

import os
import numpy as np
import pyedflib
from scipy.signal import resample_poly, butter, sosfiltfilt
from tqdm import tqdm


def resample_and_filter_edf_file_data(edf_reader: pyedflib.EdfReader, target_fs: int, hp: float, lp: float,
                                      order=5) -> np.ndarray[np.ndarray]:
    """
    Calculates the resampled channel data for each channel in the specified EDF file. After that, it applies
    a high-pass and low-pass filter.

    :param edf_reader: An instance of `pyedflib.EdfReader` representing the EDF file to be resampled.
    :type edf_reader: pyedflib.EdfReader
    :param target_fs: The target sampling frequency to resample the channels to.
    :type target_fs: int
    :param hp: The high-pass filter cutoff frequency in Hz.
    :type hp: float
    :param lp: The low-pass filter cutoff frequency in Hz.
    :type lp: float
    :param order: The order of the Butterworth filter to be applied. Defaults to `5`.
    :type order: int
    :return: A 2D NumPy array where each row corresponds to a resampled and filtered channel.
    :rtype: np.ndarray[np.ndarray]
    :raises ValueError: If the target sampling frequency is higher than any of the original frequencies
    """
    if (edf_reader.getSampleFrequencies() < target_fs).any():
        raise ValueError("Target sampling frequency must be lower than all the original frequencies.")

    altered_channel_data = np.array([
        resample_poly(edf_reader.readSignal(i), up=target_fs, down=channel_freq)
        for i, channel_freq in enumerate(edf_reader.getSampleFrequencies())
    ])

    for channel_data in altered_channel_data:
        sos = butter(N=order, Wn=[hp, lp], btype='bandpass', fs=target_fs, output='sos')
        channel_data = sosfiltfilt(sos, channel_data)

    return altered_channel_data


def resample_and_filter_edf_files(dir_path: str, target_fs: int, hp: float, lp: float, order=5) -> None:
    """
    Resamples all EDF files in the specified directory to the target sampling frequency.

    :param dir_path: The path to the directory containing the EDF files to be resampled.
    :type dir_path: str
    :param target_fs: The target sampling frequency to resample the channels to.
    :type target_fs: int
    :param hp: The high-pass filter cutoff frequency in Hz.
    :type hp: float
    :param lp: The low-pass filter cutoff frequency in Hz.
    :type lp: float
    :param order: The order of the Butterworth filter to be applied. Defaults to `5`.
    :type order: int
    :returns: None
    """
    for root, _, files in os.walk(dir_path):
        if os.path.basename(root) == 'unsorted':
            continue
        for file in tqdm(files, desc=f'Resampling files in {root}'):
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                # Save headers and resampled data from original EDF file
                with pyedflib.EdfReader(file_path) as edf_reader:
                    resampled_data = resample_and_filter_edf_file_data(edf_reader, target_fs, hp=hp, lp=lp, order=order)
                    channel_number = edf_reader.signals_in_file
                    channel_headers = edf_reader.getSignalHeaders()

                # Overwrite original EDF file with resampled data and new headers
                with pyedflib.EdfWriter(file_path, channel_number, file_type=pyedflib.FILETYPE_EDFPLUS) as edf_writer:
                    edf_writer.setSignalHeaders(channel_headers)
                    edf_writer.writeSamples(resampled_data)
                    for i in range(channel_number):
                        edf_writer.setPhysicalMaximum(i, np.ceil(np.max(resampled_data[i])))
                        edf_writer.setPhysicalMinimum(i, np.floor(np.min(resampled_data[i])))
                        edf_writer.setSamplefrequency(i, target_fs)
