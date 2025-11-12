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
    :param hp: The high-pass filter cutoff frequency in Hz. If falsy, then only a lowpass filter is applied.
    :type hp: float
    :param lp: The low-pass filter cutoff frequency in Hz. If falsy, then only a highpass filter ist applied.
    :type lp: float
    :param order: The order of the Butterworth filter to be applied. Defaults to `5`.
    :type order: int
    :return: A 2D NumPy array where each row corresponds to a resampled and filtered channel.
    :rtype: np.ndarray[np.ndarray]
    :raises ValueError: If the target sampling frequency is higher than any of the original frequencies or none of `hp`
            and `lp` is set.
    """
    if (edf_reader.getSampleFrequencies() < target_fs).any():
        raise ValueError("Target sampling frequency must be lower than all the original frequencies.")

    # Set Wn and btype based on hp and lp parameters
    if hp and lp:
        wn, btype = [hp, lp], 'bandpass'
    elif not hp and not lp:
        raise ValueError("Either hp or lp have to be set.")
    elif not hp:
        wn, btype = lp, 'lowpass'
    else:
        wn, btype = hp, 'highpass'

    # Calculate sos filters for each channel based on its original sampling frequency
    sos_filters = np.array([butter(
                                N=order,
                                Wn=wn,
                                btype=btype,
                                fs=channel_freq,
                                output='sos'
                            ) for channel_freq in edf_reader.getSampleFrequencies()])

    # Apply sos filters to each channel
    altered_channel_data = np.array([
        sosfiltfilt(filter, edf_reader.readSignal(i)) for i, filter in enumerate(sos_filters)
    ])

    # Resample each channel to the target sampling frequency
    altered_channel_data = np.array([
        resample_poly(altered_channel_data[i], up=target_fs, down=channel_freq)
        for i, channel_freq in enumerate(edf_reader.getSampleFrequencies())
    ])

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

                # Set changed headers here, because after `setSignalHeaders` call they cannot be changed
                for i, channel_header in enumerate(channel_headers):
                    channel_header['sample_frequency'] = target_fs
                    channel_header['physical_max'] = np.ceil(np.max(resampled_data[i]))
                    channel_header['physical_min'] = np.floor(np.min(resampled_data[i]))

                # Overwrite original EDF file with resampled data and new headers
                with pyedflib.EdfWriter(file_path, channel_number, file_type=pyedflib.FILETYPE_EDFPLUS) as edf_writer:
                    edf_writer.setSignalHeaders(channel_headers)
                    edf_writer.writeSamples(resampled_data)
