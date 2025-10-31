"""
This script provides functions to transform the EDF files to the montages in which the annotations were created.
Parts of this code have been copied from [Aymane's EAD repository](https://github.com/hemmouda/EAD/tree/main).
"""

import re
import os
import shutil
import pyedflib
import numpy as np


def compute_new_channel(edf_reader: pyedflib.EdfReader,
                        signal_names: tuple[str, str]) -> tuple[dict[str, str], list[float]]:
    """
    Computes a new channel as the difference between two existing signals. If only one signal is provided the new
    channel will have the same values as the provided signal.

    :param edf_reader: The EdfReader to read from.
    :type edf_reader: pyedflib.EdfReader
    :param signal_names: A tuple containing the names of the two signals to be used for the computation.
    :type signal_names: tuple[str, str]
    :returns: A dictionary containing the headers for the new channel.
    :rtype: tuple[dict[str, str], list[float]]
    """
    # Regex pattern to extract name of electrode
    reg = r"\s(.*?)-"
    ch1_data = edf_reader.readSignal(edf_reader.getSignalLabels().index(signal_names[0]))
    # If only one signal is provided, adjust data and name of new channel accordingly
    if signal_names[1]:
        ch2_data = edf_reader.readSignal(edf_reader.getSignalLabels().index(signal_names[1]))
        new_channel_data = ch1_data - ch2_data
        new_channel_name = f"{re.search(reg, signal_names[0]).group(1)}-{re.search(reg, signal_names[1]).group(1)}"
    else:
        new_channel_data = ch1_data
        new_channel_name = re.search(reg, signal_names[0]).group(1)
    new_headers = edf_reader.getSignalHeader(edf_reader.getSignalLabels().index(signal_names[0]))
    new_headers["label"] = new_channel_name
    # Round because EDF header has place for eight digits only
    new_headers["physical_min"] = np.round(np.min(new_channel_data), 2)
    new_headers["physical_max"] = np.round(np.max(new_channel_data), 2)
    return new_headers, new_channel_data


def extract_montage_channels(montage_file_path: str) -> list[tuple[str, str]]:
    """
    Extracts the tuples of signals that are necessary to compute the channels of a montage from the montage's info file.
    If one channel consists of only one signal, then the second entry of the corresponding tuple will be left empty.

    :param montage_file_path: Path to the montage's info file.
    :returns: A list of pairs that contain the channels that have to be subtracted.
    """
    result = []
    with open(montage_file_path, 'r') as f:
        for line in f:
            # Only include lines that describe one of the montage's channels
            if line.startswith('montage ='):
                channel_formula = line.split(':')[1]
                channels = channel_formula.split('--')
                # If the channel consists of only one signal, add an empty string for the second signal
                if len(channels) == 1:
                    channels.append("")
                result.append((channels[0].strip(), channels[1].strip()))
    return result


def transform_edf_file(edf_file_path: str, montage: list[tuple[str, str]]) -> None:
    """
    Transforms the given EDF file to contain the channels specified in the montage list.

    :param edf_file_path: Path to the EDF file.
    :type edf_file_path: str
    :param montage: A list containing the pairs of signals that form one channel.
    :type montage: list[tuple[str, str]]
    """
    headers, signals = [], []
    edf_reader = pyedflib.EdfReader(edf_file_path)

    for signal_pair in montage:
        header, signal = compute_new_channel(edf_reader=edf_reader, signal_names=signal_pair)
        headers.append(header)
        signals.append(signal)

    edf_reader.close()

    # Remove old version of EDF file and replace it with new one
    os.remove(edf_file_path)

    edf_writer = pyedflib.EdfWriter(
        edf_file_path, len(headers), file_type=pyedflib.FILETYPE_EDFPLUS
    )
    edf_writer.setSignalHeaders(headers)
    edf_writer.writeSamples(signals)
    edf_writer.close()


def convert_and_store_edf_files(input_path: str, output_path: str) -> None:
    """
    Converts the EDF files to the montages in which the artifact annotations were created and stores them in the output
    folder. This function does not alter any file in the input folder

    :param input_path: Path to the `v3.0.1` folder of the TUAR dataset. It must contain the `edf` and the `DOCS` folder.
    :type input_path: str
    :param output_path: Path to the output folder where the converted EDF files will be stored.
    :type output_path: str
    """
    # Copy files from input to output folder
    shutil.copytree(os.path.join(input_path, 'edf'), output_path)

    for root, dirs, files in os.walk(output_path):
        # If there are no files in the directory, continue with the next one
        if not files:
            continue
        montage_name = os.path.basename(root)
        montage_file_path = os.path.join(input_path, 'DOCS', f'{montage_name}_montage.txt')
        montage_channels = extract_montage_channels(montage_file_path)
        for file in files:
            if file.endswith('.edf'):
                edf_file_path = os.path.join(root, file)
                transform_edf_file(edf_file_path=edf_file_path, montage=montage_channels)
