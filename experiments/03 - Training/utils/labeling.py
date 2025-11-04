"""
This script provides functions to label EDF file data to make it suitable for training.
"""


from typing import Iterator
import pyedflib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def load_artifact_annotations(csv_path: str) -> dict[str, list[tuple[float, float]]]:
    """
    Loads artfiact annotation from the specified CSV file. The artifact annotations are returned as
    a dictionary mapping each channel to a list of tuples, where each tuple contains the start and stop time.

    :param csv_path: Path to the CSV file containing artifact annotations.
    :type csv_path: str
    :return: A dictionary mapping each channel to a list of tuples with start and stop times of artifacts.
    :rtype: dict[str, list[tuple[float, float]]]
    """
    artifacts = {}

    df = pd.read_csv(csv_path, comment='#')
    for row in df.itertuples(index=False):
        channel = row.channel
        if channel not in artifacts:
            artifacts[channel] = []
        artifacts[channel].append((row.start_time, row.stop_time))

    return artifacts


def split_edf_file_data(edf_reader: pyedflib.EdfReader, window_size_sec: int,
                        overlap_sec: int) -> dict[str, np.ndarray]:
    """
    Splits the channel data of the specified EDF file into smaller windows.

    :param edf_reader: An instance of `pyedflib.EdfReader` representing the EDF file to be split.
    :type edf_reader: pyedflib.EdfReader
    :param window_size_sec: The size of each window in seconds.
    :type window_size_sec: int
    :param overlap_sec: The overlap between consecutive windows in seconds.
    :type overlap_sec: int
    :return: A dictionary mapping each channel to a 2D NumPy array of windows, containing the windows in its rows.
    :rtype: dict[str, np.ndarray]
    """
    result = {}

    for label in edf_reader.getSignalLabels():
        channel_idx = edf_reader.getSignalLabels().index(label)
        channel_freq = edf_reader.getSampleFrequency(channel_idx)
        channel_data = edf_reader.readSignal(channel_idx)

        window_width = int(channel_freq * window_size_sec)
        step_size = int(channel_freq * (window_size_sec - overlap_sec))

        if step_size <= 0:
            raise ValueError("Overlap must be smaller than window size.")

        start_index = 0
        windows = []

        # Only whole windows are considered
        while start_index + window_width <= len(channel_data):
            end_index = start_index + window_width
            windows.append(channel_data[start_index:end_index])
            start_index += step_size

        result[label] = np.array(windows)

    return result


def generate_labels(data_windows: dict[str, np.ndarray], artifact_annotations: dict[str, list[tuple[float, float]]],
                    overlap_threshold: float, window_size_sec: float, window_overlap: float) -> dict[str, np.ndarray]:
    """
    Generates labels for the given data windows based on the specified artifact annotations and overlap threshold.

    :param data_windows: A dictionary mapping each channel to a 2D NumPy array of windows, containing the windows in
            its rows.
    :type data_windows: np.ndarray
    :param artifact_annotations: A dictionary mapping each channel to a list of tuples with start and stop times of
            artifacts.
    :type artifact_annotations: dict[str, list[tuple[float, float]]]
    :param overlap_threshold: The minimum overlap (as a fraction of the window size) required to label a window as
            containing an artifact.
    :type overlap_threshold: float
    :param window_size_sec: The size of each window in seconds.
    :type window_size_sec: float
    :param overlap: The overlap between consecutive windows in seconds.
    :type overlap: float
    :return: A dictionary mapping each channel to a NumPy array of labels (1 for artifact, 0 for no artifact) for each
            window.
    :rtype: dict[str, np.ndarray]
    """
    if not (0.0 <= overlap_threshold <= 1.0):
        raise ValueError("Overlap threshold must be between 0.0 and 1.0.")

    result = {}

    for channel, windows in data_windows.items():
        labels = np.zeros(windows.shape[0], dtype=int)

        # If the channel has no artifact annotations, all labels remain 0
        if channel not in artifact_annotations:
            result[channel] = labels
            continue

        step_size = window_size_sec - window_overlap

        for artifact_start, artifact_stop in artifact_annotations[channel]:
            for window_idx in range(windows.shape[0]):
                window_start = window_idx * step_size
                window_stop = window_start + window_size_sec

                overlap_start = max(window_start, artifact_start)
                overlap_stop = min(window_stop, artifact_stop)
                overlap_duration = max(0.0, overlap_stop - overlap_start)

                if overlap_duration / window_size_sec >= overlap_threshold:
                    labels[window_idx] = 1

        result[channel] = labels

    return result


def label_all_files(dir_path: str, window_size_sec: float, window_overlap: float,
                    overlap_treshold: float) -> Iterator[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Labels all EDF files in the specified directory according to the specified parameters.

    :param dir_path: Path to the directory containing EDF files and their corresponding artifact annotation CSV files.
    :type dir_path: str
    :param window_size_sec: The size of each window in seconds.
    :type window_size_sec: float
    :param window_overlap: The overlap between consecutive windows in seconds.
    :type window_overlap: float
    :param overlap_treshold: The minimum overlap (as a fraction of the window size) required to label a window as
            containing an artifact.
    :type overlap_treshold: float
    :return: A dictionary mapping each session name to another dictionary that maps each channel to a tuple
            containing the data windows stored as a 2D NumPy Array and their corresponding labels.
    :rtype: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    """
    for root, _, files in os.walk(dir_path):
        if os.path.basename(root) == 'unsorted':
            continue

        for file in tqdm(files, desc=f"Labeling files in {root}"):
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                session_name = os.path.splitext(os.path.basename(file_path))[0]
                csv_path = os.path.join(root, f"{session_name}.csv")

                with pyedflib.EdfReader(file_path) as edf_reader:
                    data_windows = split_edf_file_data(edf_reader, window_size_sec, window_overlap)
                    artifact_annotations = load_artifact_annotations(csv_path)
                    labels = generate_labels(data_windows, artifact_annotations,
                                             overlap_treshold, window_size_sec, window_overlap)

                yield session_name, {
                    channel: (data_windows[channel], labels[channel])
                    for channel in data_windows.keys()
                }
