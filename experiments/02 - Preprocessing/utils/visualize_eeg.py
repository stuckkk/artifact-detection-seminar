"""
This script provides a functions to visualize EEG channel data.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pyedflib
import numpy as np
import pandas as pd
from scipy.signal import welch


DEFAULT_COLORS = {
    'eyem': 'red',
    'musc': 'blue',
    'elec': 'green',
    'musc_elec': 'yellow',
    'eyem_musc': 'purple'
}


def visualize_eeg_with_artifacts(edf_file_path: str, csv_file_path: str, channel: str, interval_start: float,
                                 interval_stop: float, colors: dict[str, str] = DEFAULT_COLORS,
                                 ylim: tuple[float, float] = None) -> matplotlib.figure:
    """
    Visualizes the EEG channel data of the specified channel in the specified session in the specified time interval.

    :param edf_file_path: Path to the EDF file containing EEG data.
    :type edf_file_path: str
    :param csv_file_path: Path to the CSV file containing artifact annotations.
    :type csv_file_path: str
    :param channel: The EEG channel to visualize.
    :type channel: str
    :param interval_start: The start time of the visualized interval in seconds.
    :type interval_start: float
    :param interval_stop: The stop time of the visualized interval in seconds.
    :type interval_stop: float
    :param colors: A dictionary mapping artifact classes to colors. If provided, it has to contain values for all
                   the classes present. Defaults to `DEFAULT_COLORS`.
    :type colors: dict[str, str]
    :param ylim: Tuple specifying y-axis limits. If None, limits are determined automatically. Defaults to `None`.
    :type ylim: tuple[float, float]
    :returns: The matplotlib figure containing the visualization.
    :rtype: matplotlib.figure
    """
    reader = pyedflib.EdfReader(edf_file_path)

    channel_idx = reader.getSignalLabels().index(channel)
    channel_data = reader.readSignal(channel_idx)
    sampling_frequency = reader.getSampleFrequency(channel_idx)

    reader.close()

    df = pd.read_csv(csv_file_path, comment='#')
    df = df.query(f"channel == '{channel}' and start_time > {interval_start} and stop_time < {interval_stop}")

    # Define lower and upper bound for seconds specified seconds considering frequency
    lower_bound = int(interval_start * sampling_frequency)
    upper_bound = int(interval_stop * sampling_frequency)

    fig, axes = plt.subplots(figsize=(12, 4))
    title_str = f"EEG Signal und Artefaktannotationen {channel} zwischen Sekunden {interval_start} und {interval_stop}"

    axes.set_title(title_str)
    axes.set_xlabel("Sekunde")
    axes.set_ylabel("Amplitude in µV")
    axes.plot(
        np.arange(lower_bound, upper_bound) / sampling_frequency,
        channel_data[lower_bound:upper_bound],
        linewidth=0.8
    )

    if ylim:
        axes.set_ylim(ylim)

    for row in df.itertuples():
        axes.axvspan(row.start_time, row.stop_time, color=colors[row.label], alpha=0.3, label=row.label)

    legend_handles = [Patch(facecolor=colors[label], alpha=0.3, label=label, edgecolor=colors[label])
                      for label in df['label'].unique()]
    axes.legend(handles=legend_handles)

    return fig


def visualize_power_spectrum(edf_file_path: str, channel: str,
                             ylim: tuple[float, float] = (10e-2, 10e2)) -> matplotlib.figure:
    """
    Visualizes the power spectrum of the specified EEG channel in the specified EDF file.

    :param edf_file_path: Path to the EDF file containing EEG data.
    :type edf_file_path: str
    :param channel: The EEG channel to visualize.
    :type channel: str
    :param ylim: Tuple specifying y-axis limits. Defaults to (10e-2, 10e2).
    :type ylim: tuple[float, float]
    :returns: The matplotlib figure containing the power spectrum visualization.
    :rtype: matplotlib.figure
    """
    reader = pyedflib.EdfReader(edf_file_path)

    channel_idx = reader.getSignalLabels().index(channel)
    channel_data = reader.readSignal(channel_idx)
    sampling_frequency = reader.getSampleFrequency(channel_idx)

    reader.close()

    f, pxx = welch(channel_data, fs=sampling_frequency, nperseg=2048)

    fig, axes = plt.subplots()

    axes.set_title(f"Power Spektrum des EEG Signals von Kanal {channel}")
    axes.set_xlabel("Frequenz in Hz")
    axes.set_ylabel("Leistung/Frequenz (V²/Hz)")
    axes.set_ylim(ylim)
    axes.semilogy(f, pxx)

    return fig
