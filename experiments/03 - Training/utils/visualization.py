"""
This script provides functions to visualize the results of the model training.
"""

from matplotlib import (
    pyplot as plt,
    axes as ax,
    figure as fig,
    patches as pat
)
import numpy as np
import pandas as pd
from utils.training import get_features_and_labels


def visualize_feature_distribution(feature: str, hdf5_path: str, split: str, data_split_path: str,
                                   range_config: tuple[float, float] | float | None = 2.5
                                   ) -> tuple[list[fig.Figure], list[ax.Axes]]:
    """
    This function visualizes the distriution of the specified `feature` in the specified `split`. The `range` can be
    used to configure how outliers will be shown in the plot.

    :param feature: The feature to be visualized.
    :type feature: str
    :param hdf5_path: The path to the HDF5 file containing the features.
    :type hdf5_path: str
    :param split: The split to use when visualizing the feature disribution.
    :type split: str
    :param data_split_path: The path to the file containing the data split.
    :type data_split_path: str
    :param range_config: If an instance of `tuple[float, float]` then this is the value of the `range_config` parameter
            when creating the histogram plot. If an instance of `float` then this is the factor that is multiplied with
            the IQR to determine the `range_config` parameter. If `None` the `range_config` parameter will not be set.
            Defaults to `2.5`.
    :type range_config: tuple[float, float] | float | None
    :return: A tuple containing two lists with the instances of `fig.Figure` and `ax.Axes`, each representing a
            different plot. If the feature consists of only one number, the lists will only have one element each.
    :rtype: tuple[list[fig.Figure], list[ax.Axes]]
    """
    feature_values, labels = get_features_and_labels(hdf5_path, [feature], split, data_split_path)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    axes = []
    figs = []

    # Iterate over all the columns that make the feature
    for i, feature_value in enumerate(feature_values.T):
        fig, ax = plt.subplots()

        if isinstance(range_config, float):
            q1 = np.nanpercentile(feature_value, 25)
            q3 = np.nanpercentile(feature_value, 75)
            iqr = q3 - q1

            lower_bound = q1 - range_config * iqr
            upper_bound = q3 + range_config * iqr

            range = (lower_bound, upper_bound)
        elif isinstance(range_config, tuple):
            range = range_config
        else:
            range = None

        title = f'Verteilung des Features {feature} {f'({i+1})' if feature_values.shape[1] > 1 else ''}'
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel('Empirische Dichtefunktion')
        ax.hist(feature_value[neg_idx], range=range, density=True, histtype='step', color='blue',
                label='Kein Artefakt')
        ax.hist(feature_value[pos_idx], range=range, density=True, histtype='step', color='orange',
                label='Artefakt')
        ax.legend()
        ax.grid(True)
        axes.append(ax)
        figs.append(fig)

    return figs, axes


def visualize_model_predictions(axes: ax.Axes, channel_data: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                                start: int, stop: int, csv_file: str | None = None, channel: str | None = None,
                                channel_freq: float = 250., fontsize: int = 20) -> ax.Axes:
    """
    This function visualizes the ground truth annotations and the predictions for the specified channel data between
    the specified `start` and `stop`.

    :param axes: An instance of `Axes` to plot on.
    :type axes: ax.Axes
    :param channel_data: Array containing the data to be visualized.
    :type channel_data: np.ndarray
    :param y_true: Array containing the ground truth annotations.
    :type y_true: np.ndarray
    :param y_pred: Array containing the predictions.
    :type y_pred: np.ndarray
    :param start: Start of the interval that is to be visualized in seconds.
    :type start: int
    :param stop: Stop of the interval that is to be visualized in seconds.
    :type stop: int
    :param csv_file: The path to the csv file containing the original artifact durations. If not `None` the original
            artifact duration before dividing in windows will be plotted. Defaults to `None`.
    :type csv_file: str | None
    :param channel: The channel to display for the original artifact duration. Defaults to `None`.
    :type channel: str | None
    :param channel_freq: Sampling frequency of the data in `channel_data`. Defaults to `250.0`.
    :type channel_freq: float
    :param fontsize: The fontsize used for labels, legend, title etc. Defaults to `20`.
    :type fontsize: int
    :return: The instance of `ax.Axes` that contains the plot.
    :rtype: ax.Axes
    """

    lower_bound = int(start * channel_freq)
    upper_bound = int(stop * channel_freq)

    current_data = channel_data[lower_bound:upper_bound]

    axes.plot(
        np.arange(lower_bound, upper_bound) / channel_freq,
        current_data,
        linewidth=0.8
    )

    annotation_ranges = [(i, 1) for i in range(start, stop) if y_true[i] == 1]
    prediction_ranges = [(i, 1) for i in range(start, stop) if y_pred[i] == 1]

    min_value = np.min(current_data)
    max_value = np.max(current_data)
    values_range = max_value - min_value

    bar_height = values_range * 0.08
    bar_gap = values_range * 0.02

    pred_bar_start = min_value - bar_height - bar_gap
    gt_windows_bar_start = pred_bar_start - bar_height - bar_gap
    gt_csv_bar_start = gt_windows_bar_start - bar_height - bar_gap

    axes.broken_barh(prediction_ranges, (pred_bar_start, bar_height), color='red', alpha=0.3)
    axes.broken_barh(annotation_ranges, (gt_windows_bar_start, bar_height), color='green', alpha=0.3)

    patches = [
        pat.Patch(facecolor='red', alpha=0.3, label='Prediction windows', edgecolor='red'),
        pat.Patch(facecolor='green', alpha=0.3, label='Ground truth windows', edgecolor='green')
    ]

    if csv_file is not None:
        df = pd.read_csv(csv_file, comment='#')
        df = df[(df['channel'] == channel) & (df['start_time'] < stop) & (df['stop_time'] > start)]
        df['duration'] = df['stop_time'] - df['start_time']
        df['max_duration'] = stop - df['start_time']
        ranges = [
            (max(row['start_time'], start), min(row['duration'], row['max_duration'], stop - start))
            for _, row in df.iterrows()
        ]
        axes.broken_barh(ranges, (gt_csv_bar_start, bar_height), color='blue', alpha=0.3)
        patches.append(
            pat.Patch(facecolor='blue', alpha=0.3, label='Ground truth intervals', edgecolor='blue')
        )

    axes.legend(handles=patches, loc='upper right', fontsize=fontsize)
    axes.set_ylabel(r"Amplitude in $\mu V$", fontsize=fontsize)
    axes.set_xlabel("Time in seconds", fontsize=fontsize)
    axes.set_title(
        f"Predictions and ground truth of channel {channel} between seconds {start} and {stop}", fontsize=fontsize
    )
    axes.tick_params(labelsize=fontsize)

    return axes
