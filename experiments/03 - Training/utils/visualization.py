"""
This script provides functions to visualize the results of the model training.
"""

from matplotlib import (
    pyplot as plt,
    axes as ax,
    figure as fig
)
import numpy as np
from utils.training import get_features_and_labels


def visualize_feature_distribution(feature: str, hdf5_path: str, split: str, data_split_path: str,
                                   range_config: tuple[float, float] | float | None = 2.5) -> tuple[list[fig.Figure], list[ax.Axes]]:
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
