"""
This script provides functions to visualize the results of the model training.
"""

from matplotlib import (
    pyplot as plt,
    axes as ax
)
import numpy as np
from utils.training import get_features_and_labels


def visualize_feature_distribution(feature: str, hdf5_path: str, split: str, data_split_path: str,
                                   range: tuple[float, float] | float | None) -> list[ax.Axes]:
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
    :param range: If an instance of `tuple[float, float]` then this is the value of the `range` parameter when creating
            the histogram plot. If an instance of `float` then this is the factor that is multiplied with the IQR to
            determine the `range` parameter. If `None` the `range` parameter will not be set.
    :type range: tuple[float, float] | float | None
    :return: A list containing the diferent instances of `ax-Axes`, each representing a different plot. If the feature
            consists of only one number, the list will only have one element.
    :rtype: list[ax.Axes]
    """
    feature_values, labels = get_features_and_labels(hdf5_path, [feature], split, data_split_path)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    axes = []

    # Iterate over all the columns that make the feature
    for i, feature in enumerate(feature_values.T):
        _, ax = plt.subplots()

        if isinstance(range, float):
            q1 = np.nanpercentile(feature_values, 25)
            q3 = np.nanpercentile(feature_values, 75)
            iqr = q3 - q1

            lower_bound = q1 - range * iqr
            upper_bound = q3 + range * iqr

            range = (lower_bound, upper_bound)

        title = f'Verteilung des Features {feature} {f'({i})' if len(feature_values.T) > 1 else ''}'
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel('Empirische Dichtefunktion')
        ax.hist(feature_values[neg_idx], range=range, density=True, histtype='step', color='blue',
                label='Kein Artefakt')
        ax.hist(feature_values[pos_idx], range=range, density=True, histtype='step', color='orange',
                label='Artefakt')
        ax.legend()
        ax.grid(True)
        axes.append(ax)

    return axes
