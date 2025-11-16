"""
This script provides functions to evaluate a model's performance with the evaluation metric Aymane used.
"""

import numpy as np
import h5py
import yaml
import os
from utils.labeling import load_artifact_annotations


def get_artifact_predictions(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    This functions calculates the parts of the labels where the model predicted an artifact. Consecutive windows with
    positive labels are considered one artifact.


    :param labels: An array containing the labels for each window.
    :type labels: np.ndarray
    :return: List containing the start and end indices of the artifact predictions.
    :rtype: list[tuple[int, int]]
    """
    padded_labels = np.concatenate(([0], labels, [0]))
    diffs = np.diff(padded_labels)

    (starts,) = np.where(diffs == 1)
    (ends,) = np.where(diffs == -1)

    return list(zip(starts, ends))


def get_tp_fp_fn_for_channel(overlap_treshold: float, labels: np.ndarray,
                             artifact_annotations: list[tuple[float, float]]) -> tuple[int, int, int]:
    """
    Calculates the true positive, false positive and false negative values for the specified labels and
    artifact annotations. A prediction is considered a true positive if there exists an artifact annotation, where the
    Intersection over Union is greater then or equal to `overlap_threshold`. If such an artifact annotation cannot be
    found then that prediction is considered a false positive.
    If for a given artifact annotation there is no prediction that an IoU of at least the specified threshold that is
    counted as a false negative.

    Consecutive windows with `true` or `false` labels are counted as only one artifact.

    This function assumes that the labels represent one second windows.

    :param overlap_threshold: The portion of an artifact annotation's duration that need to be covered for it to
            be considered a true positive
    :type overlap_threshold: float
    :param labels: An array containing the predicted labels for each window.
    :type labels: np.ndarray
    :param artifact_annotations: A list containing tuples that mark the start and end second of an artifact annotation.
    :type artifact_annotations: list[tuple[float, float]]
    :return: Tuple consisting of the tp, fp and fn values.
    :rtype: tuple[int, int, int]
    """
    tp, fp = 0, 0
    artifact_predictions = get_artifact_predictions(labels)

    # This set stores the indices of the matched artifats
    matched_artifacts = set()

    for pred_start, pred_end in artifact_predictions:
        match_found = False
        pred_length = pred_end - pred_start
        for i, (annot_start, annot_end) in enumerate(artifact_annotations):
            intersection_start = max(pred_start, annot_start)
            intersection_end = min(pred_end, annot_end)
            intersection_length = intersection_end - intersection_start

            # If overlap_length is negative then there is no overlap at all
            if intersection_length <= 0:
                continue

            annot_length = annot_end - annot_start
            union_length = annot_length + pred_length - intersection_length
            iou = intersection_length / union_length

            if iou >= overlap_treshold:
                match_found = True
                matched_artifacts.add(i)

        if match_found:
            tp += 1
        else:
            fp += 1

    # All artifacts that were not found are considered false negative
    fn = len(artifact_annotations) - len(matched_artifacts)

    return tp, fp, fn


def get_iou_for_set(overlap_treshold: float, labels: np.ndarray, data_split_path: str, split: str,
                    hdf5_path: str) -> tuple[float, float, float]:
    """
    Calculates precision, recall and f1-score, where true positives, false positives and false negatives are determined
    by whether or not for a given prediction there exists an artifact with an IoU of at least `overlap_treshold`. For
    further information, please refer to the documentatio of `get_tp_fp_fn_for_channel`.

    :param overlap_threshold: The portion of an artifact annotation's duration that need to be covered for it to
            be considered a true positive
    :param labels: An array containing the predicted labels for each window.
    :type labels: np.ndarray
    :param data_split_path: The path to the yaml file containing the data splits.
    :type data_split_path: str
    :param split: The data split that the labels refer to.
    :param split: str
    :param hdf5_path: Path to the HDF5 file containing the features and labels.
    :type hdf5_path: str
    :return: Tuple consisting of the precision, recall and f1-score values.
    :rtype: tuple[float, float, float]
    """
    if overlap_treshold < 0 or overlap_treshold > 1:
        raise ValueError("Parameter overlap_threshold must be in [0, 1]")

    with open(data_split_path) as f:
        data_split = yaml.safe_load(f)
    relevant_patients = data_split[split]

    # Define base path to TUAR folder, where the artifact annotation CSV files are located
    tuar_path = f'../../../../tuar_processed/{split}'

    tp, fp, fn = 0, 0, 0
    labels_offset = 0

    with h5py.File(hdf5_path) as f:
        for session in f:
            if session.split('_')[0] not in relevant_patients:
                continue
            artifact_annotations = load_artifact_annotations(os.path.join(tuar_path, session) + '.csv')
            for channel in f[session]:
                channel_length = len(f[session][channel]['labels'])
                tp_channel, fp_channel, fn_channel = get_tp_fp_fn_for_channel(
                    overlap_treshold,
                    labels[labels_offset:labels_offset + channel_length],
                    artifact_annotations.get(channel, [])
                )
                tp += tp_channel
                fp += fp_channel
                fn += fn_channel
                labels_offset += channel_length

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    denominator = precision + recall

    if np.isclose(denominator, 0.):
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
