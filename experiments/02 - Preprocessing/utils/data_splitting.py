"""
This script provides functions to split the data according to Aymane's data split YAML file. It also maps the
human names back to the original code-names used in the files.
"""

import pandas as pd
import yaml
import os
import shutil
from tqdm import tqdm


def get_mapped_names(data_split_file: str, name_mappings_file: str) -> dict[str, list[str]]:
    """
    Reads the data split YAML file and maps the human names to the original names used in the files.

    :param data_split_file: Path to the YAML file containing the data split.
    :type data_split_file: str
    :param name_mappings_file: Path to the CSV file containing the name mappings.
    :type name_mappings_file: str
    :returns: Dictionary with mapped names for each data split.
    :rtype: dict[str, list[str]]
    """
    with open(data_split_file, 'r') as f:
        data_split = yaml.safe_load(f)

    df_name_mapping = pd.read_csv(name_mappings_file).set_index('New human-readable name')

    for key in data_split:
        data_split[key] = [df_name_mapping.loc[name]['Original code-name'] for name in data_split[key]]

    return data_split


def sort_files_by_set(split: dict[str, list[str]], dir_path: str) -> None:
    """
    Sorts the files in seperate folders according to the provided data split. All files that are not included in any
    set will be moved to a folder named `unsorted`.

    :param split: Dictionary with mapped names for each data split.
    :type split: dict[str, list[str]]
    :param dir_path: Path to the directory containing the files to be sorted.
    :type dir_path: str
    :returns: None
    """
    for root, _, files in os.walk(dir_path):
        for file in tqdm(files, desc=f'Sorting files in {root}'):
            code_name = file.split('_')[0]
            destination_folder = 'unsorted'
            for key in split:
                if code_name in split[key]:
                    destination_folder = key
                    break
            dest_dir_path = os.path.join(dir_path, destination_folder)
            os.makedirs(dest_dir_path, exist_ok=True)
            dest_file_path = os.path.join(dest_dir_path, file)
            shutil.move(os.path.join(root, file), dest_file_path)
