"""
The function provided in this script maps the human names from Aymane's data split back to the original names used in
the files.
"""

import pandas as pd
import yaml

data_split_file = '../../../EAD/code/preprocessing/data_split.yaml'
name_mappings_file = '../../../EAD/code/molding/name_mappings.csv'


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