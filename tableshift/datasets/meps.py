"""
Utilities for the MEPS dataset.

This is a public data source and no special action is required
to access it.

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift
"""
import pandas as pd
from tableshift.core.features import Feature, FeatureList, cat_dtype
import json

import json
import os
# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory of the current file
current_file_directory = os.path.dirname(current_file_path)

# Open the JSON file
with open(f'{current_file_directory}/meps_feature_types.json', 'r') as file:
   # Use json.load to load the data from the file
   feature_types = json.load(file)

with open(f'{current_file_directory}/meps_feature_demographic_types.json', 'r') as file:
   # Use json.load to load the data from the file
   feature_demographic_types = json.load(file)

type_dict = {'int8': int, 'int16': int, 'int64': int, 'category': cat_dtype}

MEPS_GLOBAL_FEATURES = [Feature(x, type_dict[feature_types[x]]) for x in feature_types]
MEPS_DEMOGRAPHIC_FEATURES = [Feature(x, type_dict[feature_types[x]]) for x in feature_demographic_types]

MEPS_FEATURES = FeatureList(features=[
        Feature('TOTEXP19', int, is_target=True,
                name_extended="Measure of health care utilization at the end of the year"),
] + MEPS_GLOBAL_FEATURES)

MEPS_FEATURES_CAUSAL = FeatureList(features=[
        Feature('TOTEXP19', int, is_target=True,
                name_extended="Measure of health care utilization at the end of the year"),
] + MEPS_DEMOGRAPHIC_FEATURES )

def preprocess_meps(df:pd.DataFrame)->pd.DataFrame:
        df[MEPS_FEATURES.target] = (1.0*(df[MEPS_FEATURES.target] > 3)).astype(int)
        return df