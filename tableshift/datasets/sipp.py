"""
Utilities for the SIPP dataset.

This is a public data source and no special action is required
to access it.

For more information on datasets and access, see:
* https://arxiv.org/abs/2206.11673
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
with open(f'{current_file_directory}/sipp_feature_types.json', 'r') as file:
   # Use json.load to load the data from the file
   feature_types = json.load(file)

with open(f'{current_file_directory}/sipp_feature_demographic_types.json', 'r') as file:
   # Use json.load to load the data from the file
   feature_demographic_types = json.load(file)

type_dict = {'int8': int, 'int16': int, 'int64': int, 'category': cat_dtype, "float64": float}

SIPP_GLOBAL_FEATURES = [Feature(x, type_dict[feature_types[x]]) for x in feature_types]
SIPP_DEMOGRAPHIC_FEATURES = [Feature(x, type_dict[feature_types[x]]) for x in feature_demographic_types]

SIPP_FEATURES = FeatureList(features=[
        Feature('OPM_RATIO', int, is_target=True,
                name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_GLOBAL_FEATURES)

SIPP_FEATURES_CAUSAL = FeatureList(features=[
        Feature('OPM_RATIO', int, is_target=True,
                name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_DEMOGRAPHIC_FEATURES )

def preprocess_sipp(df:pd.DataFrame)->pd.DataFrame:
        df[SIPP_FEATURES.target] = (1.0*(df[SIPP_FEATURES.target] >= 3)).astype(int)
        return df