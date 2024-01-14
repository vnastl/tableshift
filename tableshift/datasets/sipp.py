"""
Utilities for the SIPP dataset.

This is a public data source and no special action is required
to access it.

For more information on datasets and access, see:
* https://arxiv.org/abs/2206.11673
"""
import pandas as pd
from tableshift.core.features import Feature, FeatureList, cat_dtype
from tableshift.datasets.robustness import select_subset_minus_one, select_superset_plus_one

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


list_additional_causal = [
        'ORIGIN',
        'HEALTHDISAB',
        'HEALTH_HEARING',
        'HEALTH_SEEING',
        'HEALTH_COGNITIVE',
        'HEALTH_AMBULATORY',
        'HEALTH_SELF_CARE',
        'HEALTH_ERRANDS_DIFFICULTY',
        'HEALTH_CORE_DISABILITY',
        'HEALTH_SUPPLEMENTAL_DISABILITY',
                          ]
SIPP_ADDITIONAL_CAUSAL_FEATURES = [feature for feature in SIPP_GLOBAL_FEATURES if feature.name in list_additional_causal]

SIPP_FEATURES_CAUSAL = FeatureList(features=[
        Feature('OPM_RATIO', int, is_target=True,
                name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_DEMOGRAPHIC_FEATURES + SIPP_ADDITIONAL_CAUSAL_FEATURES)

target = Feature('OPM_RATIO', int, is_target=True,
                name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals")
domain = [feature for feature in SIPP_GLOBAL_FEATURES if feature.name == 'CITIZENSHIP_STATUS'][0]
causal_features = SIPP_FEATURES_CAUSAL.features.copy()
causal_features.remove(target)
causal_features.remove(domain)
causal_subsets = select_subset_minus_one(causal_features)
SIPP_FEATURES_CAUSAL_SUBSETS = []
for subset in causal_subsets:
    subset.append(target)
    subset.append(domain)
    SIPP_FEATURES_CAUSAL_SUBSETS.append(FeatureList(subset))
SIPP_FEATURES_CAUSAL_SUBSETS_NUMBER = len(causal_subsets)


list_anticausal = [
        'LIVING_QUARTERS_TYPE',
        'FOOD_ASSISTANCE',
        'TANF_ASSISTANCE',
        'SNAP_ASSISTANCE',
        'WIC_ASSISTANCE',
        'MEDICAID_ASSISTANCE',
]

SIPP_DOMAIN_ARGUABLYCAUSAL_FEATURE = [feature for feature in SIPP_GLOBAL_FEATURES if feature.name not in list_anticausal]
SIPP_FEATURES_ARGUABLYCAUSAL = FeatureList(features=[
        Feature('OPM_RATIO', int, is_target=True,
                name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_DOMAIN_ARGUABLYCAUSAL_FEATURE)
arguablycausal_supersets = select_superset_plus_one(SIPP_FEATURES_ARGUABLYCAUSAL.features, SIPP_FEATURES.features)
SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS = []
for superset in arguablycausal_supersets:
    SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS.append(FeatureList(superset))
SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER = len(arguablycausal_supersets)


list_domain_anticausal = list_anticausal + ['CITIZENSHIP_STATUS']
SIPP_DOMAIN_ANTICAUSAL_FEATURES = [feature for feature in SIPP_GLOBAL_FEATURES if feature.name in list_domain_anticausal]
SIPP_FEATURES_ANTICAUSAL = FeatureList(features=[
        Feature('OPM_RATIO', int, is_target=True,
                name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_DOMAIN_ANTICAUSAL_FEATURES)

def preprocess_sipp(df:pd.DataFrame)->pd.DataFrame:
        df[SIPP_FEATURES.target] = (1.0*(df[SIPP_FEATURES.target] >= 3)).astype(int)
        return df