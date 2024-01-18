"""
Utilities for the MEPS dataset.

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

from tableshift.datasets.robustness import select_subset_minus_one, select_superset_plus_one
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

# Note in load_meps: X["INSCOV19"] = X["INSCOV19"].replace({'1 ANY PRIVATE':0.0, '2 PUBLIC ONLY':1.0, '3 UNINSURED':2.0})

MEPS_FEATURES = FeatureList(features=[
        Feature('TOTEXP19', int, is_target=True,
                name_extended="Measure of health care utilization at the end of the year"),
] + MEPS_GLOBAL_FEATURES)

list_additional_causal = [
        "AGE31X",
        "PAYDR31",
        "REGION31",
        "SICPAY31",]

MEPS_ADDITIONAL_CAUSAL_FEATURES = [feature for feature in MEPS_GLOBAL_FEATURES if feature.name in list_additional_causal]

MEPS_FEATURES_CAUSAL = FeatureList(features=[
        Feature('TOTEXP19', int, is_target=True,
                name_extended="Measure of health care utilization at the end of the year"),
] + MEPS_DEMOGRAPHIC_FEATURES + MEPS_ADDITIONAL_CAUSAL_FEATURES)

target = Feature('TOTEXP19', int, is_target=True,
                name_extended="Measure of health care utilization at the end of the year")
domain = [feature for feature in MEPS_GLOBAL_FEATURES if feature.name == 'INSCOV19'][0]
causal_features = MEPS_FEATURES_CAUSAL.features.copy()
causal_features.remove(target)
causal_features.remove(domain)
causal_subsets = select_subset_minus_one(causal_features)
MEPS_FEATURES_CAUSAL_SUBSETS = []
for subset in causal_subsets:
    subset.append(target)
    subset.append(domain)
    MEPS_FEATURES_CAUSAL_SUBSETS.append(FeatureList(subset))
MEPS_FEATURES_CAUSAL_SUBSETS_NUMBER = len(causal_subsets)

list_others = ["JOBORG31", "BSNTY31", "NUMEMP31", "MORE31", "STJBMM31", "STJBYY31", "VACMPY31", "VAPROX31", "VASPUN31", "VACMPM31", "VASPMH31", "VASPOU31", "VAPRHT31", "VAWAIT31", "VAWAIT31", "VALOCT31", "VANTWK31", "VANEED31", "VAOUT31", "VAPAST31", "VACOMP31", "VAMREC31", "VAGTRC31", "VACARC31", "VAPROB31", "VAREP31", "VACARE31", "VAPCPR31", "VAPROV31", "VAPCOT31", "VAPCCO31", "VAPCRC31", "VAPCSN31", "VAPCRF31", "VAPCSO31", "VAPCOU31", "VAPCUN31", "VASPCL31", "VAPACT31", "VACTDY31", "VARECM31", "VAMOBL31", "VACOPD31", "VADERM31", "VAGERD31", "VAHRLS31", "VABACK31", "VAJTPN31", "VARTHR31", "VAGOUT31", "VANECK31", "VAFIBR31", "VATMD31", "VACOST31", "VAPTSD31", "VABIPL31", "VADEPR31", "VAMOOD31", "VAPROS31", "VARHAB31", "VAMNHC31", "VAGCNS31", "VARXMD31", "VACRGV31", "VALCOH31", "RNDFLG31", "HRWGIM31", "HRHOW31", "VERFLG31", "REFPRS31", "REFRL31X", "FCRP1231", "FMRS1231", "FAMS1231", "RESP31", "PROXY31", "BEGRFM31", "BEGRFY31", "ENDRFM31", "ENDRFY31", "INSCOP31", "INSC1231", "ELGRND31", "MOPID31X", "DAPID31X", "RUSIZE31", "RUSIZE31", "RUCLAS31", "PSTATS31", "SPOUID31", "SPOUIN31"]

MEPS_DOMAIN_ARGUABLYCAUSAL_FEATURE = [feature for feature in MEPS_GLOBAL_FEATURES if feature.name not in list_others]
MEPS_FEATURES_ARGUABLYCAUSAL = FeatureList(features=[
        Feature('TOTEXP19', int, is_target=True,
                name_extended="Measure of health care utilization at the end of the year"),
] + MEPS_DOMAIN_ARGUABLYCAUSAL_FEATURE)
arguablycausal_supersets = select_superset_plus_one(MEPS_FEATURES_ARGUABLYCAUSAL.features, MEPS_FEATURES.features)
MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS = []
for superset in arguablycausal_supersets:
    MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS.append(FeatureList(superset))
MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER = len(arguablycausal_supersets)


def preprocess_meps(df:pd.DataFrame)->pd.DataFrame:
        df[MEPS_FEATURES.target] = (1.0*(df[MEPS_FEATURES.target] > 3)).astype(int)
        df = df[df[domain.name] != 2]
        return df


