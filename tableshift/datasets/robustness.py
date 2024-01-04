import numpy as np
import pandas as pd
import torch
from itertools import combinations

def select_powerset(x):
    # Generate powerset of columns
    powerset = []
    for r in range(len(x.columns) + 1):
        combinations_r = list(combinations(x.columns, r))
        for item in combinations_r:
            powerset.append(list(item))
    for item in powerset:
        if len(item) == 0:
            powerset.remove(item)
    return powerset

def select_subset_minus_one(x):
    # Generate powerset of columns
    subsets = []
    r = len(x)-1
    combinations_r = list(combinations(x, r))
    for item in combinations_r:
        subsets.append(list(item))
    return subsets

def select_superset_plus_one(x,all):
    # Generate powerset of columns
    superset = []
    feature_names_x = [feature.name for feature in x]
    feature_names_all = [feature.name for feature in all]
    feature_names_additional = list(set(feature_names_all).difference(feature_names_x))
    additional = [feature for feature in all if feature.name in feature_names_additional]
    for item in additional:
        superset.append(x+[item])