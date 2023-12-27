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