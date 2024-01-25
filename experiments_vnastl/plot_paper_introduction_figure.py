#%%
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import ast
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_context("paper", font_scale=1.75)

from tableshift import get_dataset
from  statsmodels.stats.proportion import proportion_confint
from paretoset import paretoset
from scipy.spatial import ConvexHull

from experiments_vnastl.plot_config_colors import *
from experiments_vnastl.plot_experiment import get_results
from experiments_vnastl.plot_experiment_balanced import get_results as get_results_balanced
from experiments_vnastl.plot_experiment_causalml import get_results as get_results_causalml
from experiments_vnastl.plot_experiment_anticausal import get_results as get_results_anticausal
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 1200

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")

dic_title = {
    "acsemployment":'Employment',
    "acsfoodstamps": 'Food Stamps',
    "acsincome": 'Income',
    "acspubcov": 'PublicCoverage',
    "acsunemployment": 'Unemployment',
    "anes": 'Voting',
    "assistments": 'ASSISTments',
    "brfss_blood_pressure":'Hypertension',
    "brfss_diabetes": 'Diabetes',
    "college_scorecard": 'College Scorecard',
    "diabetes_readmission": 'Hospital Readmission',
    "meps": 'Utilization',
    "mimic_extract_los_3": 'ICU Length of Stay',
    "mimic_extract_mort_hosp": 'Hospital Mortality',
    "nhanes_lead": 'Childhood Lead',
    "physionet": 'Sepsis',
    "sipp": 'Poverty',
}

 #%%
experiments = [
                "acsfoodstamps",
                "acsincome",
                "acspubcov",
                "acsunemployment",
                "anes",
                "assistments",
                "brfss_blood_pressure",
                "brfss_diabetes",
                "college_scorecard",
                "diabetes_readmission",
                "mimic_extract_mort_hosp",
                "mimic_extract_los_3",
                "nhanes_lead",
                "physionet",
                "meps",
                "sipp",
                ]




eval_experiments = pd.DataFrame()
for index, experiment_name in enumerate(experiments):
    eval_all, _, _ = get_results(experiment_name)
    eval_all["task"] = dic_title[experiment_name]

    eval_plot = pd.DataFrame()
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all['features']==set]
        eval_feature = eval_feature[eval_feature["ood_test"] == eval_feature["ood_test"].max()]
        eval_feature.drop_duplicates(inplace=True)
        eval_plot = pd.concat([eval_plot,eval_feature])
    eval_experiments = pd.concat([eval_experiments,eval_plot])
    dic_shift = {}
    dic_shift_acc ={}

#%%
fig = plt.figure(figsize=(12, 6))
plt.xlabel(f"tasks")
plt.ylabel(f"out-of-domain accuracy")
#############################################################################
# plot ood accuracy
#############################################################################
markers = {'constant': 'D','all': 's', 'causal': 'o', 'arguablycausal':'^'}

sets = list(eval_experiments["features"].unique())
sets.sort()
for set in sets:
    eval_plot_features = eval_experiments[eval_experiments['features']==set]
    plt.errorbar(
            x=eval_plot_features['task'],
            y=eval_plot_features['ood_test'],
            yerr=eval_plot_features['ood_test_ub']-eval_plot_features['ood_test'],
            color=eval(f"color_{set}"), ecolor=eval(f"color_{set}"),fmt=markers[set],
            markersize=markersize, capsize=capsize, label=set)
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot_features
    shift_acc["type"] = set
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    shift_acc['id_test_var'] = ((shift_acc['id_test_ub']-shift_acc['id_test']))**2
    shift_acc['ood_test_var'] = ((shift_acc['ood_test_ub']-shift_acc['ood_test']))**2
    shift_acc['gap_var'] = shift_acc['id_test_var']+shift_acc['ood_test_var']
    dic_shift_acc[set] = shift_acc

plt.tick_params(axis='x', labelrotation = 90)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1),fancybox=True, ncol=5)


fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_introduction.pdf"), bbox_inches='tight')


#%%
fig = plt.figure(figsize=(12, 6))
plt.xlabel(f"tasks")
plt.ylabel(f"shift gap")
#############################################################################
# plot shift gap
#############################################################################
shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
sets = list(eval_experiments["features"].unique())
sets.sort()
for set in sets:
    shift_acc_plot = shift_acc[shift_acc['features']==set]
    plt.errorbar(
            x=shift_acc_plot['task'],
            y=shift_acc_plot['gap'],
            yerr=shift_acc_plot['gap_var']**0.5,
            color=eval(f"color_{set}"), ecolor=eval(f"color_{set}"),fmt=markers[set],
            markersize=markersize, capsize=capsize, label=set)
plt.tick_params(axis='x', labelrotation = 90)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1),fancybox=True, ncol=5)
fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_introduction_shift.pdf"), bbox_inches='tight')