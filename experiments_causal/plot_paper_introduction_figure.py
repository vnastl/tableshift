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
import matplotlib.markers as mmark
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_context("paper", font_scale=1.75)
sns.set_style("whitegrid")

from tableshift import get_dataset
from  statsmodels.stats.proportion import proportion_confint
from paretoset import paretoset
from scipy.spatial import ConvexHull

from experiments_causal.plot_config_colors import *
from experiments_causal.plot_experiment import get_results
from experiments_causal.plot_experiment_balanced import get_results as get_results_balanced
from experiments_causal.plot_experiment_causalml import get_results as get_results_causalml
from experiments_causal.plot_experiment_anticausal import get_results as get_results_anticausal
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
list_mak = [mmark.MarkerStyle('s'),mmark.MarkerStyle('D'),mmark.MarkerStyle('o'),mmark.MarkerStyle('X')]
list_lines = ["","","",""]
list_lab = ['All','Arguably causal','Causal', 'Constant',]
list_color  = [color_all, color_arguablycausal, color_causal, color_constant,]

from matplotlib.legend_handler import HandlerBase
class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup,xdescent, ydescent,
                        width, height, fontsize,trans):
        return [plt.Line2D([width/2], [height/2.],ls="",
                       marker=tup[1],markersize=markersize*1.5,color=tup[0], transform=trans)]
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
fig = plt.figure(figsize=(10, 5))
plt.xlabel(f"Tasks")
plt.ylabel(f"Out-of-domain accuracy")

#############################################################################
# plot ood accuracy
#############################################################################
markers = {'constant': 'X','all': 's', 'causal': 'o', 'arguablycausal':'D'}

sets = list(eval_experiments["features"].unique())
sets.sort()

for index, set in enumerate(sets):
    eval_plot_features = eval_experiments[eval_experiments['features']==set]
    eval_plot_features = eval_plot_features.sort_values('ood_test')
    plt.errorbar(
            x=eval_plot_features['task'],
            y=eval_plot_features['ood_test'],
            yerr=eval_plot_features['ood_test_ub']-eval_plot_features['ood_test'],
            color=eval(f"color_{set}"), ecolor=color_error,
            fmt=markers[set],
            markersize=markersize, capsize=capsize, label=set.capitalize() if set != 'arguablycausal' else 'Arguably causal', zorder=len(sets)-index)
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot_features
    shift_acc["type"] = set
    shift_acc["gap"] =  shift_acc["ood_test"] - shift_acc["id_test"]
    shift_acc['id_test_var'] = ((shift_acc['id_test_ub']-shift_acc['id_test']))**2
    shift_acc['ood_test_var'] = ((shift_acc['ood_test_ub']-shift_acc['ood_test']))**2
    shift_acc['gap_var'] = shift_acc['id_test_var']+shift_acc['ood_test_var']
    dic_shift_acc[set] = shift_acc

plt.tick_params(axis='x', labelrotation = 90)

plt.legend(list(zip(list_color,list_mak,list_lines)), list_lab, 
          handler_map={tuple:MarkerHandler()},loc='upper center', bbox_to_anchor=(0.5, 1.2),fancybox=True, ncol=4)
plt.ylim(top=1.0)
plt.grid(axis='x')
fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_introduction.pdf"), bbox_inches='tight')


#%%
fig = plt.figure(figsize=(10, 5))
plt.xlabel(f"Tasks")
plt.ylabel(f"Shift gap (higher better)")
#############################################################################
# plot shift gap
#############################################################################
shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
sets = list(eval_experiments["features"].unique())
sets.sort()

for index, set in enumerate(sets):
    shift_acc_plot = shift_acc[shift_acc['features']==set]
    shift_acc_plot = shift_acc_plot.sort_values('ood_test')
    plt.errorbar(
            x=shift_acc_plot['task'],
            y=shift_acc_plot['gap'],
            yerr=shift_acc_plot['gap_var']**0.5,
            color=eval(f"color_{set}"), ecolor=color_error,
            fmt=markers[set],
            markersize=markersize, capsize=capsize, label=set.capitalize() if set != 'arguablycausal' else 'Arguably causal', zorder=len(sets)-index)
    
plt.axhline(y=0, color='black', linestyle='--',)
plt.tick_params(axis='x', labelrotation = 90)

list_mak.append("_")
list_lines.append("")
list_lab.append('Same performance')
list_color.append("black")
plt.legend(list(zip(list_color,list_mak,list_lines)), list_lab, 
          handler_map={tuple:MarkerHandler()},loc='upper center', bbox_to_anchor=(0.5, 1.2),fancybox=True, ncol=5)

plt.grid(axis='x')
fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_introduction_shift.pdf"), bbox_inches='tight')