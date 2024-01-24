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
                            "meps",
                            "mimic_extract_mort_hosp",
                            "mimic_extract_los_3",
                            "nhanes_lead",
                            "physionet",
                            "sipp",
                            ]


for index, experiment_name in enumerate(experiments):
    fig = plt.figure(figsize=(12, 16))
    ax = fig.subplots(3, 2, gridspec_kw={'width_ratios':[0.3,0.2],'hspace':0.5, 'wspace':0.3})       # create 3x2 subplots on fig
    # plt.suptitle(dic_title[experiment_name])        # set suptitle for subfig1
    
    eval_all, causal_features, extra_features = get_results(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}
    dic_shift_acc ={}

    ax[0,0].set_xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    ax[0,0].set_ylabel(f"out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0,0].errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt="D",
            color=color_constant, ecolor=color_constant,
            markersize=markersize, capsize=capsize, label="constant")
    ax[0,0].hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
                color=color_constant, linewidth=linewidth_shift, alpha=0.7)
    # get pareto set for shift vs accuracy
    shift_acc = eval_constant
    shift_acc["type"] = "constant"
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    dic_shift_acc["constant"] = shift_acc
    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    errors = ax[0,0].errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                color=color_all, ecolor=color_all,
                markersize=markersize, capsize=capsize, label="all")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    ax[0,0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_all, linewidth=linewidth_shift, alpha=0.7)
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot[mask]
    shift_acc["type"] = "all"
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    dic_shift_acc["all"] = shift_acc

    #############################################################################
    # plot errorbars and shift gap for causal features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="causal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    errors = ax[0,0].errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                color=color_causal, ecolor=color_causal,
                markersize=markersize, capsize=capsize, label="causal")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    ax[0,0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_causal, linewidth=linewidth_shift, alpha=0.7)
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot[mask]
    shift_acc["type"] = "causal"
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    dic_shift_acc["causal"] = shift_acc
    #############################################################################
    # plot errorbars and shift gap for arguablycausal features
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        eval_plot = eval_all[eval_all['features']=="arguablycausal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        errors = ax[0,0].errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="^",
                    color=color_arguablycausal, ecolor=color_arguablycausal,
                    markersize=markersize, capsize=capsize, label="arguably\ncausal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "arguably\ncausal"
        dic_shift["arguablycausal"] = shift
        ax[0,0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_arguablycausal, linewidth=linewidth_shift, alpha=0.7)
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[mask]
        shift_acc["type"] = "arguablycausal"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["arguablycausal"] = shift_acc

    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax[0,0].set_xlim()
    ymin, ymax = ax[0,0].set_ylim()
    ax[0,0].plot([xmin, eval_constant['id_test'].values[0]],
                [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    ax[0,0].plot([eval_constant['id_test'].values[0], eval_constant['id_test'].values[0]],
                [ymin,eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    ax[0,0].fill_between([xmin, eval_constant['id_test'].values[0]],
                     [ymin,ymin],
                     [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                        color=color_constant, alpha=0.1)
    
    #############################################################################
    # plot pareto dominated area for all features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]  
    #get extra points for the plot
    new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    ax[0,0].plot(points['id_test'],points['ood_test'],color=color_all,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[0,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.1)

    #############################################################################
    # plot pareto dominated area for causal features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="causal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    #get extra points for the plot
    new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    ax[0,0].plot(points['id_test'],points['ood_test'],color=color_causal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[0,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.1)

    #############################################################################
    # plot pareto dominated area for arguablycausal features
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        eval_plot = eval_all[eval_all['features']=="arguablycausal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        # get extra points for the plot
        new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        ax[0,0].plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)
    
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0,0].plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)


    #############################################################################
    # Plot ood accuracy as bars
    #############################################################################
    ax[0,1].set_ylabel("out-of-domain accuracy")
    # add constant shift gap
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift

    shift = pd.concat(dic_shift.values(), ignore_index=True)
    shift.drop_duplicates(inplace=True)
    shift = shift.iloc[[0,2,1,3],:]
    
    if (eval_all['features'] == "arguablycausal").any():
        ax[0,1].bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_arguablycausal,color_causal,color_constant],
                              ecolor=color_error,align='center', capsize=capsize,
                              bottom=ymin)
        ax[0,1].tick_params(axis='x', labelrotation = 90)
        
    #############################################################################
    # Plot shift gap as bars
    #############################################################################
    ax[1,1].set_ylabel("shift gap")

    shift["gap"] = shift["id_test"] - shift["ood_test"]
    shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
    shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
    shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
    if (eval_all['features'] == "arguablycausal").any():
        ax[1,1].bar(shift["type"], shift["gap"],
                              yerr=shift['gap_var']**0.5,
                              color=[color_all,color_arguablycausal,color_causal,color_constant],
                              ecolor=color_error,align='center', capsize=capsize)
        ax[1,1].tick_params(axis='x', labelrotation = 90)

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        ax[1,0].set_xlabel("1 - shift gap")
        ax[1,0].set_ylabel("out-of-domain accuracy")
        shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
        markers = {'constant': 'D','all': 's', 'causal': 'o', 'arguablycausal':'^'}
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc['type']==type]
            type_shift['id_test_var'] = ((type_shift['id_test_ub']-type_shift['id_test']))**2
            type_shift['ood_test_var'] = ((type_shift['ood_test_ub']-type_shift['ood_test']))**2
            type_shift['gap_var'] = type_shift['id_test_var']+type_shift['ood_test_var']

            # Get markers
            ax[1,0].errorbar(x=1-type_shift["gap"],
                         y=type_shift["ood_test"],
                         xerr= type_shift['gap_var']**0.5,
                         yerr= type_shift['ood_test_ub']-type_shift['ood_test'],
                        color=eval(f"color_{type}"), ecolor=eval(f"color_{type}"),
                        fmt=marker, markersize=markersize, capsize=capsize,  label="arguably\ncausal" if type == 'arguablycausal' else f"{type}",
                        zorder=3)
        xmin, xmax = ax[1,0].get_xlim()
        ymin, ymax = ax[1,0].get_ylim()
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc['type']==type]
            # Get 1 - shift gap
            type_shift["1-gap"] = 1-type_shift['gap']
            # Calculate the pareto set
            points = type_shift[['1-gap','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            #get extra points for the plot
            new_row = pd.DataFrame({'1-gap':[xmin,max(points['1-gap'])], 'ood_test':[max(points['ood_test']),ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points.sort_values('1-gap',inplace=True)
            ax[1,0].plot(points['1-gap'],points['ood_test'],color=eval(f"color_{type}"),linestyle=(0, (1, 1)),linewidth=linewidth_bound)
            new_row = pd.DataFrame({'1-gap':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[1,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{type}"),alpha=0.1)
    
    eval_all, causal_features, extra_features = get_results_balanced(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    ax[2,0].set_xlabel(f"balanced in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    ax[2,0].set_ylabel(f"balanced out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[2,0].errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt="D",
            color=color_constant, ecolor=color_constant,
            markersize=markersize, capsize=capsize, label="constant")
    ax[2,0].hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
                color=color_constant, linewidth=linewidth_shift, alpha=0.7)

    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    errors = ax[2,0].errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                color=color_all, ecolor=color_all,
                markersize=markersize, capsize=capsize, label="all")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    ax[2,0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_all, linewidth=linewidth_shift, alpha=0.7)
    
    
    #############################################################################
    # plot errorbars and shift gap for causal features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="causal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    errors = ax[2,0].errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                color=color_causal, ecolor=color_causal,
                markersize=markersize, capsize=capsize, label="causal")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    ax[2,0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_causal, linewidth=linewidth_shift, alpha=0.7)

    
    #############################################################################
    # plot errorbars and shift gap for arguablycausal features
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        eval_plot = eval_all[eval_all['features']=="arguablycausal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        errors = ax[2,0].errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="^",
                    color=color_arguablycausal, ecolor=color_arguablycausal,
                    markersize=markersize, capsize=capsize, label="arguably causal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "arguably\ncausal"
        dic_shift["arguablycausal"] = shift
        ax[2,0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_arguablycausal, linewidth=linewidth_shift, alpha=0.7)

    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax[2,0].get_xlim()
    ymin, ymax = ax[2,0].get_ylim()
    ax[2,0].plot([xmin, eval_constant['id_test'].values[0]],
                [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    ax[2,0].plot([eval_constant['id_test'].values[0], eval_constant['id_test'].values[0]],
                [ymin,eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    ax[2,0].fill_between([xmin, eval_constant['id_test'].values[0]],
                     [ymin,ymin],
                     [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                        color=color_constant, alpha=0.1)
    
    #############################################################################
    # plot pareto dominated area for all features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    #get extra points for the plot
    new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    ax[2,0].plot(points['id_test'],points['ood_test'],color=color_all,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[2,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.1)

    #############################################################################
    # plot pareto dominated area for causal features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="causal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
    #get extra points for the plot
    new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    ax[2,0].plot(points['id_test'],points['ood_test'],color=color_causal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
    new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[2,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.1)

    #############################################################################
    # plot pareto dominated area for arguablycausal features
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        eval_plot = eval_all[eval_all['features']=="arguablycausal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        # get extra points for the plot
        new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        ax[2,0].plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[2,0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)
    
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################

    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[2,0].plot([start_lim, end_lim], [start_lim, end_lim], color='black')


    #############################################################################
    # Plot ood accuracy as bars
    #############################################################################
    ax[2,1].set_ylabel("balanced out-of-domain accuracy")

    # add constant shift gap
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift

    shift = pd.concat(dic_shift.values(), ignore_index=True)
    shift = shift.iloc[[0,2,1,3],:]
    
    # shift["gap"] = shift["id_test"] - shift["ood_test"]
    if (eval_all['features'] == "arguablycausal").any():
        barlist = ax[2,1].bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_arguablycausal,color_causal,color_constant],
                              ecolor=color_error,align='center', capsize=10,
                              bottom=ymin)
        ax[2,1].tick_params(axis='x', labelrotation = 90)
            
    with open(str(Path(__file__).parents[0]/f"legend.pkl"), 'rb') as f:
        legend = pickle.load(f)

    fig.legend(legend["lines"], legend["labels"], loc='upper center', bbox_to_anchor=(0.5, 0.04),fancybox=True, ncol=4)
    # plt.tight_layout()
    fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_appendix_{experiment_name}.pdf"), bbox_inches='tight')






#%%

experiment_groups = {
    'group1':[
                "acsfoodstamps",
                "acsincome",
                "acsunemployment",
                "anes",
                #  "assistments",
                "college_scorecard",
                ],
    'group2':[
                "diabetes_readmission",
                "mimic_extract_mort_hosp",
                "mimic_extract_los_3",
                ],}
for experiment_group, experiments in experiment_groups.items():
    fig = plt.figure(figsize=(16.8, 4.5*len(experiments)))
    subfigs = fig.subfigures(len(experiments), 1, hspace=0.2) # create 4x1 subfigures

    for index, experiment_name in enumerate(experiments):
        subfig = subfigs[index]
        subfig.subplots_adjust(wspace=0.3)
        ax = subfig.subplots(1, 3, gridspec_kw={'width_ratios':[0.3,0.2,0.2]})
        subfig.suptitle(dic_title[experiment_name])        # set suptitle for subfig1
        eval_all, causal_features = get_results_causalml(experiment_name)
        eval_constant = eval_all[eval_all['features']=="constant"]
        dic_shift = {}

        ax[0].set_xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
        ax[0].set_ylabel(f"out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")
        
        ##############################################################################
        # plot errorbars and shift gap for constant
        #############################################################################
        errors = ax[0].errorbar(
                x=eval_constant['id_test'],
                y=eval_constant['ood_test'],
                xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
                yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt="D",
                color=color_constant, ecolor=color_constant,
                markersize=markersize, capsize=capsize, label="constant")
        ax[0].hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
                    color=color_constant, linewidth=linewidth_shift, alpha=0.7)
        
        #############################################################################
        # plot errorbars and shift gap for all features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="all"]
        eval_plot = eval_plot[(eval_plot['model']!='irm')&(eval_plot['model']!='vrex')&(eval_plot['model']!='tableshift:irm')&(eval_plot['model']!='tableshift:vrex')]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= eval_constant['id_test'].values[0]]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= eval_constant['id_test'].values[0]]
        errors = ax[0].errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                    color=color_all, ecolor=color_all,
                    markersize=markersize, capsize=capsize, label="all")
        # highlight bar
        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "all"
        dic_shift["all"] = shift
        ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_all, linewidth=linewidth_shift, alpha=0.7)
        
        #############################################################################
        # plot errorbars and shift gap for causal features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="causal"]
        eval_plot = eval_plot[(eval_plot['model']!='irm')&(eval_plot['model']!='vrex')&(eval_plot['model']!='tableshift:irm')&(eval_plot['model']!='tableshift:vrex')]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= eval_constant['id_test'].values[0]]
        errors = ax[0].errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                    color=color_causal, ecolor=color_causal,
                    markersize=markersize, capsize=capsize, label="causal")
        # highlight bar
        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "causal"
        dic_shift["causal"] = shift
        ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_causal, linewidth=linewidth_shift, alpha=0.7)
        
        #############################################################################
        # plot errorbars and shift gap for arguablycausal features
        #############################################################################
        if (eval_all['features'] == "arguablycausal").any():
            eval_plot = eval_all[eval_all['features']=="arguablycausal"]
            eval_plot = eval_plot[(eval_plot['model']!='irm')&(eval_plot['model']!='vrex')&(eval_plot['model']!='tableshift:irm')&(eval_plot['model']!='tableshift:vrex')]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            markers = eval_plot[mask]
            markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            errors = ax[0].errorbar(
                        x=markers['id_test'],
                        y=markers['ood_test'],
                        xerr=markers['id_test_ub']-markers['id_test'],
                        yerr=markers['ood_test_ub']-markers['ood_test'], fmt="^",
                        color=color_arguablycausal, ecolor=color_arguablycausal,
                        markersize=markersize, capsize=capsize, label="arguably\ncausal")
            # highlight bar
            shift = markers
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = " arguably\ncausal"
            dic_shift["arguablycausal"] = shift
            ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                    color=color_arguablycausal, linewidth=linewidth_shift, alpha=0.7)

        #############################################################################
        # plot errorbars and shift gap for causal ml
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="all"]
        # eval_plot = eval_plot[eval_plot['model'].isin(['irm', 'vrex'])]
        # eval_plot.sort_values('model',inplace=True)

        for causalml in ['irm', 'vrex']:
            eval_model = eval_plot[(eval_plot['model']==causalml)|(eval_plot['model']==f"tableshift:{causalml}")]
            points = eval_model[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            markers = eval_model[mask]
            markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            errors = ax[0].errorbar(
                        x=markers['id_test'],
                        y=markers['ood_test'],
                        xerr=markers['id_test_ub']-markers['id_test'],
                        yerr=markers['ood_test_ub']-markers['ood_test'], fmt="v",
                        color=eval(f"color_{causalml}"), ecolor=eval(f"color_{causalml}"), 
                        markersize=markersize, capsize=capsize, label="causal ml")
            # highlight bar
        
            shift = markers
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = causalml
            dic_shift[causalml] = shift
            ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                    color=eval(f"color_{causalml}"), linewidth=linewidth_shift, alpha=0.7  )

        #############################################################################
        # plot pareto dominated area for constant
        #############################################################################
        xmin, xmax = ax[0].get_xlim()
        ymin, ymax = ax[0].get_ylim()
        ax[0].plot([xmin, eval_constant['id_test'].values[0]],
                    [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                    color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        ax[0].plot([eval_constant['id_test'].values[0], eval_constant['id_test'].values[0]],
                    [ymin,eval_constant['ood_test'].values[0]],
                    color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        ax[0].fill_between([xmin, eval_constant['id_test'].values[0]],
                        [ymin,ymin],
                        [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                            color=color_constant, alpha=0.1)
        
        #############################################################################
        # plot pareto dominated area for all features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="all"]
        eval_plot = eval_plot[(eval_plot['model']!='irm')&(eval_plot['model']!='vrex')&(eval_plot['model']!='tableshift:irm')&(eval_plot['model']!='tableshift:vrex')]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= eval_constant['id_test'].values[0]]
        #get extra points for the plot
        new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        ax[0].plot(points['id_test'],points['ood_test'],color=color_all,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.1)

        #############################################################################
        # plot pareto dominated area for causal features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="causal"]
        eval_plot = eval_plot[(eval_plot['model']!='irm')&(eval_plot['model']!='vrex')&(eval_plot['model']!='tableshift:irm')&(eval_plot['model']!='tableshift:vrex')]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= eval_constant['id_test'].values[0]]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= eval_constant['id_test'].values[0]]
        #get extra points for the plot
        new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        ax[0].plot(points['id_test'],points['ood_test'],color=color_causal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.1)

        #############################################################################
        # plot pareto dominated area for arguablycausal features
        #############################################################################
        if (eval_all['features'] == "arguablycausal").any():
            eval_plot = eval_all[eval_all['features']=="arguablycausal"]
            eval_plot = eval_plot[(eval_plot['model']!='irm')&(eval_plot['model']!='vrex')&(eval_plot['model']!='tableshift:irm')&(eval_plot['model']!='tableshift:vrex')]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            # get extra points for the plot
            new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points.sort_values('id_test',inplace=True)
            ax[0].plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
            new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)

        #############################################################################
        # plot pareto dominated area for causalml
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="all"]
        # eval_plot = eval_plot[eval_plot['model'].isin(['irm', 'vrex'])]
        # eval_plot.sort_values('model',inplace=True)

        for causalml in ['irm', 'vrex']:
            # Calculate the pareto set
            points = eval_plot[(eval_plot['model']==causalml)|(eval_plot['model']==f"tableshift:{causalml}")][['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            #get extra points for the plot
            new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points.sort_values('id_test',inplace=True)
            ax[0].plot(points['id_test'],points['ood_test'],color=eval(f"color_{causalml}"),linestyle=(0, (1, 1)),linewidth=linewidth_bound)
            new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{causalml}"),alpha=0.1)
        #############################################################################
        # Add legend & diagonal, save plot
        #############################################################################
        # Plot the diagonal line
        start_lim = max(xmin, ymin)
        end_lim = min(xmax, ymax)
        ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color='black')

        ############################################################################
        # Plot ood accuracy as bars
        #############################################################################
        # plt.title(
        # f"{dic_title[experiment_name]}")
        ax[1].set_ylabel("out-of-domain accuracy")

        # add constant shift gap
        shift = eval_constant
        shift["type"] = "constant"
        dic_shift["constant"] = shift

        shift = pd.concat(dic_shift.values(), ignore_index=True)
        shift.drop_duplicates(inplace=True)
        shift = shift.iloc[[0,2,1,3,4,5],:]
        
        barlist = ax[1].bar(shift["type"], shift["ood_test"]-ymin,
                                yerr=shift['ood_test_ub']-shift['ood_test'],
                                color=[color_all,color_arguablycausal,color_causal]+[color_irm,color_vrex,]+[color_constant],
                                ecolor=color_error,align='center', capsize=10,
                                bottom=ymin)
        ax[1].tick_params(axis='x', labelrotation = 90)
        #############################################################################
        # Plot shift gap as bars
        #############################################################################
        ax[2].set_ylabel("shift gap")

        shift["gap"] = shift["id_test"] - shift["ood_test"]
        shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
        shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
        shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
        barlist = ax[2].bar(shift["type"], shift["gap"],
                        yerr=shift['gap_var']**0.5,
                        ecolor=color_error,align='center', capsize=10,
                        color=[color_all,color_arguablycausal,color_causal]+[color_irm,color_vrex]+[color_constant])
        ax[2].tick_params(axis='x', labelrotation = 90)
        
    with open(str(Path(__file__).parents[0]/f"legend_causalml.pkl"), 'rb') as f:
        legend = pickle.load(f)

    fig.legend(legend["lines"], legend["labels"], loc='upper center', bbox_to_anchor=(0.5, -0.03 if experiment_group=='group1' else -0.05),fancybox=True, ncol=5)

    fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_appendix_causalml_{experiment_group}.pdf"), bbox_inches='tight')



# %%

experiment_groups = {
    'group1':[
                "acsincome",
                "acsunemployment",
                "brfss_diabetes",
                "brfss_blood_pressure",
                "sipp"
                ],}
for experiment_group, experiments in experiment_groups.items():
    fig = plt.figure(figsize=(16.8, 4.5*len(experiments)))
    subfigs = fig.subfigures(len(experiments), 1, hspace=0.2) # create 4x1 subfigures

    for index, experiment_name in enumerate(experiments):
        subfig = subfigs[index]
        subfig.subplots_adjust(wspace=0.3)
        ax = subfig.subplots(1, 3, gridspec_kw={'width_ratios':[0.3,0.2,0.2]})
        subfig.suptitle(dic_title[experiment_name])        # set suptitle for subfig1
        eval_all, causal_features, extra_features = get_results_anticausal(experiment_name)
        eval_constant = eval_all[eval_all['features']=="constant"]
        dic_shift = {}
        dic_shift_acc ={}

        ax[0].set_xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
        ax[0].set_ylabel(f"out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")

        #############################################################################
        # plot errorbars and shift gap for constant
        #############################################################################
        errors = ax[0].errorbar(
                x=eval_constant['id_test'],
                y=eval_constant['ood_test'],
                xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
                yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt="D",
                color=color_constant, ecolor=color_constant,
                markersize=markersize, capsize=capsize, label="constant")
        ax[0].hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
                    color=color_constant, linewidth=linewidth_shift, alpha=0.7)
        # get pareto set for shift vs accuracy
        shift_acc = eval_constant
        shift_acc["type"] = "constant"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["constant"] = shift_acc
        #############################################################################
        # plot errorbars and shift gap for all features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="all"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        errors = ax[0].errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                    color=color_all, ecolor=color_all,
                    markersize=markersize, capsize=capsize, label="all")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "all"
        dic_shift["all"] = shift
        ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_all, linewidth=linewidth_shift, alpha=0.7)
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[mask]
        shift_acc["type"] = "all"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["all"] = shift_acc

        #############################################################################
        # plot errorbars and shift gap for causal features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="causal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        errors = ax[0].errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                    color=color_causal, ecolor=color_causal,
                    markersize=markersize, capsize=capsize, label="causal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "causal"
        dic_shift["causal"] = shift
        ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_causal, linewidth=linewidth_shift, alpha=0.7)
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[mask]
        shift_acc["type"] = "causal"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["causal"] = shift_acc
        #############################################################################
        # plot errorbars and shift gap for arguablycausal features
        #############################################################################
        if (eval_all['features'] == "arguablycausal").any():
            eval_plot = eval_all[eval_all['features']=="arguablycausal"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            markers = eval_plot[mask]
            markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            errors = ax[0].errorbar(
                        x=markers['id_test'],
                        y=markers['ood_test'],
                        xerr=markers['id_test_ub']-markers['id_test'],
                        yerr=markers['ood_test_ub']-markers['ood_test'], fmt="^",
                        color=color_arguablycausal, ecolor=color_arguablycausal,
                        markersize=markersize, capsize=capsize, label="arguably\ncausal")
            # highlight bar
            shift = eval_plot[mask]
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = "arguably\ncausal"
            dic_shift["arguablycausal"] = shift
            ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                    color=color_arguablycausal, linewidth=linewidth_shift, alpha=0.7)
            # get pareto set for shift vs accuracy
            shift_acc = eval_plot[mask]
            shift_acc["type"] = "arguablycausal"
            shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
            dic_shift_acc["arguablycausal"] = shift_acc
        
        #############################################################################
        # plot errorbars and shift gap for anticausal features
        #############################################################################
        if (eval_all['features'] == "anticausal").any():
            eval_plot = eval_all[eval_all['features']=="anticausal"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            markers = eval_plot[mask]
            markers = markers[markers["id_test"] >=(eval_constant['id_test'].values[0] -0.01)]
            errors = ax[0].errorbar(
                        x=markers['id_test'],
                        y=markers['ood_test'],
                        xerr=markers['id_test_ub']-markers['id_test'],
                        yerr=markers['ood_test_ub']-markers['ood_test'], fmt="P",
                        color=color_anticausal, ecolor=color_anticausal,
                        markersize=markersize, capsize=capsize, label="anticausal")
            # highlight bar
            shift = eval_plot[mask]
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = "anti\ncausal"
            dic_shift["anticausal"] = shift
            ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                    color=color_anticausal, linewidth=linewidth_shift, alpha=0.7)
        #############################################################################
        # plot pareto dominated area for constant
        #############################################################################
        xmin, xmax = ax[0].set_xlim()
        ymin, ymax = ax[0].set_ylim()
        ax[0].plot([xmin, eval_constant['id_test'].values[0]],
                    [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                    color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        ax[0].plot([eval_constant['id_test'].values[0], eval_constant['id_test'].values[0]],
                    [ymin,eval_constant['ood_test'].values[0]],
                    color=color_constant,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        ax[0].fill_between([xmin, eval_constant['id_test'].values[0]],
                        [ymin,ymin],
                        [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                            color=color_constant, alpha=0.1)
        
        #############################################################################
        # plot pareto dominated area for all features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="all"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]  
        #get extra points for the plot
        new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        ax[0].plot(points['id_test'],points['ood_test'],color=color_all,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.1)

        #############################################################################
        # plot pareto dominated area for causal features
        #############################################################################
        eval_plot = eval_all[eval_all['features']=="causal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        #get extra points for the plot
        new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        ax[0].plot(points['id_test'],points['ood_test'],color=color_causal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.1)

        #############################################################################
        # plot pareto dominated area for arguablycausal features
        #############################################################################
        if (eval_all['features'] == "arguablycausal").any():
            eval_plot = eval_all[eval_all['features']=="arguablycausal"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            # get extra points for the plot
            new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points.sort_values('id_test',inplace=True)
            ax[0].plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)
            new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)
        
        #############################################################################
        # plot pareto dominated area for anticausal features
        #############################################################################
        if (eval_all['features'] == "anticausal").any():
            eval_plot = eval_all[eval_all['features']=="anticausal"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
            # get extra points for the plot
            new_row = pd.DataFrame({'id_test':[xmin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points.sort_values('id_test',inplace=True)
            ax[0].plot(points['id_test'],points['ood_test'],color=color_anticausal,linestyle=(0, (1, 1)),linewidth=linewidth_bound)

            new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_anticausal,alpha=0.1)
        
        #############################################################################
        # Add legend & diagonal, save plot
        #############################################################################
        # Plot the diagonal line
        start_lim = max(xmin, ymin)
        end_lim = min(xmax, ymax)
        ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)

        #############################################################################
        # Plot ood accuracy as bars
        #############################################################################
        ax[1].set_ylabel("out-of-domain accuracy")
        # add constant shift gap
        shift = eval_constant
        shift["type"] = "constant"
        dic_shift["constant"] = shift

        shift = pd.concat(dic_shift.values(), ignore_index=True)
        shift.drop_duplicates(inplace=True)
        shift = shift.iloc[[0,2,1,3,4],:]
        
        if (eval_all['features'] == "arguablycausal").any():
            ax[1].bar(shift["type"], shift["ood_test"]-ymin,
                                yerr=shift['ood_test_ub']-shift['ood_test'],
                                color=[color_all,color_arguablycausal,color_causal,color_anticausal,color_constant],
                                ecolor=color_error,align='center', capsize=capsize,
                                bottom=ymin)
            ax[1].tick_params(axis='x', labelrotation = 90)
            
        #############################################################################
        # Plot shift gap as bars
        #############################################################################
        ax[2].set_ylabel("shift gap")

        shift["gap"] = shift["id_test"] - shift["ood_test"]
        shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
        shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
        shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
        if (eval_all['features'] == "arguablycausal").any():
            ax[2].bar(shift["type"], shift["gap"],
                                yerr=shift['gap_var']**0.5,
                                color=[color_all,color_arguablycausal,color_causal,color_anticausal,color_constant],
                                ecolor=color_error,align='center', capsize=capsize)
            ax[2].tick_params(axis='x', labelrotation = 90)

with open(str(Path(__file__).parents[0]/f"legend_anticausal.pkl"), 'rb') as f:
    legend = pickle.load(f)

fig.legend(legend["lines"], legend["labels"], loc='upper center', bbox_to_anchor=(0.5, -0.03),fancybox=True, ncol=5)
fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_appendix_anticausal_{experiment_group}.pdf"), bbox_inches='tight')