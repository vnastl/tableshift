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
import matplotlib.markers as mmark
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import seaborn as sns
sns.set_context("paper", font_scale=1)
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
from experiments_causal.plot_experiment_causal_robust import get_results as get_results_causal_robust
from experiments_causal.plot_experiment_arguablycausal_robust import get_results as get_results_arguablycausal_robust
from experiments_causal.plot_experiment_causal_robust import dic_robust_number as dic_robust_number_causal
from experiments_causal.plot_experiment_arguablycausal_robust import dic_robust_number as dic_robust_number_arguablycausal
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
    "meps": 'MEPS: Utilization',
    "mimic_extract_los_3": 'ICU Length of Stay',
    "mimic_extract_mort_hosp": 'Hospital Mortality',
    "nhanes_lead": 'Childhood Lead',
    "physionet": 'Sepsis',
    "sipp": 'SIPP: Poverty',
}
list_mak = [mmark.MarkerStyle('s'),mmark.MarkerStyle('D'),mmark.MarkerStyle('o'),mmark.MarkerStyle('X')]
list_lab = ['All','Arguably causal','Causal', 'Constant']
list_color  = [color_all, color_arguablycausal, color_causal, color_constant]

from matplotlib.legend_handler import HandlerBase
class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup,xdescent, ydescent,
                        width, height, fontsize,trans):
        return [plt.Line2D([width/2], [height/2.],ls="",
                       marker=tup[1],markersize=markersize,color=tup[0], transform=trans)]
    
#%%
fig = plt.figure(figsize=[8*2, 4.8*4])
experiments = ["brfss_diabetes","acsunemployment","acsincome","mimic_extract_mort_hosp"] 

(subfig1, subfig2, subfig3, subfig4) = fig.subfigures(4, 1, hspace=0.1) # create 4x1 subfigures

subfigs = (subfig1, subfig2, subfig3, subfig4)


ax1 = subfig1.subplots(1, 2, gridspec_kw={'width_ratios':[0.5,0.5]})       # create 1x4 subplots on subfig1
ax2 = subfig2.subplots(1, 2, gridspec_kw={'width_ratios':[0.5,0.5]})       # create 1x4 subplots on subfig2
ax3 = subfig3.subplots(1, 2, gridspec_kw={'width_ratios':[0.5,0.5]})       # create 1x4 subplots on subfig2
ax4 = subfig4.subplots(1, 2, gridspec_kw={'width_ratios':[0.5,0.5]})       # create 1x4 subplots on subfig2
axes = (ax1, ax2, ax3, ax4)

for index, experiment_name in enumerate(experiments):
    sns.set_style('whitegrid')
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes[index]
    subfig.suptitle(dic_title[experiment_name])        # set suptitle for subfig1
    eval_all, causal_features, extra_features = get_results(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[0].set_xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    ax[0].set_ylabel(f"Out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")
    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt='X',
            color=color_constant, ecolor=color_error,
            markersize=markersize, capsize=capsize, label="constant")
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
                color=color_all, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="all")

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
                color=color_causal, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="causal")
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
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt='D',
                    color=color_arguablycausal, ecolor=color_error,
                    markersize=markersize, capsize=capsize, label="arguably\ncausal")

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
                        color=color_constant, alpha=0.05)
    
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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.05)

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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.05)

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
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.05)
    
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    ax[1].set_xlabel("Shift gap")
    ax[1].set_ylabel("Out-of-domain accuracy")

    shift_acc = eval_all
    shift_acc["gap"] = eval_all["ood_test"] - eval_all["id_test"]
    shift_acc["type"] = shift_acc["features"]

    markers = {'constant': 'X','all': 's', 'causal': 'o', 'arguablycausal':'D'}
    for type, marker in markers.items():
        type_shift = shift_acc[shift_acc['type']==type]
        type_shift = type_shift[type_shift["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        points = type_shift[['gap','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        type_shift = type_shift[mask]
        type_shift['id_test_var'] = ((type_shift['id_test_ub']-type_shift['id_test']))**2
        type_shift['ood_test_var'] = ((type_shift['ood_test_ub']-type_shift['ood_test']))**2
        type_shift['gap_var'] = type_shift['id_test_var']+type_shift['ood_test_var']

        # Get markers
        ax[1].errorbar(x=type_shift["gap"],
                        y=type_shift["ood_test"],
                        xerr= type_shift['gap_var']**0.5,
                        yerr= type_shift['ood_test_ub']-type_shift['ood_test'],
                        color=eval(f"color_{type}"), ecolor=color_error,
                        fmt=marker, markersize=markersize, capsize=capsize,  label="arguably\ncausal" if type == 'arguablycausal' else f"{type}",
                        zorder=3)
        
    xmin, xmax = ax[1].get_xlim()
    ymin, ymax = ax[1].get_ylim()
    for type, marker in markers.items():
        type_shift = shift_acc[shift_acc['type']==type]
        type_shift = type_shift[type_shift["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        points = type_shift[['gap','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        #get extra points for the plot
        new_row = pd.DataFrame({'gap':[xmin,max(points['gap'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('gap',inplace=True)
        new_row = pd.DataFrame({'gap':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[1].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{type}"),alpha=0.05)
        hull_points = points[hull.vertices]
        # Convert each array to a set of tuples
        matching_rows = np.where(np.all(hull_points == new_row.to_numpy()[0], axis=1))
        hull_points = np.delete(hull_points, matching_rows[0][0], axis=0)
        hull_points = hull_points[hull_points[:, 0].argsort()]
        if hull_points[-1,1] > hull_points[-2,1]:
            # Select the last two rows
            last_two_rows = hull_points[-2:]
            # Swap the rows
            swapped_rows = last_two_rows[::-1]
            # Replace the original rows with the swapped ones
            hull_points[-2:] = swapped_rows
        ax[1].plot(hull_points[:,0],hull_points[:,1],color=eval(f"color_{type}"),linestyle=(0, (1, 1)),linewidth=linewidth_bound)
            

fig.legend(list(zip(list_color,list_mak)), list_lab,
           handler_map={tuple:MarkerHandler()},loc='upper center', bbox_to_anchor=(0.5, -0.03),fancybox=True, ncol=5)


fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_main_result.pdf"), bbox_inches='tight')


#%% 
#############################################################################
# Next figure
#############################################################################

fig = plt.figure(figsize=[8, 4.8])

experiments = ["acsunemployment"]
subfig1 = fig.subfigures(1, 1, hspace=0.1) # create 4x1 subfigures

subfigs = (subfig1,)
ax1 = subfig1.subplots(1, 1,)       # create 1x4 subplots on subfig1
axes = (ax1,)

for index, experiment_name in enumerate(experiments):
    sns.set_style('whitegrid')
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes
    subfig.suptitle(dic_title[experiment_name])        # set suptitle for subfig1

    eval_all, causal_features, extra_features = get_results_balanced(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[0].set_xlabel(f"balanced in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    ax[0].set_ylabel(f"balanced Out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt='X',
            color=color_constant, ecolor=color_error,
            markersize=markersize, capsize=capsize, label="constant")
    # ax[0].hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
    #             color=color_constant, linewidth=linewidth_shift, alpha=0.7)

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
                color=color_all, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="all")
    
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
                color=color_causal, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="causal")
    
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
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt='D',
                    color=color_arguablycausal, ecolor=color_error,
                    markersize=markersize, capsize=capsize, label="arguably causal")

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
                        color=color_constant, alpha=0.05)
    
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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.05)

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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.05)

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
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.05)
    
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################

    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color='black')

fig.legend(list(zip(list_color,list_mak)), list_lab, 
          handler_map={tuple:MarkerHandler()},loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, ncol=5)


fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_main_balanced.pdf"), bbox_inches='tight')






# %%
#############################################################################
# Next figure
#############################################################################

fig = plt.figure(figsize=[8*2, 4.8])

experiments = ["acsincome"]
subfig1 = fig.subfigures(1, 1, hspace=0.1) # create 4x1 subfigures

subfigs = (subfig1,)
ax1 = subfig1.subplots(1, 2, gridspec_kw={'width_ratios':[0.5,0.5]})       # create 1x4 subplots on subfig1
axes = (ax1,)

for index, experiment_name in enumerate(experiments):
    sns.set_style('whitegrid')
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes[index]
    subfig.suptitle(dic_title[experiment_name])        # set suptitle for subfig1

    eval_all, causal_features = get_results_causalml(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[0].set_xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    ax[0].set_ylabel(f"Out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")
    
    ##############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt='X',
            color=color_constant, ecolor=color_error,
            markersize=markersize, capsize=capsize, label="constant")
    
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
                color=color_all, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="all")
    
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
                color=color_causal, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="causal")
    
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
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt='D',
                    color=color_arguablycausal, ecolor=color_error,
                    markersize=markersize, capsize=capsize, label="arguably\ncausal")
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
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="v" if causalml == "irm" else "^",
                    color=eval(f"color_{causalml}"), ecolor=color_error,
                    markersize=markersize, capsize=capsize, label="causal ml")

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
                        color=color_constant, alpha=0.05)
    
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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.05)

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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.05)

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
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.05)

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
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{causalml}"),alpha=0.05)
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color='black')

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    ax[1].set_xlabel("Shift gap")
    ax[1].set_ylabel("Out-of-domain accuracy")

    shift_acc = eval_all
    shift_acc["gap"] = eval_all["ood_test"] - eval_all["id_test"]
    shift_acc["type"] = shift_acc["features"]

    markers = {'constant': 'X','all': 's', 'causal': 'o', 'arguablycausal':'D', 'irm':"v", "vrex": "^"}
    for type, marker in markers.items():
        if type not in ['irm','vrex']:
            type_shift = shift_acc[shift_acc['type']==type]
            type_shift = type_shift[(type_shift['model']!='irm')&(type_shift['model']!='vrex')&(type_shift['model']!='tableshift:irm')&(type_shift['model']!='tableshift:vrex')]
            type_shift = type_shift[type_shift["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        else:
            type_shift = shift_acc[shift_acc['type']=="all"]
            type_shift = type_shift[(type_shift['model']==type)|(type_shift['model']==f"tableshift:{type}")]
        points = type_shift[['gap','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        type_shift = type_shift[mask]
        type_shift['id_test_var'] = ((type_shift['id_test_ub']-type_shift['id_test']))**2
        type_shift['ood_test_var'] = ((type_shift['ood_test_ub']-type_shift['ood_test']))**2
        type_shift['gap_var'] = type_shift['id_test_var']+type_shift['ood_test_var']

        # Get markers
        ax[1].errorbar(x=type_shift["gap"],
                        y=type_shift["ood_test"],
                        xerr= type_shift['gap_var']**0.5,
                        yerr= type_shift['ood_test_ub']-type_shift['ood_test'],
                        color=eval(f"color_{type}"), ecolor=color_error,
                        fmt=marker, markersize=markersize, capsize=capsize,  label="arguably\ncausal" if type == 'arguablycausal' else f"{type}",
                        zorder=3)
        
    xmin, xmax = ax[1].get_xlim()
    ymin, ymax = ax[1].get_ylim()
    for type, marker in markers.items():
        if type not in ['irm','vrex']:
            type_shift = shift_acc[shift_acc['type']==type]
            type_shift = type_shift[(type_shift['model']!='irm')&(type_shift['model']!='vrex')&(type_shift['model']!='tableshift:irm')&(type_shift['model']!='tableshift:vrex')]
            type_shift = type_shift[type_shift["id_test"] >= (eval_constant['id_test'].values[0] -0.01)]
        else:
            type_shift = shift_acc[shift_acc['type']=="all"]
            type_shift = type_shift[(type_shift['model']==type)|(type_shift['model']==f"tableshift:{type}")]
        points = type_shift[['gap','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        #get extra points for the plot
        new_row = pd.DataFrame({'gap':[xmin,max(points['gap'])], 'ood_test':[max(points['ood_test']),ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('gap',inplace=True)
        new_row = pd.DataFrame({'gap':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[1].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{type}"),alpha=0.05)
        hull_points = points[hull.vertices]
        # Convert each array to a set of tuples
        matching_rows = np.where(np.all(hull_points == new_row.to_numpy()[0], axis=1))
        hull_points = np.delete(hull_points, matching_rows[0][0], axis=0)
        hull_points = hull_points[hull_points[:, 0].argsort()]
        if hull_points[-1,1] > hull_points[-2,1]:
            # Select the last two rows
            last_two_rows = hull_points[-2:]
            # Swap the rows
            swapped_rows = last_two_rows[::-1]
            # Replace the original rows with the swapped ones
            hull_points[-2:] = swapped_rows
        ax[1].plot(hull_points[:,0],hull_points[:,1],color=eval(f"color_{type}"),linestyle=(0, (1, 1)),linewidth=linewidth_bound)

list_color_causalml = list_color.copy()
list_color_causalml.remove(color_constant)
list_color_causalml.append(color_irm)
list_color_causalml.append(color_vrex)
list_color_causalml.append(color_constant)
list_mak_causalml = list_mak.copy()
list_mak_causalml.remove(list_mak[-1])
list_mak_causalml.append("v")
list_mak_causalml.append("^")
list_mak_causalml.append(list_mak[-1])
list_lab_causalml = list_lab.copy()
list_lab_causalml.remove("Constant")
list_lab_causalml.append("IRM")
list_lab_causalml.append("REx")
list_lab_causalml.append("Constant")
fig.legend(list(zip(list_color_causalml,list_mak_causalml)), list_lab_causalml, 
          handler_map={tuple:MarkerHandler()},loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, ncol=6)


fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_main_causalml.pdf"), bbox_inches='tight')







# %%
#############################################################################
# Next figure
#############################################################################

fig = plt.figure(figsize=[8*2, 4.8])

experiments = ["brfss_diabetes"]
subfig1 = fig.subfigures(1, 1, hspace=0.1) # create 4x1 subfigures

subfigs = (subfig1,)
ax1 = subfig1.subplots(1, 2, gridspec_kw={'width_ratios':[0.5,0.5]})       # create 1x4 subplots on subfig1
axes = (ax1,)

for index, experiment_name in enumerate(experiments):
    sns.set_style('whitegrid')
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes[index]
    subfig.suptitle(dic_title[experiment_name])        # set suptitle for subfig1


    eval_all, causal_features, extra_features = get_results_anticausal(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}
    dic_shift_acc ={}

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[0].set_xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    ax[0].set_ylabel(f"Out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt='X',
            color=color_constant, ecolor=color_error,
            markersize=markersize, capsize=capsize, label="constant")
    # ax[0].hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
    #             color=color_constant, linewidth=linewidth_shift, alpha=0.7)
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
                color=color_all, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="all")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    # ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
    #            color=color_all, linewidth=linewidth_shift, alpha=0.7)
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot[eval_plot['ood_test'] ==eval_plot['ood_test'].max()].drop_duplicates()
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
                color=color_causal, ecolor=color_error,
                markersize=markersize, capsize=capsize, label="causal")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    # ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
    #            color=color_causal, linewidth=linewidth_shift, alpha=0.7)
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot[eval_plot['ood_test'] ==eval_plot['ood_test'].max()].drop_duplicates()
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
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt='D',
                    color=color_arguablycausal, ecolor=color_error,
                    markersize=markersize, capsize=capsize, label="arguably\ncausal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "arguably\ncausal"
        dic_shift["arguablycausal"] = shift
        # ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
        #         color=color_arguablycausal, linewidth=linewidth_shift, alpha=0.7)
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[eval_plot['ood_test'] ==eval_plot['ood_test'].max()].drop_duplicates()
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
                    color=color_anticausal, ecolor=color_error,
                    markersize=markersize, capsize=capsize, label="anticausal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "anti\ncausal"
        dic_shift["anticausal"] = shift
        shift_acc = eval_plot[eval_plot['ood_test'] ==eval_plot['ood_test'].max()].drop_duplicates()
        shift_acc["type"] = "anticausal"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["anticausal"] = shift_acc
        # ax[0].hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
        #         color=color_anticausal, linewidth=linewidth_shift, alpha=0.7)
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
                        color=color_constant, alpha=0.05)
    
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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.05)

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
    ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.05)

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
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.05)
    
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
        ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_anticausal,alpha=0.05)
    
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)

    # #############################################################################
    # # Plot ood accuracy as bars
    # #############################################################################
    # ax[1].set_ylabel("Out-of-domain accuracy")
    # # add constant shift gap
    # shift = eval_constant
    # shift["type"] = "constant"
    # dic_shift["constant"] = shift

    # shift = pd.concat(dic_shift.values(), ignore_index=True)
    # shift.drop_duplicates(inplace=True)
    # shift = shift.iloc[[0,2,1,3,4],:]
    
    # if (eval_all['features'] == "arguablycausal").any():
    #     ax[1].bar(shift["type"], shift["ood_test"]-ymin,
    #                           yerr=shift['ood_test_ub']-shift['ood_test'],
    #                           color=[color_all,color_arguablycausal,color_causal,color_anticausal,color_constant],
    #                           ecolor=color_error,
    #                           bottom=ymin)
    #     ax[1].tick_params(axis='x', labelrotation = 90)
        
    # #############################################################################
    # # Plot shift gap as bars
    # #############################################################################
    # ax[2].set_ylabel("Shift gap")

    # shift["gap"] = shift["id_test"] - shift["ood_test"]
    # shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
    # shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
    # shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
    # if (eval_all['features'] == "arguablycausal").any():
    #     ax[2].bar(shift["type"], shift["gap"],
    #                           yerr=shift['gap_var']**0.5,
    #                           color=[color_all,color_arguablycausal,color_causal,color_anticausal,color_constant],
    #                           ecolor=color_error,
    #     ax[2].tick_params(axis='x', labelrotation = 90)

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        ax[1].set_xlabel("Shift gap")
        ax[1].set_ylabel("Out-of-domain accuracy")
        shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
        markers = {'constant': 'X','all': 's', 'causal': 'o', 'arguablycausal':'D', 'anticausal':'P'}
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc['type']==type]
            type_shift['id_test_var'] = ((type_shift['id_test_ub']-type_shift['id_test']))**2
            type_shift['ood_test_var'] = ((type_shift['ood_test_ub']-type_shift['ood_test']))**2
            type_shift['gap_var'] = type_shift['id_test_var']+type_shift['ood_test_var']

            # Get markers
            ax[1].errorbar(x=-type_shift["gap"],
                         y=type_shift["ood_test"],
                         xerr= type_shift['gap_var']**0.5,
                         yerr= type_shift['ood_test_ub']-type_shift['ood_test'],
                        color=eval(f"color_{type}"), ecolor=color_error,
                        fmt=marker, markersize=markersize, capsize=capsize,  label="arguably\ncausal" if type == 'arguablycausal' else f"{type}",
                        zorder=3)
            
        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc['type']==type]
            # Get 1 - shift gap
            type_shift["-gap"] = -type_shift['gap']
            # Calculate the pareto set
            points = type_shift[['-gap','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            #get extra points for the plot
            new_row = pd.DataFrame({'-gap':[xmin,max(points['-gap'])], 'ood_test':[max(points['ood_test']),ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points.sort_values('-gap',inplace=True)
            ax[1].plot(points['-gap'],points['ood_test'],color=eval(f"color_{type}"),linestyle=(0, (1, 1)),linewidth=linewidth_bound)
            new_row = pd.DataFrame({'-gap':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[1].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{type}"),alpha=0.05)
with open(str(Path(__file__).parents[0]/f"legend_anticausal.pkl"), 'rb') as f:
    legend = pickle.load(f)

list_color_anticausal = list_color.copy()
list_color_anticausal.remove(color_constant)
list_color_anticausal.append(color_anticausal)
list_color_anticausal.append(color_constant)
list_mak_anticausal = list_mak.copy()
list_mak_anticausal.remove(list_mak[-1])
list_mak_anticausal.append("P")
list_mak_anticausal.append(list_mak[-1])
list_lab_anticausal = list_lab.copy()
list_lab_anticausal.remove("Constant")
list_lab_anticausal.append("Anticausal")
list_lab_anticausal.append("Constant")
fig.legend(list(zip(list_color_anticausal,list_mak_anticausal)), list_lab_anticausal, 
          handler_map={tuple:MarkerHandler()},loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, ncol=5)


fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_main_anticausal.pdf"), bbox_inches='tight')
# %%
#%% 
#############################################################################
# Next figure
#############################################################################

fig = plt.figure(figsize=[8*2, 4.8*2])

experiments = ["brfss_diabetes"]
subfig1 = fig.subfigures(1, 1, hspace=0.1) # create 4x1 subfigures

subfigs = (subfig1,)
ax1 = subfig1.subplots(2, 2, gridspec_kw={'width_ratios':[0.5,0.5],'hspace':0.5})       # create 1x4 subplots on subfig1
axes = (ax1,)

for index, experiment_name in enumerate(experiments):
    sns.set_style('whitegrid')
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes[index]
    subfig.suptitle(dic_title[experiment_name], y=0.95)        # set suptitle for subfig1

    eval_all, _ = get_results_causal_robust(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[0,0].set_ylabel(f"Out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")
    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    
    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    
    #############################################################################
    # plot errorbars and shift gap for causal features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="causal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift

    #############################################################################
    # plot errorbars and shift gap for robustness tests
    #############################################################################
    for index in range(dic_robust_number_causal[experiment_name]):
        if (eval_all['features'] == f"test{index}").any():
            eval_plot = eval_all[eval_all['features']==f"test{index}"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            shift = eval_plot[mask]
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = f"test {index}"
            dic_shift[f"test{index}"] = shift
            
    #############################################################################
    # Plot ood accuracy as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    ax[0,0].set_ylabel("Out-of-domain accuracy")

    # add constant shift gap
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift

    shift = pd.concat(dic_shift.values(), ignore_index=True)
    # shift["gap"] = shift["id_test"] - shift["ood_test"]
    barlist = ax[0,0].bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_causal]+[color_causal_robust for index in range(dic_robust_number_causal[experiment_name])]+[color_constant],
                              ecolor=color_error,align='center', capsize=5,
                              bottom=ymin)
    ax[0,0].tick_params(axis='x', labelrotation = 90)

    #############################################################################
    # Plot shift gap as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    ax[0,1].set_ylabel("Shift gap")
    
    shift = pd.concat(dic_shift.values(), ignore_index=True)
    shift["gap"] = shift["id_test"] - shift["ood_test"]
    shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
    shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
    shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
    barlist = ax[0,1].bar(shift["type"], -shift["gap"],
                      yerr=shift['gap_var']**0.5,ecolor=color_error,align='center', capsize=5,
                      color=[color_all,color_causal]+[color_causal_robust for index in range(dic_robust_number_causal[experiment_name])]+[color_constant])
    ax[0,1].tick_params(axis='x', labelrotation = 90)


    eval_all, _ = get_results_arguablycausal_robust(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    
    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    
    #############################################################################
    # plot errorbars and shift gap for arguably causal features
    #############################################################################
    eval_plot = eval_all[eval_all['features']=="arguablycausal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "arg. causal"
    dic_shift["arguablycausal"] = shift

    #############################################################################
    # plot errorbars and shift gap for robustness tests
    #############################################################################
    for index in range(dic_robust_number_arguablycausal[experiment_name]):
        if (eval_all['features'] == f"test{index}").any():
            eval_plot = eval_all[eval_all['features']==f"test{index}"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            shift = eval_plot[mask]
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = f"test {index}"
            dic_shift[f"test{index}"] = shift
            
    #############################################################################
    # Plot ood accuracy as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    ax[1,0].set_ylabel("Out-of-domain accuracy")

    # add constant shift gap
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift

    shift = pd.concat(dic_shift.values(), ignore_index=True)
    # shift["gap"] = shift["id_test"] - shift["ood_test"]
    barlist = ax[1,0].bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_arguablycausal]+[color_arguablycausal_robust for index in range(dic_robust_number_arguablycausal[experiment_name])]+[color_constant],
                              ecolor=color_error,align='center', capsize=5,
                              bottom=ymin)
    ax[1,0].tick_params(axis='x', labelrotation = 90)

    #############################################################################
    # Plot shift gap as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    ax[1,1].set_ylabel("Shift gap")
    
    shift = pd.concat(dic_shift.values(), ignore_index=True)
    shift["gap"] = shift["id_test"] - shift["ood_test"]
    shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
    shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
    shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
    barlist = ax[1,1].bar(shift["type"], -shift["gap"],
                      yerr=shift['gap_var']**0.5,ecolor=color_error,align='center', capsize=5,
                      color=[color_all,color_arguablycausal]+[color_arguablycausal_robust for index in range(dic_robust_number_arguablycausal[experiment_name])]+[color_constant])
    ax[1,1].tick_params(axis='x', labelrotation = 90)

fig.savefig(str(Path(__file__).parents[0]/f"plots_paper/plot_main_robust.pdf"), bbox_inches='tight')
# %%