#%%
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap

from tableshift import get_dataset
from  statsmodels.stats.proportion import proportion_confint
from paretoset import paretoset
from scipy.spatial import ConvexHull

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")
#%%
# experiment_name = "anes"
# experiments = ["anes","anes_causal"]
# domain_label = 'VCF0112'

# experiment_name = "college_scorecard"
# experiments=["college_scorecard","college_scorecard_causal","college_scorecard_causal_no_tuition_fee"]
# domain_label = 'CCBASIC'

experiment_name = "brfss_diabetes"
experiments = ["brfss_diabetes","brfss_diabetes_causal","brfss_diabetes_anticausal"]
domain_label = 'PRACE1'

# experiment_name = "acsemployment"
# experiments = ["acsemployment","acsemployment_causal", "acsemployment_anticausal"]
# domain_label ='SCHL'

# experiment_name = "acsunemployment"
# experiments = ["acsunemployment","acsunemployment_causal", "acsunemployment_anticausal"]
# domain_label ='SCHL'

# experiment_name = "acspubcov"
# experiments = ["acspubcov","acspubcov_causal"]
# domain_label = 'DIS'

# experiment_name = "acsfoodstamps"
# experiments = ["acsfoodstamps","acsfoodstamps_causal"]
# domain_label = 'DIVISION'

# experiment_name = "physionet"
# experiments = ["physionet","physionet_causal", "physionet_anticausal"] 
# domain_label = 'ICULOS'


cache_dir="tmp"

eval_all = pd.DataFrame()
feature_selection = []
for experiment in experiments:
    file_info = []
    RESULTS_DIR = Path(__file__).parents[0] / experiment
    for filename in os.listdir(RESULTS_DIR):
        if filename == ".DS_Store":
            pass
        else:
            file_info.append(filename)

    def get_feature_selection(experiment):
        if experiment.endswith('_causal'):
            if 'causal' not in feature_selection: 
                feature_selection.append('causal') 
            return 'causal'
        elif experiment.endswith('_causal_no_tuition_fee'):
            if 'causal without tuition' not in feature_selection: 
                feature_selection.append('causal without tuition')
            return 'causal without tuition'
        elif experiment.endswith('_anticausal'):
            if 'anticausal' not in feature_selection: 
                feature_selection.append('anticausal')
            return 'anticausal'
        else:
            if 'all' not in feature_selection: 
                feature_selection.append('all')
            return 'all'
    
    for run in file_info:
        with open(str(RESULTS_DIR / run), "rb") as file:
            print(str(RESULTS_DIR / run))
            eval_json = json.load(file)
            eval_pd = pd.DataFrame([{
                'id_test':eval_json['id_test'],
                'id_test_lb':eval_json['id_test_conf'][0],
                'id_test_ub':eval_json['id_test_conf'][1],
                'ood_test':eval_json['ood_test'],
                'ood_test_lb':eval_json['ood_test_conf'][0],
                'ood_test_ub':eval_json['ood_test_conf'][1],
                'features': get_feature_selection(experiment),
                'model':run.split("_")[0]}])
            if get_feature_selection(experiment) == 'causal':
                causal_features = eval_json['features']
                causal_features.remove(domain_label)
            if get_feature_selection(experiment) == 'causal without tuition' or get_feature_selection(experiment) == 'anticausal':
                extra_features = eval_json['features']
                extra_features.remove(domain_label)
            eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
#%%
RESULTS_DIR = Path(__file__).parents[0]
filename = f"{experiment_name}_constant"
if filename in os.listdir(RESULTS_DIR):
    with open(str(RESULTS_DIR / filename), "rb") as file:
            print(str(RESULTS_DIR / filename))
            eval_constant = json.load(file)
else:
    eval_constant = {}
    dset = get_dataset(experiment_name, cache_dir)
    for test_split in ["id_test","ood_test"]:
        X_te, y_te, _, _ = dset.get_pandas(test_split)
        majority_class = y_te.mode()[0]
        count = y_te.value_counts()[majority_class]
        nobs = len(y_te)
        acc = count / nobs
        acc_conf = proportion_confint(count, nobs, alpha=0.05, method='beta')

        eval_constant[test_split] =  acc
        eval_constant[test_split + "_conf"] = acc_conf
    with open(str(RESULTS_DIR / filename), "w") as file:
        json.dump(eval_constant, file)

eval_pd = pd.DataFrame([{
        'id_test':eval_constant['id_test'],
        'id_test_lb':eval_constant['id_test_conf'][0],
        'id_test_ub':eval_constant['id_test_conf'][1],
        'ood_test':eval_constant['ood_test'],
        'ood_test_lb':eval_constant['ood_test_conf'][0],
        'ood_test_ub':eval_constant['ood_test_conf'][1],
        'features':'constant',
        'model':'constant'}])
eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    
eval_all.to_csv(str(Path(__file__).parents[0]/f"{experiment_name}_eval.csv"))
print(eval_all)

#%%
def do_plot(mymin,mymax,mytextx,mytexty,myname,axmin=[0.5,0.5],axmax=[1.0,1.0]):
    plt.title(
        f"Tableshift: {experiment_name}")
    plt.xlabel(f"id accuracy")
    plt.ylabel(f"ood accuracy")

    ## All features
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    markers = eval_plot[mask]
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                color="tab:blue", ecolor="tab:blue", label="top all features")
    # get extra points for the plot
    new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    plt.plot(points['id_test'],points['ood_test'],color="tab:blue",linestyle="dotted")

    new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color="tab:blue",alpha=0.3)
    
    ## Causal features
    eval_plot = eval_all[eval_all['features']=="causal"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    markers = eval_plot[mask]
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                color="tab:orange", ecolor="tab:orange", label="top causal features")
    # get extra points for the plot
    new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    # dotted = points.to_numpy()
    # hull = ConvexHull(dotted,incremental=True)
    # plt.plot(dotted[list(hull.vertices[1:])+[hull.vertices[0]], 0], dotted[list(hull.vertices[1:])+[hull.vertices[0]], 1],
    #          color="tab:orange",linestyle="dotted")
    # plt.plot(dotted[hull.vertices, 0][1:], dotted[hull.vertices, 1][1:],color="tab:orange",linestyle="dotted")
    points.sort_values('id_test',inplace=True)
    plt.plot(points['id_test'],points['ood_test'],color="tab:orange",linestyle="dotted")

    new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    filled = points.to_numpy()
    hull = ConvexHull(filled,incremental=True)
    plt.fill(filled[hull.vertices, 0], filled[hull.vertices, 1], color="tab:orange",alpha=0.3)
    
    ## Extra set of features
    if experiment_name == "college_scorecard":
        eval_plot = eval_all[eval_all['features']=="causal without tuition"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        markers = eval_plot[mask]
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                    color="tab:pink", ecolor="tab:pink", label="top causal features without tuition")
        # get extra points for the plot
        new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        # dotted = points.to_numpy()
        # hull = ConvexHull(dotted,incremental=True)
        # plt.plot(dotted[list(hull.vertices[1:])+[hull.vertices[0]], 0], dotted[list(hull.vertices[1:])+[hull.vertices[0]], 1],
        #          color="tab:pink",linestyle="dotted")
        # plt.plot(dotted[hull.vertices, 0][1:], dotted[hull.vertices, 1][1:],color="tab:pink",linestyle="dotted")
        points.sort_values('id_test',inplace=True)
        plt.plot(points['id_test'],points['ood_test'],color="tab:pink",linestyle="dotted")

        new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        filled = points.to_numpy()
        hull = ConvexHull(filled,incremental=True)
        plt.fill(filled[hull.vertices, 0], filled[hull.vertices, 1], color="tab:pink",alpha=0.3)
    
    if experiment_name in ["acsemployment", "acsunemployment", "physionet"]:
        eval_plot = eval_all[eval_all['features']=="anticausal"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        markers = eval_plot[mask]
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                    color="tab:purple", ecolor="tab:purple", label="top anticausal features")
        # get extra points for the plot
        new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        # dotted = points.to_numpy()
        # hull = ConvexHull(dotted,incremental=True)
        # plt.plot(dotted[list(hull.vertices[1:])+[hull.vertices[0]], 0], dotted[list(hull.vertices[1:])+[hull.vertices[0]], 1],
        #          color="tab:purple",linestyle="dotted")
        # plt.plot(dotted[hull.vertices, 0][1:], dotted[hull.vertices, 1][1:],color="tab:purple",linestyle="dotted")
        points.sort_values('id_test',inplace=True)
        plt.plot(points['id_test'],points['ood_test'],color="tab:purple",linestyle="dotted")

        new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        filled = points.to_numpy()
        hull = ConvexHull(filled,incremental=True)
        plt.fill(filled[hull.vertices, 0], filled[hull.vertices, 1], color="tab:purple",alpha=0.3)

    ## Constant
    eval_plot = eval_all[eval_all['features']=="constant"]
    plot_constant = plt.plot(
            eval_plot['id_test'],
            eval_plot['ood_test'],
            marker="D",linestyle="None",
            color="tab:red", 
            label="constant")
    # errors = plt.errorbar(
    #         x=eval_plot['id_test'],
    #         y=eval_plot['ood_test'],
    #         xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
    #         yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="D",
    #         color="tab:red", ecolor="tab:red", label="constant")
    plt.plot([0, eval_plot['id_test'].values[0]],
                [eval_plot['ood_test'].values[0],eval_plot['ood_test'].values[0]],
                color="tab:red",linestyle="dotted")
    plt.plot([eval_plot['id_test'].values[0], eval_plot['id_test'].values[0]],
                [0,eval_plot['ood_test'].values[0]],
                color="tab:red",linestyle="dotted")
    plt.fill_between([0, eval_plot['id_test'].values[0]],
                        [0,0],
                        [eval_plot['ood_test'].values[0],eval_plot['ood_test'].values[0]],
                        color="tab:red", alpha=0.1)
    # eval_plot = eval_all[eval_all['features']=="constant"]


    # plt.fill_between([0, max(eval_plot['id_test'])],
    #                         [0,0],
    #                         [eval_all[eval_all['id_test']==max(eval_all['id_test'])]['ood_test'].values[0],
    #                          eval_all[eval_all['id_test']==max(eval_all['id_test'])]['ood_test'].values[0]],
    #                         color=colormap(19), alpha=0.2)

    # Get the lines and labels
    lines, labels = plt.gca().get_legend_handles_labels()

    # Remove duplicates
    newLabels, newLines = [], []
    for line, label in zip(lines, labels):
        if label not in newLabels:
            newLabels.append(label)
            newLines.append(line)

    # Create a legend with only distinct labels
    plt.legend(newLines, newLabels, title="Feature")

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black')

    plt.xlim((axmin[0],axmax[0]))
    plt.ylim((axmin[1],axmax[1]))

    # Add text below the plot
    
    if experiment_name == "acsunemployment" or experiment_name == "physionet":
        plt.text(mytextx, mytexty,f'Causal features: {causal_features} \n Anticausal features: {extra_features}')
    else:
        plt.text(mytextx, mytexty,f'Causal features: {causal_features}')

    plt.savefig(str(Path(__file__).parents[0]/myname), bbox_inches='tight')
    plt.show()

# %%
if experiment_name == "acsfoodstamps":
    mymin = 0.5
    mymax = 1
    mytextx = 0.5
    mytexty = 0.4
    myname = f"plots/plot_{experiment_name}"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "acspubcov":
    mymin = 0.2
    mymax = 1
    mytextx = 0.2
    mytexty = 0.05
    myname = f"plots/plot_{experiment_name}"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "acsemployment":
    mymin = 0.45
    mymax = 1
    mytextx = 0.45
    mytexty = 0.32
    myname = f"plots/plot_folktable_acsemployment"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "acsunemployment":
    mymin = 0.5
    mymax = 1
    mytextx = 0.5
    mytexty = 0.4
    myname = f"plots/plot_acsunemployment"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "anes":
    mymin = 0.5
    mymax = 1
    mytextx = 0.5
    mytexty = 0.4
    myname = f"plots/plot_{experiment_name}"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "college_scorecard":
    mymin = 0.5
    mymax = 1
    mytextx = 0.5
    mytexty = 0.4
    myname = f"plots/plot_{experiment_name}"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "brfss_diabetes":
    mymin = 0.5
    mymax = 1
    mytextx = 0.5
    mytexty = 0.4
    myname = f"plots/plot_{experiment_name}"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "physionet":
    mymin = 0.5
    mymax = 1
    mytextx = 0.5
    mytexty = 0.4
    myname = f"plots/plot_{experiment_name}"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

 # %%
if experiment_name == "acsfoodstamps":
    mymin = 0.75
    mymax = 0.86
    mytextx = 0.75
    mytexty = 0.73
    myname = f"plots/plot_{experiment_name}_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "acspubcov":
    mymin = 0.2
    axminx = 0.58
    axminy = 0.35
    mymax = 0.83
    mytextx = 0.58
    mytexty = 0.25
    myname = f"plots/plot_{experiment_name}_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

elif experiment_name == "acsemployment":
    mymin = 0.90
    axmin = 0.94
    mymax = 1
    mytextx = 0.94
    mytexty = 0.925
    myname = f"plots/plot_folktable_acsemployment_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[axmin,axmin],[mymax,mymax])

elif experiment_name == "acsunemployment":
    mymin = 0.90
    axminx = 0.96
    axminy = 0.94
    mymax = 0.98
    mytextx = 0.96
    mytexty = 0.93
    myname = f"plots/plot_acsunemployment_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

if experiment_name == "anes":
    mymin = 0.58
    mymax = 0.85
    mytextx = 0.58
    mytexty = 0.53
    myname = f"plots/plot_{experiment_name}_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

elif experiment_name == "college_scorecard":
    mymin = 0.65
    axminx = 0.86
    axminy = 0.65 
    mymax = 0.96
    mytextx = 0.86
    mytexty = 0.58
    myname = f"plots/plot_{experiment_name}_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

elif experiment_name == "physionet":
    mymin = 0.90
    axminx = 0.985
    axminy = 0.92
    axmaxy = 0.93
    mymax = 0.989
    mytextx = 0.985
    mytexty = 0.918
    myname = f"plots/plot_{experiment_name}_zoom"

    do_plot(mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,axmaxy])


# %%
