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

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

#%%
if __name__ == '__main__':
    experiments=["college_scorecard","college_scorecard_causal"]
    cache_dir="tmp"
    
    eval_all = pd.DataFrame()
    for experiment in experiments:
        file_info = []
        RESULTS_DIR = Path(__file__).parents[0] / experiment
        for filename in os.listdir(RESULTS_DIR):
            if filename == ".DS_Store":
                pass
            else:
                file_info.append(filename)
    
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
                    'features':'causal' if experiment.endswith('causal') else 'all',
                    'model':run.split("_")[0]}])
                eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
    #%%    
    eval_constant = {}
    dset = get_dataset("college_scorecard", cache_dir)
    X, y, _, _ = dset.get_pandas("train")
    majority_class = y.mode()[0]
    for test_split in ["id_test","ood_test"]:
        X_te, y_te, _, _ = dset.get_pandas(test_split)
        count = y_te.value_counts()[majority_class]
        nobs = len(y_te)
        acc = count / nobs
        acc_conf = proportion_confint(count, nobs, alpha=0.05, method='beta')

        eval_constant[test_split] =  acc
        eval_constant[test_split + "_conf"] = acc_conf

    eval_pd = pd.DataFrame([{
            'id_test':eval_constant['id_test'],
            'id_test_lb':eval_constant['id_test_conf'][0],
            'id_test_ub':eval_constant['id_test_conf'][1],
            'ood_test':eval_constant['ood_test'],
            'ood_test_lb':eval_constant['ood_test_conf'][0],
            'ood_test_ub':eval_constant['ood_test_conf'][1],
            'features':'constant',
            'model':run.split("_")[0]}])
    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

        
    eval_all.to_csv(str(Path(__file__).parents[0]/"eval.csv"))
    print(eval_all)
    #%%
    plt.title(
        f"Tableshift: college_scorecard")
    plt.xlabel(f"id accuracy)")
    plt.ylabel(f"ood accuracy")
    colors = pd.factorize(eval_all["model"])[0]
    colormap = cm.get_cmap('tab10')
    for i, features in enumerate(["all","causal","constant"]):
        eval_plot = eval_all[eval_all['features']==features]
        errors = plt.errorbar(
            x=eval_plot['id_test'],
            y=eval_plot['ood_test'],
            xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
            yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
            color=colormap(i), ecolor=colormap(i), label=str(features))
    plt.legend(title="Feature")
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.xlim((eval_constant['id_test']-0.01, 1))
    plt.ylim((eval_constant['ood_test']-0.01,1))
    plt.show()
    plt.savefig(str(Path(__file__).parents[0]/"plot_college_scorecard_causal_vs_all"))


    # %%
    plt.title(
        f"Tableshift: college_scorecard")
    plt.xlabel(f"id accuracy)")
    plt.ylabel(f"ood accuracy")
    colors = pd.factorize(eval_all["model"])[0]
    colormap = cm.get_cmap('tab20')
    eval_causal = eval_all[eval_all['features']=="causal"]
    unique_models = eval_all['model'].unique()
    for j, model in enumerate(unique_models):
        eval_plot = eval_causal[eval_causal['model']==model]
        errors = plt.errorbar(
            x=eval_plot['id_test'],
            y=eval_plot['ood_test'],
            xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
            yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
            color=colormap(j), ecolor=colormap(j), label=str(model))
    for i, features in enumerate(["constant","all"]):
        i = i+j+1
        eval_plot = eval_all[eval_all['features']==features]
        plots = plt.scatter(
            x=eval_plot['id_test'],
            y=eval_plot['ood_test'],
            color=colormap(i))
    plt.fill_between([0, max(eval_plot['id_test'])],
                     [0,0],
                     [max(eval_plot['ood_test']),max(eval_plot['ood_test'])],
                     color=colormap(i), alpha=0.3)
    plt.legend(title="Model (causal)")
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.xlim((eval_constant['id_test']-0.01, 1))
    plt.ylim((eval_constant['ood_test']-0.01,1))
    plt.show()
    plt.savefig(str(Path(__file__).parents[0]/"plot_college_scorecard_causal_models"))

    # %%
    plt.title(
        f"Tableshift: college_scorecard")
    plt.xlabel(f"id accuracy)")
    plt.ylabel(f"ood accuracy")
    colors = pd.factorize(eval_all["model"])[0]
    colormap = cm.get_cmap('tab20')
    unique_models = eval_all['model'].unique()
    for j, model in enumerate(unique_models):
        eval_plot = eval_all[eval_all['model']==model]
        errors = plt.errorbar(
            x=eval_plot['id_test'],
            y=eval_plot['ood_test'],
            xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
            yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
            color=colormap(j), ecolor=colormap(j), label=str(model))
    plt.legend(title="Model")
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.xlim((eval_constant['id_test']-0.01, 1))
    plt.ylim((eval_constant['ood_test']-0.01,1))
    plt.show()
    plt.savefig(str(Path(__file__).parents[0]/"plot_college_scorecard_models"))

# %%
