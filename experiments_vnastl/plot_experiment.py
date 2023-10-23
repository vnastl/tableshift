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

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")
#%%
if __name__ == '__main__':
    # experiment_name = "college_scorecard"
    # experiments=["college_scorecard","college_scorecard_causal","college_scorecard_causal_no_tuition_fee"]
    experiment_name = "acsunemployment"
    experiments = ["acsunemployment","acsunemployment_causal", "acsunemployment_anticausal"]
    # experiment_name = "acspubcov"
    # experiments = ["acspubcov","acspubcov_causal"]
    # experiment_name = "acsfoodstamps"
    # experiments = ["acsfoodstamps","acsfoodstamps_causal"]
    # experiment_name = "physionet"
    # experiments = ["physionet"] #,"physionet_causal", "physionet_anticausal"] 
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
                eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
    #%%    
    eval_constant = {}
    dset = get_dataset(experiment_name, cache_dir)
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
            'model':'constant'}])
    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

        
    eval_all.to_csv(str(Path(__file__).parents[0]/f"{experiment_name}_eval.csv"))
    print(eval_all)
    #%%
    plt.title(
        f"Tableshift: {experiment_name}")
    plt.xlabel(f"id accuracy)")
    plt.ylabel(f"ood accuracy")
    colors = pd.factorize(eval_all["model"])[0]
    colormap = cm.get_cmap('tab10')
    for i, features in enumerate(feature_selection):
        eval_plot = eval_all[eval_all['features']==features]
        errors = plt.errorbar(
            x=eval_plot['id_test'],
            y=eval_plot['ood_test'],
            xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
            yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
            color=colormap(i), ecolor=colormap(i), label=str(features))
    plt.legend(title="Feature")
    plt.fill_between([0, eval_all[eval_all['ood_test']==max(eval_all['ood_test'])]['id_test'].values[0]],
                            [0,0],
                            [max(eval_all['ood_test']),max(eval_all['ood_test'])],
                            color=colormap(19), alpha=0.2)
    plt.fill_between([0, max(eval_all['id_test'])],
                            [0,0],
                            [eval_all[eval_all['id_test']==max(eval_all['id_test'])]['ood_test'].values[0],
                             eval_all[eval_all['id_test']==max(eval_all['id_test'])]['ood_test'].values[0]],
                            color=colormap(19), alpha=0.2)
    
    eval_plot = eval_all[eval_all['features']=="constant"]
    plt.fill_between([0, min(eval_plot['id_test'])],0,1,
                        color="tab:grey")
    plt.fill_between([0, 1],0,max(eval_plot['ood_test']),
                        color="tab:grey")
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    # plt.xlim((eval_constant['id_test']-0.01, max(eval_all['id_test'])+0.01))
    # plt.ylim((eval_constant['ood_test']-0.01,max(eval_all['ood_test'])+0.01))
    plt.savefig(str(Path(__file__).parents[0]/f"plot_{experiment_name}_causal_vs_all"))
    plt.show()


    # %%
    if "causal" in feature_selection:
        plt.title(
            f"Tableshift: {experiment_name}")
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
        
        eval_plot = eval_all[eval_all['features']=="all"]
        errors = plt.errorbar(
                x=eval_plot['id_test'],
                y=eval_plot['ood_test'],
                xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
                yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
                color=colormap(19), ecolor=colormap(19), label=str(model), alpha=0.3)
        plt.fill_between([0, max(eval_plot['id_test'])],
                            [0,0],
                            [max(eval_plot['ood_test']),max(eval_plot['ood_test'])],
                            color=colormap(19), alpha=0.3)
        
        
        eval_plot = eval_all[eval_all['features']=="constant"]
        plt.fill_between([0, min(eval_plot['id_test'])],0,1,
                            color="tab:grey")
        plt.fill_between([0, 1],0,max(eval_plot['ood_test']),
                            color="tab:grey")

        plt.legend(title="Model (causal)")
        # Plot the diagonal line
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim((eval_constant['id_test']-0.01, max(eval_all['id_test'])+0.01))
        plt.ylim((eval_constant['ood_test']-0.01,max(eval_all['ood_test'])+0.01))
        plt.savefig(str(Path(__file__).parents[0]/f"plot_{experiment_name}_causal_models"))
        plt.show()
    
    # %%
    if "causal without tuition" in feature_selection:
        plt.title(
            f"Tableshift: {experiment_name}")
        plt.xlabel(f"id accuracy)")
        plt.ylabel(f"ood accuracy")
        colors = pd.factorize(eval_all["model"])[0]
        colormap = cm.get_cmap('tab20')
        eval_causal = eval_all[eval_all['features']=="causal without tuition"]
        unique_models = eval_all['model'].unique()
        for j, model in enumerate(unique_models):
            eval_plot = eval_causal[eval_causal['model']==model]
            errors = plt.errorbar(
                x=eval_plot['id_test'],
                y=eval_plot['ood_test'],
                xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
                yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
                color=colormap(j), ecolor=colormap(j), label=str(model))
        
        eval_plot = eval_all[eval_all['features']=="all"]
        errors = plt.errorbar(
                x=eval_plot['id_test'],
                y=eval_plot['ood_test'],
                xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
                yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
                color=colormap(19), ecolor=colormap(19), label=str(model), alpha=0.3)
        plt.fill_between([0, max(eval_plot['id_test'])],
                            [0,0],
                            [max(eval_plot['ood_test']),max(eval_plot['ood_test'])],
                            color=colormap(20), alpha=0.3)
        
        eval_plot = eval_all[eval_all['features']=="constant"]
        plt.fill_between([0, min(eval_plot['id_test'])],0,1,
                            color="tab:grey")
        plt.fill_between([0, 1],0,max(eval_plot['ood_test']),
                            color="tab:grey")

        plt.legend(title="Model (causal with no tuition)")
        # Plot the diagonal line
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim((eval_constant['id_test']-0.01, max(eval_all['id_test'])+0.01))
        plt.ylim((eval_constant['ood_test']-0.01,max(eval_all['ood_test'])+0.01))
        plt.savefig(str(Path(__file__).parents[0]/f"plot_{experiment_name}_causal_models_without_tuition"))
        plt.show()
    
    if "anticausal" in feature_selection:
        plt.title(
            f"Tableshift: {experiment_name}")
        plt.xlabel(f"id accuracy)")
        plt.ylabel(f"ood accuracy")
        colors = pd.factorize(eval_all["model"])[0]
        colormap = cm.get_cmap('tab20')
        eval_causal = eval_all[eval_all['features']=="anticausal"]
        unique_models = eval_all['model'].unique()
        for j, model in enumerate(unique_models):
            eval_plot = eval_causal[eval_causal['model']==model]
            errors = plt.errorbar(
                x=eval_plot['id_test'],
                y=eval_plot['ood_test'],
                xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
                yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
                color=colormap(j), ecolor=colormap(j), label=str(model))
        
        eval_plot = eval_all[eval_all['features']=="all"]
        errors = plt.errorbar(
                x=eval_plot['id_test'],
                y=eval_plot['ood_test'],
                xerr=eval_plot['id_test_ub']-eval_plot['id_test'],
                yerr=eval_plot['ood_test_ub']-eval_plot['ood_test'], fmt="o",
                color=colormap(19), ecolor=colormap(19), label=str(model), alpha=0.3)
        plt.fill_between([0, max(eval_plot['id_test'])],
                            [0,0],
                            [max(eval_plot['ood_test']),max(eval_plot['ood_test'])],
                            color=colormap(20), alpha=0.3)
        
        eval_plot = eval_all[eval_all['features']=="constant"]
        plt.fill_between([0, min(eval_plot['id_test'])],0,1,
                            color="tab:grey")
        plt.fill_between([0, 1],0,max(eval_plot['ood_test']),
                            color="tab:grey")

        plt.legend(title="Model (anticausal)")
        # Plot the diagonal line
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim((eval_constant['id_test']-0.01, max(eval_all['id_test'])+0.01))
        plt.ylim((eval_constant['ood_test']-0.01,max(eval_all['ood_test'])+0.01))
        plt.savefig(str(Path(__file__).parents[0]/f"plot_{experiment_name}_anticausal_models"))
        plt.show()

    # %%
    plt.title(
        f"Tableshift: {experiment_name}")
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
            color=colormap(j), ecolor=colormap(j), label=str(model), alpha=0.5)
    plt.legend(title="Model")
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    eval_plot = eval_all[eval_all['features']=="constant"]
    plt.fill_between([0, min(eval_plot['id_test'])],0,1,
                        color="tab:grey")
    plt.fill_between([0, 1],0,max(eval_plot['ood_test']),
                        color="tab:grey")
    plt.xlim((eval_constant['id_test']-0.01, max(eval_all['id_test'])+0.01))
    plt.ylim((eval_constant['ood_test']-0.01,max(eval_all['ood_test'])+0.01))
    plt.savefig(str(Path(__file__).parents[0]/f"plot_{experiment_name}_models"))
    plt.show()
    

# %%
