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
import seaborn as sns
sns.set_context("paper", font_scale=1.9)


from tableshift import get_dataset
from  statsmodels.stats.proportion import proportion_confint
from paretoset import paretoset
from scipy.spatial import ConvexHull

from experiments_vnastl.plot_config_colors import *
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from tqdm import tqdm

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")
#%%

ANTICAUSAL = False

# %%
def get_dic_experiments_value(name):
    return [name, f"{name}_causal", f"{name}_arguablycausal"]
def get_dic_experiments_value_anticausal(name):
    return [name, f"{name}_causal", f"{name}_arguablycausal",f"{name}_anticausal"]
if ANTICAUSAL:
    dic_experiments = {
        "acsincome": get_dic_experiments_value_anticausal("acsincome"),
        "acsunemployment": get_dic_experiments_value_anticausal("acsunemployment"),
        "brfss_diabetes": get_dic_experiments_value_anticausal("brfss_diabetes"),
        "brfss_blood_pressure": get_dic_experiments_value_anticausal("brfss_blood_pressure"),
        "sipp": get_dic_experiments_value_anticausal("sipp")
    }
else:
    dic_experiments = {
        "acsemployment": get_dic_experiments_value("acsemployment"),
        "acsfoodstamps":  get_dic_experiments_value("acsfoodstamps"),
        "acsincome": get_dic_experiments_value("acsincome"),
        "acspubcov":  get_dic_experiments_value("acspubcov"),
        "acsunemployment":  get_dic_experiments_value("acsunemployment"),
        "anes": get_dic_experiments_value("anes"),
        "assistments":  get_dic_experiments_value("assistments"),
        "brfss_blood_pressure": get_dic_experiments_value("brfss_blood_pressure"),
        "brfss_diabetes": get_dic_experiments_value("brfss_diabetes"),
        "college_scorecard":  get_dic_experiments_value("college_scorecard"),
        "diabetes_readmission": get_dic_experiments_value("diabetes_readmission"),
        "meps":  get_dic_experiments_value("meps"),
        "mimic_extract_los_3":  get_dic_experiments_value("mimic_extract_los_3"),
        "mimic_extract_mort_hosp":  get_dic_experiments_value("mimic_extract_mort_hosp"),
        "nhanes_lead":  get_dic_experiments_value("nhanes_lead"),
        "physionet": get_dic_experiments_value("physionet"),
        "sipp":  get_dic_experiments_value("sipp"),
    }

dic_domain_label = {
    "acsemployment":'SCHL',
    "acsfoodstamps": 'DIVISION',
    "acsincome": 'DIVISION',
    "acspubcov": 'DIS',
    "acsunemployment": 'SCHL',
    "anes": 'VCF0112', # region
    "assistments": 'school_id',
    "brfss_blood_pressure":'BMI5CAT',
    "brfss_diabetes": 'PRACE1',
    "college_scorecard": 'CCBASIC',
    "diabetes_readmission": 'admission_source_id',
    "meps": 'INSCOV19',
    "mimic_extract_los_3": 'insurance',
    "mimic_extract_mort_hosp": 'insurance',
    "nhanes_lead": 'INDFMPIRBelowCutoff',
    "physionet": 'ICULOS', # ICU length of stay
    "sipp": 'CITIZENSHIP_STATUS',
}

dic_id_domain = {
    "acsemployment":'High school diploma or higher',
    "acsfoodstamps": 'Other U.S. Census divisions',
    "acsincome": 'Other U.S. Census divisions', # Mid-Atlantic, East North Central, West North Central, South Atlantic, East South Central, West South Central, Mountain, Pacific
    "acspubcov": 'Without disability',
    "acsunemployment": 'High school diploma or higher',
    "anes": 'Other U.S. Census regions', # region
    "assistments": 'approximately 700 schools',
    "brfss_blood_pressure":'Underweight and normal weight',
    "brfss_diabetes": 'White',
    "college_scorecard": 'Carnegie Classification: other institutional types',
    "diabetes_readmission": 'Other admission sources',
    "meps": 'Public insurance',
    "mimic_extract_los_3": 'Private, Medicaid, Government, Self Pay',
    "mimic_extract_mort_hosp": 'Private, Medicaid, Government, Self Pay',
    "nhanes_lead": 'poverty-income ratio > 1.3',
    "physionet": 'ICU length of stay <= 47 hours', # ICU length of stay
    "sipp": 'U.S. citizen',
}

dic_ood_domain = {
    "acsemployment":'No high school diploma',
    "acsfoodstamps": 'East South Central',
    "acsincome": 'New England',
    "acspubcov": 'With disability',
    "acsunemployment": 'No high school diploma',
    "anes": 'South', # region
    "assistments": '10 new schools',
    "brfss_blood_pressure":'Overweight and obese',
    "brfss_diabetes": 'Non white',
    "college_scorecard": "Special Focus Institutions [Faith-related, art & design and other fields],\n Baccalaureate/Associates Colleges,\n Master's Colleges and Universities [larger programs]",
    "diabetes_readmission": 'Emergency Room',
    "meps": 'Private insurance',
    "mimic_extract_los_3": 'Medicare',
    "mimic_extract_mort_hosp": 'Medicare',
    "nhanes_lead": 'poverty-income ratio <= 1.3',
    "physionet": 'ICU length of stay > 47 hours', # ICU length of stay
    "sipp": 'non U.S. citizen',
}

dic_title = {
    "acsemployment":'Tableshift: Employment',
    "acsfoodstamps": 'Tableshift: Food Stamps',
    "acsincome": 'Tableshift: Income',
    "acspubcov": 'Tableshift: PublicCoverage',
    "acsunemployment": 'Tableshift: Unemployment',
    "anes": 'Tableshift: Voting',
    "assistments": 'Tableshift: ASSISTments',
    "brfss_blood_pressure":'Tableshift: Hypertension',
    "brfss_diabetes": 'Tableshift: Diabetes',
    "college_scorecard": 'Tableshift: College Scorecard',
    "diabetes_readmission": 'Tableshift: Hospital Readmission',
    "meps": 'MEPS: Utilization',
    "mimic_extract_los_3": 'Tableshift: ICU Length of Stay',
    "mimic_extract_mort_hosp": 'Tableshift: Hospital Mortality',
    "nhanes_lead": 'Tableshift: Childhood Lead',
    "physionet": 'Tableshift: Sepsis', # ICU length of stay
    "sipp": 'SIPP: Poverty',
}

sns.set_style("white")

def get_results(experiment_name):
    cache_dir="tmp"
    experiments = dic_experiments[experiment_name]
    domain_label = dic_domain_label[experiment_name]

    eval_all = pd.DataFrame()
    feature_selection = []
    for experiment in experiments:
        file_info = []
        RESULTS_DIR = Path(__file__).parents[0] / "results" / experiment
        for filename in tqdm(os.listdir(RESULTS_DIR)):
            if filename == ".DS_Store":
                pass
            else:
                file_info.append(filename)

        def get_feature_selection(experiment):
            if experiment.endswith('_causal'):
                if 'causal' not in feature_selection: 
                    feature_selection.append('causal') 
                return 'causal'
            elif experiment.endswith('_arguablycausal'):
                if 'arguablycausal' not in feature_selection: 
                    feature_selection.append('arguablycausal')
                return 'arguablycausal'
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
                # print(str(RESULTS_DIR / run))
                eval_json = json.load(file)
                eval_pd = pd.DataFrame([{
                    'id_test':eval_json['id_test'],
                    'id_test_lb':eval_json['id_test' + '_conf'][0],
                    'id_test_ub':eval_json['id_test' + '_conf'][1],
                    'ood_test':eval_json['ood_test'],
                    'ood_test_lb':eval_json['ood_test' + '_conf'][0],
                    'ood_test_ub':eval_json['ood_test' + '_conf'][1],
                    'validation':eval_json['validation'],
                    'features': get_feature_selection(experiment),
                    'model':run.split("_")[0]}])
                if get_feature_selection(experiment) == 'causal':
                    causal_features = eval_json['features']
                    causal_features.remove(domain_label)
                if get_feature_selection(experiment) == 'arguablycausal' or get_feature_selection(experiment) == 'anticausal':
                    extra_features = eval_json['features']
                    extra_features.remove(domain_label)
                else:
                    extra_features = []
                eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    RESULTS_DIR = Path(__file__).parents[0] / "results" 
    filename = f"{experiment_name}_constant"
    if filename in os.listdir(RESULTS_DIR):
        with open(str(RESULTS_DIR / filename), "rb") as file:
                # print(str(RESULTS_DIR / filename))
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

    list_model_data = []
    for set in eval_all['features'].unique():
        eval_feature = eval_all[eval_all['features']==set]
        for model in eval_feature['model'].unique():
            model_data = eval_feature[eval_feature['model']==model]
            model_data = model_data[model_data["validation"] == model_data["validation"].max()]
            list_model_data.append(model_data)
    eval_all = pd.concat(list_model_data)
    
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
    # print(eval_all)
    return eval_all, causal_features, extra_features

#%%

def do_plot(experiment_name,mymin,myname):
    if experiment_name not in dic_experiments.keys():
        if ANTICAUSAL:
            print(f"There are no anticausal features for {experiment_name}")
            return None
        else:
            ValueError(f"There is no experiment named {experiment_name}")
    eval_all, causal_features, extra_features = get_results(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}
    dic_shift_acc ={}

    # plt.title(
    #     f"{dic_title[experiment_name]}")
    plt.xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    plt.ylabel(f"out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = plt.errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt="D",
            color=color_constant, ecolor=color_constant,
            markersize=7, capsize=3, label="constant")
    plt.hlines(y=eval_constant['ood_test'].values[0], xmin=eval_constant['ood_test'].values[0], xmax=eval_constant['id_test'].values[0],
                color=color_constant, linewidth=3, alpha=0.7)
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
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                color=color_all, ecolor=color_all,
                markersize=7, capsize=3, label="all")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_all, linewidth=3, alpha=0.7)
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
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                color=color_causal, ecolor=color_causal,
                markersize=7, capsize=3, label="causal")
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_causal, linewidth=3, alpha=0.7)
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
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="^",
                    color=color_arguablycausal, ecolor=color_arguablycausal,
                    markersize=7, capsize=3, label="arguably\ncausal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "arguably\ncausal"
        dic_shift["arguablycausal"] = shift
        plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_arguablycausal, linewidth=3, alpha=0.7)
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
        if not myname.endswith("zoom"):
            print(markers["model"].values)
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="P",
                    color=color_anticausal, ecolor=color_anticausal,
                    markersize=7, capsize=3, label="anticausal")
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "anti\ncausal"
        dic_shift["anticausal"] = shift
        plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_anticausal, linewidth=3, alpha=0.7)

    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, eval_constant['id_test'].values[0]],
                [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle="dotted")
    plt.plot([eval_constant['id_test'].values[0], eval_constant['id_test'].values[0]],
                [ymin,eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle="dotted")
    plt.fill_between([xmin, eval_constant['id_test'].values[0]],
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
    plt.plot(points['id_test'],points['ood_test'],color=color_all,linestyle="dotted")
    new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.1)

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
    plt.plot(points['id_test'],points['ood_test'],color=color_causal,linestyle="dotted")
    new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal,alpha=0.1)

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
        plt.plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle="dotted")
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)

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
        plt.plot(points['id_test'],points['ood_test'],color=color_anticausal,linestyle="dotted")

        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_anticausal,alpha=0.1)
    
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Get the lines and labels
    lines, labels = plt.gca().get_legend_handles_labels()

    # Remove duplicates
    newLabels, newLines = [], []
    for line, label in zip(lines, labels):
        if label not in newLabels:
            newLabels.append(label)
            newLines.append(line)

    # Create a legend with only distinct labels
    plt.legend(newLines, newLabels, loc='upper left')

    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    plt.plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)
    
    if (eval_all['features'] == "anticausal").any():
        plt.savefig(f"{str(Path(__file__).parents[0]/myname)}_anticausal.pdf", bbox_inches='tight')
        plt.show()
    else:
        plt.savefig(f"{str(Path(__file__).parents[0]/myname)}.pdf", bbox_inches='tight')
        plt.show()

    #############################################################################
    # Plot ood accuracy as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    plt.ylabel("out-of-domain accuracy")

    # add constant shift gap
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift

    shift = pd.concat(dic_shift.values(), ignore_index=True)
    # shift["gap"] = shift["id_test"] - shift["ood_test"]
    if (eval_all['features'] == "arguablycausal").any():
        if (eval_all['features'] == "anticausal").any():
            barlist = plt.bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_causal,color_arguablycausal,color_anticausal,color_constant],
                              ecolor=color_error,align='center', capsize=10,
                              bottom=ymin)
            plt.savefig(str(Path(__file__).parents[0]/f"{myname}_anticausal_ood_accuracy.pdf"), bbox_inches='tight')
            plt.show()
        else:
            barlist = plt.bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_causal,color_arguablycausal,color_constant],
                              ecolor=color_error,align='center', capsize=10,
                              bottom=ymin)
            plt.savefig(str(Path(__file__).parents[0]/f"{myname}_ood_accuracy.pdf"), bbox_inches='tight')
            plt.show()
    else:
        barlist = plt.bar(shift["type"], shift["ood_test"], color=[color_all,color_causal,color_constant])
        plt.savefig(str(Path(__file__).parents[0]/f"{myname}_ood_accuracy.pdf"), bbox_inches='tight')
        plt.show()


    #############################################################################
    # Plot shift gap as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    plt.ylabel("shift gap")

    # # add constant shift gap
    # shift = eval_constant
    # shift["type"] = "constant"
    # dic_shift["constant"] = shift

    # shift = pd.concat(dic_shift.values(), ignore_index=True)
    shift["gap"] = shift["id_test"] - shift["ood_test"]
    shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
    shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
    shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
    if (eval_all['features'] == "arguablycausal").any():
        if (eval_all['features'] == "anticausal").any():
            barlist = plt.bar(shift["type"], shift["gap"],
                              yerr=shift['gap_var']**0.5,
                              color=[color_all,color_causal,color_arguablycausal,color_anticausal,color_constant],
                              ecolor=color_error,align='center', capsize=10)
            plt.savefig(str(Path(__file__).parents[0]/f"{myname}_anticausal_shift.pdf"), bbox_inches='tight')
            plt.show()
        else:
            barlist = plt.bar(shift["type"], shift["gap"],
                              yerr=shift['gap_var']**0.5,
                              color=[color_all,color_causal,color_arguablycausal,color_constant],
                              ecolor=color_error,align='center', capsize=10)
            plt.savefig(str(Path(__file__).parents[0]/f"{myname}_shift.pdf"), bbox_inches='tight')
            plt.show()
    else:
        barlist = plt.bar(shift["type"], shift["gap"], color=[color_all,color_causal,color_constant])
        plt.savefig(str(Path(__file__).parents[0]/f"{myname}_shift.pdf"), bbox_inches='tight')
        plt.show()

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    if (eval_all['features'] == "arguablycausal").any():
        # plt.title(
        # f"{dic_title[experiment_name]}")
        plt.xlabel("1 - shift gap")
        plt.ylabel("out-of-domain accuracy")
        shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
        markers = {'constant': 'D','all': 's', 'causal': 'o', 'arguablycausal':'^'}
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc['type']==type]
            type_shift['id_test_var'] = ((type_shift['id_test_ub']-type_shift['id_test']))**2
            type_shift['ood_test_var'] = ((type_shift['ood_test_ub']-type_shift['ood_test']))**2
            type_shift['gap_var'] = type_shift['id_test_var']+type_shift['ood_test_var']

            # Get markers
            plt.errorbar(x=1-type_shift["gap"],
                         y=type_shift["ood_test"],
                         xerr= type_shift['gap_var']**0.5,
                         yerr= type_shift['ood_test_ub']-type_shift['ood_test'],
                        color=eval(f"color_{type}"), ecolor=eval(f"color_{type}"),
                        fmt=marker, markersize=7, capsize=3,  label="arguably\ncausal" if type == 'arguablycausal' else f"{type}",
                        zorder=3)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
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
            plt.plot(points['1-gap'],points['ood_test'],color=eval(f"color_{type}"),linestyle="dotted")
            new_row = pd.DataFrame({'1-gap':[xmin], 'ood_test':[ymin]},)
            points = pd.concat([points,new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{type}"),alpha=0.1)
            
        # Get the lines and labels
        lines, labels = plt.gca().get_legend_handles_labels()

        # Remove duplicates
        newLabels, newLines = [], []
        for line, label in zip(lines, labels):
            if label not in newLabels:
                newLabels.append(label)
                newLines.append(line)
            
        # Create a legend with only distinct labels
        plt.legend(newLines, newLabels, loc='upper left')

        # plt.legend(newLines, newLabels, loc='upper left', bbox_to_anchor=(1, 1))
            # TODO add pareto dominate lines & areas, see constant, maybe even add the multiple paretos
        plt.savefig(str(Path(__file__).parents[0]/f"{myname}_shift_accuracy.pdf"), bbox_inches='tight')
        plt.show()

# %%
def plot_experiment(experiment_name):
    if experiment_name == "acsemployment":
        mymin = 0.45
        myname = f"plots_paper/plot_folktable_acsemployment"
        do_plot(experiment_name,mymin,myname)
        
    elif experiment_name == "acsfoodstamps":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "acsincome":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "acspubcov":
        mymin = 0.2
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "acsunemployment":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "anes":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "assistments":
        mymin = 0.4 if ANTICAUSAL else 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "brfss_diabetes":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "brfss_blood_pressure":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "college_scorecard":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "diabetes_readmission":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "meps":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "mimic_extract_los_3":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "mimic_extract_mort_hosp":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "nhanes_lead":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)


    elif experiment_name == "physionet":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

    elif experiment_name == "sipp":
        mymin = 0.5
        myname = f"plots_paper/plot_{experiment_name}"
        do_plot(experiment_name,mymin,myname)

# %%

completed_experiments = [
                        # "acsemployment", # old
                         "acsfoodstamps",
                         "acsincome",
                        #  "acspubcov",
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
for experiment_name in completed_experiments:
    plot_experiment(experiment_name)
