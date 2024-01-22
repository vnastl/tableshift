#%%
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import ast

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

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")
#%%
def get_dic_experiments_value(name):
    return [name, f"{name}_causal", f"{name}_arguablycausal"]

dic_experiments = {
        "acsemployment": get_dic_experiments_value("acsemployment"),
        "acsfoodstamps":  get_dic_experiments_value("acsfoodstamps"),
        "acsincome": get_dic_experiments_value("acsincome"),
        "acsunemployment":  get_dic_experiments_value("acsunemployment"),
        "anes": get_dic_experiments_value("anes"),
        "assistments":  get_dic_experiments_value("assistments"),
        "brfss_diabetes": get_dic_experiments_value("brfss_diabetes"),
        "college_scorecard":  get_dic_experiments_value("college_scorecard"),
        "diabetes_readmission": get_dic_experiments_value("diabetes_readmission"),
        "meps":  get_dic_experiments_value("meps"),
        "mimic_extract_los_3":  get_dic_experiments_value("mimic_extract_los_3"),
        "mimic_extract_mort_hosp":  get_dic_experiments_value("mimic_extract_mort_hosp"),
    }
 #%%
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
    "college_scorecard": "\nSpecial Focus Institutions [Faith-related, art & design and other fields],\n Baccalaureate/Associates Colleges,\n Master's Colleges and Universities [larger programs]",
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

dic_tableshift = {
        "acsfoodstamps":  'Food Stamps',
        "acsincome": 'Income',
        "acspubcov":  'Public Health Ins.',
        "acsunemployment":  'Unemployment',
        "anes": 'Voting',
        "assistments":  'ASSISTments',
        "brfss_blood_pressure": 'Hypertension',
        "brfss_diabetes": 'Diabetes',
        "college_scorecard":  'College Scorecard',
        "diabetes_readmission": 'Hospital Readmission',
        "mimic_extract_los_3":   'ICU Length of Stay',
        "mimic_extract_mort_hosp": 'ICU Hospital Mortality',
        "nhanes_lead": 'Childhood Lead',
        "physionet": 'Sepsis',
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
        RESULTS_DIR = Path(__file__).parents[0]  / "results" / experiment
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
            elif experiment.endswith('_arguablycausal'):
                if 'arguablycausal' not in feature_selection: 
                    feature_selection.append('arguablycausal')
                return 'arguablycausal'
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
                eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    RESULTS_DIR = Path(__file__).parents[0]  / "results" 
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

    if experiment_name in dic_tableshift.keys():
        tableshift_results = pd.read_csv(str(Path(__file__).parents[0].parents[0]/"results"/"best_id_accuracy_results_by_task_and_model.csv"))
        tableshift_results = tableshift_results[tableshift_results['task']==dic_tableshift[experiment_name]]

        tableshift_results['test_accuracy_clopper_pearson_95%_interval'] = tableshift_results['test_accuracy_clopper_pearson_95%_interval'].apply(lambda s: ast.literal_eval(s) if s is not np.nan else np.nan)
        tableshift_results_id = tableshift_results[tableshift_results['in_distribution']==True]
        tableshift_results_id.reset_index(inplace=True)
        tableshift_results_ood = tableshift_results[tableshift_results['in_distribution']==False]
        tableshift_results_ood.reset_index(inplace=True)
        for model in tableshift_results['estimator'].unique():
            model_tableshift_results_id = tableshift_results_id[tableshift_results_id['estimator']==model]
            model_tableshift_results_id.reset_index(inplace=True)
            model_tableshift_results_ood = tableshift_results_ood[tableshift_results_ood['estimator']==model]
            model_tableshift_results_ood.reset_index(inplace=True)
            try:
                eval_pd = pd.DataFrame([{
                            'id_test':model_tableshift_results_id['test_accuracy'][0],
                            'id_test_lb':model_tableshift_results_id['test_accuracy_clopper_pearson_95%_interval'][0][0],
                            'id_test_ub':model_tableshift_results_id['test_accuracy_clopper_pearson_95%_interval'][0][1],
                            'ood_test':model_tableshift_results_ood['test_accuracy'][0],
                            'ood_test_lb':model_tableshift_results_ood['test_accuracy_clopper_pearson_95%_interval'][0][0],
                            'ood_test_ub':model_tableshift_results_ood['test_accuracy_clopper_pearson_95%_interval'][0][1],
                            'validation':np.nan,
                            'features': 'all',
                            'model':f"tableshift:{model_tableshift_results_id['estimator'][0].lower()}"}])
            except:
                print(experiment_name,model)
            eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    # eval_all.to_csv(str(Path(__file__).parents[0]/f"{experiment_name}_eval.csv"))
    # print(eval_all)
    return eval_all, causal_features

#%%

def do_plot(experiment_name,mymin,myname):

    eval_all, causal_features = get_results(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    # plt.title(
    #     f"{dic_title[experiment_name]}")
    plt.xlabel(f"in-domain accuracy") #\n({dic_id_domain[experiment_name]})")
    plt.ylabel(f"out-of-domain accuracy") #\n({dic_ood_domain[experiment_name]})")
    
    ##############################################################################
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
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                color=color_all, ecolor=color_all,
                markersize=7, capsize=3, label="all")
    # highlight bar
    shift = markers
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_all, linewidth=3, alpha=0.7)
    
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
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o",
                color=color_causal, ecolor=color_causal,
                markersize=7, capsize=3, label="causal")
    # highlight bar
    shift = markers
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_causal, linewidth=3, alpha=0.7)
    
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
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="^",
                    color=color_arguablycausal, ecolor=color_arguablycausal,
                    markersize=7, capsize=3, label="arguably\ncausal")
        # highlight bar
        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = " arguably\ncausal"
        dic_shift["arguablycausal"] = shift
        plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_arguablycausal, linewidth=3, alpha=0.7)

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
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="v",
                    color=eval(f"color_{causalml}"), ecolor=eval(f"color_{causalml}"), 
                    markersize=7, capsize=3, label="causal ml")
        # highlight bar
    
        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = causalml
        dic_shift[causalml] = shift
        plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=eval(f"color_{causalml}"), linewidth=3, alpha=0.7  )

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
        plt.plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle="dotted")
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)

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
        plt.plot(points['id_test'],points['ood_test'],color=eval(f"color_{causalml}"),linestyle="dotted")
        new_row = pd.DataFrame({'id_test':[xmin], 'ood_test':[ymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=eval(f"color_{causalml}"),alpha=0.1)
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
    plt.plot([start_lim, end_lim], [start_lim, end_lim], color='black')
    
    plt.savefig(f"{str(Path(__file__).parents[0]/myname)}_causalml.pdf", bbox_inches='tight')
    plt.show()

    ############################################################################
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
    shift.drop_duplicates(inplace=True)
    barlist = plt.bar(shift["type"], shift["ood_test"]-ymin,
                              yerr=shift['ood_test_ub']-shift['ood_test'],
                              color=[color_all,color_causal, color_arguablycausal]+[color_irm,color_vrex,]+[color_constant],
                              ecolor=color_error,align='center', capsize=10,
                              bottom=ymin)
    plt.savefig(str(Path(__file__).parents[0]/f"{myname}_causalml_ood_accuracy.pdf"), bbox_inches='tight')
    plt.show()

    #############################################################################
    # Plot shift gap as bars
    #############################################################################
    # plt.title(
    # f"{dic_title[experiment_name]}")
    plt.ylabel("shift gap")

    shift["gap"] = shift["id_test"] - shift["ood_test"]
    shift['id_test_var'] = ((shift['id_test_ub']-shift['id_test']))**2
    shift['ood_test_var'] = ((shift['ood_test_ub']-shift['ood_test']))**2
    shift['gap_var'] = shift['id_test_var']+shift['ood_test_var']
    barlist = plt.bar(shift["type"], shift["gap"],
                      yerr=shift['gap_var']**0.5,
                      ecolor=color_error,align='center', capsize=10,
                      color=[color_all,color_causal, color_arguablycausal]+[color_irm,color_vrex]+[color_constant])
    plt.savefig(str(Path(__file__).parents[0]/f"{myname}_causalml_shift.pdf"), bbox_inches='tight')
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
        mymin = 0.5
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
                        
                         "acsfoodstamps",
                         "acsincome",
                        
                         "acsunemployment",
                         "anes",
                        #  "assistments",
                        
                         "college_scorecard",
                         "diabetes_readmission",
                        
                         "mimic_extract_mort_hosp",
                         "mimic_extract_los_3",
                        
                         ]
for experiment_name in completed_experiments:
    plot_experiment(experiment_name)
