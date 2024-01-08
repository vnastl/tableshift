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
sns.set_context("paper", font_scale=1.5)

from tableshift import get_dataset
from  statsmodels.stats.proportion import proportion_confint
from paretoset import paretoset
from scipy.spatial import ConvexHull

from tableshift.datasets import ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")
#%%

dic_experiments = {
    "acsincome": ["acsincome","acsincome_arguablycausal", "acsincome_arguablycausal_test_0"]+[f"acsincome_arguablycausal_test_{index}" for index in range(2,ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER-1)],
    "acsfoodstamps": ["acsfoodstamps","acsfoodstamps_arguablycausal"]+[f"acsfoodstamps_arguablycausal_test_{index}" for index in range(ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER-1)],
    "brfss_diabetes": ["brfss_diabetes","brfss_diabetes_arguablycausal"]+[f"brfss_diabetes_arguablycausal_test_{index}" for index in range(BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER-1)],
}
 #%%
dic_robust_number = {
    "acsincome": ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acsfoodstamps": ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "brfss_diabetes":BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
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

# color_all = "tab:blue"
# color_arguablycausal = "tab:orange"
# color_arguablycausal = "tab:green"
# color_anticausal = "tab:grey"
# color_constant = "tab:red"
color_all = "#0173b2"
color_arguablycausal = "#d55e00"#  "#de8f05"
color_robust = "#ece133"
# color_arguablycausal = "#d55e00"
# color_anticausal = "#029e73"
color_constant = "#949494"
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
        for filename in os.listdir(RESULTS_DIR):
            if filename == ".DS_Store":
                pass
            else:
                file_info.append(filename)

        def get_feature_selection(experiment):
            if experiment.endswith('_arguablycausal'):
                if 'causal' not in feature_selection: 
                    feature_selection.append('causal') 
                return 'causal'
            elif experiment[-1].isdigit():
                if f'test{experiment[-1]}' not in feature_selection: 
                    feature_selection.append(f'test{experiment[-1]}')
                return f'test{experiment[-1]}'
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
                    'features': get_feature_selection(experiment),
                    'model':run.split("_")[0]}])
                if get_feature_selection(experiment) == 'causal':
                    causal_features = eval_json['features']
                    causal_features.remove(domain_label)
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
    return eval_all, causal_features

#%%

def do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,axmin=[0.5,0.5],axmax=[1.0,1.0]):

    eval_all, causal_features = get_results(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    plt.title(
        f"{dic_title[experiment_name]}")
    plt.xlabel(f"in-domain accuracy\n({dic_id_domain[experiment_name]})")
    plt.ylabel(f"out-of-domain accuracy\n({dic_ood_domain[experiment_name]})")
    ## All features
    eval_plot = eval_all[eval_all['features']=="all"]
    eval_plot.sort_values('id_test',inplace=True)
    # Calculate the pareto set
    points = eval_plot[['id_test','ood_test']]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= eval_constant['id_test'].values[0]]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= eval_constant['id_test'].values[0]]
    # if not myname.endswith("zoom"):
    #     print(markers["model"].values)
    errors = plt.errorbar(
                x=markers['id_test'],
                y=markers['ood_test'],
                xerr=markers['id_test_ub']-markers['id_test'],
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                color=color_all, ecolor=color_all, label="top all features")
    # highlight bar
    shift = points[points["ood_test"] == points["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_all, linewidth=3, alpha=0.7  )
    # get extra points for the plot
    new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    plt.plot(points['id_test'],points['ood_test'],color=color_all,linestyle="dotted")

    new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.1, zorder = 0)
    
    ## Arguably causal features
    eval_plot = eval_all[eval_all['features']=="causal"]
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
                yerr=markers['ood_test_ub']-markers['ood_test'], fmt="o", 
                color=color_arguablycausal, ecolor=color_arguablycausal, label="top arguably causal features")
    # highlight bar
    shift = points[points["ood_test"] == points["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_arguablycausal, linewidth=3, alpha=0.7)
    # get extra points for the plot
    new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    plt.plot(points['id_test'],points['ood_test'],color=color_arguablycausal,linestyle="dotted")

    new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    filled = points.to_numpy()
    hull = ConvexHull(filled,incremental=True)
    plt.fill(filled[hull.vertices, 0], filled[hull.vertices, 1], color=color_arguablycausal,alpha=0.1)

    ## robustness test
    for index in range(dic_robust_number[experiment_name]-1):
        if (eval_all['features'] == f"test{index}").any():
            eval_plot = eval_all[eval_all['features']==f"test{index}"]
            eval_plot.sort_values('id_test',inplace=True)
            # Calculate the pareto set
            points = eval_plot[['id_test','ood_test']]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= eval_constant['id_test'].values[0]]
            markers = eval_plot[mask]
            markers = markers[markers["id_test"] >= eval_constant['id_test'].values[0]]
            # if not myname.endswith("zoom"):
            #     print(markers["model"].values)
            errors = plt.errorbar(
                        x=markers['id_test'],
                        y=markers['ood_test'],
                        xerr=markers['id_test_ub']-markers['id_test'],
                        yerr=markers['ood_test_ub']-markers['ood_test'], fmt="v",
                        color=color_robust, ecolor=color_robust, zorder = 1,
                        label="robustness test for arguably causal features")
            # highlight bar
            shift = points[points["ood_test"] == points["ood_test"].max()]
            shift["type"] = f"test {index}"
            dic_shift[f"test{index}"] = shift
            plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                    color=color_robust, linewidth=3, alpha=0.7  )

    ## Constant
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift
    errors = plt.errorbar(
            x=eval_constant['id_test'],
            y=eval_constant['ood_test'],
            xerr=eval_constant['id_test_ub']-eval_constant['id_test'],
            yerr=eval_constant['ood_test_ub']-eval_constant['ood_test'], fmt="D",
            color=color_constant, ecolor=color_constant, label="constant")
    plt.plot([0, eval_constant['id_test'].values[0]],
                [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle="dotted")
    plt.plot([eval_constant['id_test'].values[0], eval_constant['id_test'].values[0]],
                [0,eval_constant['ood_test'].values[0]],
                color=color_constant,linestyle="dotted")
    plt.fill_between([0, eval_constant['id_test'].values[0]],
                        [0,0],
                        [eval_constant['ood_test'].values[0],eval_constant['ood_test'].values[0]],
                        color=color_constant, alpha=0.1)

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
    plt.plot([0, 1], [0, 1], color='black')

    plt.xlim((axmin[0],axmax[0]))
    plt.ylim((axmin[1],axmax[1]))

    # # Add text below the plot
    # if (eval_all['features'] == "arguablycausal").any():
    #     print(f'Causal features: {causal_features} \nArguably causal features: {extra_features}')
    # else:
    #     # plt.text(mytextx, mytexty,f'Causal features: {causal_features}')
    #     print(f'Causal features: {causal_features}')
    # if (eval_all['features'] == "anticausal").any():
    #     # plt.text(mytextx, mytexty,f'Causal features: {causal_features} \n Anticausal features: {extra_features}')
    #     print(f'Anticausal features: {extra_features}')
    # if experiment_name == 'college_scorecard':
    #     # plt.text(mytextx, mytexty,f'Causal features: {causal_features} \n Causal features without tuition: {extra_features}')
    #     print(f'Causal features without tuition: {extra_features}')
        
    
    plt.savefig(f"{str(Path(__file__).parents[0]/myname)}_arguablycausal_robust.pdf", bbox_inches='tight')
    plt.show()

    if not myname.endswith("zoom"):
        # sns.set_style("whitegrid")
        plt.title(
        f"{dic_title[experiment_name]}")
        plt.ylabel("shift gap")
        shift = pd.concat(dic_shift.values(), ignore_index=True)
        shift["gap"] = shift["id_test"] - shift["ood_test"]
        barlist = plt.bar(shift["type"], shift["gap"], color=[color_all,color_arguablycausal]+[color_robust for index in range(dic_robust_number[experiment_name]-1)]+[color_constant])
        barlist[0].set_hatch('--')
        barlist[1].set_hatch('oo')
        for index in range(2,dic_robust_number[experiment_name]+1):
            barlist[index].set_hatch('//')
        plt.xticks(rotation=45)
        plt.savefig(str(Path(__file__).parents[0]/f"{myname}_arguablycausal_robust_shift.pdf"), bbox_inches='tight')
        plt.show()
        # sns.set_style("white")

# %%
def plot_experiment(experiment_name):
    if experiment_name == "acsemployment":
        mymin = 0.45
        mymax = 1
        mytextx = 0.45
        mytexty = 0.32
        myname = f"plots_paper/plot_folktable_acsemployment"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsfoodstamps":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsincome":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acspubcov":
        mymin = 0.2
        mymax = 1
        mytextx = 0.2
        mytexty = 0.05
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsunemployment":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_acsunemployment"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "anes":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "assistments":
        mymin = 0.4
        mymax = 1
        mytextx = 0.4
        mytexty = 0.3
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "brfss_diabetes":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "brfss_blood_pressure":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "college_scorecard":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "diabetes_readmission":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "meps":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_los_3":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_mort_hosp":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "nhanes_lead":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])


    elif experiment_name == "physionet":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "sipp":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

# %% ZOOM
def plot_experiment_zoom(experiment_name):
    if experiment_name == "acsemployment":
        mymin = 0.90
        axmin = 0.94
        mymax = 1
        mytextx = 0.94
        mytexty = 0.925
        myname = f"plots_paper/plot_folktable_acsemployment_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axmin,axmin],[mymax,mymax])

    elif experiment_name == "acsfoodstamps":
        mymin = 0.8
        mymax = 0.86
        mytextx = 0.75
        mytexty = 0.73
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsincome":
        mymin = 0.79
        mymax = 0.83
        mytextx = 0.58
        mytexty = 0.55
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acspubcov":
        mymin = 0.2
        axminx = 0.58
        axminy = 0.35
        mymax = 0.83
        mytextx = 0.58
        mytexty = 0.25
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "acsunemployment":
        mymin = 0.94
        mymax = 0.98
        mytextx = 0.94
        mytexty = 0.93
        myname = f"plots_paper/plot_acsunemployment_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    if experiment_name == "anes":
        mymin = 0.58
        mymax = 0.85
        mytextx = 0.58
        mytexty = 0.53
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "assistments":
        mymin = 0.4
        axminx = 0.68
        axminy = 0.43 
        mymax = 0.96
        mytextx = 0.68
        mytexty = 0.35
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "brfss_blood_pressure":
        mymin = 0.55
        mymax = 0.68
        mytextx = 0.55
        mytexty = 0.5
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "brfss_diabetes":
        mymin = 0.81
        mymax = 0.88
        mytextx = 0.81
        mytexty = 0.79
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "college_scorecard":
        mymin = 0.65
        axminx = 0.86
        axminy = 0.65 
        mymax = 0.96
        mytextx = 0.86
        mytexty = 0.58
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "diabetes_readmission":
        mymin = 0.5
        axminx = 0.55
        axminy = 0.5
        mymax = 0.7
        mytextx = 0.55
        mytexty = 0.45
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "meps":
        mymin = 0.5
        mymax = 0.85
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_los_3":
        mymin = 0.5
        mymax = 0.71
        mytextx = 0.5
        mytexty = 0.45
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_mort_hosp":
        mymin = 0.85
        mymax = 0.95
        mytextx = 0.85
        mytexty = 0.82
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "nhanes_lead":
        mymin = 0.91
        mymax = 0.98
        mytextx = 0.90
        mytexty = 0.9
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "physionet":
        mymin = 0.92
        # axminx = 0.985
        # axminy = 0.92
        # axmaxy = 0.93
        mymax = 0.99
        mytextx = 0.985
        mytexty = 0.918
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "sipp":
        mymin = 0.4
        mymax = 0.95
        mytextx = 0.4
        mytexty = 0.3
        myname = f"plots_paper/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])


# %%

completed_experiments = [
                        # "acsemployment", # old
                         "acsfoodstamps",
                         "acsincome",
                        #  "acspubcov", # old
                        #  "acsunemployment", # old
                        #  "anes",
                        #  "assistments",
                        #  "brfss_blood_pressure",
                         "brfss_diabetes",
                        #  "college_scorecard", # old
                        #  "diabetes_readmission",
                        #  "meps"
                        #  "mimic_extract_mort_hosp",
                        #  "mimic_extract_los_3",
                        #  "nhanes_lead",
                        #  "physionet", # old 
                        #  "sipp",
                         ]
for experiment_name in completed_experiments:
    plot_experiment(experiment_name)
    plot_experiment_zoom(experiment_name)
