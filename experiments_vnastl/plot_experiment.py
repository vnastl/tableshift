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
sns.set_context("talk")


from tableshift import get_dataset
from  statsmodels.stats.proportion import proportion_confint
from paretoset import paretoset
from scipy.spatial import ConvexHull

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import os
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")
#%%

dic_experiments = {
    "acsemployment": ["acsemployment","acsemployment_causal", "acsemployment_anticausal"],
    "acsfoodstamps": ["acsfoodstamps","acsfoodstamps_causal"],
    "acsincome": ["acsincome","acsincome_causal"],
    "acspubcov": ["acspubcov","acspubcov_causal"],
    "acsunemployment": ["acsunemployment","acsunemployment_causal", "acsunemployment_anticausal"],
    "anes": ["anes","anes_causal"],
    "assistments": ["assistments","assistments_causal"],
    "brfss_blood_pressure": ["brfss_blood_pressure", "brfss_blood_pressure_causal"],
    "brfss_diabetes": ["brfss_diabetes","brfss_diabetes_causal","brfss_diabetes_anticausal"],
    "college_scorecard": ["college_scorecard","college_scorecard_causal","college_scorecard_causal_no_tuition_fee"],
    "diabetes_readmission": ["diabetes_readmission", "diabetes_readmission_causal"],
    "meps": ["meps", "meps_causal"],
    "mimic_extract_los_3": ["mimic_extract_los_3","mimic_extract_los_3_causal"],
    "mimic_extract_mort_hosp": ["mimic_extract_mort_hosp","mimic_extract_mort_hosp_causal"],
    "nhanes_lead": ["nhanes_lead", "nhanes_lead_causal"],
    "physionet":["physionet","physionet_causal", "physionet_anticausal"],
    "sipp": ["sipp", "sipp_causal"],
}
anticausal = ["acsemployment", "acsunemployment", "physionet", "brfss_diabetes"]

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


color_all = "tab:blue"
color_causal = "tab:orange"
color_causal2 = "tab:green"
color_constant = "tab:red"

def get_results(experiment_name):
    cache_dir="tmp"
    experiments = dic_experiments[experiment_name]
    domain_label = dic_domain_label[experiment_name]

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
            # elif experiment.endswith('_causal2'):
            #     if 'causal2' not in feature_selection: 
            #         feature_selection.append('causal2')
            #     return 'causal2'
            # elif experiment.endswith('_causal_no_tuition_fee'):
            #     if 'causal without tuition' not in feature_selection: 
            #         feature_selection.append('causal without tuition')
            #     return 'causal without tuition'
            # elif experiment.endswith('_anticausal'):
            #     if 'anticausal' not in feature_selection: 
            #         feature_selection.append('anticausal')
            #     return 'anticausal'
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
                if get_feature_selection(experiment) == 'causal2' or get_feature_selection(experiment) == 'causal without tuition' or get_feature_selection(experiment) == 'anticausal':
                    extra_features = eval_json['features']
                    extra_features.remove(domain_label)
                else:
                    extra_features = []
                eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

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
    # print(eval_all)
    return eval_all, causal_features, extra_features

#%%
def do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,axmin=[0.5,0.5],axmax=[1.0,1.0]):

    eval_all, causal_features, extra_features = get_results(experiment_name)
    eval_constant = eval_all[eval_all['features']=="constant"]
    dic_shift = {}

    plt.title(
        f"Tableshift: {experiment_name}")
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
    if not myname.endswith("zoom"):
        print(markers["model"].values)
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
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.3)
    
    ## Causal features
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
                color=color_causal, ecolor=color_causal, label="top causal features")
    # highlight bar
    shift = points[points["ood_test"] == points["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
               color=color_causal, linewidth=3, alpha=0.7)
    # get extra points for the plot
    new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    points.sort_values('id_test',inplace=True)
    plt.plot(points['id_test'],points['ood_test'],color=color_causal,linestyle="dotted")
    # dotted = points.to_numpy()
    # hull = ConvexHull(dotted,incremental=True)
    # plt.plot(dotted[list(hull.vertices[1:])+[hull.vertices[0]], 0], dotted[list(hull.vertices[1:])+[hull.vertices[0]], 1],
    #          color=color_causal,linestyle="dotted")
    # plt.plot(dotted[hull.vertices, 0][1:], dotted[hull.vertices, 1][1:],color=color_causal,linestyle="dotted")
    # points.sort_values('id_test',inplace=True)
    # plt.plot(points['id_test'],points['ood_test'],color=color_causal,linestyle="dotted")

    new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
    points = pd.concat([points,new_row], ignore_index=True)
    filled = points.to_numpy()
    hull = ConvexHull(filled,incremental=True)
    plt.fill(filled[hull.vertices, 0], filled[hull.vertices, 1], color=color_causal,alpha=0.3)
    # plt.fill(filled['id_test'],filled['ood_test'], color=color_causal,alpha=0.3)

    ## Causal features 2
    if (eval_all['features'] == "causal2").any():
        eval_plot = eval_all[eval_all['features']=="causal2"]
        eval_plot.sort_values('id_test',inplace=True)
        # Calculate the pareto set
        points = eval_plot[['id_test','ood_test']]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= eval_constant['id_test'].values[0]]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= eval_constant['id_test'].values[0]]
        if not myname.endswith("zoom"):
            print(markers["model"].values)
        errors = plt.errorbar(
                    x=markers['id_test'],
                    y=markers['ood_test'],
                    xerr=markers['id_test_ub']-markers['id_test'],
                    yerr=markers['ood_test_ub']-markers['ood_test'], fmt="s",
                    color=color_causal2, ecolor=color_causal2, label="top arguably causal features")
        # highlight bar
        shift = points[points["ood_test"] == points["ood_test"].max()]
        shift["type"] = "causal2"
        dic_shift["causal2"] = shift
        plt.hlines(y=shift["ood_test"], xmin=shift["ood_test"], xmax=shift['id_test'],
                color=color_causal2, linewidth=3, alpha=0.7  )
        # get extra points for the plot
        new_row = pd.DataFrame({'id_test':[mymin,max(points['id_test'])], 'ood_test':[max(points['ood_test']),mymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points.sort_values('id_test',inplace=True)
        plt.plot(points['id_test'],points['ood_test'],color=color_causal2,linestyle="dotted")

        new_row = pd.DataFrame({'id_test':[mymin], 'ood_test':[mymin]},)
        points = pd.concat([points,new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color_all,alpha=0.3)

    ## Constant
    shift = eval_constant
    shift["type"] = "constant"
    dic_shift["constant"] = shift
    plot_constant = plt.plot(
            eval_constant['id_test'],
            eval_constant['ood_test'],
            marker="D",linestyle="None",
            color=color_constant,
            label="constant")
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
                        color=color_constant, alpha=0.3)

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

    # Add text below the plot
    if (eval_all['features'] == "causal2").any():
        print(f'Causal features: {causal_features} \n Arguably causal features: {extra_features}')
    if experiment_name in anticausal:
        # plt.text(mytextx, mytexty,f'Causal features: {causal_features} \n Anticausal features: {extra_features}')
        print(f'Causal features: {causal_features} \n Anticausal features: {extra_features}')
    elif experiment_name == 'college_scorecard':
        # plt.text(mytextx, mytexty,f'Causal features: {causal_features} \n Causal features without tuition: {extra_features}')
        print(f'Causal features: {causal_features} \n Causal features without tuition: {extra_features}')
    else:
        # plt.text(mytextx, mytexty,f'Causal features: {causal_features}')
        print(f'Causal features: {causal_features}')

    plt.savefig(str(Path(__file__).parents[0]/myname), bbox_inches='tight')
    plt.show()

    if not myname.endswith("zoom"):
        plt.title(
            f"Tableshift: {experiment_name}")
        plt.ylabel("shift gap")
        shift = pd.concat(dic_shift.values(), ignore_index=True)
        shift["gap"] = shift["id_test"] - shift["ood_test"]
        if (eval_all['features'] == "causal2").any():
            plt.bar(shift["type"], shift["gap"], color=[color_all,color_causal,color_causal2,color_constant])
        else:
            plt.bar(shift["type"], shift["gap"], color=[color_all,color_causal,color_constant])
        plt.savefig(str(Path(__file__).parents[0]/f"{myname}_shift"), bbox_inches='tight')
        plt.show()

# %%
def plot_experiment(experiment_name):
    if experiment_name == "acsemployment":
        mymin = 0.45
        mymax = 1
        mytextx = 0.45
        mytexty = 0.32
        myname = f"plots/plot_folktable_acsemployment"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsfoodstamps":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsincome":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acspubcov":
        mymin = 0.2
        mymax = 1
        mytextx = 0.2
        mytexty = 0.05
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsunemployment":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_acsunemployment"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "anes":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "assistments":
        mymin = 0.4
        mymax = 1
        mytextx = 0.4
        mytexty = 0.3
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "brfss_diabetes":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "brfss_blood_pressure":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "college_scorecard":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "diabetes_readmission":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "meps":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_los_3":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_mort_hosp":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "nhanes_lead":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])


    elif experiment_name == "physionet":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "sipp":
        mymin = 0.5
        mymax = 1
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

# %% ZOOM
def plot_experiment_zoom(experiment_name):
    if experiment_name == "acsemployment":
        mymin = 0.90
        axmin = 0.94
        mymax = 1
        mytextx = 0.94
        mytexty = 0.925
        myname = f"plots/plot_folktable_acsemployment_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axmin,axmin],[mymax,mymax])

    elif experiment_name == "acsfoodstamps":
        mymin = 0.75
        mymax = 0.86
        mytextx = 0.75
        mytexty = 0.73
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acsincome":
        mymin = 0.58
        mymax = 0.85
        mytextx = 0.58
        mytexty = 0.55
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "acspubcov":
        mymin = 0.2
        axminx = 0.58
        axminy = 0.35
        mymax = 0.83
        mytextx = 0.58
        mytexty = 0.25
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "acsunemployment":
        mymin = 0.94
        mymax = 0.98
        mytextx = 0.94
        mytexty = 0.93
        myname = f"plots/plot_acsunemployment_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    if experiment_name == "anes":
        mymin = 0.58
        mymax = 0.85
        mytextx = 0.58
        mytexty = 0.53
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "assistments":
        mymin = 0.4
        axminx = 0.68
        axminy = 0.43 
        mymax = 0.96
        mytextx = 0.68
        mytexty = 0.35
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "brfss_blood_pressure":
        mymin = 0.55
        mymax = 0.68
        mytextx = 0.55
        mytexty = 0.5
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "brfss_diabetes":
        mymin = 0.81
        mymax = 0.88
        mytextx = 0.81
        mytexty = 0.79
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "college_scorecard":
        mymin = 0.65
        axminx = 0.86
        axminy = 0.65 
        mymax = 0.96
        mytextx = 0.86
        mytexty = 0.58
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "diabetes_readmission":
        mymin = 0.5
        axminx = 0.55
        axminy = 0.5
        mymax = 0.7
        mytextx = 0.55
        mytexty = 0.45
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[axminx,axminy],[mymax,mymax])

    elif experiment_name == "meps":
        mymin = 0.5
        mymax = 0.85
        mytextx = 0.5
        mytexty = 0.4
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_los_3":
        mymin = 0.5
        mymax = 0.71
        mytextx = 0.5
        mytexty = 0.45
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "mimic_extract_mort_hosp":
        mymin = 0.85
        mymax = 0.95
        mytextx = 0.85
        mytexty = 0.82
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "nhanes_lead":
        mymin = 0.91
        mymax = 0.98
        mytextx = 0.90
        mytexty = 0.9
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "physionet":
        mymin = 0.92
        # axminx = 0.985
        # axminy = 0.92
        # axmaxy = 0.93
        mymax = 0.99
        mytextx = 0.985
        mytexty = 0.918
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])

    elif experiment_name == "sipp":
        mymin = 0.4
        mymax = 0.95
        mytextx = 0.4
        mytexty = 0.3
        myname = f"plots/plot_{experiment_name}_zoom"

        do_plot(experiment_name,mymin,mymax,mytextx,mytexty,myname,[mymin,mymin],[mymax,mymax])


# %%

completed_experiments = [
                        # "acsemployment", # old
                        #  "acsfoodstamps", # old
                        #  "acsincome",
                        #  "acspubcov", # old
                        #  "acsunemployment", # old
                        #  "anes",
                        #  "assistments",
                        #  "brfss_blood_pressure",
                        #  "brfss_diabetes",
                        #  "college_scorecard", # old
                         "diabetes_readmission",
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
