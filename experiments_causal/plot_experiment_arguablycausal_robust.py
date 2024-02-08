import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import ast
import os

from tableshift import get_dataset
from statsmodels.stats.proportion import proportion_confint
from tableshift.datasets import (
    ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    PHYSIONET_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    NHANES_LEAD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
)


def get_dic_experiments_value(name, superset):
    return [name, f"{name}_arguablycausal"] + [
        f"{name}_arguablycausal_test_{index}" for index in range(superset)
    ]


dic_robust_number = {
    "acsincome": ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acsfoodstamps": ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "brfss_diabetes": BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "brfss_blood_pressure": BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "diabetes_readmission": DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "anes": ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acsunemployment": ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "assistments": ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "college_scorecard": COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "diabetes_readmission": DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "sipp": SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acspubcov": ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "meps": MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "physionet": PHYSIONET_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "nhanes_lead": NHANES_LEAD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
}

dic_experiments = {
    "acsincome": get_dic_experiments_value("acsincome", dic_robust_number["acsincome"]),
    "acsfoodstamps": get_dic_experiments_value(
        "acsfoodstamps", dic_robust_number["acsfoodstamps"]
    ),
    "brfss_diabetes": get_dic_experiments_value(
        "brfss_diabetes", dic_robust_number["brfss_diabetes"]
    ),
    "brfss_blood_pressure": get_dic_experiments_value(
        "brfss_blood_pressure", dic_robust_number["brfss_blood_pressure"]
    ),
    "diabetes_readmission": get_dic_experiments_value(
        "diabetes_readmission", dic_robust_number["diabetes_readmission"]
    ),
    "anes": get_dic_experiments_value("anes", dic_robust_number["anes"]),
    "acsunemployment": get_dic_experiments_value(
        "acsunemployment", dic_robust_number["acsunemployment"]
    ),
    "assistments": get_dic_experiments_value(
        "assistments", dic_robust_number["assistments"]
    ),
    "college_scorecard": get_dic_experiments_value(
        "college_scorecard", dic_robust_number["college_scorecard"]
    ),
    "diabetes_readmission": get_dic_experiments_value(
        "diabetes_readmission", dic_robust_number["diabetes_readmission"]
    ),
    "sipp": get_dic_experiments_value("sipp", dic_robust_number["sipp"]),
    "acspubcov": get_dic_experiments_value("acspubcov", dic_robust_number["acspubcov"]),
    "meps": get_dic_experiments_value("meps", dic_robust_number["meps"]),
    "physionet": get_dic_experiments_value("physionet", dic_robust_number["physionet"]),
    "nhanes_lead": get_dic_experiments_value(
        "nhanes_lead", dic_robust_number["nhanes_lead"]
    ),
}

dic_domain_label = {
    "acsemployment": "SCHL",
    "acsfoodstamps": "DIVISION",
    "acsincome": "DIVISION",
    "acspubcov": "DIS",
    "acsunemployment": "SCHL",
    "anes": "VCF0112",  # region
    "assistments": "school_id",
    "brfss_blood_pressure": "BMI5CAT",
    "brfss_diabetes": "PRACE1",
    "college_scorecard": "CCBASIC",
    "diabetes_readmission": "admission_source_id",
    "meps": "INSCOV19",
    "mimic_extract_los_3": "insurance",
    "mimic_extract_mort_hosp": "insurance",
    "nhanes_lead": "INDFMPIRBelowCutoff",
    "physionet": "ICULOS",  # ICU length of stay
    "sipp": "CITIZENSHIP_STATUS",
}

dic_id_domain = {
    "acsemployment": "High school diploma or higher",
    "acsfoodstamps": "Other U.S. Census divisions",
    # Mid-Atlantic, East North Central, West North Central, South Atlantic, East South Central, West South Central, Mountain, Pacific
    "acsincome": "Other U.S. Census divisions",
    "acspubcov": "Without disability",
    "acsunemployment": "High school diploma or higher",
    "anes": "Other U.S. Census regions",  # region
    "assistments": "approximately 700 schools",
    "brfss_blood_pressure": "Underweight and normal weight",
    "brfss_diabetes": "White",
    "college_scorecard": "Carnegie Classification: other institutional types",
    "diabetes_readmission": "Other admission sources",
    "meps": "Public insurance",
    "mimic_extract_los_3": "Private, Medicaid, Government, Self Pay",
    "mimic_extract_mort_hosp": "Private, Medicaid, Government, Self Pay",
    "nhanes_lead": "poverty-income ratio > 1.3",
    "physionet": "ICU length of stay <= 47 hours",  # ICU length of stay
    "sipp": "U.S. citizen",
}

dic_ood_domain = {
    "acsemployment": "No high school diploma",
    "acsfoodstamps": "East South Central",
    "acsincome": "New England",
    "acspubcov": "With disability",
    "acsunemployment": "No high school diploma",
    "anes": "South",  # region
    "assistments": "10 new schools",
    "brfss_blood_pressure": "Overweight and obese",
    "brfss_diabetes": "Non white",
    "college_scorecard": "\nSpecial Focus Institutions [Faith-related, art & design and other fields],\n Baccalaureate/Associates Colleges,\n Master's Colleges and Universities [larger programs]",
    "diabetes_readmission": "Emergency Room",
    "meps": "Private insurance",
    "mimic_extract_los_3": "Medicare",
    "mimic_extract_mort_hosp": "Medicare",
    "nhanes_lead": "poverty-income ratio <= 1.3",
    "physionet": "ICU length of stay > 47 hours",  # ICU length of stay
    "sipp": "non U.S. citizen",
}

dic_title = {
    "acsemployment": "Tableshift: Employment",
    "acsfoodstamps": "Tableshift: Food Stamps",
    "acsincome": "Tableshift: Income",
    "acspubcov": "Tableshift: PublicCoverage",
    "acsunemployment": "Tableshift: Unemployment",
    "anes": "Tableshift: Voting",
    "assistments": "Tableshift: ASSISTments",
    "brfss_blood_pressure": "Tableshift: Hypertension",
    "brfss_diabetes": "Tableshift: Diabetes",
    "college_scorecard": "Tableshift: College Scorecard",
    "diabetes_readmission": "Tableshift: Hospital Readmission",
    "meps": "MEPS: Utilization",
    "mimic_extract_los_3": "Tableshift: ICU Length of Stay",
    "mimic_extract_mort_hosp": "Tableshift: Hospital Mortality",
    "nhanes_lead": "Tableshift: Childhood Lead",
    "physionet": "Tableshift: Sepsis",  # ICU length of stay
    "sipp": "SIPP: Poverty",
}

dic_tableshift = {
    "acsfoodstamps": "Food Stamps",
    "acsincome": "Income",
    "acspubcov": "Public Health Ins.",
    "acsunemployment": "Unemployment",
    "anes": "Voting",
    "assistments": "ASSISTments",
    "brfss_blood_pressure": "Hypertension",
    "brfss_diabetes": "Diabetes",
    "college_scorecard": "College Scorecard",
    "diabetes_readmission": "Hospital Readmission",
    "mimic_extract_los_3": "ICU Length of Stay",
    "mimic_extract_mort_hosp": "ICU Hospital Mortality",
    "nhanes_lead": "Childhood Lead",
    "physionet": "Sepsis",
}


def get_results(experiment_name):
    cache_dir = "tmp"
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
            if experiment.endswith("_arguablycausal"):
                if "arguablycausal" not in feature_selection:
                    feature_selection.append("arguablycausal")
                return "arguablycausal"
            elif experiment.endswith("_los_3"):
                feature_selection.append("all")
                return "all"
            elif experiment[-2].isdigit():
                if f"test{experiment[-2]}" not in feature_selection:
                    feature_selection.append(f"test{experiment[-2:]}")
                return f"test{experiment[-2:]}"
            elif experiment[-1].isdigit():
                if f"test{experiment[-1]}" not in feature_selection:
                    feature_selection.append(f"test{experiment[-1]}")
                return f"test{experiment[-1]}"
            else:
                if "all" not in feature_selection:
                    feature_selection.append("all")
                return "all"

        for run in file_info:
            with open(str(RESULTS_DIR / run), "rb") as file:
                # print(str(RESULTS_DIR / run))
                try:
                    eval_json = json.load(file)
                    eval_pd = pd.DataFrame(
                        [
                            {
                                "id_test": eval_json["id_test"],
                                "id_test_lb": eval_json["id_test" + "_conf"][0],
                                "id_test_ub": eval_json["id_test" + "_conf"][1],
                                "ood_test": eval_json["ood_test"],
                                "ood_test_lb": eval_json["ood_test" + "_conf"][0],
                                "ood_test_ub": eval_json["ood_test" + "_conf"][1],
                                "validation": (
                                    eval_json["validation"]
                                    if "validation" in eval_json
                                    else np.nan
                                ),
                                "features": get_feature_selection(experiment),
                                "model": run.split("_")[0],
                            }
                        ]
                    )
                    if get_feature_selection(experiment) == "arguablycausal":
                        causal_features = eval_json["features"]
                        causal_features.remove(domain_label)
                    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                except:
                    print(str(RESULTS_DIR / run))

    RESULTS_DIR = Path(__file__).parents[0] / "results"
    filename = f"{experiment_name}_constant"
    if filename in os.listdir(RESULTS_DIR):
        with open(str(RESULTS_DIR / filename), "rb") as file:
            # print(str(RESULTS_DIR / filename))
            eval_constant = json.load(file)
    else:
        eval_constant = {}
        dset = get_dataset(experiment_name, cache_dir)
        for test_split in ["id_test", "ood_test"]:
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            majority_class = y_te.mode()[0]
            count = y_te.value_counts()[majority_class]
            nobs = len(y_te)
            acc = count / nobs
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")

            eval_constant[test_split] = acc
            eval_constant[test_split + "_conf"] = acc_conf
        with open(str(RESULTS_DIR / filename), "w") as file:
            json.dump(eval_constant, file)

    list_model_data = []
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        for model in eval_feature["model"].unique():
            model_data = eval_feature[eval_feature["model"] == model]
            if not set[-1].isdigit():
                model_data = model_data[
                    model_data["validation"] == model_data["validation"].max()
                ]
            list_model_data.append(model_data)
    eval_all = pd.concat(list_model_data)

    eval_pd = pd.DataFrame(
        [
            {
                "id_test": eval_constant["id_test"],
                "id_test_lb": eval_constant["id_test_conf"][0],
                "id_test_ub": eval_constant["id_test_conf"][1],
                "ood_test": eval_constant["ood_test"],
                "ood_test_lb": eval_constant["ood_test_conf"][0],
                "ood_test_ub": eval_constant["ood_test_conf"][1],
                "features": "constant",
                "model": "constant",
            }
        ]
    )
    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    if experiment_name in dic_tableshift.keys():
        tableshift_results = pd.read_csv(
            str(
                Path(__file__).parents[0].parents[0]
                / "results"
                / "best_id_accuracy_results_by_task_and_model.csv"
            )
        )
        tableshift_results = tableshift_results[
            tableshift_results["task"] == dic_tableshift[experiment_name]
        ]

        tableshift_results["test_accuracy_clopper_pearson_95%_interval"] = (
            tableshift_results["test_accuracy_clopper_pearson_95%_interval"].apply(
                lambda s: ast.literal_eval(s) if s is not np.nan else np.nan
            )
        )
        tableshift_results_id = tableshift_results[
            tableshift_results["in_distribution"] == True
        ]
        tableshift_results_id.reset_index(inplace=True)
        tableshift_results_ood = tableshift_results[
            tableshift_results["in_distribution"] == False
        ]
        tableshift_results_ood.reset_index(inplace=True)
        for model in tableshift_results["estimator"].unique():
            model_tableshift_results_id = tableshift_results_id[
                tableshift_results_id["estimator"] == model
            ]
            model_tableshift_results_id.reset_index(inplace=True)
            model_tableshift_results_ood = tableshift_results_ood[
                tableshift_results_ood["estimator"] == model
            ]
            model_tableshift_results_ood.reset_index(inplace=True)
            try:
                eval_pd = pd.DataFrame(
                    [
                        {
                            "id_test": model_tableshift_results_id["test_accuracy"][0],
                            "id_test_lb": model_tableshift_results_id[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][0],
                            "id_test_ub": model_tableshift_results_id[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][1],
                            "ood_test": model_tableshift_results_ood["test_accuracy"][0],
                            "ood_test_lb": model_tableshift_results_ood[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][0],
                            "ood_test_ub": model_tableshift_results_ood[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][1],
                            "validation": np.nan,
                            "features": "all",
                            "model": f"tableshift:{model_tableshift_results_id['estimator'][0].lower()}",
                        }
                    ]
                )
            except:
                print(experiment_name, model)
            eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
    return eval_all, causal_features
