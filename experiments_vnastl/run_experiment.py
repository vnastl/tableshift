import argparse
import logging
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import accuracy_score

from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
from tableshift.core.tabular_dataset import TabularDataset
from tableshift.datasets import ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER

from experiments_vnastl.metrics import balanced_accuracy_score

import json
from  statsmodels.stats.proportion import proportion_confint

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(experiment, dset, model, debug: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    config = get_default_config(model, dset)
    estimator = get_estimator(model, **config)
    estimator = train(estimator, dset, config=config)

    if not isinstance(estimator, torch.nn.Module):
        evaluation = {}
        # Case: non-pytorch estimator; perform test-split evaluation.
        test_splits = ["id_test","ood_test"] if dset.is_domain_split else ["test"]
        for test_split in test_splits:
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            yhat_te = estimator.predict(X_te)

            # Calculate accuracy
            acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
            evaluation[test_split] = acc
            nobs = len(y_te)
            count = nobs*acc
            # beta : Clopper-Pearson interval based on Beta distribution
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method='beta')
            evaluation[test_split + "_conf"] = acc_conf
            print(f"training completed! {test_split} accuracy: {acc:.4f}")

            # Calculate balanced accuracy
            balanced_acc, balanced_acc_se = balanced_accuracy_score(target=y_te, prediction=yhat_te)
            evaluation[test_split + "_balanced"] = balanced_acc
            balanced_acc_conf = (balanced_acc-1.96*balanced_acc_se, balanced_acc+1.96*balanced_acc_se)
            evaluation[test_split + "_balanced" + "_conf"] = balanced_acc_conf
            print(f"training completed! {test_split} balanced accuracy: {balanced_acc:.4f}")

        with open(f'experiments_vnastl/{experiment}/{model}_eval.json', 'w') as f:
            # Use json.dump to write the dictionary into the file
            evaluation["features"] = dset.predictors
            json.dump(evaluation, f)
        
    else:
        # Case: pytorch estimator; eval is already performed + printed by train().
        print("training completed!")
        with open(f'experiments_vnastl/{experiment}/{model}_eval.json', 'w') as f:
            # Use json.dump to write the dictionary into the file
            evaluation = estimator.fit_metrics
            evaluation_balanced = estimator.fit_metrics_balanced
            for test_split in ["id_test","ood_test"]:
                # Get accuracy
                # Fetch predictions and labels for a sklearn model.
                X_te, y_te, _, _ = dset.get_pandas(test_split)
                nobs = len(y_te)
                acc = evaluation[test_split]
                count = nobs*acc
                acc_conf = proportion_confint(count, nobs, alpha=0.05, method='beta')
                evaluation[test_split + "_conf"] = acc_conf

                # Get balanced accuracy
                balanced_acc = evaluation_balanced["score"][test_split]
                balanced_acc_se = evaluation_balanced["se"][test_split]
                evaluation[test_split + "_balanced"] = balanced_acc
                balanced_acc_conf = (balanced_acc-1.96*balanced_acc_se, balanced_acc+1.96*balanced_acc_se)
                evaluation[test_split + "_balanced" + "_conf"] = balanced_acc_conf
                print(f"training completed! {test_split} balanced accuracy: {balanced_acc:.4f}")
            evaluation["features"] = dset.predictors
            json.dump(evaluation, f)
    return

if __name__ == "__main__":
    ROOT_DIR = Path("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift/experiments_vnastl")
    experiments = []
    # for index in range(ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER-1):
    #     experiments.append("acsincome_causal_test_"+f"{index}")
    #     RESULTS_DIR = ROOT_DIR / f"acsincome_causal_test_{index}"
    #     RESULTS_DIR.mkdir(exist_ok=True, parents=False)
    # for index in range(ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER-1):
    #     experiments.append("acsincome_arguablycausal_test_"+f"{index}")
    #     RESULTS_DIR = ROOT_DIR / f"acsincome_arguablycausal_test_{index}"
    #     RESULTS_DIR.mkdir(exist_ok=True, parents=False)
    experiments.append("acsincome_arguablycausal_test_1")
    RESULTS_DIR = ROOT_DIR / f"acsincome_arguablycausal_test_1"
    RESULTS_DIR.mkdir(exist_ok=True, parents=False)
    # experiments = ["acsincome" ,"acsincome_causal", "acsincome_arguablycausal","acsincome_anticausal",]
    # experiments=["acspubcov", "acspubcov_causal"]
    # experiments = ["acsunemployment","acsunemployment_causal", "acsunemployment_anticausal"] 
    # experiments=["acsfoodstamps", "acsfoodstamps_causal"]
    # for index in range(ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER-1):
    #     experiments.append("acsfoodstamps_causal_test_"+f"{index}")
    #     RESULTS_DIR = ROOT_DIR / f"acsfoodstamps_causal_test_{index}"
    #     RESULTS_DIR.mkdir(exist_ok=True, parents=False)
    # for index in range(ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER-1):
    #     experiments.append("acsfoodstamps_arguablycausal_test_"+f"{index}")
    #     RESULTS_DIR = ROOT_DIR / f"acsfoodstamps_arguablycausal_test_{index}"
    #     RESULTS_DIR.mkdir(exist_ok=True, parents=False)
    # experiments = ["anes","anes_causal"]
    # experiments = ["assistments","assistments_causal"]
    # experiments = ["brfss_diabetes_causal","brfss_diabetes_anticausal"] #,"brfss_diabetes"]
    # experiments = ["brfss_blood_pressure_causal","brfss_blood_pressure"]
    # experiments=["college_scorecard","college_scorecard_causal"]
    # experiments = ["nhanes_lead", "nhanes_lead_causal"]
    # experiments = ["diabetes_readmission"] #, "diabetes_readmission_causal"]
    # experiments = ["meps","meps_causal"]
    # experiments = ["mimic_extract_los_3","mimic_extract_los_3_causal"] 
    # experiments = ["mimic_extract_mort_hosp","mimic_extract_mort_hosp_causal"]
    # experiments = ["physionet","physionet_causal", "physionet_anticausal"]
    # experiments = ["sipp","sipp_causal"]

    cache_dir="tmp"

    for experiment in experiments:
        dset = get_dataset(experiment, cache_dir)
        # X, y, _, _ = dset.get_pandas("train")
        models = [
            # "ft_transformer",
            "histgbm",
            # "mlp",
            # "saint",
            # "tabtransformer",
            # "resnet",
            # "xgb",
            # "aldro",
            # "dro",
            # "node",
            # "group_dro",
            # "label_group_dro",
            # "irm",
            # "vrex",
            # "mixup",
            # "dann",
            # "mmd",
            # "lightgbm",
            # "deepcoral"
            ]
        for model in models:
            main(experiment=experiment,dset=dset,model=model,debug=False)
