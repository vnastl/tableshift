import argparse
import logging
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import scipy

from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
from tableshift.core.tabular_dataset import TabularDataset
from tableshift.datasets import ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER

from experiments_causal.metrics import balanced_accuracy_score

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
        test_splits = ["id_test","ood_test","validation"] if dset.is_domain_split else ["test"]
        for test_split in test_splits:
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            X_te = X_te.astype(float)
            yhat_te = estimator.predict(X_te)
            evaluation[test_split+"_proba"] = estimator.predict_proba(X_te)[:,1].tolist()
            evaluation[test_split+"_preds"] = yhat_te.tolist()
            evaluation[test_split+"_true"] = y_te.tolist()
            
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

        with open(f'experiments_vnastl/results/drafts/{experiment}_{model}_eval.json', 'w') as f:
            # Use json.dump to write the dictionary into the file
            evaluation["features"] = dset.predictors
            json.dump(evaluation, f)
        
    else:
        # Case: pytorch estimator; eval is already performed + printed by train().
        print("training completed!")
        with open(f'experiments_vnastl/results/drafts/{experiment}_{model}_eval.json', 'w') as f:
            # Use json.dump to write the dictionary into the file
            evaluation = estimator.fit_metrics
            evaluation_balanced = estimator.fit_metrics_balanced
            for test_split in ["id_test","ood_test"]:
                # Get accuracy
                # Fetch predictions and labels for a sklearn model.
                X_te, y_te, _, _ = dset.get_pandas(test_split)
                evaluation[test_split+"_true"] = y_te.tolist()
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
            logits = estimator.logits
            for test_split in estimator.logits:
                evaluation[test_split+"_logits"] = logits[test_split][0].tolist()
                evaluation[test_split+"_proba"] = scipy.special.expit(logits[test_split][0]).tolist()
                evaluation[test_split+"_preds"] = logits[test_split][1].tolist()
            json.dump(evaluation, f)
    return

if __name__ == "__main__":
    ROOT_DIR = Path("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift/experiments_causal")
    experiments = []
    experiments = ["diabetes_readmission", "diabetes_readmission_causal", "diabetes_readmission_arguablycausal"]

    cache_dir="tmp"

    for experiment in experiments:
        dset = get_dataset(experiment, cache_dir)
        models = [
            "ft_transformer",
            "histgbm",
            "mlp",
            "saint",
            "tabtransformer",
            "resnet",
            "xgb",
            "aldro",
            "dro",
            "node",
            "group_dro",
            "label_group_dro",
            "irm",
            "vrex",
            "mixup",
            "dann",
            "mmd",
            "lightgbm",
            "deepcoral"
            ]
        for model in models:
            main(experiment=experiment,dset=dset,model=model,debug=False)
