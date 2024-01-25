#%%
import json
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import random
import seaborn as sns

from tqdm import tqdm

def do_plot(experiment,model):
    with open(f'results/drafts/{experiment}_{model}_eval.json', 'r') as f:
        evaluation = json.load(f)

    test_split = "id_test"
    # logits_id = evaluation[test_split+"_logits"] 
    proba = evaluation[test_split+"_proba"] 
    # preds_id = evaluation[test_split+"_preds"]
    true = evaluation[test_split+"_true"] 

    x = np.arange(len(proba))
    df = pd.DataFrame({f"{test_split}_proba":proba, "true":true})
    df.sort_values("true",inplace=True,ignore_index=True)

    sns.scatterplot(df[f"{test_split}_proba"])
    sns.scatterplot(df["true"])
    plt.ylim((0,1))
    plt.hlines(y=0.5,xmin=0,xmax=len(proba), colors="tab:red")
    plt.show()

    test_split = "ood_test"
    # logits_id = evaluation[test_split+"_logits"] 
    proba = evaluation[test_split+"_proba"] 
    # preds_id = evaluation[test_split+"_preds"]
    true = evaluation[test_split+"_true"] 

    x = np.arange(len(proba))
    df = pd.DataFrame({f"{test_split}_proba":proba, "true":true})
    df.sort_values("true",inplace=True,ignore_index=True)

    sns.scatterplot(df[f"{test_split}_proba"])
    sns.scatterplot(df["true"])
    plt.ylim((0,1))
    plt.hlines(y=0.5,xmin=0,xmax=len(proba), colors="tab:red")
    plt.show()
    

#%%
experiment = "diabetes_readmission"
# # %%
# model = "mlp"
# do_plot(experiment,model)
# %%
model = "tabtransformer"
do_plot(experiment,model)
# %%
model = "xgb"
do_plot(experiment,model)

