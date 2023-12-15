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

axmin = [0.5,0.5]
axmax = [1,1]


pacs = pd.DataFrame([
                    [62.86, 66.97, 89.50, 57.51],
                    [61.67, 67.41, 84.31, 63.91],
                    [62.70, 69.73, 78.65, 64.45],
                    [62.64, 65.98, 90.44, 58.76],
                    [66.23, 66.88, 88.00, 58.96],
                    [66.80, 69.70, 87.90, 56.30],
                    [64.10, 66.80, 90.20, 60.10],
                    [64.40, 68.60, 90.10, 58.40],
                    [67.04, 67.97, 89.74, 59.81],
                    [65.52, 69.90, 89.16, 63.37],
                    [64.70, 72.30, 86.10, 65.00],
                    [66.60, 73.36, 88.12, 66.19],
                    [70.35, 72.46, 90.68, 67.33],
                    [79.42, 75.25, 96.03, 71.35],
                    [80.50, 77.80, 94.80, 72.80],
                    [79.48, 77.13, 94.30, 75.30],
                    [84.20, 78.10, 95.30, 74.70],
                    [81.28, 77.16, 96.09, 72.29],
                    [83.58, 77.66, 95.47, 76.30],
                    [87.20, 79.20, 97.60, 70.30],
                    [83.01, 79.39, 96.83, 78.62],
                    ],
                    columns=["Art","Cartoon","Photo","Sketch"])
pacs = pacs*0.01
shift = pacs[pacs["Sketch"] == pacs["Sketch"].max()]

indomain = "Photo"
outofdomain = "Sketch"
axmin = [0.5,0.5]
axmax = [1,1]

plt.title(f"PACS")
plt.xlabel(f"in-domain accuracy\n({indomain})")
plt.ylabel(f"out-of-domain accuracy\n({outofdomain})")
plt.scatter(pacs[indomain], pacs[outofdomain], color='tab:blue')
plt.hlines(y=shift[outofdomain], xmin=shift[outofdomain], xmax=shift[indomain],
               color='tab:blue', linewidth=3, alpha=0.7  )
               # Plot the diagonal line
plt.plot([0, 1], [0, 1], color='#627313')

plt.xlim((axmin[0],axmax[0]))
plt.ylim((axmin[1],axmax[1]))
plt.tight_layout()
plt.savefig(str(Path(__file__).parents[0]/f"pacs_shift"), bbox_inches='tight')
plt.show()

# %%
acc_on_the_line = pd.read_csv("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/accuracy-on-the-line/results.csv")
wilds = acc_on_the_line[acc_on_the_line["train_set"]=="FMoW-train"]
wilds = wilds[wilds["shift_set"]=="FMoW-ood_test"]
shift = wilds[wilds["shift_accuracy"] == wilds["shift_accuracy"].max()]

axmin = [0.5,0.5]
axmax = [0.66,0.66]


plt.title(f"fMoW-WILDS")
plt.xlabel(f"in-domain accuracy\n(from before 2013)")
plt.ylabel(f"out-of-domain accuracy\n(from 2016 and after)")
plt.scatter(wilds["test_accuracy"], wilds["shift_accuracy"], color='tab:blue')
plt.hlines(y=shift["shift_accuracy"], xmin=shift["shift_accuracy"], xmax=shift["test_accuracy"],
               color='tab:blue', linewidth=3, alpha=0.7  )
               # Plot the diagonal line
plt.plot([0, 1], [0, 1], color='#627313')

plt.xlim((axmin[0],axmax[0]))
plt.ylim((axmin[1],axmax[1]))
plt.tight_layout()
plt.savefig(str(Path(__file__).parents[0]/f"wilds_shift"), bbox_inches='tight')
plt.show()

# %%
tableshift = pd.read_csv("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift/results/best_id_accuracy_results_by_task_and_model.csv")
readmission = tableshift[tableshift["task"] =="Hospital Readmission"]
readmission_id = readmission[readmission["in_distribution"]==True]
readmission_ood = readmission[readmission["in_distribution"]==False]
readmission_ood.rename(columns={"test_accuracy":"shift_accuracy"},inplace=True)
readmission = pd.merge(readmission_id,readmission_ood,on="estimator")

shift = readmission[readmission["shift_accuracy"] == readmission["shift_accuracy"].max()]

axmin = [0.5,0.5]
axmax = [0.7,0.7]

plt.title(f"Hospital readmission")
plt.xlabel(f"in-domain accuracy\n(other admission sources)")
plt.ylabel(f"out-of-domain accuracy\n(emergency room)")
plt.scatter(readmission["test_accuracy"], readmission["shift_accuracy"], color='tab:blue')
plt.hlines(y=shift["shift_accuracy"], xmin=shift["shift_accuracy"], xmax=shift["test_accuracy"],
               color='tab:blue', linewidth=3, alpha=0.7  )
               # Plot the diagonal line
plt.plot([0, 1], [0, 1], color='#627313')

plt.xlim((axmin[0],axmax[0]))
plt.ylim((axmin[1],axmax[1]))
plt.tight_layout()
plt.savefig(str(Path(__file__).parents[0]/f"hospital_readmission_shift"), bbox_inches='tight')
plt.show()

# %%
