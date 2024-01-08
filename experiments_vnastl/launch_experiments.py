#!/usr/bin/env python3
"""
Python script to launch condor jobs for task.
"""
# %%
import sys
import random
import dataclasses
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tableshift.datasets import ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER, BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER


if __name__ == '__main__':
    import htcondor
    import classad

# Number of task to run per algorithm, per dataset
N_TRIALS    = 1
# N_TRIALS    = 30
# N_TRIALS    = 100

# Cluster settings
JOB_MIN_BID = 100    # htcondor bid (min. is 15 apparently...)
JOB_CPUS = 1     # number of CPUs per experiment (per cluster job)
JOB_MEMORY_GB = 128   # GBs of memory
BIG_JOB_MEMORY_GB = 256

VERBOSE = True

TASKS = [
    # "acsincome",
    # "acsincome_causal",
    # "acsincome_arguablycausal",
    # "acsincome_anticausal",

    # "acspubcov",
    # "acspubcov_causal",

    # "acsfoodstamps",
    # "acsfoodstamps_causal",
    # "acsfoodstamps_arguablycausal",

    # "acsunemployment",
    # "acsunemployment_causal",
    # "acsunemployment_arguablycausal",

    # "anes",
    # "anes_causal",

    # "assistments",
    # "assistments_causal",

    # "brfss_diabetes",
    # "brfss_diabetes_causal",
    # "brfss_diabetes_arguablycausal",
    # "brfss_diabetes_anticausal",

    # "brfss_blood_pressure",
    # "brfss_blood_pressure_causal",

    # "college_scorecard",
    # "college_scorecard_causal",

    # "nhanes_lead", 
    # "nhanes_lead_causal",

    # "diabetes_readmission", 
    # "diabetes_readmission_causal",

    # "mimic_extract_los_3",
    # "mimic_extract_los_3_causal",

    # "mimic_extract_mort_hosp",
    # "mimic_extract_mort_hosp_causal",

    # "physionet", 
    # "physionet_causal",

    # "sipp", 
    # "sipp_causal",

    # "meps", 
    # "meps_causal",
]

# Robustness checks
# for index in range(ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER):
#         TASKS.append("acsincome_causal_test_"+f"{index}")

# for index in range(ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#         TASKS.append("acsincome_arguablycausal_test_"+f"{index}")

# for index in range(ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER):
#         TASKS.append("acsfoodstamps_causal_test_"+f"{index}")

# for index in range(ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#         TASKS.append("acsfoodstamps_arguablycausal_test_"+f"{index}")

# for index in range(BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER):
#         TASKS.append("brfss_diabetes_causal_test_"+f"{index}")

# for index in range(BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#         TASKS.append("brfss_diabetes_arguablycausal_test_"+f"{index}")
TASKS.append("brfss_diabetes_arguablycausal_test_4")
              
# Useful directories
if __name__ == '__main__':
    ROOT_DIR = Path("/home")
    # ROOT_DIR = Path("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-vs-noncausal/code")

    # Data directory
    DATA_DIR = ROOT_DIR / "fast/vnastl/data"  # TODO check again if it is in fact the fast driver

    # Results directory
    RESULTS_DIR = ROOT_DIR / "vnastl/results" #/tableshift/experiments_vnastl"
    RESULTS_DIR.mkdir(exist_ok=True, parents=False)

    # Directory to save cluster logs and job stdout/stderr
    CLUSTER_LOGS_SAVE_DIR =  ROOT_DIR / "fast/vnastl/cluster-logs"
    CLUSTER_LOGS_SAVE_DIR.mkdir(exist_ok=True)

    CLUSTER_LOGS_SAVE_ERR_DIR = CLUSTER_LOGS_SAVE_DIR / "error"
    CLUSTER_LOGS_SAVE_ERR_DIR.mkdir(exist_ok=True)

    CLUSTER_LOGS_SAVE_OUT_DIR = CLUSTER_LOGS_SAVE_DIR / "output"
    CLUSTER_LOGS_SAVE_OUT_DIR.mkdir(exist_ok=True)

    CLUSTER_LOGS_SAVE_LOG_DIR = CLUSTER_LOGS_SAVE_DIR / "logs"
    CLUSTER_LOGS_SAVE_LOG_DIR.mkdir(exist_ok=True)

####################################################
#  START of: details on which task to run.  #
####################################################
@dataclasses.dataclass
class ExperimentConfigs:
    name: str
    job_memory_gb: int  # = JOB_MEMORY_GB

    n_trials: int = N_TRIALS
    job_cpus: int = JOB_CPUS
    job_gpus: int = 0
    job_bid: int = JOB_MIN_BID

    def __post_init__(self):
        self.job_bid = max(self.job_bid, JOB_MIN_BID)       # enforce min bid

if __name__ == '__main__':
    all_task = []
    for task in TASKS:
        all_task.append(ExperimentConfigs(
                    name=task,
                    job_memory_gb=JOB_MEMORY_GB))
        
##################################################
#  END of: details on which task to run.  #
##################################################

    def launch_task_jobs(
        task: str,
        exp_obj: ExperimentConfigs,
    ):
        """Launches the cluster jobs to execute all `n_trials` of a given experiment.

        Parameters
        ----------
        task : str
            The name of the task/data to use.
        exp_obj : ExperimentConfigs
            The detailed configs to run an experiment.
        """

        # Name/prefix for cluster logs related to this job
        cluster_job_err_name = str(
            CLUSTER_LOGS_SAVE_ERR_DIR
            / f"launch_{exp_obj.name}_$(Cluster).$(Process)"
        )

        cluster_job_out_name = str(
            CLUSTER_LOGS_SAVE_OUT_DIR
            / f"launch_{exp_obj.name}_$(Cluster).$(Process)"
        )

        cluster_job_log_name = str(
            CLUSTER_LOGS_SAVE_LOG_DIR
            / f"launch_{exp_obj.name}_$(Cluster).$(Process)"
        )

        EXP_RESULTS_DIR = RESULTS_DIR
        EXP_RESULTS_DIR.mkdir(exist_ok=True, parents=False)

        # Construct job description
        job_description = htcondor.Submit({
            "executable": "/home/vnastl/miniconda3/envs/tableshift/bin/python3",  # correct env for the python executable
            # "arguments": "foo.py",    # NOTE: used for testing
            "arguments": (
                "/home/vnastl/tableshift/experiments_vnastl/launch_one_experiment.py "
                f"--task {exp_obj.name} "
                f"--DATA_DIR {str(DATA_DIR)} "
                f"--RESULTS_DIR {str(EXP_RESULTS_DIR)} "
                f"--CLUSTER_LOGS_SAVE_DIR {str(CLUSTER_LOGS_SAVE_DIR)} "
                f"--N_TRIALS {N_TRIALS} "
                f"--JOB_CPUS {JOB_CPUS} "
                f"--JOB_MEMORY_GB {JOB_MEMORY_GB} "
                f"--JOB_MIN_BID {JOB_MIN_BID} "
            ),
            "output": f"{cluster_job_out_name}.out",
            "error": f"{cluster_job_err_name}.err",
            "log": f"{cluster_job_log_name}.log",
            "request_cpus": f"{exp_obj.job_cpus}",
            "request_gpus": f"{exp_obj.job_gpus}",
            "request_memory": f"{exp_obj.job_memory_gb}GB",
            # "request_disk": "2GB",
            "jobprio": f"{exp_obj.job_bid - 1000}",
            "notify_user": "vivian.nastl@tuebingen.mpg.de",
            "notification": "error",
            # "job_seed_macro": f"$(Process) + {random.randrange(int(1e9))}",      # add random salt to all job seeds
            # "job_seed": "$INT(job_seed_macro)",

            # Concurrency limits:
            # > each job uses this amount of resources out of a pool of 10k
            # "concurrency_limits": "user.theoremfivepointsix:10000",     # 1 job
            "concurrency_limits": "user.theoremfivepointsix:100",     # 100 jobs in parallel
            # "concurrency_limits": "user.theoremfivepointsix:50",     # 200 jobs in parallel

            "+MaxRunningPrice": 100,
            # "+RunningPriceExceededAction": classad.quote("restart"),
        })

        # Submit `n_trials` jobs to the scheduler
        schedd = htcondor.Schedd()
        submit_result = schedd.submit(job_description, count=exp_obj.n_trials)

        if VERBOSE:
            print(
                f"Launched {submit_result.num_procs()} processes with "
                f"cluster-ID={submit_result.cluster()}\n")

    # Log all task that we want to run
    num_task = len(all_task)
    print(
        f"\nLaunching the following tasks (n={num_task}):\n")
    
    # For each task
    print(
            f"\n*** *** ***\n"
            f"Launching {len(all_task)} * {N_TRIALS} = "
            f"{N_TRIALS * len(all_task)} "
            f"tasks"
            f"\n*** *** ***\n"
        )

    for i, exp_obj in enumerate(all_task):
            print(f"{i}. Launching {exp_obj.n_trials} trials for the task '{exp_obj.name}'")
            launch_task_jobs(task=task, exp_obj=exp_obj)

