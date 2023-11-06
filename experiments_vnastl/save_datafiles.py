import os
import pickle
import pandas as pd
from tableshift import get_dataset, get_iid_dataset
os.chdir("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/tableshift")

experiments = [
                # "anes",
                # "acspubcov",
                # "college_scorecard",
                # "acsunemployment",
                # "physionet",
                # "acsfoodstamps",
                # "assistments",
                # "brfss_diabetes",
                "acsincome",
                "meps"
                ]
cache_dir="tmp"

for experiment in experiments:
    dset = get_dataset(experiment, cache_dir)

    with open(f'experiments_vnastl/saved_datafiles/{experiment}_dset.pkl', 'wb') as f:
        # Unpickle the object
        pickle.dump(dset, f)

    X_id_train, y_id_train, _, _ = dset.get_pandas("train")
    X_id_test, y_id_test, _, _ = dset.get_pandas("id_test")
    X_id_val, y_id_val, _, _ = dset.get_pandas("validation")

    X_id = pd.concat([X_id_train, X_id_test, X_id_val])
    y_id = pd.concat([y_id_train,y_id_test,y_id_val])

    X_ood_test, y_ood_test, _, _ = dset.get_pandas("ood_test")
    X_ood_val, y_ood_val, _, _ = dset.get_pandas("ood_validation")

    X_ood = pd.concat([X_ood_test,X_ood_val])
    y_ood = pd.concat([y_ood_test,y_ood_val])

    with open(f'experiments_vnastl/saved_datafiles/{experiment}_xy.pkl', 'wb') as f:
        # Unpickle the object
        pickle.dump({
            "X_id": X_id,
            "X_ood": X_ood,
            "y_id":y_id,
            "y_ood": y_ood,
        }, f)

    dset_iid = get_iid_dataset(experiment, cache_dir)
    with open(f'experiments_vnastl/saved_datafiles/{experiment}_dset_iid.pkl', 'wb') as f:
        # Unpickle the object
        pickle.dump(dset_iid, f)