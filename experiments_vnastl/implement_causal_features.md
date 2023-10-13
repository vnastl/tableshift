# Implement causal features (in the tableshift environment)
1. Add configuration in tableshift/tableshift/configs/benchmark_configs.py
   In particular, copy the configuration from the original benchmark dataset and change the name (e.g. add "_causal")
2. Go to the benchmark dataset file in tableshift/tableshift/datasets and add the FeatureList of causal features
3. Add task configuration in tableshift/tableshift/core/tasks.py
   In particular, copy the task configuration from the original benchmark dataset and adapt the dataset name as well as name of the FeatureList
4. Use the dataset name for experiment in tableshift/experiments_vnastl/run_experiment.py

# Change causal features (after first implementation)
Go to the benchmark dataset file in tableshift/tableshift/datasets and change the FeatureList of causal features