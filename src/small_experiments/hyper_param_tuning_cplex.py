import numpy as np
from src.real_datasets import (
    BreastCancerDataset,
    WineQualityRedDataset,
    WineQualityWhiteDataset,
    SouthGermanCreditDataset,
    CropMappingDataset,
    s1, s2, s3
)
from src.synthetic_datasets import (
    ClusterDataset,
    TwoClusterDataset,
    DiffusedBenchmark,
    PrismDataset,
    TruncatedNormalPrism,
)
from src.solvers import cplex_solver
import os
import subprocess
import re

datasets = {
    "Breast Cancer": BreastCancerDataset(),
    "Wine Quality Red": WineQualityRedDataset(),
    "Wine Quality White": WineQualityWhiteDataset(),
    "South German Credit": SouthGermanCreditDataset(),
    "Crop Mapping": CropMappingDataset(),
    "Cluster 8": ClusterDataset(d=8),
    "Two Cluster 8": TwoClusterDataset(d=8),
    "Cluster": ClusterDataset(d=11),
    "Two Cluster": TwoClusterDataset(d=11),
    "Diffused Benchmark": DiffusedBenchmark(),
    "Prism": PrismDataset(),
    "Truncated Normal Prism": TruncatedNormalPrism(),
}

ds_list = []

for dataset_name, dataset in datasets.items():
    P, N = dataset.generate()
    theta_0, theta_1, theta, lambda_param = dataset.params()
    ds_name = "".join(dataset_name.lower().split(' '))
    ds_list.append(ds_name)
    cplex_solver(
        theta=theta,
        theta0=theta_0,
        theta1=theta_1,
        P=P,
        N=N,
        lambda_param=lambda_param,
        dataset_name=ds_name,
        run=False
    )

# tune_time_limit = 120  # Adjust the tuning time limit

best_params = {}
timelimit = 100

# for ds in ds_list:
files_str = " ".join([f'"read {file}"' for file in ds_list])
# command = f'/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex -c "read {ds}.lp" "tools tune"'
command = f'/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex -c "set timelimit {timelimit}" {files_str} "tools tune"'

print(f"Running command: {command}")
process = subprocess.run(command, shell=True, text=True)
output = process.stdout
