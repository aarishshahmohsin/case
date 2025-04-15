import numpy as np
import time
from real_datasets import (
    BreastCancerDataset,
    WineQualityRedDataset,
    WineQualityWhiteDataset,
    SouthGermanCreditDataset,
    CropMappingDataset,
    s1,
    s2,
    s3,
)
from synthetic_datasets import (
    ClusterDataset,
    TwoClusterDataset,
    DiffusedBenchmark,
    PrismDataset,
    TruncatedNormalPrism,
)
from solvers import cplex_solver, gurobi_solver, scip_solver
import pandas as pd

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
    's1': s1(),
    's2': s2(),
    's4': s3(),
}


times = 5
results = {}

final_res = []


for dataset_name, dataset in datasets.items():
    P, N = dataset.generate()
    theta_0, theta_1, theta, lambda_param = dataset.params()
    #   final_res = []
    for i in range(times):
        res_gurobi = scip_solver(
            theta=theta,
            theta0=theta_0,
            theta1=theta_1,
            P=P,
            N=N,
            lambda_param=lambda_param,
            dataset_name=dataset_name
        )
        t = time.time()
        print(dataset_name)
        print(res_gurobi['Reach'])
        print(res_gurobi['Time taken'])
        final_res.append([dataset_name, res_gurobi['Reach'], res_gurobi['Time taken']])
        e = time.time()
        print(e - t)
        

print(final_res)
