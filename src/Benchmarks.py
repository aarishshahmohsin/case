import numpy as np
from real_datasets import (
    BreastCancerDataset,
    WineQualityRedDataset,
    WineQualityWhiteDataset,
    SouthGermanCreditDataset,
    CropMappingDataset,
)
from synthetic_datasets import (
    ClusterDataset,
    TwoClusterDataset,
    DiffusedBenchmark,
    PrismDataset,
    TruncatedNormalPrism,
)
from solvers import cplex_solver, gurobi_solver
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
}

times = 5
results = {}

final_res = []


for dataset_name, dataset in datasets.items():
    P, N = dataset.generate()
    theta_0, theta_1, theta, lambda_param = dataset.params()
    final_res = []
    for i in range(times):
        res_gurobi = gurobi_solver(
            theta=theta,
            theta0=theta_0,
            theta1=theta_1,
            P=P,
            N=N,
            lambda_param=lambda_param,
            dataset_name=dataset_name
        )
        # res_cplex = cplex_solver(
        #     theta=theta,
        #     theta0=theta_0,
        #     theta1=theta_1,
        #     P=P,
        #     N=N,
        #     lambda_param=lambda_param,
        # )
        # dataset_results = [res_gurobi, res_cplex]
        # print(dataset_name, dataset_results[0]['Reach'], dataset_results[1]['Reach'])
        # final_res.append([dataset_name,dataset_results[0]['Reach'], dataset_results[1]['Reach']])
    # results[dataset_name] = final_res

# for name, res in results.items():
#     print(res[0][0]["Reach"], res[0][1]["Reach"])

print(final_res)
