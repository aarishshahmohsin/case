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
from utils import plot_P_N, plot_P_N_3d

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
    "Prism": PrismDataset(d=11),
    '360 prism': PrismDataset(num_positive=360, s=0.909, d0=3, d=11),
    "Truncated Normal Prism": TruncatedNormalPrism(),
    's1': s1(),
    's2': s2(),
    's3': s3(),
}


times = 1
results = {}

final_res = []


for dataset_name, dataset in datasets.items():
    P, N = dataset.generate()
    # plot_P_N_3d(P, N)
    # print(P.shape, N.shape)
    # print(P, N)
    # plot_P_N(P, N)
    # sums = []
    # for i in range(len(P)):
    #     sum = 0
    #     for j in range(3):
    #         sum += P[i][j]
    #     sums.append(float(sum))

    # sums.sort()
    # with open('sorted_prism.txt', 'w') as f:
    #     for i in sums:
    #         f.write(f'{i}\n')
    # print(sums)
    # with open('prism.txt', 'w') as f:
    #     for i in P:
    #         f.write(f'{i}\n')
    #     for i in N:
    #         f.write(f'{i}\n')
    theta_0, theta_1, theta, lambda_param = dataset.params()
    # # # final_res = []
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
        

# print(final_res)
