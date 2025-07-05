import numpy as np
import time
from src.datasets.real_datasets import (
    BreastCancerDataset,
    WineQualityRedDataset,
    WineQualityWhiteDataset,
    SouthGermanCreditDataset,
    CropMappingDataset,
    s1,
    s2,
    s3,
)
from src.datasets.synthetic_datasets import (
    ClusterDataset,
    TwoClusterDataset,
    DiffusedBenchmark,
    PrismDataset,
    TruncatedNormalPrism,
)
from solvers.solvers import cplex_solver, gurobi_solver, scip_solver, separating_hyperplane, scip_solver_c
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
    's1': s1(),
    's2': s2(),
    's3': s3(),
    '360 prism': PrismDataset(num_positive=360, s=0.707, d0=2, d=11),
    "Truncated Normal Prism": TruncatedNormalPrism(),
    "Prism": PrismDataset(d=11),
}


times = 1
results = {}

final_res = []
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50]
import numpy as np

    
results_df = pd.DataFrame(columns=["Dataset", "Solver", "Initial Reach", "Time Taken", "Final Reach"])
rows = [] 


for i in range(times):
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

    for dataset_name, dataset in datasets.items():
        # P,N = dataset.generate()
        P,N = dataset.generate(normalize=True)
        theta_0, theta_1, theta, lambda_param = dataset.params()

        res_gurobi = gurobi_solver(
            theta=theta,
            theta0=theta_0,
            theta1=theta_1,
            P=P,
            N=N,
            lambda_param=lambda_param,
            dataset_name=dataset_name,
            run=True,
            seeds=seeds[i],
        )

        res_scip = scip_solver(
            theta=theta,
            theta0=theta_0,
            theta1=theta_1,
            P=P,
            N=N,
            lambda_param=lambda_param,
            dataset_name=dataset_name,
            run=True,
            seeds=seeds[i],
        )

        res_cplex = cplex_solver(
            theta=theta,
            theta0=theta_0,
            theta1=theta_1,
            P=P,
            N=N,
            lambda_param=lambda_param,
            dataset_name=dataset_name,
            run=True,
            seeds=seeds[i],
        )

        res_scip_c = scip_solver_c(
            theta=theta,
            theta0=theta_0,
            theta1=theta_1,
            P=P,
            N=N,
            lambda_param=lambda_param,
            dataset_name=dataset_name,
            run=True,
            seeds=seeds[i],
        )

        for solver_name, res in [("Gurobi", res_gurobi)]:
            if res:
                row = {
                    "Dataset": dataset_name,
                    "Solver": solver_name,
                    "Initial Reach": int(res['Initial reach']),
                    "Time Taken": res['Time taken'],
                    "Final Reach": res['Reach']
                }
                print(res['X'])
                print(res['Y'])
                print(row)
                rows.append(row)


        # t = time.time()
        # print(dataset_name)
        # print(res_gurobi['Reach'])
        # print(res_gurobi['Time taken'])
        # final_res.append([dataset_name, res_gurobi['Reach'], res_gurobi['Time taken']])
        # e = time.time()
        # print(e - t)


results_df = pd.DataFrame(rows)
results_df.to_csv("experiment_results.csv", index=False)

# print(final_res)
