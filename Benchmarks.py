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
from src.solvers.solvers import (
    cplex_solver,
    gurobi_solver,
    scip_solver,
    scip_solver_c,
)
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
    's1': s1(),
    's2': s2(),
    's3': s3(),
    '360 prism': PrismDataset(num_positive=360, s=0.707, d0=2, d=11),
    "Truncated Normal Prism": TruncatedNormalPrism(),
    "Prism": PrismDataset(d=11),
}

N_ITERATIONS = 1
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50]
rows = []

xs = []
ys = []



results_df = pd.DataFrame(
    columns=["Dataset", "Solver", "Initial Reach", "Time Taken", "Final Reach"] 
)

solvers = {
    "Gurobi": gurobi_solver, 
    "CPlex": cplex_solver,
    "SCIP": scip_solver,
    "SCIP_C": scip_solver_c,
}


for i in range(N_ITERATIONS):
    for dataset_name, dataset in datasets.items():
        P, N = dataset.generate(normalize=True)
        # P, N = dataset.generate()
        theta_0, theta_1, theta, lambda_param = dataset.params()

        for solver_name, solver in solvers.items():
            res = solver(
                theta=theta,
                theta0=theta_0,
                theta1=theta_1,
                P=P,
                N=N,
                lambda_param=lambda_param,
                dataset_name=dataset_name,
                run=True,
                seeds=SEEDS[i],

            )

            if res:
                row = {
                    "Dataset": dataset_name,
                    "Solver": solver_name,
                    "Initial Reach": int(res["Initial reach"]),
                    "Time Taken": res["Time taken"],
                    "Final Reach": res["Reach"],
                }
                print(res['X'])
                xs.append(res['X'])
                print(res['Y'])
                ys.append(res['Y'])
                print(row)
                print(solver_name)
                rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv("experiment_results.csv", index=False)

final_ar = []
for i in range(len(xs)):
    final_ar.append('X')
    final_ar.append(xs[i])
    final_ar.append('Y')
    final_ar.append(ys[i])

for line in final_ar:
    print(line)
