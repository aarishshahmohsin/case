import numpy as np
import time
from scip_c_wrapper import call_scip_solver  # Use your new C wrapper
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
from solvers_modified import separating_hyperplane
import pandas as pd
from utils import plot_P_N, plot_P_N_3d
from constants import epsilon_N, epsilon_P, epsilon_R

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

results_df = pd.DataFrame(columns=["Dataset", "Solver", "Initial Reach", "Time Taken", "Final Reach"])
rows = [] 

try:
    for i in range(times):
        for dataset_name, dataset in datasets.items():
            print(f"Processing {dataset_name}, iteration {i+1}/{times}")
            
            # Generate dataset
            P, N = dataset.generate()
            theta_0, theta_1, theta, lambda_param = dataset.params()
            
            # Get initial hyperplane using your existing function
            init_w, init_c, reach = separating_hyperplane(
                P, N, epsilon_P, epsilon_N, epsilon_R, theta, lambda_param, 
                num_trials=10000, seeds=seeds
            )
            
            # Convert numpy arrays to ensure they're the right type
            P = np.array(P, dtype=np.float64)
            N = np.array(N, dtype=np.float64)
            init_w = np.array(init_w, dtype=np.float64).tolist()
            
            # Call the C SCIP solver
            res_scip = call_scip_solver(
                P=P, 
                N=N, 
                init_w=init_w, 
                init_c=float(init_c), 
                theta=float(theta), 
                lambda_param=float(lambda_param) if lambda_param is not None else None,
                epsilon_P=epsilon_P, 
                epsilon_N=epsilon_N, 
                epsilon_R=epsilon_R,
                dataset_name=dataset_name.replace(" ", "_"),  # Remove spaces for filename
                print_output=False,  # Set to True if you want to see SCIP output
                time_limit=300.0
            )
            
            # Check if there was an error
            if "Error" in res_scip:
                print(f"Error in {dataset_name}: {res_scip['Error']}")
                continue
            
            # Add result to rows
            for solver_name, res in [("SCIP_C", res_scip)]:
                row = {
                    "Dataset": dataset_name,
                    "Solver": solver_name,
                    "Initial Reach": int(res['Initial reach']),
                    "Time Taken": res['Time taken'],
                    "Final Reach": res['Reach']
                }
                print(row)
                rows.append(row)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Save results even if there was an error
    if rows:
        results_df = pd.DataFrame(rows)
        results_df.to_csv("experiment_results.csv", index=False)
        print(f"Saved {len(rows)} results to experiment_results.csv")
    else:
        print("No results to save")