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
from src.solvers.new_solvers import (
    gurobi_solver,
)
import pandas as pd

datasets = {
    "Breast Cancer": BreastCancerDataset(),
    # "Wine Quality Red": WineQualityRedDataset(),
    # "Wine Quality White": WineQualityWhiteDataset(),
    # "South German Credit": SouthGermanCreditDataset(),
    # "Crop Mapping": CropMappingDataset(),
    # "Cluster 8": ClusterDataset(d=8),
    # "Two Cluster 8": TwoClusterDataset(d=8),
    # "Cluster": ClusterDataset(d=11),
    # "Two Cluster": TwoClusterDataset(d=11),
    # "Diffused Benchmark": DiffusedBenchmark(),
    # 's1': s1(),
    # 's2': s2(),
    # 's3': s3(),
    # '360 prism': PrismDataset(num_positive=360, s=0.707, d0=2, d=11),
    # "Truncated Normal Prism": TruncatedNormalPrism(),
    # "Prism": PrismDataset(d=11),
}


times = 1
xs = [] 
ys = []

final_res = []
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50]
import numpy as np


results_df = pd.DataFrame(
    columns=["Dataset", "Solver", "Initial Reach", "Time Taken", "Final Reach"]
)
rows = []

# dataset = PrismDataset(num_positive=360, s=0.707, d0=1, d=2)
dataset = BreastCancerDataset() 
P, N = dataset.generate() 
print(P.shape, N.shape)
P = P[:, :2]
N = N[:, :2]
theta_0, theta_1, theta, lambda_param = dataset.params()
res_gurobi = gurobi_solver(
    theta=theta,
    theta0=theta_0,
    theta1=theta_1,
    P=P,
    N=N,
    lambda_param=lambda_param,
    dataset_name="breast cancer",
    run=True,
    seeds=42,
)

print(res_gurobi['Intermediate steps'])
print(len(res_gurobi['Intermediate steps']))
print(res_gurobi['Reach'])

intermediate_solutions = res_gurobi['Intermediate steps']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_hyperplanes(P, N, intermediate_solutions, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(np.min(np.vstack((P, N))[:, 0]) - 1, np.max(np.vstack((P, N))[:, 0]) + 1)
    ax.set_ylim(np.min(np.vstack((P, N))[:, 1]) - 1, np.max(np.vstack((P, N))[:, 1]) + 1)
    ax.set_title("Animated Hyperplanes from Gurobi")

    # Initialize scatter plots
    pos_scatter = ax.scatter([], [], s=80, label="Positive (P)")
    neg_scatter = ax.scatter([], [], s=80, label="Negative (N)")
    hyperplane_line, = ax.plot([], [], 'k--', linewidth=2, label="Hyperplane")

    ax.legend()

    def update(frame):
        sol = intermediate_solutions[frame]
        w = np.array(sol['w'])
        c = sol['c']
        x = np.array(sol['x'])
        y = np.array(sol['y'])

        # Compute hyperplane line: w0 * x + w1 * y = c => y = (c - w0*x)/w1
        x_vals = np.array(ax.get_xlim())
        if w[1] != 0:
            y_vals = (c - w[0]*x_vals) / w[1]
        else:
            y_vals = np.array([0, 0])  # Avoid divide by zero

        hyperplane_line.set_data(x_vals, y_vals)

        # Colors based on indicator variables
        pos_colors = np.where(x >= 0.5, 'green', 'lightgray')
        neg_colors = np.where(y >= 0.5, 'red', 'lightgray')

        pos_scatter.set_offsets(P)
        pos_scatter.set_color(pos_colors)
        neg_scatter.set_offsets(N)
        neg_scatter.set_color(neg_colors)

        ax.set_title(f"Step {frame+1}/{len(intermediate_solutions)} | Objective: {sol['obj']:.2f}")

        return pos_scatter, neg_scatter, hyperplane_line

    anim = FuncAnimation(fig, update, frames=len(intermediate_solutions), interval=300, blit=False, repeat=True)

    if save_path:
        anim.save(save_path, fps=1)
    else:
        plt.show()


animate_hyperplanes(P, N, intermediate_solutions)
