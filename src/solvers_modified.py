import numpy as np
import random
# from pyscipopt import Model, quicksum
import time
import gc
from constants import (
    epsilon_N,
    epsilon_P,
    epsilon_R,
    TIME_LIMIT,
    PRINT_OUTPUT,
    RAM_LIMIT,
)


def separating_hyperplane(P, N, eps_P, eps_N, eps_R, theta, lamb, num_trials=10000, seeds=None):
    """
    Finds the initial separating hyperplane using the provided algorithm.

    Args:
        P (numpy.ndarray): Set of positive samples (numpy arrays).
        N (numpy.ndarray): Set of negative samples (numpy arrays).
        eps_P (float): Parameter for positive samples.
        eps_N (float): Parameter for negative samples.
        eps_R (float): Regularization parameter.
        theta (float): Scaling parameter.
        lamb (float): Lambda parameter for the optimization.
        num_trials (int): Number of random trials.

    Returns:
        tuple: Optimal hyperplane (w, c, reach), where w is the normal vector, c is the bias, and reach is the number of true positives.
    """
    if seeds: np.random.seed(seed=seeds)
    dim = P.shape[1]  # Dimension of the feature space
    L = -np.inf
    best_h = None

    for _ in range(num_trials):
        # Choose a random unit vector w
        w = np.random.randn(dim)
        w /= np.linalg.norm(w)

        # Choose a random point c in the unit hypercube
        c = np.random.uniform(0, 1, dim)
        c = -np.dot(w, c)

        # Compute x_tilde and y_tilde arrays
        distances_P = np.dot(P, w) - c
        distances_N = np.dot(N, w) - c

        x_tilde = np.where(distances_P >= eps_P, 1, 0)
        y_tilde = np.where(distances_N > -eps_N, 1, 0)

        # Compute V_tilde
        V_tilde = max(
            0, ((theta - 1) * np.sum(x_tilde) + theta * np.sum(y_tilde) + theta * eps_R)
        )

        # Compute L_tilde
        L_tilde = np.sum(x_tilde) - V_tilde * lamb

        # Update L and best_h
        if L_tilde > L:
            L = L_tilde
            best_h = (w, c, np.sum(x_tilde))

    return best_h


# def scip_solver(
#     *,
#     theta,
#     theta0,
#     theta1,
#     P,
#     N,
#     epsilon_P=epsilon_P,
#     epsilon_N=epsilon_N,
#     epsilon_R=epsilon_R,
#     lambda_param=None,
#     dataset_name='random_name',
#     run=True,
#     seeds=None,
# ):
#     """
#     Solves the wide-reach classification problem using SCIP for given positive and negative samples.

#     Parameters:
#         theta (float): Precision threshold.
#         theta0 (float): Parameter for optimization.
#         theta1 (float): Parameter for optimization.
#         P (numpy.ndarray): Feature matrix of positive samples (n_positive, n_features).
#         N (numpy.ndarray): Feature matrix of negative samples (n_negative, n_features).
#         epsilon_P (float, optional): Parameter for positive samples.
#         epsilon_N (float, optional): Parameter for negative samples.
#         epsilon_R (float, optional): Regularization parameter.
#         lambda_param (float, optional): Lambda parameter for the optimization.
#         dataset_name (str, optional): Name for output files.
#         run (bool, optional): Whether to solve the model or just create it.

#     Returns:
#         dict: Contains the reach, hyperplane parameters, bias, and precision violation, or an error message.
#     """
#     X = np.vstack((P, N))

#     # Update indices for P and N after combining
#     num_positive = P.shape[0]

#     P_indices = range(num_positive)
#     N_indices = range(num_positive, X.shape[0])
    
#     # Parameters
#     if not lambda_param:
#         lambda_param = (num_positive + 1) * theta1

#     initial_h = separating_hyperplane(
#         P, N, epsilon_P, epsilon_N, epsilon_R, theta, lambda_param, num_trials=10000, seeds=seeds
#     )

#     # Create the SCIP model
#     model = Model("Wide-Reach Classification")

#     if not PRINT_OUTPUT:
#         model.hideOutput()
    
#     if TIME_LIMIT:
#         model.setRealParam('limits/time', TIME_LIMIT)
    
#     # Adjusting hyperparameters
#     # model.setIntParam('randomization/randomhift', 42)
#     # model.setRealParam('limits/gap', 0.0001)
#     # model.setRealParam('limits/absgap', 0.0001)
#     model.setIntParam('separating/maxrounds', -1)  # Unlimited cutting plane rounds
    
#     # Decision variables
#     x = {}
#     for i in P_indices:
#         x[i] = model.addVar(vtype="B", name=f"x_{i}")
    
#     y = {}
#     for j in range(len(N_indices)):
#         y[j] = model.addVar(vtype="B", name=f"y_{j}")
    
#     w = {}
#     for d in range(X.shape[1]):
#         w[d] = model.addVar(lb=None, ub=None, name=f"w_{d}")
    
#     c = model.addVar(lb=None, ub=None, name="c")
#     V = model.addVar(lb=0, name="V")

#     # Set initial solution if available
#     if initial_h is not None:
#         init_w, init_c, reach = initial_h
#         print(f"Initial reach = {reach}")
#         distances_P = np.dot(P, init_w) - init_c
#         distances_N = np.dot(N, init_w) - init_c

#         xs = distances_P >= epsilon_P
#         ys = distances_N > -epsilon_N
    
        
#     # Constraint: Precision constraint violation
#     precision_expr = (theta - 1) * quicksum(x[i] for i in P_indices) + \
#                     theta * quicksum(y[j] for j in range(len(N_indices))) + \
#                     theta * epsilon_R
    
#     model.addCons(V >= precision_expr, name="PrecisionConstraint")

#     # Constraints: Classification constraints for positive samples
#     for i, p_idx in enumerate(P_indices):
#         pos_expr = 1 + quicksum(w[d] * X[p_idx, d] for d in range(X.shape[1])) - c - epsilon_P
#         model.addCons(x[i] <= pos_expr, name=f"Positive_{i}")

#     for j, n_idx in enumerate(N_indices):
#         neg_expr = quicksum(w[d] * X[n_idx, d] for d in range(X.shape[1])) - c + epsilon_N
#         model.addCons(y[j] >= neg_expr, name=f"Negative_{j}")

#     objective = quicksum(x[i] for i in P_indices) - lambda_param * V
#     model.setObjective(objective, "maximize")

#     model.writeProblem(f"{dataset_name}.lp")

#     if run:
#         if initial_h is not None:
#             init_w, init_c, reach = initial_h
#             distances_P = np.dot(P, init_w) - init_c
#             distances_N = np.dot(N, init_w) - init_c

#             xs = distances_P >= epsilon_P
#             ys = distances_N > -epsilon_N
            
#             print(f"Applying initial solution with reach={reach}")
            
#             sol = model.createPartialSol()
            
#             for i in P_indices:
#                 model.setSolVal(sol, x[i], 1.0 if xs[i] else 0.0)
            
#             for j in range(len(N_indices)):
#                 model.setSolVal(sol, y[j], 1.0 if ys[j] else 0.0)
            
#             for d in range(X.shape[1]):
#                 model.setSolVal(sol, w[d], init_w[d])
            
#             model.setSolVal(sol, c, init_c)
#             v_val = max(0, ((theta - 1) * xs.sum() + theta * ys.sum() + theta * epsilon_R))
#             model.setSolVal(sol, V, v_val)
            
#             # Try adding the solution as a heuristic (optional)
#             try:
#                 model.addSol(sol)
#             except:
#                 print("Warning: Could not add initial solution as heuristic")
            
#             # Free the partial solution (optional)
#             del sol

            
#         start_time = time.time()
#         model.optimize()
#         end_time = time.time()

#         if model.getStatus() == "optimal" or model.getStatus() == "timelimit":
#             results = {
#                 "Initial reach": reach,
#                 "Reach": sum(model.getVal(x[i]) for i in P_indices),
#                 "Hyperplane w": [model.getVal(w[d]) for d in range(X.shape[1])],
#                 "Bias c": model.getVal(c),
#                 "X": [model.getVal(x[d]) for d in range(len(P))],
#                 "Y": [model.getVal(y[d]) for d in range(len(N))],
#                 "Precision Violation V": model.getVal(V),
#                 "Node Count": model.getNNodes(),
#                 "Time taken": end_time - start_time,
#             }
#         else:
#             results = {"Error": f"No optimal solution found. Status: {model.getStatus()}"}

#         model.freeProb()
#         del model
#         gc.collect()

#         return results

#     else:
#         return None
