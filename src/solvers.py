import numpy as np
from gurobipy import Model as GurobiModel, GRB, Env
from docplex.mp.model import Model as CplexModel
import time
import gc

from constants import (
    epsilon_N as eps_N,
    epsilon_P as eps_P,
    epsilon_R as eps_R,
    TIME_LIMIT,
    PRINT_OUTPUT,
    RAM_LIMIT,
)


def gurobi_solver(
    *, theta, P, N, epsilon_P=eps_P, epsilon_N=eps_N, epsilon_R=eps_R, lambda_param=None
):
    """
    Solves the wide-reach classification problem for given positive and negative samples.

    Parameters:
        theta (float): Precision threshold.
        P (numpy.ndarray): Feature matrix of positive samples (n_positive, n_features).
        N (numpy.ndarray): Feature matrix of negative samples (n_negative, n_features).

    Returns:
        dict: Contains the reach, hyperplane parameters, bias, and precision violation, or an error message.
    """
    X = np.vstack((P, N))

    # Update indices for P and N after combining
    num_positive = P.shape[0]
    P_indices = range(num_positive)
    N_indices = range(num_positive, X.shape[0])

    # Parameters
    if not lambda_param:
        lambda_param = (num_positive + 1) / theta

    # Create the Gurobi model
    model = GurobiModel("Wide-Reach Classification")
    env = Env()

    if TIME_LIMIT:
        model.setParam("TimeLimit", 120)
    if not PRINT_OUTPUT:
        model.setParam("OutputFlag", 0)
    if RAM_LIMIT:
        model.setParam("MemLimit", 4096)

    # Decision variables
    x = model.addVars(num_positive, vtype=GRB.BINARY, name="x")
    y = model.addVars(len(N_indices), vtype=GRB.BINARY, name="y")
    w = model.addVars(X.shape[1], lb=-GRB.INFINITY, name="w")
    c = model.addVar(lb=-GRB.INFINITY, name="c")
    V = model.addVar(lb=0, name="V")

    # Objective: Maximize the reach minus penalty for precision violation
    model.setObjective(sum(x[i] for i in P_indices) - lambda_param * V, GRB.MAXIMIZE)

    # Constraint: Precision constraint violation
    model.addConstr(
        V
        >= (theta - 1) * sum(x[i] for i in P_indices)
        + theta * sum(y[j] for j in range(len(N_indices)))
        + theta * epsilon_R,
        "PrecisionConstraint",
    )

    # Constraints: Classification constraints for positive samples
    for i, p_idx in enumerate(P_indices):
        model.addConstr(
            x[i]
            <= 1 + sum(w[d] * X[p_idx, d] for d in range(X.shape[1])) - c - epsilon_P,
            name=f"Positive_{i}",
        )

    # Constraints: Classification constraints for negative samples
    for j, n_idx in enumerate(N_indices):
        model.addConstr(
            y[j] >= sum(w[d] * X[n_idx, d] for d in range(X.shape[1])) - c + epsilon_N,
            name=f"Negative_{j}",
        )

    # Solve the model
    start_time = time.time()
    model.optimize()
    end_time = time.time()

    # Check and return results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        results = {
            "Reach": sum(x[i].x for i in P_indices),
            "Hyperplane w": [w[d].x for d in range(X.shape[1])],
            "Bias c": c.x,
            "X": [x[d].x for d in range(len(P))],
            "Y": [y[d].x for d in range(len(N))],
            "Precision Violation V": V.x,
            "Node Count": model.NodeCount,
            "Time taken": end_time - start_time,
        }
        # return results
    else:
        results = {"Error": "No optimal solution found."}

    # dispose everything
    model.dispose()
    env.dispose()

    del model
    del env

    gc.collect()

    return results


def cplex_solver(
    *, theta, P, N, epsilon_P=eps_P, epsilon_N=eps_N, epsilon_R=eps_R, lambda_param=None
):
    """
    Solves the wide-reach classification problem using DOcplex for given positive and negative samples.

    Parameters:
        theta (float): Precision threshold.
        P (numpy.ndarray): Feature matrix of positive samples (n_positive, n_features).
        N (numpy.ndarray): Feature matrix of negative samples (n_negative, n_features).

    Returns:
        dict: Contains the reach, hyperplane parameters, bias, and precision violation, or an error message.
    """
    X = np.vstack((P, N))

    # Update indices for P and N after combining
    num_positive = P.shape[0]
    P_indices = range(num_positive)
    N_indices = range(num_positive, X.shape[0])

    # Parameters
    if not lambda_param:
        lambda_param = (num_positive + 1) / theta

    # Create the DOcplex model
    model = CplexModel(name="Wide-Reach Classification")

    if not PRINT_OUTPUT:
        model.set_log_output(None)
        model.set_log_output_as_stream(None)
        model.parameters.mip.display.set(0)
        model.context.cplex_parameters.read.datacheck = 0
        model.context.solver.log_output = False
        model.parameters.mip.display = 0

    if TIME_LIMIT:
        model.set_time_limit(time_limit=TIME_LIMIT)
    if RAM_LIMIT:
        model.parameters.workmem = 4096

    # Decision variables
    x = model.binary_var_list(num_positive, name="x")
    y = model.binary_var_list(len(N_indices), name="y")
    w = model.continuous_var_list(X.shape[1], lb=-model.infinity, name="w")
    c = model.continuous_var(lb=-model.infinity, name="c")
    V = model.continuous_var(lb=0, name="V")

    # Objective: Maximize the reach minus penalty for precision violation
    model.maximize(model.sum(x[i] for i in P_indices) - lambda_param * V)

    # Constraint: Precision constraint violation
    model.add_constraint(
        V
        >= (theta - 1) * model.sum(x[i] for i in P_indices)
        + theta * model.sum(y[j] for j in range(len(N_indices)))
        + theta * epsilon_R,
        ctname="PrecisionConstraint",
    )

    # Constraints: Classification constraints for positive samples
    for i, p_idx in enumerate(P_indices):
        model.add_constraint(
            x[i]
            <= 1
            + model.sum(w[d] * X[p_idx, d] for d in range(X.shape[1]))
            - c
            - epsilon_P,
            ctname=f"Positive_{i}",
        )

    # Constraints: Classification constraints for negative samples
    for j, n_idx in enumerate(N_indices):
        model.add_constraint(
            y[j]
            >= model.sum(w[d] * X[n_idx, d] for d in range(X.shape[1])) - c + epsilon_N,
            ctname=f"Negative_{j}",
        )

    # Solve the model
    start_time = time.time()
    solution = model.solve(log_output=True)
    end_time = time.time()

    # Check and return results
    if solution:
        results = {
            "Reach": sum(solution.get_value(x[i]) for i in P_indices),
            "Hyperplane w": [solution.get_value(w[d]) for d in range(X.shape[1])],
            "Bias c": solution.get_value(c),
            "X": [solution.get_value(x[d]) for d in range(len(P))],
            "Y": [solution.get_value(y[d]) for d in range(len(N))],
            "Precision Violation V": solution.get_value(V),
            "Node Count": model.get_solve_details().nb_nodes_processed,
            "Time taken": end_time - start_time,
        }
    else:
        results = {"Error": "No optimal solution found."}

    # dispose
    model.end()
    del model

    gc.collect()

    return results
