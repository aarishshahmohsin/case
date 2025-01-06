import gurobipy as grb
from gurobipy import GRB
import cplex
import numpy as np
from constants import (
    RAM_LIMIT,
    TIME_LIMIT,
    PRINT_OUTPUT,
    epsilon_R,
    epsilon_N,
    epsilon_P,
)


class GurobiSolver:
    def __init__(
        self,
        *,
        P,
        N,
        theta,
        eps_R=epsilon_R,
        eps_P=epsilon_P,
        eps_N=epsilon_N,
        lambda_param,
    ):
        self.P = P
        self.N = N
        self.theta = theta
        self.epsilon_R = eps_R
        self.epsilon_P = eps_P
        self.epsilon_N = eps_N
        self.lambda_param = lambda_param
        self.mdl = grb.Model("Wide-Reach_Classification")
        if not PRINT_OUTPUT:
            self.mdl.setParam("OutputFlag", 0)
        if TIME_LIMIT:
            self.mdl.setParam("TimeLimit", TIME_LIMIT)
        if RAM_LIMIT:
            self.mdl.setParam("MemLimit", RAM_LIMIT)
        # self.mdl.setParam(GRB.Param.MIPGap, 0.01)  # gap
        # self.mdl.setParam(GRB.Param.Heuristics, 0)
        # self.mdl.setParam(GRB.Param.NodeMethod, 2)
        #

    def build_model(self):
        d = self.P.shape[1]
        num_P = len(self.P)
        num_N = len(self.N)

        # Decision variables
        self.x_vars = self.mdl.addVars(num_P, vtype=grb.GRB.BINARY, name="x")
        self.y_vars = self.mdl.addVars(num_N, vtype=grb.GRB.BINARY, name="y")
        self.w = self.mdl.addVars(d, vtype=grb.GRB.CONTINUOUS, name="w")
        self.c = self.mdl.addVar(vtype=grb.GRB.CONTINUOUS, name="c")
        self.V = self.mdl.addVar(vtype=grb.GRB.CONTINUOUS, name="V")

        # Regularization constraint based on Equation 4
        self.mdl.addConstr(
            self.V
            >= (self.theta - 1) * grb.quicksum(self.x_vars[i] for i in range(num_P))
            + self.theta * grb.quicksum(self.y_vars[i] for i in range(num_N))
            + self.theta * self.epsilon_R
        )
        self.mdl.addConstr(self.V >= 0)

        # Positive class constraints
        for i, s in enumerate(self.P):
            dot_product = grb.quicksum(s[j] * self.w[j] for j in range(d))
            self.mdl.addConstr(
                self.x_vars[i] <= 1 + dot_product - self.c - self.epsilon_P
            )

        # Negative class constraints
        for i, s in enumerate(self.N):
            dot_product = grb.quicksum(s[j] * self.w[j] for j in range(d))
            self.mdl.addConstr(self.y_vars[i] >= dot_product - self.c + self.epsilon_N)

        # Objective function
        objective = (
            grb.quicksum(self.x_vars[i] for i in range(num_P))
            - self.lambda_param * self.V
        )
        self.mdl.setObjective(objective, grb.GRB.MAXIMIZE)

    def solve(self):
        self.build_model()
        self.mdl.optimize()

        if self.mdl.status in [grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT]:
            node_count = self.mdl.NodeCount

            if self.mdl.status == grb.GRB.OPTIMAL:
                reach = np.sum([self.x_vars[i].x for i in range(len(self.P))])
            else:
                reach = np.sum([self.x_vars[i].x for i in range(len(self.P))])
            best_obj = self.mdl.ObjVal
        else:
            print(
                "No feasible or optimal solution found. Returning best solution found."
            )
            node_count = self.mdl.NodeCount
            reach = np.sum([self.x_vars[i].getAttr("x") for i in range(len(self.P))])
            best_obj = self.mdl.ObjVal

        return reach, int(node_count), best_obj


class CplexSolver:
    def __init__(
        self,
        *,
        P,
        N,
        theta,
        eps_R=epsilon_R,
        eps_P=epsilon_P,
        eps_N=epsilon_N,
        lambda_param,
    ):
        self.P = P
        self.N = N
        self.theta = theta
        self.epsilon_R = eps_R
        self.epsilon_P = eps_P
        self.epsilon_N = eps_N
        self.lambda_param = lambda_param
        self.mdl = cplex.Cplex()
        if not PRINT_OUTPUT:
            self.mdl.set_log_stream(None)
            self.mdl.set_error_stream(None)
            self.mdl.set_warning_stream(None)
            self.mdl.set_results_stream(None)
        if TIME_LIMIT:
            self.mdl.parameters.timelimit.set(TIME_LIMIT)
        if RAM_LIMIT:
            self.mdl.parameters.workmem.set(RAM_LIMIT)

    def build_model(self):
        d = self.P.shape[1]
        num_P = len(self.P)
        num_N = len(self.N)

        # Variables
        self.x_vars = [f"x_{i}" for i in range(num_P)]
        self.y_vars = [f"y_{i}" for i in range(num_N)]
        self.w_vars = [f"w_{j}" for j in range(d)]
        self.c_var = "c"
        self.V_var = "V"

        # Add variables
        self.mdl.variables.add(names=self.x_vars, types="B" * num_P)
        self.mdl.variables.add(names=self.y_vars, types="B" * num_N)
        self.mdl.variables.add(names=self.w_vars, types="C" * d)
        self.mdl.variables.add(names=[self.c_var], types="C")
        self.mdl.variables.add(names=[self.V_var], types="C", lb=[0.0])

        # Regularization constraint based on Equation 4
        regularization_indices = [self.V_var] + self.x_vars + self.y_vars
        regularization_values = (
            [1.0] + [(self.theta - 1)] * num_P + [self.theta] * num_N
        )
        self.mdl.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=regularization_indices, val=regularization_values)
            ],
            senses=["G"],
            rhs=[self.theta * self.epsilon_R],
        )

        # Positive class constraints
        for i, s in enumerate(self.P):
            indices = [self.x_vars[i]] + self.w_vars + [self.c_var]
            values = [1.0] + [-v for v in s] + [1.0]
            self.mdl.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                senses=["L"],
                rhs=[1 - self.epsilon_P],
            )

        # Negative class constraints
        for i, s in enumerate(self.N):
            indices = [self.y_vars[i]] + self.w_vars + [self.c_var]
            values = [1.0] + list(s) + [-1.0]
            self.mdl.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                senses=["G"],
                rhs=[self.epsilon_N],
            )

        # Objective function
        obj = [1.0] * num_P + [0.0] * num_N + [0.0] * d + [0.0, -self.lambda_param]
        self.mdl.objective.set_linear(
            list(
                zip(
                    self.x_vars + self.y_vars + self.w_vars + [self.c_var, self.V_var],
                    obj,
                )
            )
        )
        self.mdl.objective.set_sense(self.mdl.objective.sense.maximize)

    def solve(self):
        self.build_model()
        self.mdl.solve()

        status = self.mdl.solution.get_status()

        if status in [
            self.mdl.solution.status.MIP_optimal,
            self.mdl.solution.status.MIP_feasible,
        ]:
            x_values = self.mdl.solution.get_values(self.x_vars)
            reach = sum(
                int(x > 0.5) for x in x_values
            )  # Count positive correctly classified
            node_count = self.mdl.solution.progress.get_num_nodes_processed()
            return reach, int(node_count), self.mdl
        elif status == self.mdl.solution.status.abort_time_limit:
            print("Best feasible solution found within the time limit.")
            x_values = self.mdl.solution.get_values(self.x_vars)
            reach = sum(int(x > 0.5) for x in x_values)
            node_count = self.mdl.solution.progress.get_num_nodes_processed()
            return reach, int(node_count), self.mdl
        else:
            print("No feasible solution found.")
            return None, None, None
