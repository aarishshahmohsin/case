import gurobipy as grb
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
        lambda_param
    ):
        self.P = P
        self.N = N
        self.theta = theta
        self.epsilon_R = eps_R
        self.epsilon_P = eps_P
        self.epsilon_N = eps_N
        self.lambda_param = lambda_param
        self.mdl = grb.Model("MILP_with_lambda")
        if not PRINT_OUTPUT:
            self.mdl.setParam("OutputFlag", 0)
        if TIME_LIMIT:
            self.mdl.setParam("TimeLimit", 2 * 60)
        if RAM_LIMIT:
            self.mdl.setParam("MemLimit", 4096)

    def build_model(self):
        d = self.P.shape[1]  # Dimension of feature vectors
        num_P = len(self.P)
        num_N = len(self.N)

        # Decision variables
        self.x_vars = self.mdl.addVars(num_P, vtype=grb.GRB.BINARY, name="x")
        self.y_vars = self.mdl.addVars(num_N, vtype=grb.GRB.BINARY, name="y")
        self.w = self.mdl.addVars(
            d, vtype=grb.GRB.CONTINUOUS, name="w"
        )  # Weight vector
        self.c = self.mdl.addVar(vtype=grb.GRB.CONTINUOUS, name="c")  # Bias term
        self.V = self.mdl.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=0, name="V"
        )  # Regularization term

        # Regularization constraint
        self.mdl.addConstr(
            self.V
            >= (self.theta - 1) * grb.quicksum(self.x_vars[i] for i in range(num_P))
            + self.theta * grb.quicksum(self.y_vars[i] for i in range(num_N))
            + self.theta * self.epsilon_R
        )

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
            c_value = self.c.x
            w_values = [self.w[i].x for i in range(self.P.shape[1])]

            reach = np.sum(
                np.array(
                    [
                        (
                            0
                            if np.dot(w_values, s) + np.sum(c_value) < self.epsilon_P
                            else 1
                        )
                        for s in self.P
                    ]
                )
            )

            return reach, int(node_count), self.mdl
        else:
            print("No feasible solution found.")
            return None, None, None
