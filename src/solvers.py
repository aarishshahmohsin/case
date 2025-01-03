import gurobipy as grb
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
            self.mdl.parameters.timelimit.set(2 * 60)
        if RAM_LIMIT:
            self.mdl.parameters.workmem.set(4096)

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

        # Regularization constraint
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
            values = [1.0] + list(s) + [-1.0]
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

        try:
            # Check if a feasible or optimal solution is available
            if status in [
                self.mdl.solution.status.MIP_optimal,
                self.mdl.solution.status.MIP_feasible,
                self.mdl.solution.status.abort_time_limit,
            ]:
                # Retrieve the incumbent value (objective value)
                obj_value = self.mdl.solution.get_objective_value()

                # Retrieve solution values for `w` and `c` to calculate `reach`
                w_values = self.mdl.solution.get_values(self.w_vars)
                c_value = self.mdl.solution.get_values(self.c_var)

                # Calculate `reach`
                # reach = sum(
                #     1 for s in self.P if np.dot(w_values, s) + c_value >= self.epsilon_P
                # )

                node_count = self.mdl.solution.progress.get_num_nodes_processed()

                return int(obj_value), int(node_count), self.mdl
            else:
                # Handle cases where no feasible solution is found
                if status == self.mdl.solution.status.abort_time_limit:
                    # Attempt to retrieve the best solution so far
                    try:
                        obj_value = self.mdl.solution.get_objective_value()
                        print(f"Best incumbent value after time limit: {obj_value}")

                        # Retrieve solution values for `w` and `c` to calculate `reach`
                        w_values = self.mdl.solution.get_values(self.w_vars)
                        c_value = self.mdl.solution.get_values(self.c_var)

                        # Calculate `reach`
                        reach = sum(
                            1
                            for s in self.P
                            if np.dot(w_values, s) + c_value >= self.epsilon_P
                        )
                        print(f"Reach after time limit: {reach}")

                        # Get the number of processed nodes
                        node_count = (
                            self.mdl.solution.progress.get_num_nodes_processed()
                        )

                        return int(obj_value), int(node_count), self.mdl
                    except Exception as e:
                        print(f"Error retrieving best solution after time limit: {e}")
                        return None, None, None
                else:
                    print(f"No feasible solution found. Status code: {status}")
                    return None, None, None
        except Exception as e:
            print(f"Error during solving process: {e}")
            return None, None, None
