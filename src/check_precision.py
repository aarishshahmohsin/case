import gurobipy as gp
from docplex.mp.model import Model
from gurobipy import GRB

def check_gurobi_precision():
    model = gp.Model("precision_test")
    x = model.addVar(name="x")
    model.setObjective(x, GRB.MAXIMIZE)
    constraint_value = 1.1234567890123456789
    model.addConstr(x <= constraint_value, "c0")
    model.optimize()
    if model.status == GRB.OPTIMAL:
        x_value = x.X
        constraint_str = f"{constraint_value:.20f}"
        x_str = f"{x_value:.20f}"
        matching_digits = 0
        for c1, c2 in zip(constraint_str, x_str):
            if c1 == c2:
                matching_digits += 1
            else:
                break
        matching_digits -= 2
        if matching_digits >= 15:
            print(f"Gurobi uses 64-bit floats. Matching digits: {matching_digits}")
        elif matching_digits >= 11:
            print(f"Gurobi uses 48-bit floats. Matching digits: {matching_digits}")
        else:
            print(f"Gurobi uses an unknown format. Matching digits: {matching_digits}")
    else:
        print("No optimal solution found.")



def check_docplex_precision():
    model = Model("precision_test")
    x = model.continuous_var(name="x")
    model.maximize(x)
    constraint_value = 1.1234567890123456789
    model.add_constraint(x <= constraint_value, "c0")
    solution = model.solve()
    if solution:
        x_value = solution.get_value(x)
        constraint_str = f"{constraint_value:.20f}"
        x_str = f"{x_value:.20f}"
        matching_digits = 0
        for c1, c2 in zip(constraint_str, x_str):
            if c1 == c2:
                matching_digits += 1
            else:
                break
        matching_digits -= 2
        if matching_digits >= 15:
            print(f"DOcplex uses 64-bit floats. Matching digits: {matching_digits}")
        elif matching_digits >= 11:
            print(f"DOcplex uses 48-bit floats. Matching digits: {matching_digits}")
        else:
            print(f"DOcplex uses an unknown format. Matching digits: {matching_digits}")
    else:
        print("No optimal solution found.")

check_docplex_precision()
check_gurobi_precision()