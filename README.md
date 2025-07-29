# CASE

## Installation

### 1. Python Dependencies

Install the required solver APIs:

```bash
pip install gurobipy docplex pyscipopt
```

> Ensure that your Gurobi and CPLEX licenses are correctly set up beforehand.

---

### 2. Install SCIP (with C API support)

####  Official instructions

Follow SCIP's instructions at:
[https://www.scipopt.org/doc-7.0.0/html/MAKE.php](https://www.scipopt.org/doc-7.0.0/html/MAKE.php)

#### ️ Build Commands

Compile SCIP with different LP solvers:

```bash
# For Gurobi LP solver
make LPS=grb SHARED=true  

# For CPLEX LP solver
make LPS=cpx SHARED=true  

# For SoPlex LP solver (default)
make SHARED=true  
```

####  Set SCIP path in Makefile

Edit `c_api_exp/Makefile`:

```makefile
SCIP_PATH = /home/aarish/scip_install_try/scipoptsuite-9.2.2/scip
```

#### ️ Build custom C API

```bash
make -C c_api_exp/
```

####  Set shared object path in Python

Edit `src/constants.py`:

```python
LIB_PATH = "/home/aarish/case/c_api_exp/scip_solver.so"
```

---

### 3. Download the Data

#### Download Link

[Google Drive: Data Folder](https://drive.google.com/drive/folders/1pUdUXI8ewrO2PLj9abAnXaN-wxSLaXhW?usp=sharing)

Extract and place it at your desired location, then update the path in:

```python
# src/constants.py
DATA_PATH = "/home/aarish/case/data"
```

---

## Run Experiments

Before running, export library paths:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/home/aarish/scip_install_try/scipoptsuite-9.2.2/scip/lib/shared:$LD_LIBRARY_PATH
```

> You may add these lines to your `.bashrc` for persistence.

Run main benchmark script:

```bash
python3 /home/aarish/case/Benchmarks_scip_c.py
```

All experiment configurations are located in `Benchmarks.py`.

---

##  Feasibility Pump Logging

To log the Feasibility Pump iterations:

1. Replace the `heur_feaspump.c` file in the SCIP installation source e.g. 
```
scipoptsuite-9.2.2/scip/src/scip/heur_feaspump.c
```
2. Rebuild SCIP using `make` as above.
3. This enables logging of lp and rounding

---

##  Visualizations

Use the appropriate script depending on the dimension of your problem:

```bash
python3 visualize_2d.py
python3 visualize_3d.py
```

