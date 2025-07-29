import ctypes
import numpy as np
from ctypes import Structure, POINTER, c_double, c_int, c_char_p, c_long
import os


# Define the Matrix structure to match the C code
class Matrix(Structure):
    _fields_ = [("data", POINTER(c_double)), ("rows", c_int), ("cols", c_int)]


# Define the SolverResults structure to match the C code
class SolverResults(Structure):
    _fields_ = [
        ("initial_reach", c_double),
        ("reach", c_double),
        ("hyperplane_w", POINTER(c_double)),
        ("bias_c", c_double),
        ("x_vals", POINTER(c_double)),
        ("y_vals", POINTER(c_double)),
        ("precision_violation_v", c_double),
        ("node_count", c_long),
        ("time_taken", c_double),
        ("status", c_int),
        ("error_msg", c_char_p),
    ]


class SCIPSolver:
    def __init__(self, lib_path="/home/aarish/case/c_api_exp/scip_solver.so"):
        """
        Initialize the SCIP solver wrapper.

        Args:
            lib_path: Path to the compiled shared library
        """
        # Load the shared library
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Shared library not found at {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # Define function signatures
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Setup the C function signatures for proper calling convention."""

        # scip_solver function
        self.lib.scip_solver.argtypes = [
            c_double,  # theta
            c_double,  # theta0
            c_double,  # theta1
            POINTER(Matrix),  # P
            POINTER(Matrix),  # N
            c_double,  # epsilon_P
            c_double,  # epsilon_N
            c_double,  # epsilon_R
            c_double,  # lambda_param
            c_char_p,  # dataset_name
            c_int,  # run
            POINTER(c_double),  # init_w
            c_double,  # init_c
            c_int,  # print_output
            c_double,  # time_limit
        ]
        self.lib.scip_solver.restype = POINTER(SolverResults)

        # Helper functions
        self.lib.create_matrix.argtypes = [c_int, c_int]
        self.lib.create_matrix.restype = POINTER(Matrix)

        self.lib.free_matrix.argtypes = [POINTER(Matrix)]
        self.lib.free_matrix.restype = None

        self.lib.set_element.argtypes = [POINTER(Matrix), c_int, c_int, c_double]
        self.lib.set_element.restype = None

        self.lib.free_solver_results.argtypes = [POINTER(SolverResults)]
        self.lib.free_solver_results.restype = None

    def _numpy_to_matrix(self, np_array):
        """Convert numpy array to C Matrix structure."""
        rows, cols = np_array.shape
        matrix_ptr = self.lib.create_matrix(rows, cols)

        # Fill the matrix with data
        for i in range(rows):
            for j in range(cols):
                self.lib.set_element(matrix_ptr, i, j, float(np_array[i, j]))

        return matrix_ptr

    def _extract_results(self, results_ptr, n_features, n_positive, n_negative):
        """Extract results from C structure to Python dictionary."""
        if not results_ptr:
            return {"Error": "Null results pointer"}

        results = results_ptr.contents

        if results.status < 0:
            error_msg = (
                results.error_msg.decode("utf-8")
                if results.error_msg
                else "Unknown error"
            )
            return {"Error": error_msg}

        # Extract hyperplane weights if available
        hyperplane_w = []
        if results.hyperplane_w:
            for i in range(n_features):
                hyperplane_w.append(results.hyperplane_w[i])

        # Extract x values if available
        x_vals = []
        if results.x_vals:
            for i in range(n_positive):
                x_vals.append(results.x_vals[i])

        # Extract y values if available
        y_vals = []
        if results.y_vals:
            for i in range(n_negative):
                y_vals.append(results.y_vals[i])

        return {
            "Initial reach": results.initial_reach,
            "Reach": results.reach,
            "Hyperplane w": hyperplane_w,
            "Bias c": results.bias_c,
            "X": x_vals,
            "Y": y_vals,
            "Precision Violation V": results.precision_violation_v,
            "Node Count": results.node_count,
            "Time taken": results.time_taken,
            "Status": (
                "Optimal"
                if results.status == 0
                else "Time limit" if results.status == 1 else "Error"
            ),
        }

    def solve(
        self,
        P,
        N,
        init_w,
        init_c,
        theta,
        lambda_param=None,
        epsilon_P=0.01,
        epsilon_N=0.01,
        epsilon_R=0.001,
        theta0=0.1,
        theta1=0.2,
        dataset_name="test",
        print_output=False,
        time_limit=300.0,
    ):
        """
        Solve the wide-reach classification problem.

        Args:
            P: numpy array of positive samples (n_positive, n_features)
            N: numpy array of negative samples (n_negative, n_features)
            init_w: initial hyperplane weights (list or numpy array)
            init_c: initial bias (float)
            theta: precision threshold (float)
            lambda_param: lambda parameter (float, optional)
            epsilon_P: positive samples parameter (float)
            epsilon_N: negative samples parameter (float)
            epsilon_R: regularization parameter (float)
            theta0: parameter for optimization (float)
            theta1: parameter for optimization (float)
            dataset_name: name for output files (string)
            print_output: whether to show SCIP output (bool)
            time_limit: time limit in seconds (float)

        Returns:
            dict: Results dictionary with reach, hyperplane parameters, etc.
        """
        # Convert numpy arrays to C matrices
        P_matrix = self._numpy_to_matrix(P)
        N_matrix = self._numpy_to_matrix(N)

        # Convert init_w to C array
        init_w_array = (c_double * len(init_w))(*init_w)

        # Set lambda parameter if not provided
        if lambda_param is None:
            lambda_param = (P.shape[0] + 1) * theta1

        try:
            # Call the C function
            results_ptr = self.lib.scip_solver(
                c_double(theta),
                c_double(theta0),
                c_double(theta1),
                P_matrix,
                N_matrix,
                c_double(epsilon_P),
                c_double(epsilon_N),
                c_double(epsilon_R),
                c_double(lambda_param),
                dataset_name.encode("utf-8"),
                c_int(1),  # run = True
                init_w_array,
                c_double(init_c),
                c_int(1 if print_output else 0),
                c_double(time_limit),
            )

            # Extract results
            results = self._extract_results(
                results_ptr, P.shape[1], P.shape[0], N.shape[0]
            )

            # Free C memory
            if results_ptr:
                self.lib.free_solver_results(results_ptr)

        finally:
            # Free matrices
            self.lib.free_matrix(P_matrix)
            self.lib.free_matrix(N_matrix)

        return results


# Global solver instance
_solver = None


def call_scip_solver(
    P,
    N,
    init_w,
    init_c,
    theta,
    lambda_param=None,
    epsilon_P=0.01,
    epsilon_N=0.01,
    epsilon_R=0.001,
    **kwargs,
):
    """
    Convenience function that matches your original interface.

    Args:
        P: numpy array of positive samples
        N: numpy array of negative samples
        init_w: initial hyperplane weights
        init_c: initial bias
        theta: precision threshold
        lambda_param: lambda parameter (optional)
        epsilon_P: positive samples parameter
        epsilon_N: negative samples parameter
        epsilon_R: regularization parameter
        **kwargs: additional arguments passed to solver

    Returns:
        dict: Results dictionary
    """
    global _solver

    # Initialize solver if not already done
    if _solver is None:
        _solver = SCIPSolver()

    return _solver.solve(
        P=P,
        N=N,
        init_w=init_w,
        init_c=init_c,
        theta=theta,
        lambda_param=lambda_param,
        epsilon_P=epsilon_P,
        epsilon_N=epsilon_N,
        epsilon_R=epsilon_R,
        **kwargs,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the wrapper
    np.random.seed(42)

    # Generate test data
    P = np.random.randn(10, 5) + 1  # Positive samples
    N = np.random.randn(15, 5) - 1  # Negative samples

    # Initial hyperplane
    init_w = [0.1, 0.2, -0.1, 0.3, -0.2]
    init_c = 0.5

    # Test the solver
    try:
        results = call_scip_solver(
            P=P,
            N=N,
            init_w=init_w,
            init_c=init_c,
            theta=0.8,
            lambda_param=None,
            epsilon_P=0.01,
            epsilon_N=0.01,
            epsilon_R=0.001,
        )

        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Compile the C code as a shared library")
        print("2. Ensure the shared library is in the current directory")
