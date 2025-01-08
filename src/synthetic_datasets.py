import numpy as np
import math
from scipy.special import factorial
from scipy.stats import rayleigh, maxwell
from utils import Dataset
from constants import D_MAX


class ClusterDataset(Dataset):
    def __init__(self, n=400, d=8) -> None:
        self.n = n
        self.d = d
        self.s = (factorial(d) / factorial(D_MAX)) ** (1 / d)  # Side length of simplex
        self.theta0 = 99
        self.theta1 = 100
        self._generate()

    def generate_hypercube_points(self, n, d):
        return np.random.uniform(0, 1, size=(n, d))

    def generate_simplex_points(self, n, d, s):
        points = np.random.dirichlet(np.ones(d), size=n)
        return points * s

    def _generate(self):
        n_negative = 4 * self.n // 5
        n_positive_simplex = self.n // 10
        n_positive_remaining = self.n // 10

        negative_points = self.generate_hypercube_points(n_negative, self.d)

        positive_simplex_points = self.generate_simplex_points(
            n_positive_simplex, self.d, self.s
        )

        positive_remaining_points = self.generate_hypercube_points(
            n_positive_remaining, self.d
        )

        self.X = np.vstack(
            (negative_points, positive_simplex_points, positive_remaining_points)
        )

        self.y = np.hstack(
            (np.zeros(n_negative), np.ones(n_positive_simplex + n_positive_remaining))
        )


class TwoClusterDataset(Dataset):
    def __init__(self, n=400, d=8) -> None:
        self.n = n
        self.d = d
        self.s = (factorial(d) / factorial(D_MAX)) ** (1 / d)  # Side length of simplex
        self.theta0 = 99
        self.theta1 = 100
        self._generate()

    def generate_hypercube_points(self, n, d):
        return np.random.uniform(0, 1, size=(n, d))

    def generate_simplex_points(self, n, d, s):
        points = np.random.dirichlet(np.ones(d), size=n)
        return points * s

    def generate_diametrically_opposite(self, n, d, s):
        points = np.random.dirichlet(np.ones(d), size=n)
        points = points * s
        return np.ones(d) - points

    def _generate(self):
        n_negative = 4 * self.n // 5
        n_positive_simplex = self.n // 20
        n_positive_simplex_opp = self.n // 20
        n_positive_remaining = self.n // 10

        negative_points = self.generate_hypercube_points(n_negative, self.d)

        positive_simplex_points = self.generate_simplex_points(
            n_positive_simplex, self.d, self.s
        )
        positive_simplex_points_opp = self.generate_diametrically_opposite(
            n_positive_simplex_opp, self.d, self.s
        )

        positive_remaining_points = self.generate_hypercube_points(
            n_positive_remaining, self.d
        )

        self.X = np.vstack(
            (
                negative_points,
                positive_simplex_points,
                positive_simplex_points_opp,
                positive_remaining_points,
            )
        )

        self.y = np.hstack(
            (
                np.zeros(n_negative),
                np.ones(
                    n_positive_simplex + n_positive_simplex_opp + n_positive_remaining
                ),
            )
        )


class DiffusedBenchmark(Dataset):
    def __init__(self, n=400, d=8) -> None:
        self.n = n
        self.d = d
        self.theta0 = 99
        self.theta1 = 100
        self._generate()

    def generate_hypercube_points(self, n, d):
        return np.random.uniform(0, 1, size=(n, d))

    def _generate(self):
        n_negative = self.n // 2
        n_positive = self.n // 2

        negative_points = self.generate_hypercube_points(n_negative, self.d)
        positive_points = self.generate_hypercube_points(n_positive, self.d)

        self.X = np.vstack((negative_points, positive_points))
        self.y = np.hstack((np.zeros(n_negative), np.ones(n_positive)))


class PrismDataset(Dataset):
    def __init__(self, n=400, d=11, f=8, p_mode="P"):
        """
        Parameters:
        n (int): Total number of samples (default: 400).
        d (int): Dimension of the dataset (default: 11).
        f (float): Factor controlling negatives per positive (default: 8).
        p_mode (str): Positive sample mode, "P" (default) or "P/4".
        """

        self.theta0 = 99
        self.theta1 = 100

        self.n = n
        self.d = d
        self.f = f
        self.p_mode = p_mode

        # Total number of positive and negative samples
        self.n_positive = n // 33
        self.n_negative = n - self.n_positive

        # Ratio of positives to negatives
        self.p_to_n_ratio = self.n_positive / self.n_negative

        # Volume (v) of the simplex
        self.v = (
            (self.n_positive if self.p_mode == "P" else self.n_positive / 4) * self.f
        ) / self.n_negative

        # Calculate d0 based on factorial constraints
        self.d0 = self._compute_d0()
        print(self.d0)

        # Side length of the simplex
        self.s = (self.v * factorial(self.d0)) ** (1 / self.d0)
        self._generate()

    def _compute_d0(self):
        """Calculate d0 based on |N| / (p * f)."""
        for d0 in range(1, self.d + 1):
            if factorial(d0) <= self.n_negative / (
                (self.n_positive if self.p_mode == "P" else self.n_positive / 4)
                * self.f
            ):
                continue
            return d0 - 1
        return self.d

    def generate_hypercube_points(self, n, d):
        """Generate points uniformly in a hypercube."""
        return np.random.uniform(0, 1, size=(n, d))

    def generate_prism_points(self, n, d, d0, s):
        k = np.random.exponential(scale=s, size=(n, d0))
        P = k / sum(k)
        padding = np.random.uniform(0, 1, size=(n, d - d0))  # Random padding
        return np.hstack((P, padding))

    def _generate(self):
        """
        Generate the Prism dataset.

        Returns:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        y (numpy.ndarray): Labels array of shape (n,).
        dict: Counts of each type of sample.
        """
        # Positive samples
        n_positive_prism = (
            self.n_positive if self.p_mode == "P" else self.n_positive // 4
        )
        n_positive_hypercube = self.n_positive - n_positive_prism

        # Negative samples
        # n_negative_prism = int(self.n_negative * self.v)
        n_negative_hypercube = self.n_negative

        # Generate samples
        positive_prism_points = self.generate_prism_points(
            n_positive_prism, self.d, self.d0, self.s
        )
        positive_hypercube_points = self.generate_hypercube_points(
            n_positive_hypercube, self.d
        )
        #
        # negative_prism_points = self.generate_prism_points(
        #     n_negative_prism, self.d, self.d0, self.s
        # )
        negative_hypercube_points = self.generate_hypercube_points(
            n_negative_hypercube, self.d
        )

        # Combine into dataset
        self.X = np.vstack(
            (
                # negative_prism_points,
                negative_hypercube_points,
                positive_prism_points,
                positive_hypercube_points,
            )
        )
        self.y = np.hstack(
            (
                np.zeros(n_negative_hypercube),
                np.ones(n_positive_prism + n_positive_hypercube),
            )
        )

        # Count of each type
        # sample_counts = {
        #     "positive_in_prism": n_positive_prism,
        #     "positive_not_in_prism": n_positive_hypercube,
        #     "negative_in_prism": n_negative_prism,
        #     "negative_not_in_prism": n_negative_hypercube,
        # }


class L1PrismDataset(Dataset):
    def __init__(self, n=400, d=11, f=8, p_mode="P", d0=2):
        """
        Parameters:
        n (int): Total number of samples (default: 400).
        d (int): Dimension of the dataset (default: 11).
        f (float): Factor controlling negatives per positive (default: 8).
        p_mode (str): Positive sample mode, "P" (default) or "P/4".
        """

        self.theta0 = 99
        self.theta1 = 100

        self.n = n
        self.d = d
        self.f = f
        self.p_mode = p_mode

        # Total number of positive and negative samples
        self.n_positive = n // 5
        self.n_negative = n - self.n_positive

        self.d0 = d0

        self._generate()

    def calculate_params(P, N, d=11, f=8, s=1):
        """
        Calculate parameters for the prism and multivariate normal models.

        Parameters:
            P (int): Number of positive samples.
            N (int): Number of negative samples.
            d (int): Dimension (default: 11).
            f (float): Ratio of negatives to positives (default: 8).
            s (float): Maximum length constraint for the positives (default: 1).

        Returns:
            dict: Dictionary containing calculated parameters.
        """
        results = {}

        # Multivariate normal calculations (d0 = 2 and d0 = 3)
        def rayleigh_cdf(r, sigma):
            return rayleigh.cdf(r, scale=sigma)

        def maxwell_cdf(r, sigma):
            return maxwell.cdf(r, scale=sigma)

        # Calculate r for d0 = 2
        def calculate_r_d0_2(sigma, f, N, P):
            pdf_val = rayleigh.pdf(1, sigma)
            if pdf_val == 0:
                return s  # Return max constraint if division is invalid
            r = math.sqrt(
                (4 * f * P * rayleigh.cdf(s, sigma)) / (math.pi * N * pdf_val)
            )
            return min(r, s)  # Ensure r <= s

        # Calculate r for d0 = 3
        def calculate_r_d0_3(sigma, f, N, P):
            pdf_val = maxwell.pdf(1, sigma)
            if pdf_val == 0:
                return s  # Return max constraint if division is invalid
            r = ((6 * f * P * maxwell.cdf(s, sigma)) / (math.pi * N * pdf_val)) ** (
                1 / 3
            )
            return min(r, s)  # Ensure r <= s

        sigma_2 = 1  # Initialize sigma for d0 = 2
        sigma_3 = 1  # Initialize sigma for d0 = 3

        r_2 = calculate_r_d0_2(sigma_2, f, N, P)
        r_3 = calculate_r_d0_3(sigma_3, f, N, P)

        results["multivariate_normal"] = {
            "2": {
                "sigma": sigma_2,
                "r": r_2,
                "density": (
                    (rayleigh_cdf(r_2, sigma_2) / rayleigh_cdf(1, sigma_2))
                    / (math.pi * r_2**2)
                    if r_2 > 0
                    else None
                ),
                "s": s,
            },
            "3": {
                "sigma": sigma_3,
                "r": r_3,
                "density": (
                    (maxwell_cdf(r_3, sigma_3) / maxwell_cdf(1, sigma_3))
                    / ((4 / 3) * math.pi * r_3**3)
                    if r_3 > 0
                    else None
                ),
                "s": s,
            },
        }

        return results

    def _generate(self):
        params = self.calculate_params(self.n_positive, self.n_negative, f=8, s=1)

        sigma = params["multivariate_normal"][str(self.d0)]["sigma"]
        r = params["multivariate_normal"][str(self.d0)]["r"]

        def generate_positive_samples(P_size, d, d0, sigma, r):
            samples = []
            while len(samples) < P_size:
                point = np.random.normal(loc=0, scale=sigma, size=d0)
                length = np.linalg.norm(point, ord=1)
                if length <= r and np.all(point >= 0):
                    full_point = np.zeros(d)
                    full_point[:d0] = point
                    samples.append(full_point)
            return np.array(samples)

        def generate_negative_samples(N_size, d, d0, s):
            samples = []
            while len(samples) < N_size:
                point = np.random.uniform(low=0, high=s, size=d0)
                if np.sum(point) <= s:
                    full_point = np.zeros(d)
                    full_point[:d0] = point
                    samples.append(full_point)
            return np.array(samples)

        P = generate_positive_samples(self.n_positive, self.d, self.d0, sigma, r)
        N = generate_negative_samples(self.n_negative, self.d, self.d0, 1)

        self.X = np.vstack((P, N))
        self.y = np.hstack((np.ones(self.n_positive), np.zeros(self.n_negative)))
