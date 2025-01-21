import numpy as np
from scipy.special import factorial
from scipy.stats import truncnorm
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
    def __init__(
        self, d=11, P_size=180, N_size=(32 * 180), f=8, background_noise=False
    ):
        """
        Generates a synthetic binary classification dataset using the Prism benchmark.

        Parameters:
        d (int): Total number of dimensions.
        P_size (int): Total number of positive samples.
        N_size (int): Total number of negative samples.
        f (float): Desired ratio of negatives to positives inside the prism.
        background_noise (bool): If True, half of the positive samples are background noise.
        """
        self.d = d
        self.P_size = P_size
        self.N_size = N_size
        self.f = f
        self.background_noise = background_noise
        self.theta0 = 1
        self.theta1 = 100

        # Generate the dataset
        self.X, self.y = self._generate()

    def _generate(self):
        """Generates the dataset."""
        # Adjust positive sample count if background noise is enabled
        if self.background_noise:
            P_prism = self.P_size // 2
        else:
            P_prism = self.P_size

        # Find the largest d0 such that d0! <= N_size / (P_prism * f)
        d0 = 1
        while factorial(d0) <= self.N_size / (P_prism * self.f):
            d0 += 1
        d0 -= 1  # Step back to last valid value

        # Compute the side length s of the d0-dimensional simplex
        s = ((P_prism * self.f * factorial(d0)) / self.N_size) ** (1 / d0)

        # Generate negative samples uniformly in [0,1]^d
        X_neg = np.random.uniform(0, 1, size=(self.N_size, self.d))

        # Generate positive samples inside the prism
        X_pos = np.zeros((P_prism, self.d))

        for i in range(P_prism):
            # Generate a point inside the d0-dimensional simplex
            simplex_point = np.sort(np.random.uniform(0, s, d0))
            simplex_point -= np.insert(
                simplex_point[:-1], 0, 0
            )  # Convert to barycentric
            simplex_point = np.random.permutation(simplex_point)  # Shuffle dimensions

            # Fill the first d0 dimensions with the simplex point
            X_pos[i, :d0] = simplex_point

            # Fill the remaining dimensions uniformly in [0,1]
            X_pos[i, d0:] = np.random.uniform(0, 1, self.d - d0)

        # Generate background noise if enabled
        if self.background_noise:
            X_bg = np.random.uniform(0, 1, size=(self.P_size - P_prism, self.d))
            X_pos = np.vstack((X_pos, X_bg))

        # Concatenate the negative and positive samples
        X = np.vstack((X_neg, X_pos))
        y = np.hstack((np.zeros(self.N_size), np.ones(self.P_size)))

        # Shuffle the dataset
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        return X[indices], y[indices]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns a single sample and its label."""
        return self.X[idx], self.y[idx]


class TruncatedNormalPrism(Dataset):
    def __init__(self, P=180, N=None, d=11, d0=3, r=0.6, sigma=0.4016, f=8):
        """
        Parameters:
        P (int): Number of positive samples (default: 180).
        N (int): Number of negative samples (default: 32 * P).
        d (int): Dimensionality of the dataset (default: 11).
        d0 (int): Dimensions for the truncated normal (default: 3).
        r (float): Radius for the truncated normal constraint (default: 0.6).
        sigma (float): Standard deviation for the truncated normal (default: 0.4016).
        f (float): Desired ratio of negatives to positives (default: 8).
        """
        self.P = P
        self.N = N if N is not None else 32 * P
        self.d = d
        self.d0 = d0
        self.r = r
        self.sigma = sigma
        self.f = f
        self.theta0 = 1
        self.theta1 = 10

        # Generate dataset
        self.X, self.y = self._generate()

    def _generate(self):
        """Generates the dataset."""
        # Generate negative samples
        negative_samples = np.random.uniform(0, 1, size=(self.N, self.d))

        # Generate positive samples
        positive_samples = []
        while len(positive_samples) < self.P:
            # Sample from truncated normal for d0 dimensions
            truncated_sample = truncnorm.rvs(
                -self.r / self.sigma,
                self.r / self.sigma,
                scale=self.sigma,
                size=self.d0,
            )
            # Uniformly sample the remaining d-d0 dimensions
            remaining_dims = np.random.uniform(0, 1, size=self.d - self.d0)
            # Concatenate to form the complete sample
            sample = np.concatenate([truncated_sample, remaining_dims])
            # Validate sample: Norm constraint and non-negativity
            if np.linalg.norm(sample[: self.d0], ord=1) <= self.r and np.all(
                sample >= 0
            ):
                positive_samples.append(sample)

        positive_samples = np.array(positive_samples)

        # Combine positive and negative samples
        X = np.vstack((negative_samples, positive_samples))
        y = np.hstack((np.zeros(self.N), np.ones(self.P)))

        return X, y
