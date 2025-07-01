from scipy.special import factorial
from scipy.stats import truncnorm
from solvers.utils import Dataset
from constants import D_MAX
import numpy as np
from math import factorial
from itertools import combinations
from scipy.stats import chi
from scipy.special import gamma
from scipy.optimize import root_scalar


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
        np.random.seed(0)
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
        np.random.seed(0)
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
        np.random.seed(0)
        n_negative = self.n // 2
        n_positive = self.n // 2

        negative_points = self.generate_hypercube_points(n_negative, self.d)
        positive_points = self.generate_hypercube_points(n_positive, self.d)

        self.X = np.vstack((negative_points, positive_points))
        self.y = np.hstack((np.zeros(n_negative), np.ones(n_positive)))



class PrismDataset(Dataset):
    def __init__(self, d=11, d0=3, s=0.909, num_positive=180, num_negative=5760, 
                 positive_background_ratio=0.5, num_prisms=1):
        """
        Initialize the prism dataset generator.
        
        Parameters:
        - d: Total dimensionality of the space
        - d0: Dimensionality of the base simplex
        - s: Side length of the simplex
        - num_positive: Number of positive samples (|P|)
        - num_negative: Number of negative samples (|N|)
        - positive_background_ratio: Fraction of positive samples to distribute uniformly
        - num_prisms: Number of prisms to generate (1 or 2)
        """
        self.d = d
        self.d0 = d0
        self.s = s
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.positive_background_ratio = positive_background_ratio
        self.num_prisms = num_prisms
        self.theta0 = 1
        self.theta1 = 10
        
        self._generate()

    def sample_simplex(self, n, d0, side_length):
        # Sample from a d0-simplex using the method of Dirichlet distribution
        dirichlet_samples = np.random.dirichlet([1]* (d0 + 1), n)
        # Drop the last coordinate (since it's determined by the others)
        simplex_coords = dirichlet_samples[:, :-1] * side_length
        return simplex_coords
 
        
    def _generate(self):
        np.random.seed(0)
        assert self.d > self.d0, "Total dimension d must be greater than d0"

        p_in = int(self.num_positive * (1 - self.positive_background_ratio))
        p_noise = self.num_positive - p_in

        # Generate prism positives
        simplex_part = self.sample_simplex(p_in, self.d0, self.s)
        uniform_part = np.random.uniform(0, 1, size=(p_in, self.d - self.d0))
        positives_prism = np.hstack((simplex_part, uniform_part))

        # Generate noisy positives (uniformly)
        positives_noise = np.random.uniform(0, 1, size=(p_noise, self.d))

        # Combine positives
        self.P = np.vstack((positives_prism, positives_noise))
        positives_labels = np.ones((self.num_positive, 1))

        # Generate negatives (uniformly)
        self.N = np.random.uniform(0, 1, size=(self.num_negative, self.d))
        negatives_labels = np.zeros((self.num_negative, 1))


              


class TruncatedNormalPrism(Dataset):
    def __init__(self, d=11, d0=2, r=0.6, sigma=0.4016, num_positive=180, 
                 num_negative=5760, positive_background_ratio=0.0, num_clusters=1):
        """
        Initialize the truncated normal dataset generator.
        
        Parameters:
        - d: Total dimensionality of the space
        - d0: Dimensionality of the normal distribution
        - r: Radius of the L2 ball for truncation
        - sigma: Standard deviation of the normal distribution
        - num_positive: Number of positive samples (|P|)
        - num_negative: Number of negative samples (|N|)
        - positive_background_ratio: Fraction of positive samples to distribute uniformly
        - num_clusters: Number of clusters to generate (1 or 2)
        """
        self.d = d
        self.d0 = d0
        self.r = r
        self.sigma = sigma
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.positive_background_ratio = positive_background_ratio
        self.num_clusters = num_clusters
        self.theta0 = 1
        self.theta1 = 10
        
        # Validate parameters
        if d0 > d:
            raise ValueError("d0 cannot be greater than d")
        if num_clusters not in [1, 2]:
            raise ValueError("num_clusters must be 1 or 2")
        if r > 1:
            raise ValueError("r must be ≤ 1 for unit hypercube")
            
        # Generate the dataset
        self._generate()
        
    def _chi_cdf(self, x):
        """CDF of the chi distribution with d0 degrees of freedom scaled by sigma."""
        return chi.cdf(x/self.sigma, self.d0)
    
    def _generate_truncated_normal_point(self):
        """Generate a random point from truncated normal distribution."""
        while True:
            # Sample from multivariate normal
            point = np.random.normal(0, self.sigma, self.d0)
            norm = np.linalg.norm(point)
            
            # Reject if outside unit ball or if norm > r
            if norm <= self.r and all(0 <= x <= 1 for x in point):
                return point
    
    def _generate(self):
        np.random.seed(0)
        """Generate the dataset with positive and negative samples."""
        # Generate negative samples (uniform in [0,1]^d)
        self.X_negative = np.random.uniform(0, 1, (self.num_negative, self.d))
        
        # Calculate number of cluster-positive samples
        num_cluster_positive = int(self.num_positive * (1 - self.positive_background_ratio))
        num_background_positive = self.num_positive - num_cluster_positive
        
        # Generate background positive samples (uniform in [0,1]^d)
        if num_background_positive > 0:
            X_background = np.random.uniform(0, 1, (num_background_positive, self.d))
        else:
            X_background = np.empty((0, self.d))
        
        # Generate cluster-positive samples
        X_cluster = []
        
        for _ in range(num_cluster_positive):
            # Generate point from truncated normal for first d0 dimensions
            normal_part = self._generate_truncated_normal_point()
            
            # Generate uniform point for remaining dimensions
            if self.d > self.d0:
                remaining_dims = np.random.uniform(0, 1, self.d - self.d0)
                point = np.concatenate((normal_part, remaining_dims))
            else:
                point = normal_part
                
            X_cluster.append(point)
            
        X_cluster = np.array(X_cluster)
        
        # If two clusters, generate a second one at the opposite corner
        if self.num_clusters == 2:
            X_cluster2 = []
            for _ in range(num_cluster_positive):
                # Generate point from truncated normal for first d0 dimensions (opposite corner)
                normal_part = 1 - self._generate_truncated_normal_point()
                
                # Generate uniform point for remaining dimensions
                if self.d > self.d0:
                    remaining_dims = np.random.uniform(0, 1, self.d - self.d0)
                    point = np.concatenate((normal_part, remaining_dims))
                else:
                    point = normal_part
                    
                X_cluster2.append(point)
                
            X_cluster2 = np.array(X_cluster2)
            X_cluster = np.vstack((X_cluster, X_cluster2))
        
        # Combine all positive samples
        self.X_positive = np.vstack((X_cluster, X_background)) if X_background.size > 0 else X_cluster
        
        # Combine all samples and create labels
        self.X = np.vstack((self.X_negative, self.X_positive))
        self.y = np.concatenate((
            np.zeros(self.num_negative),
            np.ones(self.num_positive)
        ))
        
  
    
    def get_dataset(self):
        """Return the complete dataset (X, y)."""
        return self.X, self.y
    
    @classmethod
    def from_ratio(cls, d, d0, num_positive, num_negative, f=8, 
                   positive_background_ratio=0.0, num_clusters=1):
        """
        Alternative constructor that calculates r and sigma based on desired ratio f = q/p.
        
        Parameters:
        - d: Total dimensionality
        - d0: Dimensionality of the normal distribution
        - num_positive: Number of positive samples (|P|)
        - num_negative: Number of negative samples (|N|)
        - f: Desired ratio q/p of negatives to positives inside cluster
        - positive_background_ratio: Fraction of positive samples to distribute uniformly
        - num_clusters: Number of clusters to generate (1 or 2)
        """
        # Calculate effective p based on background ratio and number of clusters
        p = num_positive * (1 - positive_background_ratio)
        if num_clusters == 2:
            p = p / 2  # Each cluster gets half the non-background positives
        
        # Volume of d0-dimensional L2 ball
        def ball_volume(r):
            return (np.pi ** (d0/2) * (r ** d0)) / gamma(d0/2 + 1)
        
        # Equation to solve: f = (vol(r)*|N|*F(1))/(2^d*|P|*F(r))
        def equation(sigma):
            # We'll solve for r given sigma
            def inner_eq(r):
                if r > 1:
                    return np.inf
                vol = ball_volume(r)
                F_r = chi.cdf(r/sigma, d0)
                F_1 = chi.cdf(1/sigma, d0)
                return vol * num_negative * F_1 / (2**d * p * F_r) - f
            
            # Find r that satisfies the equation for this sigma
            try:
                sol = root_scalar(inner_eq, bracket=[0.01, 1], method='brentq')
                return sol.root, sol.converged
            except:
                return np.nan, False
        
        # Find sigma that gives r ≤ 1
        # We'll do a simple search over possible sigma values
        best_sigma = None
        best_r = None
        min_diff = np.inf
        
        # Search over possible sigma values
        for sigma in np.linspace(0.1, 1.0, 100):
            r, converged = equation(sigma)
            if converged and r <= 1:
                # Calculate how close we are to desired f
                vol = ball_volume(r)
                F_r = chi.cdf(r/sigma, d0)
                F_1 = chi.cdf(1/sigma, d0)
                current_f = vol * num_negative * F_1 / (2**d * p * F_r)
                diff = abs(current_f - f)
                
                if diff < min_diff:
                    min_diff = diff
                    best_sigma = sigma
                    best_r = r
        
        if best_sigma is None:
            raise ValueError("Could not find parameters satisfying the desired ratio")
            
        return cls(d=d, d0=d0, r=best_r, sigma=best_sigma, num_positive=num_positive,
                   num_negative=num_negative, positive_background_ratio=positive_background_ratio,
                   num_clusters=num_clusters)

