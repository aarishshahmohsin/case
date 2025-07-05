import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Dataset:
    def __init__(self):
        self._extract()

    def _extract(self):
        with open(self.file_path, "r") as file:
            lines = file.readlines()
        header = list(map(int, lines[0].split()))
        num_classes, num_negative_samples, num_positive_samples = header
        negative_samples = []
        positive_samples = []
        for i in range(1, num_negative_samples + 1):
            negative_samples.append(list(map(float, lines[i].split())))
        for i in range(
            num_negative_samples + 1, num_negative_samples + num_positive_samples + 1
        ):
            positive_samples.append(list(map(float, lines[i].split())))

        X_negative = np.array(negative_samples)
        X_positive = np.array(positive_samples)

        self.X = np.vstack([X_negative, X_positive])

        y_negative = np.zeros(num_negative_samples)
        y_positive = np.ones(num_positive_samples)

        self.y = np.hstack([y_negative, y_positive])

    def generate(self, normalize=False):
        if not hasattr(self, "P"):
            positive_mask = self.y == 1
            negative_mask = self.y == 0
            self.P = self.X[positive_mask]
            self.N = self.X[negative_mask]

        if normalize:
            self.P = self._normalize_unit_sphere(self.P)
            self.N = self._normalize_unit_sphere(self.N)

        return self.P, self.N

    def params(self):
        self.theta = self.theta0 / self.theta1
        self.lambda_param = (len(self.P) + 1) * self.theta1
        return (self.theta0, self.theta1, self.theta, self.lambda_param)

    def _normalize_unit_sphere(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / row_norms
        return X


def plot_P_N(P, N):
    X = np.vstack((P, N))
    y = np.hstack((np.ones(len(P)), np.zeros(len(N))))
    plt.figure(figsize=(8, 8))
    plt.scatter(
        X[y == 0][:, 0],
        X[y == 0][:, 1],
        color="blue",
        label="Negative Samples",
        alpha=0.1,
        marker="x",
    )
    plt.scatter(
        X[y == 1][:, 0],
        X[y == 1][:, 1],
        color="red",
        label="Positive Samples",
        alpha=0.5,
        marker="x",
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()


def plot_P_N_3d(P, N):
    X = np.vstack((P, N))
    y = np.hstack((np.ones(len(P)), np.zeros(len(N))))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X[y == 0][:, 0],
        X[y == 0][:, 1],
        X[y == 0][:, 2],
        color="blue",
        label="Negative Samples",
        alpha=0.1,
        marker="x",
    )
    ax.scatter(
        X[y == 1][:, 0],
        X[y == 1][:, 1],
        X[y == 1][:, 2],
        color="red",
        label="Positive Samples",
        alpha=0.5,
        marker="x",
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()
    plt.grid()
    plt.show()


def compute_reach(P, w, c, epsilon_P):
    tau = P @ w - c  # τ(s) = s^T w - c
    consistent = tau >= epsilon_P
    return np.sum(consistent)
