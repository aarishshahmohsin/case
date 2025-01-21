import numpy as np
import matplotlib.pyplot as plt


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

    def generate(self):
        positive_mask = self.y == 1
        negative_mask = self.y == 0
        self.P = self.X[positive_mask]
        self.N = self.X[negative_mask]
        return self.P, self.N

    def params(self):
        self.theta = self.theta0 / self.theta1
        self.lambda_param = (len(self.P) + 1) * self.theta1
        return (self.theta0, self.theta1, self.theta, self.lambda_param)


def plot_P_N(P, N):
    X = np.vstack((P, N))
    y = np.hstack((np.ones(len(P)), np.zeros(len(N))))
    plt.figure(figsize=(8, 8))
    plt.scatter(
        X[y == 0][:, 0],
        X[y == 0][:, 1],
        color="blue",
        label="Negative Samples",
        alpha=0.5,
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
