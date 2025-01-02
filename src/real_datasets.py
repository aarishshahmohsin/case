import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np
from constants import DATA_PATH
import os


class Dataset:
    def __init__(self):
        pass

    def generate(self):
        positive_mask = self.y == 1
        negative_mask = self.y == 0
        self.P = self.X[positive_mask]
        self.N = self.X[negative_mask]
        return self.P, self.N

    def params(self):
        self.lambda_param = (len(self.P) + 1) * self.theta1
        self.theta = self.theta0 / self.theta1
        return (self.theta0, self.theta1, self.theta, self.lambda_param)


class BreastCancerDataset(Dataset):
    def __init__(self):
        d = load_breast_cancer()
        self.X = d["data"]
        self.y = d["target"]
        self.theta0 = 99
        self.theta1 = 100


class WineQualityRedDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(
            str(os.path.join(DATA_PATH, "wine-quality/winequality-red.csv")),
            delimiter=";",
        )
        self.X, self.y = (
            self.df.drop(columns=["quality"]).to_numpy(),
            self.df["quality"].to_numpy(),
        )
        self.y = (self.y >= 8).astype(int)
        self.theta0 = 4
        self.theta1 = 100


class WineQualityWhiteDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(
            str(os.path.join(DATA_PATH, "wine-quality/winequality-white.csv")),
            delimiter=";",
        )
        self.X, self.y = (
            self.df.drop(columns=["quality"]).to_numpy(),
            self.df["quality"].to_numpy(),
        )
        self.y = (self.y >= 8).astype(int)
        self.theta0 = 10
        self.theta1 = 100


class SouthGermanCreditDataset(Dataset):
    def __init__(self):
        self.south_german_credit_df = pd.read_csv(
            str(os.path.join(DATA_PATH, "south-german-credit/SouthGermanCredit.dat")),
            skiprows=1,
            header=None,
            delimiter=" ",
        )
        self.X, self.y = (
            self.south_german_credit_df.drop(columns=[6]).to_numpy(),
            self.south_german_credit_df[6].to_numpy(),
        )
        self.theta0 = 90
        self.theta1 = 100


class CropMappingDataset(Dataset):
    def __init__(self):
        file_path = str(os.path.join(DATA_PATH, "crops/small-sample.dat"))
        with open(file_path, "r") as file:
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

        self.theta0 = 99
        self.theta1 = 100
