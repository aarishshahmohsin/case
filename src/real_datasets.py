from constants import DATA_PATH
from utils import Dataset
import os


class BreastCancerDataset(Dataset):
    def __init__(self):
        self.file_path = str(os.path.join(DATA_PATH, "breast-cancer/wdbc.dat"))
        super().__init__()
        self.theta0 = 99
        self.theta1 = 100


class WineQualityRedDataset(Dataset):
    def __init__(self):
        self.file_path = str(
            os.path.join(DATA_PATH, "wine-quality/winequality-red.dat")
        )
        super().__init__()
        self.theta0 = 4
        self.theta1 = 100


class WineQualityWhiteDataset(Dataset):
    def __init__(self):
        self.file_path = str(
            os.path.join(DATA_PATH, "wine-quality/winequality-white.dat")
        )
        super().__init__()
        self.theta0 = 10
        self.theta1 = 100


class SouthGermanCreditDataset(Dataset):
    def __init__(self):
        self.file_path = str(
            os.path.join(DATA_PATH, "south-german-credit/SouthGermanCredit.dat")
        )
        super().__init__()
        self.theta0 = 90
        self.theta1 = 100


class CropMappingDataset(Dataset):
    def __init__(self):
        self.file_path = str(os.path.join(DATA_PATH, "crops/small-sample.dat"))
        super().__init__()
        self.theta0 = 99
        self.theta1 = 100
