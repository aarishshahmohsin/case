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
        self.theta0 = 1
        self.theta1 = 25


class WineQualityWhiteDataset(Dataset):
    def __init__(self):
        self.file_path = str(
            os.path.join(DATA_PATH, "wine-quality/winequality-white.dat")
        )
        super().__init__()
        self.theta0 = 1
        self.theta1 = 10


class SouthGermanCreditDataset(Dataset):
    def __init__(self):
        self.file_path = str(
            os.path.join(DATA_PATH, "south-german-credit/SouthGermanCredit.dat")
        )
        super().__init__()
        self.theta0 = 9
        self.theta1 = 10


class CropMappingDataset(Dataset):
    def __init__(self):
        self.file_path = str(os.path.join(DATA_PATH, "crops/small-sample.dat"))
        super().__init__()
        self.theta0 = 99
        self.theta1 = 100
