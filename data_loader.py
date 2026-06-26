"""
data_loader.py
Loads the three real-world medical datasets used in this project:

1. Breast Cancer Wisconsin (Diagnostic) - via sklearn.datasets (UCI source)
2. Pima Indians Diabetes Dataset       - via public CSV mirror
3. UCI Heart Disease (Cleveland)       - via public CSV mirror

Each loader returns (X, y, feature_names, dataset_name) where y is binary
(1 = disease present, 0 = not present).
"""

import os
import pandas as pd
import numpy as np
import urllib.request
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DIABETES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
HEART_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"


def _download_if_missing(url, local_path):
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path


def load_breast_cancer_data():
    """UCI Breast Cancer Wisconsin (Diagnostic) dataset. 569 samples, 30 features."""
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    # sklearn encodes 0=malignant, 1=benign. Flip so 1 = disease (malignant), consistent
    # with the convention used for the other two datasets in this project.
    y = pd.Series(np.where(data.target == 0, 1, 0), name="target")
    return X, y, list(X.columns), "Breast Cancer (UCI Wisconsin Diagnostic)"


def load_diabetes_data():
    """Pima Indians Diabetes dataset. 768 samples, 8 features. 1 = diabetic."""
    local_path = os.path.join(DATA_DIR, "diabetes.csv")
    _download_if_missing(DIABETES_URL, local_path)
    df = pd.read_csv(local_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].rename("target")
    return X, y, list(X.columns), "Diabetes (Pima Indians)"


def load_heart_data():
    """UCI Heart Disease (Cleveland) dataset. 303 samples, 13 features. 1 = disease present."""
    local_path = os.path.join(DATA_DIR, "heart.csv")
    _download_if_missing(HEART_URL, local_path)
    df = pd.read_csv(local_path, encoding="utf-8-sig")
    X = df.drop(columns=["target"])
    y = df["target"].rename("target")
    return X, y, list(X.columns), "Heart Disease (UCI Cleveland)"


DATASET_LOADERS = {
    "breast_cancer": load_breast_cancer_data,
    "diabetes": load_diabetes_data,
    "heart": load_heart_data,
}


def load_all():
    """Returns a dict: {dataset_key: (X, y, feature_names, display_name)}"""
    return {key: loader() for key, loader in DATASET_LOADERS.items()}


if __name__ == "__main__":
    for key, loader in DATASET_LOADERS.items():
        X, y, features, name = loader()
        print(f"{name}: X={X.shape}, positive_rate={y.mean():.3f}, features={len(features)}")
