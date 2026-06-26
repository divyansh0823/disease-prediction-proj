"""
pipeline.py
Trains and evaluates SVM, Logistic Regression, Random Forest, and XGBoost
on a given dataset, with proper scaling, train/test split, and metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

RANDOM_STATE = 42


def get_models():
    """Returns a dict of model name -> unfitted estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_dataset(X, y, dataset_name, test_size=0.2, cv_folds=5):
    """
    Trains every model on (X, y), evaluates on a held-out test split, and also
    runs stratified k-fold cross-validation for a more robust accuracy estimate.

    Returns a dict with per-model metrics and fitted artifacts needed for plotting.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for model_name, model in get_models().items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        cv_scores = cross_val_score(
            model, scaler.fit_transform(X), y, cv=cv, scoring="accuracy", n_jobs=-1
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "fpr": fpr,
            "tpr": tpr,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }

    return {
        "dataset_name": dataset_name,
        "results": results,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
    }


def summary_table(eval_output):
    """Builds a tidy comparison DataFrame across all models for one dataset."""
    rows = []
    for model_name, r in eval_output["results"].items():
        rows.append({
            "Model": model_name,
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1"], 4),
            "ROC-AUC": round(r["roc_auc"], 4),
            "CV Accuracy (mean ± std)": f"{r['cv_mean_accuracy']:.4f} ± {r['cv_std_accuracy']:.4f}",
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df
