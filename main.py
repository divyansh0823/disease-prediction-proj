"""
main.py
Entry point: trains SVM, Logistic Regression, Random Forest, and XGBoost on
the Heart Disease, Diabetes, and Breast Cancer datasets, then prints metrics
and saves comparison charts + trained models.

Usage:
    python main.py                 # run all three datasets
    python main.py --dataset heart # run just one
"""

import os
import sys
import argparse
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DATASET_LOADERS
from pipeline import evaluate_dataset, summary_table
from visualize import generate_all_plots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def run_for_dataset(key, loader_fn):
    print(f"\n{'='*70}")
    print(f"Dataset: {key}")
    print(f"{'='*70}")

    X, y, feature_names, dataset_name = loader_fn()
    print(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{y.mean()*100:.1f}% positive class")

    eval_output = evaluate_dataset(X, y, dataset_name)

    table = summary_table(eval_output)
    print(f"\nResults for {dataset_name}:\n")
    print(table.to_string(index=False))

    plot_paths = generate_all_plots(eval_output, OUTPUTS_DIR)
    print(f"\nSaved plots: {list(plot_paths.values())}")

    best_model_name = table.iloc[0]["Model"]
    best_model = eval_output["results"][best_model_name]["model"]
    scaler = eval_output["scaler"]

    model_bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "dataset_name": dataset_name,
        "model_name": best_model_name,
    }
    model_path = os.path.join(MODELS_DIR, f"{key}_best_model.joblib")
    joblib.dump(model_bundle, model_path)
    print(f"Saved best model ({best_model_name}) to: {model_path}")

    return eval_output, table


def main():
    parser = argparse.ArgumentParser(description="Disease Prediction from Medical Data")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()) + ["all"],
                         default="all", help="Which dataset to run (default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    targets = DATASET_LOADERS if args.dataset == "all" else {args.dataset: DATASET_LOADERS[args.dataset]}

    all_tables = {}
    for key, loader_fn in targets.items():
        _, table = run_for_dataset(key, loader_fn)
        all_tables[key] = table

    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (best model per dataset)")
    print(f"{'='*70}")
    for key, table in all_tables.items():
        best = table.iloc[0]
        print(f"  {key:15s} -> {best['Model']:22s} "
              f"Accuracy={best['Accuracy']:.4f}  ROC-AUC={best['ROC-AUC']:.4f}")


if __name__ == "__main__":
    main()
