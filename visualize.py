"""
visualize.py
Generates comparison charts, confusion matrices, ROC curves, and feature
importance plots for the disease prediction results.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_metric_comparison(eval_output, out_path):
    """Bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC across models."""
    rows = []
    for model_name, r in eval_output["results"].items():
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            rows.append({"Model": model_name, "Metric": metric.replace("_", "-").upper(), "Score": r[metric]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette=PALETTE)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Model Comparison — {eval_output['dataset_name']}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(eval_output, out_path):
    """Grid of confusion matrices, one per model."""
    n_models = len(eval_output["results"])
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, eval_output["results"].items()):
        cm = r["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
        ax.set_title(model_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(f"Confusion Matrices — {eval_output['dataset_name']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(eval_output, out_path):
    """Overlaid ROC curves for all models."""
    plt.figure(figsize=(7, 6))
    for color, (model_name, r) in zip(PALETTE, eval_output["results"].items()):
        plt.plot(r["fpr"], r["tpr"], label=f"{model_name} (AUC = {r['roc_auc']:.3f})",
                  color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {eval_output['dataset_name']}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(eval_output, out_path, top_n=15):
    """Feature importance from Random Forest and XGBoost (tree-based models only)."""
    tree_models = {k: v for k, v in eval_output["results"].items()
                    if hasattr(v["model"], "feature_importances_")}
    if not tree_models:
        return False

    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    feature_names = eval_output["feature_names"]
    for ax, (model_name, r) in zip(axes, tree_models.items()):
        importances = r["model"].feature_importances_
        idx = np.argsort(importances)[-top_n:]
        ax.barh(np.array(feature_names)[idx], importances[idx], color="#4C72B0")
        ax.set_title(f"{model_name} — Top {top_n} Features", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance")

    fig.suptitle(f"Feature Importance — {eval_output['dataset_name']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def generate_all_plots(eval_output, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    key = eval_output["dataset_name"].split(" ")[0].lower().replace("(", "").replace(")", "")
    paths = {}

    p1 = os.path.join(out_dir, f"{key}_metric_comparison.png")
    plot_metric_comparison(eval_output, p1)
    paths["metric_comparison"] = p1

    p2 = os.path.join(out_dir, f"{key}_confusion_matrices.png")
    plot_confusion_matrices(eval_output, p2)
    paths["confusion_matrices"] = p2

    p3 = os.path.join(out_dir, f"{key}_roc_curves.png")
    plot_roc_curves(eval_output, p3)
    paths["roc_curves"] = p3

    p4 = os.path.join(out_dir, f"{key}_feature_importance.png")
    if plot_feature_importance(eval_output, p4):
        paths["feature_importance"] = p4

    return paths
