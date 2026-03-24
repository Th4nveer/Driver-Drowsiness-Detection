"""
============================================================
 train_drowsiness.py  —  Train & save the drowsiness model
============================================================
NOTE: This version is for PRE-SCALED datasets (mean≈0, std≈1).
      Do NOT use StandardScaler — data is already standardized.

Usage:
    python train_drowsiness.py
    python train_drowsiness.py --csv cleaned_dataset.csv
    python train_drowsiness.py --csv "cleaned_dataset(1).csv"
============================================================
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve,
)

FEATURES  = ["EAR", "MAR", "Head_Tilt"]
LABEL     = "Label"
MODEL_OUT = "drowsiness_model.pkl"
PLOT_OUT  = "model_report.png"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {csv_path}: {df.shape}")
    print(f"       Columns : {df.columns.tolist()}")

    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURES + [LABEL])
    print(f"[INFO] Rows after dropna : {len(df)}")
    print(f"[INFO] Label distribution:\n{df[LABEL].value_counts().to_string()}")

    # Confirm data is pre-scaled (mean should be near 0)
    means = df[FEATURES].mean()
    print(f"\n[INFO] Feature means (should be ~0 if pre-scaled):")
    for f, m in means.items():
        print(f"       {f}: {m:.6f}")
    print()
    return df


def train(csv_path):
    df = load_data(csv_path)
    X  = df[FEATURES].values
    y  = df[LABEL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── No scaler needed — data is already standardized ──────────
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("=" * 55)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 55)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results[name] = scores
        print(f"  {name:<22}  {scores.mean():.4f} ± {scores.std():.4f}")
    print()

    # ── Train best model (Random Forest) ────────────────────────
    best_model = models["Random Forest"]
    best_model.fit(X_train, y_train)

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("=" * 55)
    print("RANDOM FOREST — TEST SET RESULTS")
    print("=" * 55)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Alert", "Drowsy"]))

    # ── Save plain model (no pipeline wrapper) ───────────────────
    joblib.dump(best_model, MODEL_OUT)
    print(f"[INFO] Model saved → {MODEL_OUT}")

    # ── Plots ────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Drowsiness Detection – Model Report", fontsize=16, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0, 0],
                xticklabels=["Alert", "Drowsy"],
                yticklabels=["Alert", "Drowsy"])
    ax[0, 0].set_title("Confusion Matrix", fontweight="bold")
    ax[0, 0].set_ylabel("Actual")
    ax[0, 0].set_xlabel("Predicted")

    # Feature importance — directly from best_model (not named_steps)
    imp    = best_model.feature_importances_
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    bars   = ax[0, 1].barh(FEATURES, imp, color=colors)
    ax[0, 1].set_title("Feature Importance", fontweight="bold")
    ax[0, 1].set_xlabel("Importance Score")
    for b, v in zip(bars, imp):
        ax[0, 1].text(v + 0.002, b.get_y() + b.get_height() / 2,
                      f"{v:.3f}", va="center")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax[1, 0].plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.3f}")
    ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1, 0].set_title("ROC Curve", fontweight="bold")
    ax[1, 0].set_xlabel("FPR")
    ax[1, 0].set_ylabel("TPR")
    ax[1, 0].legend()

    # CV comparison
    cv_means = [results[n].mean() for n in models]
    cv_stds  = [results[n].std()  for n in models]
    mnames   = ["Random\nForest", "Gradient\nBoosting", "SVM\n(RBF)"]
    bars2    = ax[1, 1].bar(mnames, cv_means, yerr=cv_stds,
                             color=["#2196F3", "#FF5722", "#9C27B0"],
                             capsize=6, alpha=0.85)
    ax[1, 1].set_title("Cross-Validation Accuracy Comparison", fontweight="bold")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].set_ylim(0.7, 1.01)
    for b, v in zip(bars2, cv_means):
        ax[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.005,
                      f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved → {PLOT_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drowsiness detection model")
    parser.add_argument(
        "--csv",
        default="cleaned_dataset.csv",
        help='Path to cleaned CSV. Example: --csv "cleaned_dataset(1).csv"'
    )
    args = parser.parse_args()
    train(args.csv)