"""
Drowsiness Detection - Model Training Script
Features: EAR, MAR, Head_Tilt
Label: 0 = Alert, 1 = Drowsy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
df = pd.read_csv("cleaned_dataset(1).csv"))
print("=" * 55)
print("DATASET INFO")
print("=" * 55)
print(f"Shape        : {df.shape}")
print(f"Features     : {['EAR', 'MAR', 'Head_Tilt']}")
print(f"Label dist   : \n{df['Label'].value_counts().to_string()}")
print()

X = df[["EAR", "MAR", "Head_Tilt"]].values
y = df["Label"].values

# ──────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples : {len(X_train)}")
print(f"Test samples  : {len(X_test)}")
print()

# ──────────────────────────────────────────────
# 3. TRAIN MODELS & COMPARE
# ──────────────────────────────────────────────
models = {
    "Random Forest"        : RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    "SVM (RBF)"            : SVC(kernel="rbf", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 55)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 55)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results[name] = scores
    print(f"{name:<25} Acc: {scores.mean():.4f} ± {scores.std():.4f}")
print()

# ──────────────────────────────────────────────
# 4. TRAIN BEST MODEL (Random Forest)
# ──────────────────────────────────────────────
best_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
best_model.fit(X_train, y_train)

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("=" * 55)
print("RANDOM FOREST - TEST SET RESULTS")
print("=" * 55)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Alert (0)", "Drowsy (1)"]))

# ──────────────────────────────────────────────
# 5. PLOTS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Drowsiness Detection - Model Report", fontsize=16, fontweight="bold")

# -- Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["Alert", "Drowsy"],
            yticklabels=["Alert", "Drowsy"])
axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

# -- Feature Importance
importances = best_model.feature_importances_
features = ["EAR", "MAR", "Head_Tilt"]
colors = ["#2196F3", "#FF5722", "#4CAF50"]
bars = axes[0, 1].barh(features, importances, color=colors)
axes[0, 1].set_title("Feature Importance", fontweight="bold")
axes[0, 1].set_xlabel("Importance Score")
for bar, val in zip(bars, importances):
    axes[0, 1].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=11)

# -- ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
axes[1, 0].plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc_score:.3f}")
axes[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[1, 0].set_title("ROC Curve", fontweight="bold")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend()

# -- Cross-val comparison
cv_means = [results[n].mean() for n in models]
cv_stds  = [results[n].std()  for n in models]
model_names = ["Random\nForest", "Gradient\nBoosting", "SVM\n(RBF)"]
bars2 = axes[1, 1].bar(model_names, cv_means, yerr=cv_stds,
                        color=["#2196F3", "#FF5722", "#9C27B0"],
                        capsize=6, alpha=0.85)
axes[1, 1].set_title("Cross-Validation Accuracy Comparison", fontweight="bold")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_ylim(0.7, 1.01)
for bar, val in zip(bars2, cv_means):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/model_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → model_report.png")

# ──────────────────────────────────────────────
# 6. SAVE MODEL
# ──────────────────────────────────────────────
joblib.dump(best_model, "drowsiness_model.pkl")
print("Model saved → drowsiness_model.pkl")

# ──────────────────────────────────────────────
# 7. HOW TO USE IN REAL-TIME PIPELINE
# ──────────────────────────────────────────────
print()
print("=" * 55)
print("HOW TO USE IN YOUR REAL-TIME PIPELINE")
print("=" * 55)
print("""
import joblib
import numpy as np
from collections import deque

# Load once at startup
model = joblib.load("drowsiness_model.pkl")

# Sliding window (last 20 frames)
window = deque(maxlen=20)
ALERT_THRESHOLD = 0.6   # 60% of window frames = drowsy

def predict_frame(ear, mar, head_tilt):
    # NOTE: your data was standardized during cleaning.
    # You MUST apply the same scaler here before predicting.
    # Either save the scaler and load it, or hardcode mean/std from training data.
    features = np.array([[ear, mar, head_tilt]])
    pred = model.predict(features)[0]
    window.append(pred)
    drowsy_ratio = sum(window) / len(window)
    alert = drowsy_ratio >= ALERT_THRESHOLD
    return alert, drowsy_ratio
""")
