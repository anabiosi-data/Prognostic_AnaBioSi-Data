"""
Stiffness-Only Logistic Regression Model
Ablation study: Predicting treatment response using only elastic modulus (kPa).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
# Path to the Excel file with per-image elastic modulus (kPa) and response label.
# Set via the KPA_EXCEL environment variable, or edit the default below.
# (Dataset available from the corresponding author on request — see README.)
EXCEL_PATH = os.environ.get("KPA_EXCEL", "clustering_all_v5.xlsx")

df = pd.read_excel(EXCEL_PATH)
print(f"Loaded {len(df)} samples")
print(f"Column names: {df.columns.tolist()}")

label_col = [c for c in df.columns if 'respon' in c.lower() or 'stable' in c.lower()][0]
kpa_col = [c for c in df.columns if 'elastic' in c.lower() or 'kpa' in c.lower()][0]

print(f"Label column: '{label_col}'")
print(f"kPa column: '{kpa_col}'")

X = df[[kpa_col]].values
y = df[label_col].values

unique_labels = np.unique(y)
print(f"Unique labels: {unique_labels}")

class_names = ['Response', 'Stable', 'Non-Response']
n_classes = 3

print(f"\nClass distribution:")
for i, name in enumerate(class_names):
    print(f"  {name}: {np.sum(y == i)} ({np.sum(y == i)/len(y)*100:.1f}%)")

# 2. TRAIN/VALIDATION/TEST SPLIT (70/15/15)
SEED = 42
np.random.seed(SEED)

split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
dev_idx, test_idx = next(split1.split(X, y))
X_dev, X_test = X[dev_idx], X[test_idx]
y_dev, y_test = y[dev_idx], y[test_idx]

val_fraction = 0.15 / 0.85
split2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED)
train_idx, val_idx = next(split2.split(X_dev, y_dev))
X_train, X_val = X_dev[train_idx], X_dev[val_idx]
y_train, y_val = y_dev[train_idx], y_dev[val_idx]

print(f"\nData splits:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 3. STANDARDIZE kPa (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# 4. FIT LOGISTIC REGRESSION
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=SEED,
    C=1.0
)
model.fit(X_train_s, y_train)

# 5. PREDICTIONS
y_pred_test = model.predict(X_test_s)
y_prob_test = model.predict_proba(X_test_s)
y_pred_val = model.predict(X_val_s)
y_prob_val = model.predict_proba(X_val_s)

# 6. METRICS
test_acc = accuracy_score(y_test, y_pred_test)
val_acc = accuracy_score(y_val, y_pred_val)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
macro_auc = roc_auc_score(y_test_bin, y_prob_test, multi_class='ovr', average='macro')

per_class_auc = {}
for i, name in enumerate(class_names):
    per_class_auc[name] = roc_auc_score(y_test_bin[:, i], y_prob_test[:, i])

macro_f1 = f1_score(y_test, y_pred_test, average='macro')
confidence = np.max(y_prob_test, axis=1)
mean_confidence = np.mean(confidence)

print("\nSTIFFNESS-ONLY LOGISTIC REGRESSION RESULTS")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Macro AUC:           {macro_auc:.4f}")
print(f"Macro F1:            {macro_f1:.4f}")
print(f"Mean Confidence:     {mean_confidence:.4f}")
print(f"\nPer-class AUC:")
for name, a in per_class_auc.items():
    print(f"  {name}: {a:.4f}")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=class_names, digits=4))

# 7. 5-FOLD CROSS-VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(
    LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=SEED),
    scaler.fit_transform(X_dev), y_dev, cv=cv, scoring='accuracy'
)
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

# 8. PLOTS
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Stiffness-Only Logistic Regression - Ablation Study', fontsize=16, fontweight='bold')

# Confusion Matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
            yticklabels=class_names, ax=ax, cbar=False,
            annot_kws={'size': 14, 'fontweight': 'bold'})
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title(f'Confusion Matrix (Accuracy = {test_acc:.3f})', fontsize=12)

# ROC Curves
ax = axes[0, 1]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_test[:, i])
    roc_auc_i = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc_i:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(f'ROC Curves (Macro AUC = {macro_auc:.3f})', fontsize=12)
ax.legend(loc='lower right', fontsize=9)

# Confidence Histogram
ax = axes[0, 2]
for i, (name, color) in enumerate(zip(class_names, colors)):
    mask = y_pred_test == i
    if mask.sum() > 0:
        ax.hist(confidence[mask], bins=15, alpha=0.6, color=color, label=name, edgecolor='white')
ax.axvline(mean_confidence, color='black', linestyle='--', lw=1.5, label=f'Mean={mean_confidence:.3f}')
ax.set_xlabel('Confidence Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Prediction Confidence Distribution', fontsize=12)
ax.legend(fontsize=9)

# Calibration Curve
ax = axes[1, 0]
for i, (name, color) in enumerate(zip(class_names, colors)):
    prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_prob_test[:, i], n_bins=8, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', color=color, lw=2, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=11)
ax.set_ylabel('Fraction of Positives', fontsize=11)
ax.set_title('Calibration Curves', fontsize=12)
ax.legend(fontsize=9)

# Decision Boundaries
ax = axes[1, 1]
kpa_range = np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 500).reshape(-1, 1)
kpa_range_s = scaler.transform(kpa_range)
probs_range = model.predict_proba(kpa_range_s)
for i, (name, color) in enumerate(zip(class_names, colors)):
    ax.plot(kpa_range, probs_range[:, i], color=color, lw=2.5, label=name)
ax.set_xlabel('Elastic Modulus (kPa)', fontsize=11)
ax.set_ylabel('Predicted Probability', fontsize=11)
ax.set_title('Decision Boundaries', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
preds_range = model.predict(kpa_range_s)
for j in range(len(preds_range) - 1):
    if preds_range[j] != preds_range[j + 1]:
        ax.axvline(kpa_range[j, 0], color='gray', linestyle=':', alpha=0.7)

# Radar Plot
ax = axes[1, 2]
report = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
ax_polar = fig.add_subplot(2, 3, 6, projection='polar')
axes[1, 2].set_visible(False)
for name, color in zip(class_names, colors):
    values = [report[name][m] for m in metrics]
    values += values[:1]
    ax_polar.plot(angles, values, 'o-', color=color, lw=2, label=name)
    ax_polar.fill(angles, values, alpha=0.1, color=color)
ax_polar.set_thetagrids(np.degrees(angles[:-1]), ['Precision', 'Recall', 'F1-score'])
ax_polar.set_ylim(0, 1)
ax_polar.set_title('Per-Class Metrics', fontsize=12, pad=20)
ax_polar.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), fontsize=9)

plt.tight_layout()

# Save figure in same folder as script
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(script_dir, 'stiffness_only_logistic_regression_results.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")
plt.savefig(fig_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# 9. SUMMARY TABLE
print("\nSUMMARY FOR COMPARISON TABLE")
print(f"Model: Logistic Regression (kPa only)")
print(f"Parameters: {n_classes * 2} (3 weights + 3 biases)")
print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss (Brier): {np.mean([brier_score_loss(y_test_bin[:, i], y_prob_test[:, i]) for i in range(3)]):.4f}")
print(f"Macro AUC: {macro_auc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Mean Confidence: {mean_confidence:.4f}")
print(f"Response Precision: {report['Response']['precision']:.4f}")
print(f"Response Recall: {report['Response']['recall']:.4f}")
print(f"Stable Precision: {report['Stable']['precision']:.4f}")
print(f"Stable Recall: {report['Stable']['recall']:.4f}")
print(f"Non-Response Precision: {report['Non-Response']['precision']:.4f}")
print(f"Non-Response Recall: {report['Non-Response']['recall']:.4f}")