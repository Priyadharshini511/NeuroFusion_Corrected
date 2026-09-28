"""
NeuroFusion AI
sMRI Branch - Nested CV OOF Prediction Analysis

READ-ONLY:
Analyzes completed 5x4 nested-CV OOF predictions.
Does NOT retrain or modify model/data/splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data/smri/cnn_dataset")

CV_DIR = ROOT / "cross_validation"
TRAINING_DIR = CV_DIR / "training_full_5x4"

OOF_FILE = TRAINING_DIR / "nested_cv_oof_predictions.csv"

OUTPUT_DIR = CV_DIR / "audit" / "prediction_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "oof_prediction_analysis_report.txt"
SUMMARY_FILE = OUTPUT_DIR / "oof_prediction_analysis_summary.csv"
FOLD_FILE = OUTPUT_DIR / "oof_prediction_analysis_by_fold.csv"

ROC_FILE = OUTPUT_DIR / "oof_roc_curve.png"
PR_FILE = OUTPUT_DIR / "oof_precision_recall_curve.png"
PROB_FILE = OUTPUT_DIR / "oof_probability_distribution.png"
FOLD_PROB_FILE = OUTPUT_DIR / "oof_fold_probability_distribution.png"


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for c in df.columns:
        cl = c.lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c

    raise KeyError(
        f"Could not find {candidates} in {list(df.columns)}"
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 72)
print("NeuroFusion AI - sMRI Nested CV OOF Prediction Analysis")
print("=" * 72)

if not OOF_FILE.exists():
    raise FileNotFoundError(
        f"OOF prediction file not found: {OOF_FILE}"
    )

df = pd.read_csv(OOF_FILE)

subject_col = find_column(
    df,
    ["subject_id", "subject", "id"]
)

label_col = find_column(
    df,
    ["label", "y_true", "target"]
)

prob_col = find_column(
    df,
    ["probability", "prob", "prediction_probability", "y_prob"]
)

fold_col = find_column(
    df,
    ["outer_fold", "fold"]
)

df[subject_col] = df[subject_col].astype(str)
df[label_col] = pd.to_numeric(
    df[label_col],
    errors="coerce"
)
df[prob_col] = pd.to_numeric(
    df[prob_col],
    errors="coerce"
)
df[fold_col] = pd.to_numeric(
    df[fold_col],
    errors="coerce"
)

df = df.dropna(
    subset=[
        subject_col,
        label_col,
        prob_col,
        fold_col,
    ]
).copy()

df[label_col] = df[label_col].astype(int)
df[fold_col] = df[fold_col].astype(int)

y = df[label_col].to_numpy()
p = df[prob_col].to_numpy()
folds = df[fold_col].to_numpy()


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n[1] Validation")

print(f"Rows: {len(df)}")
print(f"Unique subjects: {df[subject_col].nunique()}")
print(f"Duplicate subjects: {df[subject_col].duplicated().sum()}")
print(f"Probability range: {p.min():.6f} - {p.max():.6f}")
print(f"Probability finite: {np.isfinite(p).all()}")
print(f"Folds: {sorted(np.unique(folds).tolist())}")


# ============================================================
# ROC
# ============================================================

print("\n[2] ROC analysis")

roc_auc = roc_auc_score(y, p)

fpr, tpr, roc_thresholds = roc_curve(
    y,
    p
)

print(f"OOF ROC-AUC: {roc_auc:.6f}")

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"OOF ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("sMRI 3D-CNN — Nested CV OOF ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROC_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRECISION-RECALL
# ============================================================

print("\n[3] Precision-Recall analysis")

precision, recall, pr_thresholds = precision_recall_curve(
    y,
    p
)

average_precision = average_precision_score(
    y,
    p
)

positive_prevalence = float(
    np.mean(y)
)

print(
    f"Average Precision: "
    f"{average_precision:.6f}"
)

print(
    f"Positive prevalence: "
    f"{positive_prevalence:.6f}"
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"AP = {average_precision:.4f}"
)

plt.axhline(
    positive_prevalence,
    linestyle="--",
    linewidth=1,
    label=f"Baseline = {positive_prevalence:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "sMRI 3D-CNN — Nested CV OOF Precision-Recall Curve"
)
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PR_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n[4] Probability distribution")

control_probs = p[y == 0]
case_probs = p[y == 1]

print(
    f"Controls: n={len(control_probs)}, "
    f"mean={np.mean(control_probs):.6f}, "
    f"std={np.std(control_probs):.6f}, "
    f"median={np.median(control_probs):.6f}"
)

print(
    f"Schizophrenia: n={len(case_probs)}, "
    f"mean={np.mean(case_probs):.6f}, "
    f"std={np.std(case_probs):.6f}, "
    f"median={np.median(case_probs):.6f}"
)

plt.figure(figsize=(8, 6))

plt.hist(
    control_probs,
    bins=20,
    alpha=0.6,
    label="Control"
)

plt.hist(
    case_probs,
    bins=20,
    alpha=0.6,
    label="Schizophrenia"
)

plt.axvline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Decision threshold = 0.5"
)

plt.xlabel("Predicted schizophrenia probability")
plt.ylabel("Number of subjects")
plt.title(
    "sMRI 3D-CNN — Nested CV OOF Probability Distribution"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PROB_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FOLD-WISE PROBABILITY ANALYSIS
# ============================================================

print("\n[5] Fold-wise probability analysis")

fold_rows = []

for fold in sorted(np.unique(folds)):

    fold_df = df[
        df[fold_col] == fold
    ]

    yf = fold_df[label_col].to_numpy()
    pf = fold_df[prob_col].to_numpy()

    controls = pf[yf == 0]
    cases = pf[yf == 1]

    fold_auc = roc_auc_score(
        yf,
        pf
    )

    fold_rows.append(
        {
            "outer_fold": int(fold),
            "n": len(fold_df),
            "controls": int(np.sum(yf == 0)),
            "schizophrenia": int(np.sum(yf == 1)),
            "roc_auc": float(fold_auc),
            "prob_min": float(np.min(pf)),
            "prob_max": float(np.max(pf)),
            "prob_mean": float(np.mean(pf)),
            "prob_std": float(np.std(pf)),
            "control_prob_mean": float(
                np.mean(controls)
            ) if len(controls) else np.nan,
            "control_prob_std": float(
                np.std(controls)
            ) if len(controls) else np.nan,
            "case_prob_mean": float(
                np.mean(cases)
            ) if len(cases) else np.nan,
            "case_prob_std": float(
                np.std(cases)
            ) if len(cases) else np.nan,
        }
    )

fold_analysis = pd.DataFrame(fold_rows)

print(
    fold_analysis.to_string(index=False)
)

fold_analysis.to_csv(
    FOLD_FILE,
    index=False
)


# ============================================================
# PLOT FOLD PROBABILITY DISTRIBUTIONS
# ============================================================

plt.figure(figsize=(9, 6))

for fold in sorted(np.unique(folds)):

    fold_probs = p[folds == fold]

    plt.hist(
        fold_probs,
        bins=12,
        alpha=0.35,
        label=f"Fold {fold}"
    )

plt.axvline(
    0.5,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Predicted schizophrenia probability")
plt.ylabel("Number of subjects")
plt.title(
    "Nested CV OOF Probability Distribution by Outer Fold"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FOLD_PROB_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# THRESHOLD BEHAVIOR
# ============================================================

print("\n[6] Threshold behavior")

thresholds = [
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
]

threshold_rows = []

for threshold in thresholds:

    pred = (
        p >= threshold
    ).astype(int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    accuracy = (
        (tp + tn) / len(y)
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            "predicted_controls": int(
                np.sum(pred == 0)
            ),
            "predicted_schizophrenia": int(
                np.sum(pred == 1)
            ),
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    )

threshold_analysis = pd.DataFrame(
    threshold_rows
)

print(
    threshold_analysis.to_string(index=False)
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_rows = [
    {
        "metric": "subjects",
        "value": len(df),
    },
    {
        "metric": "unique_subjects",
        "value": df[subject_col].nunique(),
    },
    {
        "metric": "duplicate_subjects",
        "value": int(
            df[subject_col].duplicated().sum()
        ),
    },
    {
        "metric": "controls",
        "value": int(np.sum(y == 0)),
    },
    {
        "metric": "schizophrenia",
        "value": int(np.sum(y == 1)),
    },
    {
        "metric": "roc_auc",
        "value": roc_auc,
    },
    {
        "metric": "average_precision",
        "value": average_precision,
    },
    {
        "metric": "positive_prevalence",
        "value": positive_prevalence,
    },
    {
        "metric": "prob_min",
        "value": np.min(p),
    },
    {
        "metric": "prob_max",
        "value": np.max(p),
    },
    {
        "metric": "prob_mean",
        "value": np.mean(p),
    },
    {
        "metric": "prob_std",
        "value": np.std(p),
    },
    {
        "metric": "control_prob_mean",
        "value": np.mean(control_probs),
    },
    {
        "metric": "case_prob_mean",
        "value": np.mean(case_probs),
    },
]

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NeuroFusion AI - sMRI Nested CV OOF Prediction Analysis\n"
    )
    f.write("=" * 72 + "\n\n")

    f.write(
        "READ-ONLY ANALYSIS\n"
        "No model, dataset, checkpoint, or CV assignment was modified.\n\n"
    )

    f.write("1. DATASET\n")
    f.write("-" * 40 + "\n")
    f.write(f"Subjects: {len(df)}\n")
    f.write(
        f"Unique subjects: "
        f"{df[subject_col].nunique()}\n"
    )
    f.write(
        f"Duplicate subjects: "
        f"{df[subject_col].duplicated().sum()}\n"
    )
    f.write(
        f"Controls: {np.sum(y == 0)}\n"
    )
    f.write(
        f"Schizophrenia: {np.sum(y == 1)}\n\n"
    )

    f.write("2. ROC ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"OOF ROC-AUC: {roc_auc:.6f}\n\n"
    )

    f.write("3. PRECISION-RECALL ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Average Precision: "
        f"{average_precision:.6f}\n"
    )
    f.write(
        f"Positive prevalence: "
        f"{positive_prevalence:.6f}\n\n"
    )

    f.write("4. PROBABILITY DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Overall minimum: {np.min(p):.6f}\n"
    )
    f.write(
        f"Overall maximum: {np.max(p):.6f}\n"
    )
    f.write(
        f"Overall mean: {np.mean(p):.6f}\n"
    )
    f.write(
        f"Overall std: {np.std(p):.6f}\n"
    )
    f.write(
        f"Overall median: {np.median(p):.6f}\n\n"
    )

    f.write(
        f"Control mean: "
        f"{np.mean(control_probs):.6f}\n"
    )
    f.write(
        f"Control std: "
        f"{np.std(control_probs):.6f}\n"
    )
    f.write(
        f"Case mean: "
        f"{np.mean(case_probs):.6f}\n"
    )
    f.write(
        f"Case std: "
        f"{np.std(case_probs):.6f}\n\n"
    )

    f.write("5. FOLD-WISE ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(
        fold_analysis.to_string(index=False)
    )
    f.write("\n\n")

    f.write("6. THRESHOLD ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(
        threshold_analysis.to_string(index=False)
    )
    f.write("\n\n")

    f.write("OUTPUT FILES\n")
    f.write("-" * 40 + "\n")
    f.write(f"ROC curve: {ROC_FILE}\n")
    f.write(f"PR curve: {PR_FILE}\n")
    f.write(
        f"Probability distribution: {PROB_FILE}\n"
    )
    f.write(
        f"Fold probability distribution: "
        f"{FOLD_PROB_FILE}\n"
    )
    f.write(f"Summary: {SUMMARY_FILE}\n")
    f.write(f"Fold analysis: {FOLD_FILE}\n")


# ============================================================
# FINAL STATUS
# ============================================================

validation_pass = (
    len(df) == 172
    and df[subject_col].nunique() == 172
    and df[subject_col].duplicated().sum() == 0
    and np.isfinite(p).all()
    and np.all((p >= 0) & (p <= 1))
    and set(np.unique(folds)) == {1, 2, 3, 4, 5}
)

print("\n" + "=" * 72)
print(
    f"FINAL STATUS: "
    f"{'PASS' if validation_pass else 'FAIL'}"
)
print("=" * 72)

print("\nOutputs:")
print(REPORT_FILE)
print(SUMMARY_FILE)
print(FOLD_FILE)
print(ROC_FILE)
print(PR_FILE)
print(PROB_FILE)
print(FOLD_PROB_FILE)

print("\nNo training or dataset files were modified.")