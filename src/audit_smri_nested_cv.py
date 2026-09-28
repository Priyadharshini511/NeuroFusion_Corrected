"""
NeuroFusion AI
sMRI Branch - Full 5x4 Nested Cross-Validation Audit

READ-ONLY AUDIT:
- Does not retrain the model
- Does not modify CV assignments
- Does not modify predictions
- Does not modify checkpoints
- Validates OOF subject coverage
- Validates fold coverage
- Computes overall and fold-wise metrics
- Computes sensitivity/specificity/accuracy/balanced accuracy/F1
- Summarizes probability distributions
- Audits legacy/RMS source composition
- Compares with the earlier baseline test result
- Checks subject/fold leakage
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data/smri/cnn_dataset")

CV_DIR = ROOT / "cross_validation"
FULL_CV_DIR = CV_DIR / "training_full_5x4"

OOF_FILE = FULL_CV_DIR / "nested_cv_oof_predictions.csv"
FOLD_METRICS_FILE = FULL_CV_DIR / "nested_cv_fold_metrics.csv"
SUMMARY_FILE = FULL_CV_DIR / "nested_cv_summary.txt"

ASSIGNMENT_FILE = CV_DIR / "cv_assignments.csv"
NESTED_ASSIGNMENT_FILE = CV_DIR / "nested" / "nested_cv_assignments.csv"

BASELINE_PRED_FILE = ROOT / "evaluation" / "test_predictions.csv"
BASELINE_METRICS_FILE = ROOT / "evaluation" / "test_metrics.csv"

OUTPUT_DIR = CV_DIR / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_REPORT = OUTPUT_DIR / "nested_cv_audit_report.txt"
AUDIT_METRICS = OUTPUT_DIR / "nested_cv_audit_metrics.csv"
AUDIT_FOLD_METRICS = OUTPUT_DIR / "nested_cv_audit_fold_metrics.csv"
AUDIT_SOURCE = OUTPUT_DIR / "nested_cv_audit_source_composition.csv"


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def metric_bundle(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    auc = safe_auc(y_true, y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    sensitivity = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    f1 = float(
        f1_score(y_true, y_pred, zero_division=0)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else float("nan")
    )

    return {
        "n": len(y_true),
        "controls": int(np.sum(y_true == 0)),
        "schizophrenia": int(np.sum(y_true == 1)),
        "roc_auc": auc,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prob_min": float(np.min(y_prob)),
        "prob_max": float(np.max(y_prob)),
        "prob_mean": float(np.mean(y_prob)),
        "prob_std": float(np.std(y_prob)),
        "prob_median": float(np.median(y_prob)),
    }


def bootstrap_auc_ci(y_true, y_prob, seed=42, n_boot=2000):
    """
    Subject-level bootstrap 95% CI for OOF ROC-AUC.

    This is descriptive uncertainty estimation, not a replacement
    for the nested CV procedure.
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    auc_values = []

    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)

        y_b = y_true[indices]
        p_b = y_prob[indices]

        if len(np.unique(y_b)) < 2:
            continue

        auc_values.append(roc_auc_score(y_b, p_b))

    if len(auc_values) < 100:
        return float("nan"), float("nan"), len(auc_values)

    lower = float(np.percentile(auc_values, 2.5))
    upper = float(np.percentile(auc_values, 97.5))

    return lower, upper, len(auc_values)


# ============================================================
# START
# ============================================================

print("=" * 72)
print("NeuroFusion AI - FULL 5x4 NESTED CV AUDIT")
print("=" * 72)

for path in [
    OOF_FILE,
    FOLD_METRICS_FILE,
    ASSIGNMENT_FILE,
    NESTED_ASSIGNMENT_FILE,
]:
    require_file(path)

print("\n[1] Loading completed nested-CV outputs...")

oof = pd.read_csv(OOF_FILE)
fold_metrics_existing = pd.read_csv(FOLD_METRICS_FILE)
assignments = pd.read_csv(ASSIGNMENT_FILE)
nested_assignments = pd.read_csv(NESTED_ASSIGNMENT_FILE)

print(f"OOF rows: {len(oof)}")
print(f"Existing fold metric rows: {len(fold_metrics_existing)}")
print(f"Outer assignment rows: {len(assignments)}")
print(f"Nested assignment rows: {len(nested_assignments)}")


# ============================================================
# COLUMN DISCOVERY
# ============================================================

def find_column(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for c in df.columns:
        cl = c.lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c

    if required:
        raise KeyError(
            f"Could not find any of {candidates} in columns: "
            f"{list(df.columns)}"
        )

    return None


subject_col = find_column(
    oof,
    ["subject_id", "subject", "id"]
)

label_col = find_column(
    oof,
    ["label", "y_true", "target"]
)

prob_col = find_column(
    oof,
    ["probability", "prob", "prediction_probability", "y_prob"]
)

fold_col = find_column(
    oof,
    ["outer_fold", "fold"]
)

print("\nDetected OOF columns:")
print(f"  Subject: {subject_col}")
print(f"  Label:   {label_col}")
print(f"  Prob:    {prob_col}")
print(f"  Fold:    {fold_col}")


# ============================================================
# NORMALIZE TYPES
# ============================================================

oof[subject_col] = oof[subject_col].astype(str)
oof[label_col] = pd.to_numeric(oof[label_col], errors="coerce")
oof[prob_col] = pd.to_numeric(oof[prob_col], errors="coerce")
oof[fold_col] = pd.to_numeric(oof[fold_col], errors="coerce")

oof_valid = oof.dropna(
    subset=[
        subject_col,
        label_col,
        prob_col,
        fold_col,
    ]
).copy()

oof_valid[label_col] = oof_valid[label_col].astype(int)
oof_valid[fold_col] = oof_valid[fold_col].astype(int)


# ============================================================
# BASIC OOF VALIDATION
# ============================================================

print("\n[2] OOF subject-level validation...")

n_rows = len(oof_valid)
n_unique = oof_valid[subject_col].nunique()
duplicate_subjects = (
    oof_valid[subject_col]
    .value_counts()
)

duplicate_subjects = duplicate_subjects[
    duplicate_subjects > 1
]

expected_subjects = 172
expected_folds = {1, 2, 3, 4, 5}

oof_subject_set = set(oof_valid[subject_col])

print(f"Rows: {n_rows}")
print(f"Unique subjects: {n_unique}")
print(f"Expected subjects: {expected_subjects}")
print(f"Duplicate subjects: {len(duplicate_subjects)}")
print(
    f"Fold values: "
    f"{sorted(oof_valid[fold_col].unique().tolist())}"
)

# Each subject must occur exactly once in OOF predictions.
subject_once_pass = (
    n_rows == n_unique == expected_subjects
)

fold_values_pass = (
    set(oof_valid[fold_col].unique()) == expected_folds
)

labels_pass = set(oof_valid[label_col].unique()) == {0, 1}

probability_pass = bool(
    np.isfinite(oof_valid[prob_col]).all()
    and
    (oof_valid[prob_col].between(0, 1).all())
)


# ============================================================
# OOF METRICS
# ============================================================

print("\n[3] Overall OOF metrics...")

overall = metric_bundle(
    oof_valid[label_col].values,
    oof_valid[prob_col].values,
)

ci_low, ci_high, ci_boot = bootstrap_auc_ci(
    oof_valid[label_col].values,
    oof_valid[prob_col].values,
)

overall["roc_auc_ci95_low"] = ci_low
overall["roc_auc_ci95_high"] = ci_high
overall["bootstrap_replicates"] = ci_boot

for key, value in overall.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# FOLD-WISE METRICS
# ============================================================

print("\n[4] Fold-wise metrics...")

fold_rows = []

for fold in sorted(oof_valid[fold_col].unique()):

    fold_df = oof_valid[
        oof_valid[fold_col] == fold
    ].copy()

    metrics = metric_bundle(
        fold_df[label_col].values,
        fold_df[prob_col].values,
    )

    metrics["outer_fold"] = int(fold)

    fold_rows.append(metrics)

fold_metrics = pd.DataFrame(fold_rows)

print(
    fold_metrics[
        [
            "outer_fold",
            "n",
            "controls",
            "schizophrenia",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
        ]
    ].to_string(index=False)
)

fold_auc_mean = float(
    fold_metrics["roc_auc"].mean()
)

fold_auc_std = float(
    fold_metrics["roc_auc"].std(ddof=0)
)

print(f"\nFold AUC mean: {fold_auc_mean:.6f}")
print(f"Fold AUC std:  {fold_auc_std:.6f}")


# ============================================================
# SOURCE / LINEAGE AUDIT
# ============================================================

print("\n[5] Source / lineage composition audit...")

source_col = None

for candidate in [
    "source",
    "acquisition_source",
    "lineage",
    "normalization_source",
    "preprocessing_source",
    "data_source",
]:
    if candidate in oof_valid.columns:
        source_col = candidate
        break

# If OOF itself does not contain source, use the master manifest.
manifest_file = ROOT / "smri_subject_manifest_labeled.csv"

if source_col is None and manifest_file.exists():

    manifest = pd.read_csv(manifest_file)

    manifest_subject_col = find_column(
        manifest,
        ["subject_id", "subject", "id"],
    )

    possible_source_cols = [
        c for c in manifest.columns
        if any(
            token in c.lower()
            for token in [
                "source",
                "lineage",
                "acquisition",
            ]
        )
    ]

    if possible_source_cols:

        source_col_manifest = possible_source_cols[0]

        source_map = (
            manifest[
                [
                    manifest_subject_col,
                    source_col_manifest,
                ]
            ]
            .drop_duplicates(
                subset=[manifest_subject_col]
            )
            .rename(
                columns={
                    manifest_subject_col: subject_col,
                    source_col_manifest: "_source",
                }
            )
        )

        oof_valid = oof_valid.merge(
            source_map,
            on=subject_col,
            how="left",
        )

        source_col = "_source"

if source_col is not None:

    source_summary = (
        oof_valid
        .groupby(
            [
                source_col,
                label_col,
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
    )

    print(
        source_summary.to_string(index=False)
    )

    source_summary.to_csv(
        AUDIT_SOURCE,
        index=False,
    )

else:

    source_summary = pd.DataFrame()

    print(
        "Source column unavailable in OOF or labeled manifest."
    )


# ============================================================
# LEAKAGE AUDIT
# ============================================================

print("\n[6] Subject/fold leakage audit...")

leakage_pass = True

# Each OOF subject must belong to one outer fold.
subject_fold_counts = (
    oof_valid
    .groupby(subject_col)[fold_col]
    .nunique()
)

multi_fold_subjects = subject_fold_counts[
    subject_fold_counts > 1
]

if len(multi_fold_subjects) > 0:
    leakage_pass = False

print(
    f"Subjects appearing in >1 outer fold: "
    f"{len(multi_fold_subjects)}"
)

# Compare with official outer assignment.
assignment_subject_col = find_column(
    assignments,
    ["subject_id", "subject", "id"],
)

assignment_fold_col = find_column(
    assignments,
    ["outer_fold", "fold"],
)

official_map = (
    assignments[
        [
            assignment_subject_col,
            assignment_fold_col,
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            assignment_subject_col: subject_col,
            assignment_fold_col: "_official_fold",
        }
    )
)

comparison = oof_valid.merge(
    official_map,
    on=subject_col,
    how="left",
)

fold_mismatch = comparison[
    comparison[fold_col] != comparison["_official_fold"]
]

missing_official_fold = comparison[
    comparison["_official_fold"].isna()
]

if len(fold_mismatch) > 0:
    leakage_pass = False

if len(missing_official_fold) > 0:
    leakage_pass = False

print(
    f"OOF subjects with official-fold mismatch: "
    f"{len(fold_mismatch)}"
)

print(
    f"OOF subjects missing official outer assignment: "
    f"{len(missing_official_fold)}"
)


# ============================================================
# NESTED ASSIGNMENT CONSISTENCY
# ============================================================

print("\n[7] Nested assignment consistency...")

nested_rows = len(nested_assignments)

expected_nested_rows = expected_subjects * 5 * 4

nested_row_count_pass = (
    nested_rows == expected_nested_rows
)

print(
    f"Nested assignment rows: {nested_rows}"
)

print(
    f"Expected rows: {expected_nested_rows}"
)

print(
    f"Nested assignment row-count PASS: "
    f"{nested_row_count_pass}"
)


# ============================================================
# BASELINE COMPARISON
# ============================================================

print("\n[8] Earlier baseline comparison...")

baseline_metrics = {}

if BASELINE_METRICS_FILE.exists():

    baseline_df = pd.read_csv(
        BASELINE_METRICS_FILE
    )

    print(
        baseline_df.to_string(index=False)
    )

    for _, row in baseline_df.iterrows():

        metric_name = str(
            row.iloc[0]
        )

        try:
            metric_value = float(
                row.iloc[1]
            )
            baseline_metrics[
                metric_name
            ] = metric_value
        except Exception:
            pass

else:

    print(
        "Earlier baseline metrics file not found."
    )


# ============================================================
# CURRENT VS BASELINE SUMMARY
# ============================================================

baseline_auc = None

for key, value in baseline_metrics.items():

    if "auc" in key.lower():

        baseline_auc = value
        break

if baseline_auc is not None:

    auc_difference = (
        overall["roc_auc"] - baseline_auc
    )

else:

    auc_difference = float("nan")


# ============================================================
# FINAL AUDIT STATUS
# ============================================================

print("\n[9] Final audit status...")

checks = {
    "OOF rows = 172": n_rows == expected_subjects,
    "Unique subjects = 172": n_unique == expected_subjects,
    "No duplicate OOF subjects": len(duplicate_subjects) == 0,
    "All 5 outer folds present": fold_values_pass,
    "Binary labels valid": labels_pass,
    "Probabilities finite and in [0,1]": probability_pass,
    "Official outer fold consistency": leakage_pass,
    "Nested assignment row count": nested_row_count_pass,
}

for name, passed in checks.items():
    print(
        f"{name}: {'PASS' if passed else 'FAIL'}"
    )

final_pass = all(checks.values())

print(
    f"\nFINAL AUDIT STATUS: "
    f"{'PASS' if final_pass else 'FAIL'}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_rows = []

for key, value in overall.items():

    metrics_rows.append(
        {
            "metric": key,
            "value": value,
        }
    )

metrics_rows.extend(
    [
        {
            "metric": "fold_auc_mean",
            "value": fold_auc_mean,
        },
        {
            "metric": "fold_auc_std",
            "value": fold_auc_std,
        },
        {
            "metric": "baseline_test_auc",
            "value": baseline_auc,
        },
        {
            "metric": "baseline_vs_nested_oof_auc_difference",
            "value": auc_difference,
        },
        {
            "metric": "oof_subjects",
            "value": n_unique,
        },
        {
            "metric": "duplicate_oof_subjects",
            "value": len(duplicate_subjects),
        },
    ]
)

audit_metrics_df = pd.DataFrame(metrics_rows)

audit_metrics_df.to_csv(
    AUDIT_METRICS,
    index=False,
)

fold_metrics.to_csv(
    AUDIT_FOLD_METRICS,
    index=False,
)


# ============================================================
# WRITE REPORT
# ============================================================

with open(
    AUDIT_REPORT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "NeuroFusion AI - Full 5x4 Nested CV Audit\n"
    )
    f.write("=" * 72 + "\n\n")

    f.write(
        "This is a read-only audit of the completed "
        "nested-CV outputs.\n"
    )
    f.write(
        "No model training, checkpoint, prediction, "
        "dataset, or split files were modified.\n\n"
    )

    f.write("1. OOF VALIDATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"OOF rows: {n_rows}\n")
    f.write(f"Unique subjects: {n_unique}\n")
    f.write(f"Expected subjects: {expected_subjects}\n")
    f.write(
        f"Duplicate subjects: "
        f"{len(duplicate_subjects)}\n"
    )
    f.write(
        f"Outer folds: "
        f"{sorted(oof_valid[fold_col].unique())}\n"
    )
    f.write("\n")

    f.write("2. OVERALL OOF METRICS\n")
    f.write("-" * 40 + "\n")

    for key, value in overall.items():

        if isinstance(value, float):
            f.write(
                f"{key}: {value:.6f}\n"
            )
        else:
            f.write(
                f"{key}: {value}\n"
            )

    f.write("\n")

    f.write("3. FOLD-WISE METRICS\n")
    f.write("-" * 40 + "\n")

    f.write(
        fold_metrics.to_string(index=False)
    )
    f.write("\n\n")

    f.write(
        f"Fold AUC mean: "
        f"{fold_auc_mean:.6f}\n"
    )
    f.write(
        f"Fold AUC std: "
        f"{fold_auc_std:.6f}\n\n"
    )

    f.write("4. LEAKAGE AUDIT\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Multi-fold OOF subjects: "
        f"{len(multi_fold_subjects)}\n"
    )
    f.write(
        f"Official fold mismatches: "
        f"{len(fold_mismatch)}\n"
    )
    f.write(
        f"Missing official assignments: "
        f"{len(missing_official_fold)}\n"
    )
    f.write("\n")

    f.write("5. NESTED ASSIGNMENT\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Rows: {nested_rows}\n"
    )
    f.write(
        f"Expected: {expected_nested_rows}\n"
    )
    f.write(
        f"Row-count PASS: "
        f"{nested_row_count_pass}\n\n"
    )

    f.write("6. SOURCE COMPOSITION\n")
    f.write("-" * 40 + "\n")

    if not source_summary.empty:
        f.write(
            source_summary.to_string(index=False)
        )
    else:
        f.write(
            "Source information unavailable."
        )

    f.write("\n\n")

    f.write("7. BASELINE COMPARISON\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Earlier baseline test AUC: "
        f"{baseline_auc}\n"
    )
    f.write(
        f"Nested OOF AUC: "
        f"{overall['roc_auc']:.6f}\n"
    )
    f.write(
        f"Nested - baseline AUC: "
        f"{auc_difference}\n\n"
    )

    f.write("8. AUDIT CHECKS\n")
    f.write("-" * 40 + "\n")

    for name, passed in checks.items():
        f.write(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}\n"
        )

    f.write("\n")

    f.write(
        "FINAL AUDIT STATUS: "
        f"{'PASS' if final_pass else 'FAIL'}\n"
    )


# ============================================================
# DONE
# ============================================================

print("\nAudit outputs:")
print(f"  {AUDIT_REPORT}")
print(f"  {AUDIT_METRICS}")
print(f"  {AUDIT_FOLD_METRICS}")
print(f"  {AUDIT_SOURCE}")

print("\nNo training or dataset files were modified.")
print("=" * 72)