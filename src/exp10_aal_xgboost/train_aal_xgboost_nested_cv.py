import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ROOT = PROJECT_ROOT / "processed_data" / "smri" / "cnn_dataset"

MANIFEST_PATH = (
    PROCESSED_ROOT.parent
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

AAL_ROOT = (
    PROCESSED_ROOT.parent.parent
    / "features"
    / "aal"
)

# Exact nested-CV assignments already created for Experiment 06.
NESTED_ASSIGNMENTS = (
    PROCESSED_ROOT
    / "experiments"
    / "exp06_rms_cnn_aal"
    / "cross_validation"
    / "nested_cv_assignments.csv"
)

OUTPUT_ROOT = (
    PROCESSED_ROOT
    / "experiments"
    / "exp10_aal_xgboost"
    / "training_nested_cv"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ================================================================
# REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ================================================================
# EXPERIMENT SETTINGS
# ================================================================

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

K_VALUES = ["all", 20, 40, 60, 80]

# Controlled XGBoost search space.
XGB_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}

# RMS-only class ratio.
N_CONTROLS = 40
N_CASES = 79

SCALE_POS_WEIGHT = N_CONTROLS / N_CASES


# ================================================================
# HELPERS
# ================================================================

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_aal_matrix(manifest):
    """
    Load one AAL feature vector per subject.
    """

    feature_rows = []
    subject_ids = []

    for _, row in manifest.iterrows():

        subject_id = str(row["subject_id"])

        aal_path = Path(str(row["aal_csv"]))

        if not aal_path.is_absolute():
            aal_path = PROJECT_ROOT / aal_path

        if not aal_path.exists():
            raise FileNotFoundError(
                f"AAL file not found for {subject_id}: {aal_path}"
            )

        aal_df = pd.read_csv(aal_path)

        if "subject_id" in aal_df.columns:
            feature_cols = [
                c for c in aal_df.columns
                if c != "subject_id"
            ]
        else:
            feature_cols = list(aal_df.columns)

        # Exclude metadata if present.
        metadata_cols = {
            "acquisition_type",
            "volume",
        }

        feature_cols = [
            c for c in feature_cols
            if c not in metadata_cols
        ]

        if len(feature_cols) != 116:
            raise ValueError(
                f"{subject_id}: expected 116 AAL features, "
                f"found {len(feature_cols)}"
            )

        values = aal_df[feature_cols].iloc[0].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{subject_id}: non-finite AAL values detected"
            )

        feature_rows.append(values)
        subject_ids.append(subject_id)

    X = np.vstack(feature_rows)

    return X, np.array(subject_ids)


def get_role_ids(assignments, outer_fold, inner_fold, role):
    mask = (
        (assignments["outer_fold"] == outer_fold)
        & (assignments["inner_fold"] == inner_fold)
        & (assignments["role"] == role)
    )

    return assignments.loc[mask, "subject_id"].astype(str).tolist()


def make_pipeline(k_value, params):
    """
    Build the complete leakage-safe pipeline.
    Scaling and feature selection are fitted only on the
    training data by sklearn Pipeline.
    """

    if k_value == "all":
        selector_k = "all"
    else:
        selector_k = int(k_value)

    model = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=int(params["min_child_weight"]),
        scale_pos_weight=float(SCALE_POS_WEIGHT),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(
                score_func=f_classif,
                k=selector_k,
            )),
            ("model", model),
        ]
    )

    return pipeline


def generate_param_combinations():
    """
    Generate the controlled XGBoost hyperparameter grid.
    """

    keys = list(XGB_GRID.keys())

    values = [
        XGB_GRID[key]
        for key in keys
    ]

    combinations = []

    import itertools

    for combo in itertools.product(*values):

        params = dict(zip(keys, combo))
        combinations.append(params)

    return combinations


def inner_cv_score(
    X_outer_train,
    y_outer_train,
    outer_train_subject_ids,
    assignments,
    outer_fold,
    inner_fold,
    k_value,
    params,
):
    """
    Evaluate one hyperparameter configuration using only
    INNER_TRAIN / INNER_VALIDATION subjects belonging to the
    current outer-training partition.

    The outer test subjects are never accessed.
    """

    train_ids = get_role_ids(
        assignments,
        outer_fold,
        inner_fold,
        "INNER_TRAIN",
    )

    val_ids = get_role_ids(
        assignments,
        outer_fold,
        inner_fold,
        "INNER_VALIDATION",
    )

    outer_train_subject_ids = [
        str(sid)
        for sid in outer_train_subject_ids
    ]

    outer_train_index = {
        sid: idx
        for idx, sid in enumerate(
            outer_train_subject_ids
        )
    }

    missing_train = [
        sid
        for sid in train_ids
        if sid not in outer_train_index
    ]

    missing_val = [
        sid
        for sid in val_ids
        if sid not in outer_train_index
    ]

    if missing_train:
        raise RuntimeError(
            f"INNER_TRAIN subjects outside outer training "
            f"set in outer fold {outer_fold}, "
            f"inner fold {inner_fold}: {missing_train}"
        )

    if missing_val:
        raise RuntimeError(
            f"INNER_VALIDATION subjects outside outer training "
            f"set in outer fold {outer_fold}, "
            f"inner fold {inner_fold}: {missing_val}"
        )

    train_idx = [
        outer_train_index[sid]
        for sid in train_ids
    ]

    val_idx = [
        outer_train_index[sid]
        for sid in val_ids
    ]

    X_tr = X_outer_train[train_idx]
    y_tr = y_outer_train[train_idx]

    X_val = X_outer_train[val_idx]
    y_val = y_outer_train[val_idx]

    if len(np.unique(y_tr)) < 2:
        return np.nan

    if len(np.unique(y_val)) < 2:
        return np.nan

    pipeline = make_pipeline(
        k_value,
        params,
    )

    pipeline.fit(
        X_tr,
        y_tr,
    )

    probabilities = (
        pipeline
        .predict_proba(X_val)[:, 1]
    )

    return roc_auc_score(
        y_val,
        probabilities,
    )

def select_hyperparameters(
    X_outer_train,
    y_outer_train,
    outer_train_subject_ids,
    assignments,
    outer_fold,
):
    """
    Select k + XGBoost parameters using the four inner folds.

    Only the current outer-training subjects are available.
    """

    best_config = None
    best_score = -np.inf

    param_combinations = generate_param_combinations()

    for k_value in K_VALUES:

        for params in param_combinations:

            inner_scores = []

            for inner_fold in range(
                1,
                N_INNER_FOLDS + 1,
            ):

                score = inner_cv_score(
                    X_outer_train,
                    y_outer_train,
                    outer_train_subject_ids,
                    assignments,
                    outer_fold,
                    inner_fold,
                    k_value,
                    params,
                )

                if np.isfinite(score):
                    inner_scores.append(score)

            if not inner_scores:
                continue

            mean_score = float(
                np.mean(inner_scores)
            )

            if mean_score > best_score:

                best_score = mean_score

                best_config = {
                    "k": k_value,
                    **params,
                }

    if best_config is None:
        raise RuntimeError(
            f"No valid inner-CV configuration found "
            f"for outer fold {outer_fold}"
        )

    return best_config, best_score


# ================================================================
# LOAD MANIFEST
# ================================================================

print("=" * 70)
print("EXPERIMENT 10 — AAL XGBOOST")
print("LEAKAGE-SAFE 5×4 NESTED CV")
print("=" * 70)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )

if not NESTED_ASSIGNMENTS.exists():
    raise FileNotFoundError(
        f"Nested assignments not found:\n{NESTED_ASSIGNMENTS}"
    )


manifest = pd.read_csv(MANIFEST_PATH)

manifest["subject_id"] = (
    manifest["subject_id"]
    .astype(str)
)

# RMS-only cohort.
manifest = manifest[
    manifest["acquisition_source"].astype(str).str.upper()
    == "RMS"
].copy()

manifest = manifest.sort_values(
    "subject_id"
).reset_index(drop=True)


# ================================================================
# COHORT VALIDATION
# ================================================================

if len(manifest) != 119:
    raise ValueError(
        f"Expected 119 RMS subjects, found {len(manifest)}"
    )

if manifest["subject_id"].duplicated().any():
    raise ValueError(
        "Duplicate RMS subjects detected in manifest"
    )

label_counts = manifest["label"].value_counts().to_dict()

if int(label_counts.get(0, 0)) != N_CONTROLS:
    raise ValueError(
        f"Expected 40 controls, found {label_counts.get(0, 0)}"
    )

if int(label_counts.get(1, 0)) != N_CASES:
    raise ValueError(
        f"Expected 79 schizophrenia subjects, "
        f"found {label_counts.get(1, 0)}"
    )

print(f"RMS subjects: {len(manifest)}")
print(f"Controls: {label_counts.get(0, 0)}")
print(f"Schizophrenia: {label_counts.get(1, 0)}")


# ================================================================
# LOAD NESTED ASSIGNMENTS
# ================================================================

assignments = pd.read_csv(
    NESTED_ASSIGNMENTS
)

assignments["subject_id"] = (
    assignments["subject_id"]
    .astype(str)
)

assignment_subjects = set(
    assignments["subject_id"].unique()
)

manifest_subjects = set(
    manifest["subject_id"]
)

if assignment_subjects != manifest_subjects:
    missing = manifest_subjects - assignment_subjects
    extra = assignment_subjects - manifest_subjects

    raise ValueError(
        f"Assignment mismatch. Missing={missing}, Extra={extra}"
    )


# ================================================================
# LOAD LABELS
# ================================================================

labels_by_subject = dict(
    zip(
        manifest["subject_id"],
        manifest["label"].astype(int),
    )
)

y = np.array(
    [
        labels_by_subject[sid]
        for sid in manifest["subject_id"]
    ],
    dtype=np.int64,
)


# ================================================================
# LOAD AAL
# ================================================================

print()
print("Loading AAL features...")

X, subject_ids = load_aal_matrix(
    manifest
)

print(f"AAL matrix: {X.shape}")
print(f"AAL finite: {np.isfinite(X).all()}")

if X.shape != (119, 116):
    raise ValueError(
        f"Unexpected AAL matrix shape: {X.shape}"
    )


subject_to_index = {
    sid: idx
    for idx, sid in enumerate(subject_ids)
}


# ================================================================
# OUTER NESTED CV
# ================================================================

oof_records = []
fold_summaries = []

for outer_fold in range(1, N_OUTER_FOLDS + 1):

    print()
    print("=" * 70)
    print(f"OUTER FOLD {outer_fold}/{N_OUTER_FOLDS}")
    print("=" * 70)

    set_seed(SEED + outer_fold)

    outer_train_ids = set(
        assignments.loc[
            (assignments["outer_fold"] == outer_fold)
            & (
                assignments["role"]
                != "OUTER_TEST"
            ),
            "subject_id",
        ].astype(str)
    )

    outer_test_ids = set(
        assignments.loc[
            (assignments["outer_fold"] == outer_fold)
            & (
                assignments["role"]
                == "OUTER_TEST"
            ),
            "subject_id",
        ].astype(str)
    )

    # Every subject in the outer fold must belong to exactly
    # one of train/test.
    if outer_train_ids & outer_test_ids:
        raise RuntimeError(
            f"Outer train/test overlap in fold {outer_fold}"
        )

    if len(outer_train_ids) + len(outer_test_ids) != 119:
        raise RuntimeError(
            f"Outer partition does not contain all 119 subjects "
            f"in fold {outer_fold}"
        )

    outer_train_idx = [
        subject_to_index[sid]
        for sid in sorted(outer_train_ids)
    ]

    outer_test_idx = [
        subject_to_index[sid]
        for sid in sorted(outer_test_ids)
    ]

    X_outer_train = X[outer_train_idx]
    y_outer_train = y[outer_train_idx]

    X_outer_test = X[outer_test_idx]
    y_outer_test = y[outer_test_idx]

    # ------------------------------------------------------------
    # INNER HYPERPARAMETER SELECTION
    # ------------------------------------------------------------

    best_config, best_inner_auc = select_hyperparameters(
      X_outer_train,
      y_outer_train,
      sorted(outer_train_ids),
      assignments,
      outer_fold,
    )

    print(f"Selected k: {best_config['k']}")
    print(
        f"Selected n_estimators: "
        f"{best_config['n_estimators']}"
    )
    print(
        f"Selected max_depth: "
        f"{best_config['max_depth']}"
    )
    print(
        f"Selected learning_rate: "
        f"{best_config['learning_rate']}"
    )
    print(
        f"Selected subsample: "
        f"{best_config['subsample']}"
    )
    print(
        f"Selected colsample_bytree: "
        f"{best_config['colsample_bytree']}"
    )
    print(
        f"Selected min_child_weight: "
        f"{best_config['min_child_weight']}"
    )
    print(
        f"Best inner-CV AUC: "
        f"{best_inner_auc:.6f}"
    )

    # ------------------------------------------------------------
    # FIT FINAL OUTER-TRAIN MODEL
    # ------------------------------------------------------------

    final_pipeline = make_pipeline(
        best_config["k"],
        {
            key: value
            for key, value in best_config.items()
            if key != "k"
        },
    )

    final_pipeline.fit(
        X_outer_train,
        y_outer_train,
    )

    # ------------------------------------------------------------
    # OUTER TEST PREDICTION
    # ------------------------------------------------------------

    outer_probabilities = (
        final_pipeline
        .predict_proba(X_outer_test)[:, 1]
    )

    outer_auc = roc_auc_score(
        y_outer_test,
        outer_probabilities,
    )

    print(
        f"Outer test subjects: "
        f"{len(outer_test_ids)}"
    )

    print(
        f"Outer test AUC: "
        f"{outer_auc:.6f}"
    )

    # ------------------------------------------------------------
    # SAVE OOF PREDICTIONS
    # ------------------------------------------------------------

    for sid, true_label, probability in zip(
        sorted(outer_test_ids),
        y_outer_test,
        outer_probabilities,
    ):

        oof_records.append(
            {
                "subject_id": sid,
                "label": int(true_label),
                "probability": float(probability),
                "outer_fold": int(outer_fold),
            }
        )

    fold_summaries.append(
        {
            "outer_fold": int(outer_fold),
            "outer_test_subjects": int(
                len(outer_test_ids)
            ),
            "best_inner_auc": float(
                best_inner_auc
            ),
            "k": best_config["k"],
            "n_estimators": int(
                best_config["n_estimators"]
            ),
            "max_depth": int(
                best_config["max_depth"]
            ),
            "learning_rate": float(
                best_config["learning_rate"]
            ),
            "subsample": float(
                best_config["subsample"]
            ),
            "colsample_bytree": float(
                best_config["colsample_bytree"]
            ),
            "min_child_weight": int(
                best_config["min_child_weight"]
            ),
            "outer_auc": float(
                outer_auc
            ),
        }
    )


# ================================================================
# OOF INTEGRITY
# ================================================================

oof_df = pd.DataFrame(oof_records)

oof_path = OUTPUT_ROOT / "oof_predictions.csv"

oof_df.to_csv(
    oof_path,
    index=False,
)

print()
print("=" * 70)
print("OOF INTEGRITY")
print("=" * 70)

oof_rows = len(oof_df)
unique_subjects = oof_df["subject_id"].nunique()
duplicate_subjects = (
    oof_df["subject_id"]
    .duplicated()
    .sum()
)

print(f"OOF rows: {oof_rows}")
print(f"Unique subjects: {unique_subjects}")
print(f"Duplicate subjects: {duplicate_subjects}")

if oof_rows != 119:
    raise RuntimeError(
        f"Expected 119 OOF rows, found {oof_rows}"
    )

if unique_subjects != 119:
    raise RuntimeError(
        f"Expected 119 unique OOF subjects, "
        f"found {unique_subjects}"
    )

if duplicate_subjects != 0:
    raise RuntimeError(
        "Duplicate OOF subjects detected"
    )

print("OOF integrity: PASS")


# ================================================================
# FINAL METRICS
# ================================================================

oof_y = oof_df["label"].to_numpy(
    dtype=int
)

oof_prob = oof_df["probability"].to_numpy(
    dtype=float
)

oof_pred = (
    oof_prob >= 0.5
).astype(int)

outer_aucs = [
    row["outer_auc"]
    for row in fold_summaries
]

mean_fold_auc = float(
    np.mean(outer_aucs)
)

fold_auc_sd = float(
    np.std(
        outer_aucs,
        ddof=0,
    )
)

oof_auc = float(
    roc_auc_score(
        oof_y,
        oof_prob,
    )
)

accuracy = float(
    accuracy_score(
        oof_y,
        oof_pred,
    )
)

balanced_accuracy = float(
    balanced_accuracy_score(
        oof_y,
        oof_pred,
    )
)

precision = float(
    precision_score(
        oof_y,
        oof_pred,
        zero_division=0,
    )
)

sensitivity = float(
    recall_score(
        oof_y,
        oof_pred,
        zero_division=0,
    )
)

specificity = float(
    recall_score(
        1 - oof_y,
        1 - oof_pred,
        zero_division=0,
    )
)

f1 = float(
    f1_score(
        oof_y,
        oof_pred,
        zero_division=0,
    )
)


# ================================================================
# SAVE REPORTS
# ================================================================

fold_df = pd.DataFrame(
    fold_summaries
)

fold_df.to_csv(
    OUTPUT_ROOT / "outer_fold_results.csv",
    index=False,
)

summary = pd.DataFrame(
    [
        {
            "experiment": "EXP10_AAL_XGBOOST",
            "n_subjects": 119,
            "controls": 40,
            "schizophrenia": 79,
            "n_aal_features": 116,
            "mean_fold_auc": mean_fold_auc,
            "fold_auc_population_sd": fold_auc_sd,
            "oof_roc_auc": oof_auc,
            "oof_accuracy": accuracy,
            "oof_balanced_accuracy": balanced_accuracy,
            "oof_precision": precision,
            "oof_sensitivity": sensitivity,
            "oof_specificity": specificity,
            "oof_f1": f1,
            "oof_rows": oof_rows,
            "unique_oof_subjects": unique_subjects,
            "duplicate_oof_subjects": duplicate_subjects,
            "integrity": "PASS",
        }
    ]
)

summary.to_csv(
    OUTPUT_ROOT / "experiment_summary.csv",
    index=False,
)

report_path = (
    OUTPUT_ROOT
    / "experiment_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "EXPERIMENT 10 — AAL XGBOOST\n"
    )
    f.write(
        "LEAKAGE-SAFE 5×4 NESTED CV\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        f"RMS subjects: 119\n"
    )
    f.write(
        f"Controls: 40\n"
    )
    f.write(
        f"Schizophrenia: 79\n"
    )
    f.write(
        f"AAL features: 116\n\n"
    )

    f.write(
        "Outer fold AUCs:\n"
    )

    for i, auc in enumerate(
        outer_aucs,
        start=1,
    ):
        f.write(
            f"Fold {i}: {auc:.6f}\n"
        )

    f.write("\n")

    f.write(
        f"Mean fold AUC: "
        f"{mean_fold_auc:.6f}\n"
    )

    f.write(
        f"Fold AUC population SD: "
        f"{fold_auc_sd:.6f}\n"
    )

    f.write(
        f"OOF ROC-AUC: "
        f"{oof_auc:.6f}\n"
    )

    f.write(
        f"OOF Accuracy: "
        f"{accuracy:.6f}\n"
    )

    f.write(
        f"OOF Balanced Accuracy: "
        f"{balanced_accuracy:.6f}\n"
    )

    f.write(
        f"OOF Precision: "
        f"{precision:.6f}\n"
    )

    f.write(
        f"OOF Sensitivity: "
        f"{sensitivity:.6f}\n"
    )

    f.write(
        f"OOF Specificity: "
        f"{specificity:.6f}\n"
    )

    f.write(
        f"OOF F1: "
        f"{f1:.6f}\n"
    )

    f.write("\n")

    f.write(
        f"OOF rows: {oof_rows}\n"
    )

    f.write(
        f"Unique OOF subjects: "
        f"{unique_subjects}\n"
    )

    f.write(
        f"Duplicate OOF subjects: "
        f"{duplicate_subjects}\n"
    )

    f.write(
        "OOF integrity: PASS\n"
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 70)
print("EXPERIMENT 10 COMPLETE")
print(
    f"Outer fold AUCs: "
    f"{[round(x, 6) for x in outer_aucs]}"
)
print(
    f"Mean fold AUC: "
    f"{mean_fold_auc:.6f}"
)
print(
    f"Fold AUC population SD: "
    f"{fold_auc_sd:.6f}"
)
print(
    f"OOF ROC-AUC: "
    f"{oof_auc:.6f}"
)
print(
    f"OOF Accuracy: "
    f"{accuracy:.6f}"
)
print(
    f"OOF Balanced Accuracy: "
    f"{balanced_accuracy:.6f}"
)
print(
    f"OOF Precision: "
    f"{precision:.6f}"
)
print(
    f"OOF Sensitivity: "
    f"{sensitivity:.6f}"
)
print(
    f"OOF Specificity: "
    f"{specificity:.6f}"
)
print(
    f"OOF F1: "
    f"{f1:.6f}"
)
print(f"OOF rows: {oof_rows}")
print(
    f"Unique OOF subjects: "
    f"{unique_subjects}"
)
print(
    f"Results: {OUTPUT_ROOT}"
)
print(
    "FINAL STATUS: EXPERIMENT 10 NESTED CV PASS"
)
print("=" * 70)