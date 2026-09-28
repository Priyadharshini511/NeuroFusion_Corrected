"""
NeuroFusion AI - Exp13
VBM + Linear SVM
Nested 5x4 Cross-Validation

Scientific question:
Can VBM gray-matter morphology provide stable discriminative information
for schizophrenia classification in the validated 119-subject RMS cohort?

Important:
- Uses canonical 172-subject nested CV assignments.
- Filters canonical assignments to the 119 RMS subjects.
- Does NOT modify existing CV assignments.
- Uses final SPM smoothed modulated GM maps:
      s8mwc1<SUBJECT_ID>_rms_n4.nii
- Feature selection is performed inside each inner-CV training split.
- Scaling is performed inside the sklearn pipeline.
- Linear SVM hyperparameter C is selected using inner CV.
- Outer test subjects are never used for model selection.
- Primary metric: OOF ROC-AUC.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "cross_validation"
    / "nested"
    / "nested_cv_assignments.csv"
)

VBM_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "vbm_spm"
)

OUTPUT_ROOT = (
    VBM_ROOT
    / "experiments"
    / "exp13_vbm_linear_svm"
)

OOF_DIR = OUTPUT_ROOT / "oof"
FOLD_DIR = OUTPUT_ROOT / "fold_results"
MODEL_DIR = OUTPUT_ROOT / "models"
LOG_DIR = OUTPUT_ROOT / "logs"

N_OUTER = 5
N_INNER = 4

EXPECTED_RMS_SUBJECTS = 119
EXPECTED_CONTROLS = 40
EXPECTED_SCHIZOPHRENIA = 79

EXPECTED_X = 121
EXPECTED_Y = 145
EXPECTED_Z = 121

N_FEATURES = 5000

C_GRID = [0.01, 0.1, 1.0, 10.0]

RANDOM_STATE = 42

# Values below this are treated as background when constructing
# the unsupervised spatial support mask.
NONZERO_EPS = 1e-8


# ============================================================
# UTILITIES
# ============================================================

def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def ensure_dirs() -> None:
    for p in [
        OUTPUT_ROOT,
        OOF_DIR,
        FOLD_DIR,
        MODEL_DIR,
        LOG_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def find_vbm_file(subject_id: str) -> Path:
    subject_dir = VBM_ROOT / subject_id / "smooth"

    expected = (
        subject_dir
        / f"s8mwc1{subject_id}_rms_n4.nii"
    )

    if expected.exists():
        return expected

    # Defensive fallback: search only inside this subject's smooth folder.
    candidates = sorted(
        subject_dir.glob("s8mwc1*_rms_n4.nii")
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No final smoothed VBM found for {subject_id}: "
            f"{expected}"
        )

    raise RuntimeError(
        f"Multiple VBM files found for {subject_id}: {candidates}"
    )


# ============================================================
# ASSIGNMENTS
# ============================================================

def load_rms_assignments() -> pd.DataFrame:
    log(f"Loading canonical assignments: {ASSIGNMENTS_PATH}")

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical assignment file not found:\n"
            f"{ASSIGNMENTS_PATH}"
        )

    df = pd.read_csv(ASSIGNMENTS_PATH)

    required = {
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "outer_fold",
        "inner_fold",
        "role",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Assignment file missing columns: {sorted(missing)}"
        )

    canonical_rows = len(df)

    rms = df[
        df["acquisition_source"]
        .astype(str)
        .str.upper()
        .eq("RMS")
    ].copy()

    unique_ids = sorted(
        rms["subject_id"].astype(str).unique()
    )

    log(f"Canonical assignment rows: {canonical_rows}")
    log(f"Canonical RMS rows: {len(rms)}")
    log(f"Unique RMS subjects: {len(unique_ids)}")

    if canonical_rows != 3440:
        raise ValueError(
            f"Unexpected canonical assignment size: "
            f"{canonical_rows}; expected 3440."
        )

    if len(rms) != EXPECTED_RMS_SUBJECTS * N_OUTER * N_INNER:
        raise ValueError(
            f"Unexpected RMS assignment rows: {len(rms)}; "
            f"expected "
            f"{EXPECTED_RMS_SUBJECTS * N_OUTER * N_INNER}."
        )

    if len(unique_ids) != EXPECTED_RMS_SUBJECTS:
        raise ValueError(
            f"Unexpected RMS subject count: {len(unique_ids)}; "
            f"expected {EXPECTED_RMS_SUBJECTS}."
        )

    label_counts = (
        rms.drop_duplicates("subject_id")
        .groupby("label")
        .size()
        .to_dict()
    )

    log(f"RMS subject label counts: {label_counts}")

    if label_counts.get(0, 0) != EXPECTED_CONTROLS:
        raise ValueError(
            f"Expected 40 controls; "
            f"found {label_counts.get(0, 0)}"
        )

    if label_counts.get(1, 0) != EXPECTED_SCHIZOPHRENIA:
        raise ValueError(
            f"Expected 79 schizophrenia subjects; "
            f"found {label_counts.get(1, 0)}"
        )

    # Each subject should have exactly 20 assignment rows.
    counts = rms.groupby("subject_id").size()

    bad_counts = counts[counts != N_OUTER * N_INNER]

    if len(bad_counts) > 0:
        raise ValueError(
            "Some RMS subjects do not have exactly "
            "20 nested-CV assignment rows:\n"
            f"{bad_counts.to_string()}"
        )

    # Validate outer folds.
    outer_values = sorted(
        rms["outer_fold"].astype(int).unique()
    )

    if outer_values != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Unexpected outer folds: {outer_values}"
        )

    return rms


def build_subject_labels(
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    subject_df = (
        assignments[
            ["subject_id", "label"]
        ]
        .drop_duplicates()
        .copy()
    )

    if len(subject_df) != EXPECTED_RMS_SUBJECTS:
        raise ValueError(
            "Subject-level label table does not contain "
            f"{EXPECTED_RMS_SUBJECTS} subjects."
        )

    duplicate_ids = (
        subject_df["subject_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids:
        raise ValueError(
            f"Duplicate subject IDs detected: {duplicate_ids}"
        )

    return subject_df.sort_values("subject_id").reset_index(drop=True)


# ============================================================
# VBM LOADING
# ============================================================

def validate_vbm_image(
    img: nib.Nifti1Image,
    subject_id: str,
) -> np.ndarray:

    shape = tuple(img.shape[:3])

    if shape != (
        EXPECTED_X,
        EXPECTED_Y,
        EXPECTED_Z,
    ):
        raise ValueError(
            f"{subject_id}: unexpected VBM dimensions "
            f"{shape}; expected "
            f"{(EXPECTED_X, EXPECTED_Y, EXPECTED_Z)}"
        )

    data = np.asarray(
        img.dataobj,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"{subject_id}: VBM contains non-finite values."
        )

    return data


def load_all_vbm(
    subject_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    subject_ids = subject_df["subject_id"].astype(str).tolist()
    labels = subject_df["label"].astype(int).to_numpy()

    n_subjects = len(subject_ids)
    n_voxels = EXPECTED_X * EXPECTED_Y * EXPECTED_Z

    log(
        f"Loading {n_subjects} VBM maps "
        f"({n_voxels:,} voxels/map)..."
    )

    X = np.empty(
        (n_subjects, n_voxels),
        dtype=np.float32,
    )

    affine_reference = None
    header_reference = None

    for i, subject_id in enumerate(subject_ids):

        path = find_vbm_file(subject_id)

        log(
            f"Loading [{i + 1:03d}/{n_subjects:03d}] "
            f"{subject_id}"
        )

        img = nib.load(str(path))

        data = validate_vbm_image(
            img,
            subject_id,
        )

        if affine_reference is None:
            affine_reference = img.affine.copy()
            header_reference = img.header.copy()
        else:
            if not np.allclose(
                affine_reference,
                img.affine,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{subject_id}: affine differs from "
                    "the first VBM image."
                )

        X[i] = data.reshape(-1)

    log(
        f"VBM matrix loaded: {X.shape}, "
        f"dtype={X.dtype}"
    )

    return X, labels, subject_ids


# ============================================================
# UNSUPERVISED VBM SPATIAL SUPPORT
# ============================================================

def build_spatial_support_mask(
    X: np.ndarray,
) -> np.ndarray:

    """
    Build a spatial support mask using only image values,
    not diagnosis labels.

    A voxel is retained if it is non-background in at least
    50% of the supplied subjects.

    This is intentionally label-free.
    """

    log("Building label-free VBM spatial support mask...")

    nonzero_fraction = np.mean(
        np.abs(X) > NONZERO_EPS,
        axis=0,
    )

    mask = nonzero_fraction >= 0.50

    n = int(mask.sum())

    log(
        f"Spatial support voxels: {n:,} / {X.shape[1]:,}"
    )

    if n < N_FEATURES:
        raise ValueError(
            f"Spatial support contains only {n} voxels, "
            f"less than required k={N_FEATURES}."
        )

    return mask


# ============================================================
# NESTED CV SPLITS
# ============================================================

def get_outer_subjects(
    assignments: pd.DataFrame,
    outer_fold: int,
) -> tuple[list[str], list[str]]:

    fold_df = assignments[
        assignments["outer_fold"].astype(int) == outer_fold
    ].copy()

    # The canonical assignment file uses OUTER_TEST.
    test_ids = sorted(
        fold_df.loc[
            fold_df["role"].astype(str).str.upper()
            == "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
        .unique()
    )

    all_ids = sorted(
        assignments["subject_id"]
        .astype(str)
        .unique()
    )

    train_ids = [
        sid for sid in all_ids
        if sid not in set(test_ids)
    ]

    if len(train_ids) + len(test_ids) != EXPECTED_RMS_SUBJECTS:
        raise ValueError(
            f"Outer fold {outer_fold}: "
            f"train={len(train_ids)}, "
            f"test={len(test_ids)}."
        )

    if len(test_ids) == 0:
        raise ValueError(
            f"Outer fold {outer_fold}: "
            "no OUTER_TEST subjects found."
        )

    return train_ids, test_ids


def make_inner_predefined_split(
    assignments: pd.DataFrame,
    outer_fold: int,
    train_ids: list[str],
) -> np.ndarray:

    """
    Return an inner-CV test_fold vector.

    - -1 = used as training data in every inner split
    - 0..3 = validation fold

    The assignment file is the sole source of inner folds.
    """

    outer_df = assignments[
        assignments["outer_fold"].astype(int)
        == outer_fold
    ].copy()

    # For this outer fold, each subject should have one role
    # indicating train/validation for each inner fold.

    rows = []

    for sid in train_ids:

        sid_df = outer_df[
            outer_df["subject_id"].astype(str)
            == sid
        ].copy()

        if len(sid_df) != N_INNER:
            raise ValueError(
                f"Outer {outer_fold}, subject {sid}: "
                f"expected {N_INNER} inner rows, "
                f"found {len(sid_df)}."
            )

        for inner_fold in range(1, N_INNER + 1):

            inner_row = sid_df[
                sid_df["inner_fold"].astype(int)
                == inner_fold
            ]

            if len(inner_row) != 1:
                raise ValueError(
                    f"Outer {outer_fold}, subject {sid}, "
                    f"inner {inner_fold}: expected exactly "
                    "one assignment row."
                )

            rows.append(
                {
                    "subject_id": sid,
                    "inner_fold": inner_fold,
                    "role": str(
                        inner_row.iloc[0]["role"]
                    ).lower(),
                }
            )

    # For each inner fold, construct the validation mask
    # across train_ids.

    # The sklearn PredefinedSplit vector corresponds to
    # the X_train ordering. Each subject needs one final
    # fold assignment. We derive it from the rows where the
    # role is validation.

    subject_to_validation_fold = {}

    for sid in train_ids:

        sid_rows = [
            r for r in rows
            if r["subject_id"] == sid
        ]

        validation_rows = [
            r for r in sid_rows
            if r["role"].upper() == "INNER_VALIDATION"
        ]

        if len(validation_rows) != 1:
            raise ValueError(
                f"Outer {outer_fold}, subject {sid}: "
                f"expected exactly one validation role; "
                f"found {len(validation_rows)}."
            )

        subject_to_validation_fold[sid] = (
            validation_rows[0]["inner_fold"] - 1
        )

    split_vector = np.array(
        [
            subject_to_validation_fold[sid]
            for sid in train_ids
        ],
        dtype=int,
    )

    counts = np.bincount(
        split_vector,
        minlength=N_INNER,
    )

    if np.any(counts == 0):
        raise ValueError(
            f"Outer {outer_fold}: one or more inner "
            f"validation folds are empty: {counts.tolist()}"
        )

    return split_vector


# ============================================================
# MODEL
# ============================================================

def make_model() -> GridSearchCV:

    pipeline = Pipeline(
        steps=[
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=N_FEATURES,
                ),
            ),
            (
                "scale",
                StandardScaler(
                    with_mean=True,
                    with_std=True,
                ),
            ),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=10000,
                ),
            ),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "svm__C": C_GRID,
        },
        scoring="roc_auc",
        cv=None,  # assigned per outer fold
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    predictions = (
        scores >= 0
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    return {
        "roc_auc": float(
            roc_auc_score(y_true, scores)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "n": int(len(y_true)),
        "controls": int((y_true == 0).sum()),
        "schizophrenia": int((y_true == 1).sum()),
    }


# ============================================================
# MAIN NESTED CV
# ============================================================

def run_experiment() -> None:

    ensure_dirs()

    start_time = time.time()

    log("=" * 70)
    log("NEUROFUSION AI - EXP13 VBM + LINEAR SVM")
    log("=" * 70)

    assignments = load_rms_assignments()

    subject_df = build_subject_labels(
        assignments
    )

    # --------------------------------------------------------
    # Locate all final VBM files before starting CV.
    # --------------------------------------------------------

    log("Validating final VBM file availability...")

    missing = []

    for sid in subject_df["subject_id"]:
        try:
            find_vbm_file(str(sid))
        except Exception:
            missing.append(str(sid))

    if missing:
        raise FileNotFoundError(
            f"Missing final VBM files for "
            f"{len(missing)} subjects:\n"
            f"{missing}"
        )

    log(
        f"Final VBM availability: "
        f"{EXPECTED_RMS_SUBJECTS}/{EXPECTED_RMS_SUBJECTS}"
    )

    # --------------------------------------------------------
    # Load VBM matrix.
    # --------------------------------------------------------

    X, y, subject_ids = load_all_vbm(
        subject_df
    )

    # --------------------------------------------------------
    # Build label-free spatial support.
    # --------------------------------------------------------

    spatial_mask = build_spatial_support_mask(X)

    X_masked = X[:, spatial_mask]

    log(
        f"Masked VBM matrix: {X_masked.shape}"
    )

    if X_masked.shape[1] < N_FEATURES:
        raise ValueError(
            "Not enough supported voxels for "
            f"k={N_FEATURES}."
        )

    subject_to_index = {
        sid: i
        for i, sid in enumerate(subject_ids)
    }

    # --------------------------------------------------------
    # OOF storage.
    # --------------------------------------------------------

    oof_records = []

    fold_summaries = []

    # --------------------------------------------------------
    # Outer CV.
    # --------------------------------------------------------

    for outer_fold in range(1, N_OUTER + 1):

        fold_start = time.time()

        log("")
        log("=" * 70)
        log(f"OUTER FOLD {outer_fold}/{N_OUTER}")
        log("=" * 70)

        train_ids, test_ids = get_outer_subjects(
            assignments,
            outer_fold,
        )

        train_idx = np.array(
            [
                subject_to_index[sid]
                for sid in train_ids
            ],
            dtype=int,
        )

        test_idx = np.array(
            [
                subject_to_index[sid]
                for sid in test_ids
            ],
            dtype=int,
        )

        y_train = y[train_idx]
        y_test = y[test_idx]

        log(
            f"Outer train: {len(train_idx)} "
            f"(C={int((y_train == 0).sum())}, "
            f"S={int((y_train == 1).sum())})"
        )

        log(
            f"Outer test:  {len(test_idx)} "
            f"(C={int((y_test == 0).sum())}, "
            f"S={int((y_test == 1).sum())})"
        )

        # ----------------------------------------------------
        # Inner CV.
        # ----------------------------------------------------

        inner_split_vector = (
            make_inner_predefined_split(
                assignments,
                outer_fold,
                train_ids,
            )
        )
        
        inner_counts = np.bincount(
            inner_split_vector,
            minlength=N_INNER,
        ) 

        log(
            "Inner validation counts: "
            f"{inner_counts.tolist()}"
        )

        predefined_split = PredefinedSplit(
            test_fold=inner_split_vector
        )

        # ----------------------------------------------------
        # Build and tune model.
        #
        # IMPORTANT:
        # SelectKBest and StandardScaler are inside the
        # pipeline, so they are fitted independently inside
        # every inner split.
        # ----------------------------------------------------

        grid = make_model()

        grid.cv = predefined_split

        log(
            f"Inner hyperparameter search: "
            f"C={C_GRID}"
        )

        grid.fit(
            X_masked[train_idx],
            y_train,
        )

        best_model = grid.best_estimator_
        best_C = float(
            grid.best_params_["svm__C"]
        )
        best_inner_auc = float(
            grid.best_score_
        )

        log(
            f"Best inner C: {best_C}"
        )

        log(
            f"Best inner ROC-AUC: "
            f"{best_inner_auc:.6f}"
        )

        # ----------------------------------------------------
        # Outer test prediction.
        # ----------------------------------------------------

        test_scores = best_model.decision_function(
            X_masked[test_idx]
        )

        fold_metrics = calculate_metrics(
            y_test,
            test_scores,
        )

        fold_metrics.update(
            {
                "outer_fold": outer_fold,
                "best_C": best_C,
                "best_inner_auc": best_inner_auc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

        fold_summaries.append(
            fold_metrics
        )

        log(
            f"Outer fold {outer_fold} ROC-AUC: "
            f"{fold_metrics['roc_auc']:.6f}"
        )

        # ----------------------------------------------------
        # Store OOF predictions.
        # ----------------------------------------------------

        for sid, true_label, score in zip(
            test_ids,
            y_test,
            test_scores,
        ):
            oof_records.append(
                {
                    "subject_id": sid,
                    "label": int(true_label),
                    "outer_fold": outer_fold,
                    "oof_decision_score": float(score),
                }
            )

        # ----------------------------------------------------
        # Fold summary.
        # ----------------------------------------------------

        pd.DataFrame(
            [fold_metrics]
        ).to_csv(
            FOLD_DIR
            / f"outer_fold_{outer_fold}_metrics.csv",
            index=False,
        )

        elapsed = time.time() - fold_start

        log(
            f"Outer fold {outer_fold} completed "
            f"in {elapsed / 60:.2f} minutes."
        )

    # ========================================================
    # OOF VALIDATION
    # ========================================================

    oof_df = pd.DataFrame(oof_records)

    oof_df = oof_df.sort_values(
        "subject_id"
    ).reset_index(drop=True)

    # Exactly one OOF prediction per subject.
    if len(oof_df) != EXPECTED_RMS_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_RMS_SUBJECTS} OOF rows; "
            f"found {len(oof_df)}."
        )

    if oof_df["subject_id"].nunique() != EXPECTED_RMS_SUBJECTS:
        raise ValueError(
            "OOF subject uniqueness check failed."
        )

    if oof_df["subject_id"].duplicated().any():
        raise ValueError(
            "Duplicate OOF subject IDs detected."
        )

    if not np.all(
        np.isfinite(
            oof_df["oof_decision_score"].to_numpy()
        )
    ):
        raise ValueError(
            "Non-finite OOF decision scores detected."
        )

    y_oof = oof_df["label"].to_numpy(
        dtype=int
    )

    scores_oof = oof_df[
        "oof_decision_score"
    ].to_numpy(
        dtype=float
    )

    oof_metrics = calculate_metrics(
        y_oof,
        scores_oof,
    )

    fold_auc = np.array(
        [
            x["roc_auc"]
            for x in fold_summaries
        ],
        dtype=float,
    )

    mean_fold_auc = float(
        np.mean(fold_auc)
    )

    fold_auc_sd = float(
        np.std(
            fold_auc,
            ddof=0,
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    oof_path = (
        OOF_DIR
        / "exp13_vbm_linear_svm_oof.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    fold_summary_df = pd.DataFrame(
        fold_summaries
    )

    fold_summary_df.to_csv(
        OUTPUT_ROOT
        / "outer_fold_summary.csv",
        index=False,
    )

    summary = {
        "experiment": "Exp13",
        "name": "VBM + Linear SVM",
        "status": "PASS",
        "cohort": "RMS",
        "n_subjects": EXPECTED_RMS_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "outer_cv": N_OUTER,
        "inner_cv": N_INNER,
        "canonical_assignments": str(
            ASSIGNMENTS_PATH
        ),
        "vbm_root": str(VBM_ROOT),
        "representation": (
            "SPM smoothed modulated GM "
            "s8mwc1<SUBJECT_ID>_rms_n4.nii"
        ),
        "vbm_dimensions": [
            EXPECTED_X,
            EXPECTED_Y,
            EXPECTED_Z,
        ],
        "spatial_support_voxels": int(
            spatial_mask.sum()
        ),
        "selected_features": N_FEATURES,
        "feature_selection": (
            "SelectKBest(f_classif), fitted "
            "inside inner CV"
        ),
        "scaling": (
            "StandardScaler, fitted inside "
            "inner CV"
        ),
        "classifier": "LinearSVC",
        "class_weight": "balanced",
        "C_grid": C_GRID,
        "fold_auc": fold_auc.tolist(),
        "mean_fold_auc": mean_fold_auc,
        "fold_auc_sd": fold_auc_sd,
        "oof_roc_auc": oof_metrics["roc_auc"],
        "oof_accuracy": oof_metrics["accuracy"],
        "oof_balanced_accuracy": (
            oof_metrics["balanced_accuracy"]
        ),
        "oof_precision": oof_metrics["precision"],
        "oof_sensitivity": oof_metrics["sensitivity"],
        "oof_specificity": oof_metrics["specificity"],
        "oof_f1": oof_metrics["f1"],
        "oof_rows": len(oof_df),
        "oof_unique_subjects": (
            oof_df["subject_id"].nunique()
        ),
        "oof_integrity": "PASS",
        "runtime_minutes": (
            (time.time() - start_time) / 60.0
        ),
    }

    with open(
        OUTPUT_ROOT / "exp13_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # Save the label-free spatial mask.
    mask_3d = spatial_mask.reshape(
        EXPECTED_X,
        EXPECTED_Y,
        EXPECTED_Z,
    )

    # Use the first subject's affine.
    first_img = nib.load(
        str(find_vbm_file(subject_ids[0]))
    )

    mask_img = nib.Nifti1Image(
        mask_3d.astype(np.uint8),
        first_img.affine,
        first_img.header,
    )

    nib.save(
        mask_img,
        OUTPUT_ROOT / "spatial_support_mask.nii.gz",
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    log("")
    log("=" * 70)
    log("EXP13 FINAL RESULTS")
    log("=" * 70)

    log(
        f"Subjects:              "
        f"{EXPECTED_RMS_SUBJECTS}"
    )

    log(
        f"Controls:              "
        f"{EXPECTED_CONTROLS}"
    )

    log(
        f"Schizophrenia:         "
        f"{EXPECTED_SCHIZOPHRENIA}"
    )

    log(
        f"VBM support voxels:    "
        f"{int(spatial_mask.sum()):,}"
    )

    log(
        f"Selected voxels:       "
        f"{N_FEATURES:,}"
    )

    log(
        f"Outer fold AUCs:       "
        f"{[round(x, 6) for x in fold_auc.tolist()]}"
    )

    log(
        f"Mean fold AUC:         "
        f"{mean_fold_auc:.6f}"
    )

    log(
        f"Fold AUC SD:           "
        f"{fold_auc_sd:.6f}"
    )

    log(
        f"OOF ROC-AUC:           "
        f"{oof_metrics['roc_auc']:.6f}"
    )

    log(
        f"OOF Accuracy:          "
        f"{oof_metrics['accuracy']:.6f}"
    )

    log(
        f"OOF Balanced Accuracy: "
        f"{oof_metrics['balanced_accuracy']:.6f}"
    )

    log(
        f"OOF Sensitivity:       "
        f"{oof_metrics['sensitivity']:.6f}"
    )

    log(
        f"OOF Specificity:       "
        f"{oof_metrics['specificity']:.6f}"
    )

    log(
        f"OOF F1:                "
        f"{oof_metrics['f1']:.6f}"
    )

    log(
        f"OOF rows:              "
        f"{len(oof_df)}"
    )

    log(
        f"Unique OOF subjects:   "
        f"{oof_df['subject_id'].nunique()}"
    )

    log("")
    log("Output:")
    log(str(OUTPUT_ROOT))
    log("")
    log("EXP13 COMPLETED")
    log("=" * 70)


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "NeuroFusion AI Exp13 - "
            "VBM + Linear SVM nested CV"
        )
    )

    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help=(
            "Validate cohort, assignments and VBM "
            "files, then exit without training."
        ),
    )

    args = parser.parse_args()

    ensure_dirs()

    assignments = load_rms_assignments()

    subject_df = build_subject_labels(
        assignments
    )

    missing = []

    for sid in subject_df["subject_id"]:
        try:
            find_vbm_file(str(sid))
        except Exception:
            missing.append(str(sid))

    if missing:
        raise FileNotFoundError(
            f"Missing final VBM files for "
            f"{len(missing)} subjects:\n"
            f"{missing}"
        )

    log("")
    log("=" * 70)
    log("EXP13 SANITY CHECK")
    log("=" * 70)

    log(
        f"RMS subjects: {len(subject_df)}"
    )

    log(
        f"Controls: "
        f"{int((subject_df['label'] == 0).sum())}"
    )

    log(
        f"Schizophrenia: "
        f"{int((subject_df['label'] == 1).sum())}"
    )

    log(
        f"Final VBM files: "
        f"{len(subject_df)}/{EXPECTED_RMS_SUBJECTS}"
    )

    log(
        "Canonical nested CV: PASS"
    )

    log(
        "VBM file availability: PASS"
    )

    log("=" * 70)

    if args.sanity_check:
        log(
            "Sanity check requested; "
            "training will NOT start."
        )
        return

    run_experiment()


if __name__ == "__main__":
    main()