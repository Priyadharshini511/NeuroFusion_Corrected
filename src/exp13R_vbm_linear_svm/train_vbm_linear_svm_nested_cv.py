"""
NeuroFusion AI - Exp13R
VBM + Linear SVM
Strict Nested 5x4 Cross-Validation

Difference from Exp13:
    Exp13  : spatial support mask was constructed using all 119 subjects.
    Exp13R : spatial support mask is constructed separately inside each
             outer training partition.

All label-dependent feature selection and scaling remain inside inner CV.

Cohort:
    119 RMS subjects
    40 controls
    79 schizophrenia

Representation:
    SPM smoothed modulated GM:
    s8mwc1<SUBJECT_ID>_rms_n4.nii

Primary metric:
    OOF ROC-AUC
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# PROJECT CONFIGURATION
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
    / "exp13R_vbm_linear_svm"
)

OOF_DIR = OUTPUT_ROOT / "oof"
FOLD_DIR = OUTPUT_ROOT / "fold_results"

N_OUTER = 5
N_INNER = 4

EXPECTED_SUBJECTS = 119
EXPECTED_CONTROLS = 40
EXPECTED_SCHIZOPHRENIA = 79

VBM_SHAPE = (121, 145, 121)

N_SELECTED_FEATURES = 5000

C_GRID = [0.01, 0.1, 1.0, 10.0]

RANDOM_STATE = 42

NONZERO_EPS = 1e-8


# ============================================================
# BASIC UTILITIES
# ============================================================

def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] {message}",
        flush=True,
    )


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OOF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FOLD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# ASSIGNMENTS
# ============================================================

def load_rms_assignments() -> pd.DataFrame:

    log(
        f"Loading canonical assignments: "
        f"{ASSIGNMENTS_PATH}"
    )

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical assignment file not found:\n"
            f"{ASSIGNMENTS_PATH}"
        )

    df = pd.read_csv(
        ASSIGNMENTS_PATH
    )

    required_columns = {
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "outer_fold",
        "inner_fold",
        "role",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Assignment file is missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 3440:
        raise ValueError(
            f"Expected 3440 canonical rows; "
            f"found {len(df)}."
        )

    rms = df[
        df["acquisition_source"]
        .astype(str)
        .str.upper()
        .eq("RMS")
    ].copy()

    unique_subjects = sorted(
        rms["subject_id"]
        .astype(str)
        .unique()
    )

    log(
        f"Canonical assignment rows: {len(df)}"
    )

    log(
        f"Canonical RMS rows: {len(rms)}"
    )

    log(
        f"Unique RMS subjects: "
        f"{len(unique_subjects)}"
    )

    if len(rms) != EXPECTED_SUBJECTS * 20:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS * 20} "
            f"RMS assignment rows; found {len(rms)}."
        )

    if len(unique_subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} RMS subjects; "
            f"found {len(unique_subjects)}."
        )

    subject_labels = (
        rms[
            ["subject_id", "label"]
        ]
        .drop_duplicates()
    )

    if len(subject_labels) != EXPECTED_SUBJECTS:
        raise ValueError(
            "Subject-level RMS label table "
            "does not contain exactly 119 subjects."
        )

    label_counts = (
        subject_labels
        .groupby("label")
        .size()
        .to_dict()
    )

    log(
        f"RMS subject label counts: "
        f"{label_counts}"
    )

    if label_counts.get(0, 0) != EXPECTED_CONTROLS:
        raise ValueError(
            "Expected 40 controls; "
            f"found {label_counts.get(0, 0)}."
        )

    if label_counts.get(1, 0) != EXPECTED_SCHIZOPHRENIA:
        raise ValueError(
            "Expected 79 schizophrenia subjects; "
            f"found {label_counts.get(1, 0)}."
        )

    counts = (
        rms.groupby("subject_id")
        .size()
    )

    if not np.all(
        counts.to_numpy() == 20
    ):
        raise ValueError(
            "Every RMS subject must have "
            "exactly 20 nested-CV rows."
        )

    return rms


def get_subject_table(
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    subjects = (
        assignments[
            ["subject_id", "label"]
        ]
        .drop_duplicates()
        .copy()
    )

    subjects["subject_id"] = (
        subjects["subject_id"]
        .astype(str)
    )

    subjects["label"] = (
        subjects["label"]
        .astype(int)
    )

    subjects = (
        subjects
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    return subjects


# ============================================================
# VBM FILE HANDLING
# ============================================================

def vbm_path(
    subject_id: str,
) -> Path:

    path = (
        VBM_ROOT
        / subject_id
        / "smooth"
        / f"s8mwc1{subject_id}_rms_n4.nii"
    )

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Final VBM file not found for "
        f"{subject_id}:\n{path}"
    )


def validate_vbm(
    subject_id: str,
) -> None:

    path = vbm_path(subject_id)

    img = nib.load(
        str(path)
    )

    shape = tuple(
        img.shape[:3]
    )

    if shape != VBM_SHAPE:
        raise ValueError(
            f"{subject_id}: VBM shape {shape}; "
            f"expected {VBM_SHAPE}."
        )

    data = np.asarray(
        img.dataobj,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(data)
    ):
        raise ValueError(
            f"{subject_id}: non-finite VBM values."
        )


def load_vbm_matrix(
    subjects: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:

    subject_ids = (
        subjects["subject_id"]
        .astype(str)
        .tolist()
    )

    n_subjects = len(subject_ids)

    n_voxels = int(
        np.prod(VBM_SHAPE)
    )

    log(
        f"Loading {n_subjects} VBM maps "
        f"({n_voxels:,} voxels/map)..."
    )

    X = np.empty(
        (
            n_subjects,
            n_voxels,
        ),
        dtype=np.float32,
    )

    reference_affine = None

    for i, sid in enumerate(subject_ids):

        log(
            f"Loading "
            f"[{i + 1:03d}/{n_subjects:03d}] "
            f"{sid}"
        )

        path = vbm_path(sid)

        img = nib.load(
            str(path)
        )

        if reference_affine is None:
            reference_affine = (
                img.affine.copy()
            )
        else:
            if not np.allclose(
                reference_affine,
                img.affine,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{sid}: affine differs from "
                    "reference VBM image."
                )

        data = np.asarray(
            img.dataobj,
            dtype=np.float32,
        )

        if tuple(data.shape) != VBM_SHAPE:
            raise ValueError(
                f"{sid}: unexpected data shape "
                f"{data.shape}."
            )

        if not np.all(
            np.isfinite(data)
        ):
            raise ValueError(
                f"{sid}: non-finite values."
            )

        X[i] = data.reshape(-1)

    log(
        f"VBM matrix loaded: {X.shape}, "
        f"dtype={X.dtype}"
    )

    return X, subject_ids


# ============================================================
# STRICT OUTER-TRAINING SPATIAL MASK
# ============================================================

def build_training_spatial_mask(
    X_train: np.ndarray,
) -> np.ndarray:

    """
    IMPORTANT:
    This function is called separately for each outer fold
    using ONLY the outer-training subjects.

    No outer-test image contributes to this mask.

    The mask itself is label-free.
    """

    log(
        "Building spatial support mask "
        "from OUTER TRAINING subjects only..."
    )

    nonzero_fraction = np.mean(
        np.abs(X_train) > NONZERO_EPS,
        axis=0,
    )

    mask = (
        nonzero_fraction >= 0.50
    )

    count = int(
        mask.sum()
    )

    log(
        f"Outer-training spatial support: "
        f"{count:,} voxels"
    )

    if count < N_SELECTED_FEATURES:
        raise ValueError(
            f"Training spatial support contains "
            f"only {count} voxels; "
            f"need {N_SELECTED_FEATURES}."
        )

    return mask


# ============================================================
# NESTED CV ASSIGNMENTS
# ============================================================

def get_outer_split(
    assignments: pd.DataFrame,
    outer_fold: int,
) -> tuple[list[str], list[str]]:

    fold = assignments[
        assignments["outer_fold"].astype(int)
        == outer_fold
    ].copy()

    test_ids = sorted(
        fold.loc[
            fold["role"]
            .astype(str)
            .str.upper()
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

    test_set = set(
        test_ids
    )

    train_ids = [
        sid
        for sid in all_ids
        if sid not in test_set
    ]

    if len(test_ids) == 0:
        raise ValueError(
            f"Outer fold {outer_fold}: "
            "no OUTER_TEST subjects."
        )

    if len(train_ids) + len(test_ids) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Outer fold {outer_fold}: "
            f"train={len(train_ids)}, "
            f"test={len(test_ids)}."
        )

    return train_ids, test_ids


def make_inner_split(
    assignments: pd.DataFrame,
    outer_fold: int,
    train_ids: list[str],
) -> np.ndarray:

    """
    Creates the PredefinedSplit vector for the outer-training
    subjects.

    Each subject has exactly one INNER_VALIDATION assignment
    among the four inner folds.

    The canonical assignment file uses:
        INNER_TRAIN
        INNER_VALIDATION
        OUTER_TEST
    """

    outer = assignments[
        assignments["outer_fold"].astype(int)
        == outer_fold
    ].copy()

    fold_vector = []

    for sid in train_ids:

        sid_rows = outer[
            outer["subject_id"]
            .astype(str)
            == sid
        ]

        validation_rows = sid_rows[
            sid_rows["role"]
            .astype(str)
            .str.upper()
            == "INNER_VALIDATION"
        ]

        if len(validation_rows) != 1:
            raise ValueError(
                f"Outer {outer_fold}, "
                f"subject {sid}: expected exactly "
                "one INNER_VALIDATION row; "
                f"found {len(validation_rows)}."
            )

        inner_fold = int(
            validation_rows.iloc[0][
                "inner_fold"
            ]
        )

        if inner_fold not in {
            1, 2, 3, 4
        }:
            raise ValueError(
                f"Invalid inner fold "
                f"{inner_fold} for {sid}."
            )

        fold_vector.append(
            inner_fold - 1
        )

    split_vector = np.asarray(
        fold_vector,
        dtype=int,
    )

    counts = np.bincount(
        split_vector,
        minlength=N_INNER,
    )

    if np.any(counts == 0):
        raise ValueError(
            f"Empty inner validation fold: "
            f"{counts.tolist()}"
        )

    return split_vector


# ============================================================
# MODEL
# ============================================================

def build_grid_search(
    inner_cv,
) -> GridSearchCV:

    pipeline = Pipeline(
        steps=[
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=N_SELECTED_FEATURES,
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
        cv=inner_cv,
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
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
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
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment() -> None:

    ensure_dirs()

    experiment_start = time.time()

    log("=" * 70)
    log("NEUROFUSION AI - EXP13R")
    log("VBM + LINEAR SVM")
    log("STRICT OUTER-TRAINING SPATIAL MASK")
    log("=" * 70)

    assignments = load_rms_assignments()

    subjects = get_subject_table(
        assignments
    )

    # --------------------------------------------------------
    # Check all final VBM maps before training.
    # --------------------------------------------------------

    log(
        "Validating final VBM files..."
    )

    for sid in subjects[
        "subject_id"
    ]:
        validate_vbm(
            str(sid)
        )

    log(
        f"Final VBM files: "
        f"{len(subjects)}/{EXPECTED_SUBJECTS}"
    )

    # --------------------------------------------------------
    # Load all images once.
    #
    # The image values are fixed inputs.
    # No labels are used during image loading.
    # --------------------------------------------------------

    X, subject_ids = load_vbm_matrix(
        subjects
    )

    y = subjects[
        "label"
    ].to_numpy(
        dtype=int
    )

    subject_to_index = {
        sid: i
        for i, sid in enumerate(
            subject_ids
        )
    }

    # --------------------------------------------------------
    # OOF storage.
    # --------------------------------------------------------

    oof_records = []

    fold_results = []

    # --------------------------------------------------------
    # OUTER CV
    # --------------------------------------------------------

    for outer_fold in range(
        1,
        N_OUTER + 1,
    ):

        fold_start = time.time()

        log("")
        log("=" * 70)
        log(
            f"OUTER FOLD "
            f"{outer_fold}/{N_OUTER}"
        )
        log("=" * 70)

        train_ids, test_ids = (
            get_outer_split(
                assignments,
                outer_fold,
            )
        )

        train_idx = np.asarray(
            [
                subject_to_index[sid]
                for sid in train_ids
            ],
            dtype=int,
        )

        test_idx = np.asarray(
            [
                subject_to_index[sid]
                for sid in test_ids
            ],
            dtype=int,
        )

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        log(
            f"Outer train: "
            f"{len(train_idx)} "
            f"(C={int((y_train == 0).sum())}, "
            f"S={int((y_train == 1).sum())})"
        )

        log(
            f"Outer test:  "
            f"{len(test_idx)} "
            f"(C={int((y_test == 0).sum())}, "
            f"S={int((y_test == 1).sum())})"
        )

        # ----------------------------------------------------
        # STRICT STEP:
        # spatial support is constructed ONLY from the
        # outer-training subjects.
        # ----------------------------------------------------

        outer_train_mask = (
            build_training_spatial_mask(
                X[train_idx]
            )
        )

        X_train_masked = X[
            train_idx
        ][:, outer_train_mask]

        X_test_masked = X[
            test_idx
        ][:, outer_train_mask]

        log(
            f"Outer-train masked matrix: "
            f"{X_train_masked.shape}"
        )

        log(
            f"Outer-test masked matrix: "
            f"{X_test_masked.shape}"
        )

        # ----------------------------------------------------
        # Inner CV.
        # ----------------------------------------------------

        inner_vector = (
            make_inner_split(
                assignments,
                outer_fold,
                train_ids,
            )
        )

        inner_counts = np.bincount(
            inner_vector,
            minlength=N_INNER,
        )

        log(
            f"Inner validation counts: "
            f"{inner_counts.tolist()}"
        )

        inner_cv = PredefinedSplit(
            test_fold=inner_vector
        )

        # ----------------------------------------------------
        # Inner model selection.
        #
        # SelectKBest and StandardScaler are inside the
        # pipeline, therefore each is fitted independently
        # within every inner training split.
        # ----------------------------------------------------

        log(
            f"Inner hyperparameter search: "
            f"C={C_GRID}"
        )

        grid = build_grid_search(
            inner_cv
        )

        grid.fit(
            X_train_masked,
            y_train,
        )

        best_C = float(
            grid.best_params_[
                "svm__C"
            ]
        )

        best_inner_auc = float(
            grid.best_score_
        )

        log(
            f"Best inner C: "
            f"{best_C}"
        )

        log(
            f"Best inner ROC-AUC: "
            f"{best_inner_auc:.6f}"
        )

        # ----------------------------------------------------
        # OUTER TEST PREDICTION
        # ----------------------------------------------------

        test_scores = (
            grid.best_estimator_
            .decision_function(
                X_test_masked
            )
        )

        metrics = calculate_metrics(
            y_test,
            test_scores,
        )

        metrics.update(
            {
                "outer_fold": outer_fold,
                "best_C": best_C,
                "best_inner_auc": best_inner_auc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "outer_training_spatial_voxels": int(
                    outer_train_mask.sum()
                ),
            }
        )

        fold_results.append(
            metrics
        )

        log(
            f"Outer fold "
            f"{outer_fold} ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

        # ----------------------------------------------------
        # Save OOF predictions.
        # ----------------------------------------------------

        for sid, label, score in zip(
            test_ids,
            y_test,
            test_scores,
        ):
            oof_records.append(
                {
                    "subject_id": sid,
                    "label": int(label),
                    "outer_fold": outer_fold,
                    "oof_decision_score": float(
                        score
                    ),
                }
            )

        pd.DataFrame(
            [metrics]
        ).to_csv(
            FOLD_DIR
            / f"outer_fold_{outer_fold}_metrics.csv",
            index=False,
        )

        log(
            f"Outer fold "
            f"{outer_fold} completed in "
            f"{(time.time() - fold_start) / 60:.2f} "
            "minutes."
        )

    # ========================================================
    # OOF INTEGRITY
    # ========================================================

    oof = pd.DataFrame(
        oof_records
    )

    oof = (
        oof
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    if len(oof) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} OOF rows; "
            f"found {len(oof)}."
        )

    if (
        oof["subject_id"]
        .nunique()
        != EXPECTED_SUBJECTS
    ):
        raise ValueError(
            "OOF unique-subject check failed."
        )

    if (
        oof["subject_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate OOF subjects detected."
        )

    scores = oof[
        "oof_decision_score"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(scores)
    ):
        raise ValueError(
            "Non-finite OOF scores detected."
        )

    y_oof = oof[
        "label"
    ].to_numpy(
        dtype=int
    )

    oof_metrics = calculate_metrics(
        y_oof,
        scores,
    )

    fold_auc = np.asarray(
        [
            row["roc_auc"]
            for row in fold_results
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
        / "exp13R_vbm_linear_svm_oof.csv"
    )

    oof.to_csv(
        oof_path,
        index=False,
    )

    fold_summary = pd.DataFrame(
        fold_results
    )

    fold_summary.to_csv(
        OUTPUT_ROOT
        / "outer_fold_summary.csv",
        index=False,
    )

    summary = {
        "experiment": "Exp13R",
        "name": (
            "VBM + Linear SVM "
            "(strict outer-training spatial mask)"
        ),
        "status": "PASS",
        "cohort": "RMS",
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "outer_cv": N_OUTER,
        "inner_cv": N_INNER,
        "canonical_assignments": str(
            ASSIGNMENTS_PATH
        ),
        "vbm_root": str(
            VBM_ROOT
        ),
        "vbm_shape": list(
            VBM_SHAPE
        ),
        "selected_features": (
            N_SELECTED_FEATURES
        ),
        "spatial_mask": (
            "constructed separately from "
            "outer-training subjects only"
        ),
        "feature_selection": (
            "SelectKBest(f_classif), "
            "inside inner CV"
        ),
        "scaling": (
            "StandardScaler, "
            "inside inner CV"
        ),
        "classifier": "LinearSVC",
        "class_weight": "balanced",
        "C_grid": C_GRID,
        "fold_auc": fold_auc.tolist(),
        "mean_fold_auc": mean_fold_auc,
        "fold_auc_sd": fold_auc_sd,
        "oof_roc_auc": (
            oof_metrics["roc_auc"]
        ),
        "oof_accuracy": (
            oof_metrics["accuracy"]
        ),
        "oof_balanced_accuracy": (
            oof_metrics[
                "balanced_accuracy"
            ]
        ),
        "oof_precision": (
            oof_metrics["precision"]
        ),
        "oof_sensitivity": (
            oof_metrics["sensitivity"]
        ),
        "oof_specificity": (
            oof_metrics["specificity"]
        ),
        "oof_f1": (
            oof_metrics["f1"]
        ),
        "oof_rows": len(oof),
        "oof_unique_subjects": (
            oof["subject_id"]
            .nunique()
        ),
        "oof_integrity": "PASS",
        "runtime_minutes": (
            (time.time() - experiment_start)
            / 60.0
        ),
    }

    with open(
        OUTPUT_ROOT
        / "exp13R_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    log("")
    log("=" * 70)
    log("EXP13R FINAL RESULTS")
    log("=" * 70)

    log(
        f"Subjects:              "
        f"{EXPECTED_SUBJECTS}"
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
        f"Selected voxels/fold:  "
        f"{N_SELECTED_FEATURES}"
    )

    log(
        "Outer fold AUCs:       "
        f"{[round(x, 6) for x in fold_auc]}"
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
        f"{len(oof)}"
    )

    log(
        f"Unique OOF subjects:   "
        f"{oof['subject_id'].nunique()}"
    )

    log("")
    log(
        f"Output: {OUTPUT_ROOT}"
    )

    log("")
    log("EXP13R COMPLETED")
    log("=" * 70)


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check() -> None:

    ensure_dirs()

    assignments = load_rms_assignments()

    subjects = get_subject_table(
        assignments
    )

    log("")
    log("=" * 70)
    log("EXP13R SANITY CHECK")
    log("=" * 70)

    missing = []

    for sid in subjects[
        "subject_id"
    ]:
        try:
            validate_vbm(
                str(sid)
            )
        except Exception as exc:
            missing.append(
                (str(sid), str(exc))
            )

    log(
        f"RMS subjects: "
        f"{len(subjects)}"
    )

    log(
        f"Controls: "
        f"{int((subjects['label'] == 0).sum())}"
    )

    log(
        f"Schizophrenia: "
        f"{int((subjects['label'] == 1).sum())}"
    )

    log(
        f"Final VBM files: "
        f"{len(subjects) - len(missing)}/"
        f"{EXPECTED_SUBJECTS}"
    )

    if missing:
        log(
            f"Missing/invalid VBM files: "
            f"{len(missing)}"
        )

        for sid, reason in missing[:20]:
            log(
                f"  {sid}: {reason}"
            )

        raise RuntimeError(
            "VBM availability check failed."
        )

    # Validate canonical outer folds.
    for outer_fold in range(
        1,
        N_OUTER + 1,
    ):
        train_ids, test_ids = (
            get_outer_split(
                assignments,
                outer_fold,
            )
        )

        inner_vector = (
            make_inner_split(
                assignments,
                outer_fold,
                train_ids,
            )
        )

        counts = np.bincount(
            inner_vector,
            minlength=N_INNER,
        )

        log(
            f"Outer {outer_fold}: "
            f"train={len(train_ids)}, "
            f"test={len(test_ids)}, "
            f"inner={counts.tolist()}"
        )

    log(
        "Canonical nested CV: PASS"
    )

    log(
        "VBM file availability: PASS"
    )

    log(
        "Strict outer-training mask design: PASS"
    )

    log("=" * 70)
    log(
        "Sanity check requested; "
        "training will NOT start."
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "NeuroFusion AI Exp13R - "
            "strict VBM + Linear SVM"
        )
    )

    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help=(
            "Validate cohort, CV assignments and "
            "VBM files without training."
        ),
    )

    args = parser.parse_args()

    if args.sanity_check:
        sanity_check()
    else:
        run_experiment()


if __name__ == "__main__":
    main()