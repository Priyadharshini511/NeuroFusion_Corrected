"""
NeuroFusion AI - Experiment 14
VBM + RBF SVM
Strict Nested 5x4 CV
Strict outer-training spatial support mask

Scientific question:
Can a nonlinear RBF SVM capture VBM-based morphometric patterns
that are not captured by the linear SVM?

Cohort:
119 RMS subjects
40 controls
79 schizophrenia

Leakage controls:
1. Canonical 172-subject nested CV assignments
2. RMS subset only
3. Spatial support mask built from OUTER TRAIN subjects only
4. SelectKBest fitted inside inner CV
5. StandardScaler fitted inside inner CV
6. RBF SVM hyperparameters selected using inner ROC-AUC
7. Outer test remains untouched until final evaluation
"""

from __future__ import annotations

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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    / "exp14_vbm_rbf_svm"
)

OOF_DIR = OUTPUT_ROOT / "oof"

N_SUBJECTS_EXPECTED = 119
N_CONTROLS_EXPECTED = 40
N_SCHIZOPHRENIA_EXPECTED = 79

VBM_SHAPE = (121, 145, 121)

N_SELECTED_VOXELS = 5000

SPATIAL_SUPPORT_THRESHOLD = 0.50

RANDOM_STATE = 42

C_GRID = [0.01, 0.1, 1.0, 10.0]

GAMMA_GRID = [
    "scale",
    1e-5,
    1e-4,
    1e-3,
]


# ============================================================
# LOGGING
# ============================================================

def log(message: str = "") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================
# ASSIGNMENT LOADING
# ============================================================

def load_assignments() -> pd.DataFrame:

    log(f"Loading canonical assignments: {ASSIGNMENTS_PATH}")

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical assignment file not found:\n{ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_csv(ASSIGNMENTS_PATH)

    log(f"Canonical assignment rows: {len(assignments)}")

    required_columns = {
        "subject_id",
        "label",
        "outer_fold",
        "inner_fold",
        "role",
    }

    missing = required_columns - set(assignments.columns)

    if missing:
        raise ValueError(
            f"Assignment file missing required columns: {sorted(missing)}"
        )

    assignments["subject_id"] = assignments["subject_id"].astype(str)

    # --------------------------------------------------------
    # Determine RMS subject IDs from final VBM directories.
    # --------------------------------------------------------

    vbm_subject_dirs = sorted(
        [
            p.name
            for p in VBM_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("A")
        ]
    )

    if len(vbm_subject_dirs) != N_SUBJECTS_EXPECTED:
        raise ValueError(
            f"Expected {N_SUBJECTS_EXPECTED} VBM subject directories, "
            f"found {len(vbm_subject_dirs)}"
        )

    rms_ids = set(vbm_subject_dirs)

    rms_assignments = assignments[
        assignments["subject_id"].isin(rms_ids)
    ].copy()

    log(f"Canonical RMS rows: {len(rms_assignments)}")
    log(
        f"Unique RMS subjects: "
        f"{rms_assignments['subject_id'].nunique()}"
    )

    label_counts = (
        rms_assignments
        .drop_duplicates("subject_id")["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    log(f"RMS subject label counts: {label_counts}")

    # --------------------------------------------------------
    # Validate subject count.
    # --------------------------------------------------------

    unique_subjects = (
        rms_assignments["subject_id"]
        .drop_duplicates()
        .tolist()
    )

    if len(unique_subjects) != N_SUBJECTS_EXPECTED:
        raise ValueError(
            f"Expected {N_SUBJECTS_EXPECTED} RMS subjects, "
            f"found {len(unique_subjects)}"
        )

    # --------------------------------------------------------
    # Validate 5 x 4 nested CV structure.
    # --------------------------------------------------------

    expected_rows = N_SUBJECTS_EXPECTED * 5 * 4

    if len(rms_assignments) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} RMS assignment rows, "
            f"found {len(rms_assignments)}"
        )

    roles = set(rms_assignments["role"].unique())

    expected_roles = {
        "INNER_TRAIN",
        "INNER_VALIDATION",
        "OUTER_TEST",
    }

    if roles != expected_roles:
        raise ValueError(
            f"Unexpected CV roles: {roles}. "
            f"Expected: {expected_roles}"
        )

    duplicate_count = (
        rms_assignments
        .duplicated(
            subset=[
                "subject_id",
                "outer_fold",
                "inner_fold",
                "role",
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate CV assignment rows."
        )

    # --------------------------------------------------------
    # Validate labels are subject-consistent.
    # --------------------------------------------------------

    label_per_subject = (
        rms_assignments
        .groupby("subject_id")["label"]
        .nunique()
    )

    inconsistent_labels = label_per_subject[
        label_per_subject != 1
    ]

    if len(inconsistent_labels) > 0:
        raise ValueError(
            "Some RMS subjects have inconsistent labels."
        )

    log("Canonical nested CV validation: PASS")

    return rms_assignments


# ============================================================
# VBM PATH
# ============================================================

def get_vbm_path(subject_id: str) -> Path:

    return (
        VBM_ROOT
        / subject_id
        / "smooth"
        / f"s8mwc1{subject_id}_rms_n4.nii"
    )


# ============================================================
# VBM VALIDATION
# ============================================================

def validate_vbm_files(subject_ids: list[str]) -> None:

    log("Validating final VBM files...")

    missing = []

    for subject_id in subject_ids:

        path = get_vbm_path(subject_id)

        if not path.exists():
            missing.append(subject_id)

    if missing:

        raise FileNotFoundError(
            "Missing final VBM files:\n"
            + "\n".join(missing)
        )

    log(
        f"Final VBM files: "
        f"{len(subject_ids)}/{N_SUBJECTS_EXPECTED}"
    )


# ============================================================
# LOAD VBM MATRIX
# ============================================================

def load_vbm_matrix(
    subject_ids: list[str],
) -> np.ndarray:

    n_voxels = int(np.prod(VBM_SHAPE))

    log(
        f"Loading {len(subject_ids)} VBM maps "
        f"({n_voxels:,} voxels/map)..."
    )

    X = np.empty(
        (len(subject_ids), n_voxels),
        dtype=np.float32,
    )

    for i, subject_id in enumerate(subject_ids):

        log(
            f"Loading "
            f"[{i + 1:03d}/{len(subject_ids):03d}] "
            f"{subject_id}"
        )

        path = get_vbm_path(subject_id)

        img = nib.load(path)

        data = np.asarray(
            img.get_fdata(dtype=np.float32)
        )

        if data.shape != VBM_SHAPE:
            raise ValueError(
                f"{subject_id}: unexpected VBM shape "
                f"{data.shape}, expected {VBM_SHAPE}"
            )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"{subject_id}: VBM contains non-finite values."
            )

        X[i] = data.reshape(-1)

    log(
        f"VBM matrix loaded: {X.shape}, "
        f"dtype={X.dtype}"
    )

    return X


# ============================================================
# OUTER-TRAINING SPATIAL SUPPORT MASK
# ============================================================

def build_spatial_support_mask(
    X_outer_train: np.ndarray,
) -> np.ndarray:

    log(
        "Building spatial support mask from "
        "OUTER TRAINING subjects only..."
    )

    # Nonzero presence per subject.
    nonzero = X_outer_train != 0

    support_fraction = nonzero.mean(axis=0)

    mask = support_fraction >= SPATIAL_SUPPORT_THRESHOLD

    n_selected = int(mask.sum())

    if n_selected == 0:
        raise ValueError(
            "Spatial support mask contains zero voxels."
        )

    log(
        f"Outer-training spatial support: "
        f"{n_selected:,} voxels"
    )

    return mask


# ============================================================
# BUILD RBF SVM PIPELINE
# ============================================================

def build_pipeline() -> Pipeline:

    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=RANDOM_STATE,
        cache_size=4096,
    )

    pipeline = Pipeline(
        steps=[
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=N_SELECTED_VOXELS,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "svm",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# INNER CV
# ============================================================

def build_inner_predefined_split(
    outer_assignments: pd.DataFrame,
    outer_train_ids: set[str],
) -> tuple[np.ndarray, list[str]]:

    train_assignments = outer_assignments[
        outer_assignments["subject_id"].isin(outer_train_ids)
    ].copy()

    # --------------------------------------------------------
    # We need exactly one row per subject for the inner split.
    #
    # The canonical assignment table contains one row for
    # every subject/outer_fold/inner_fold/role combination.
    #
    # For inner validation:
    #   INNER_VALIDATION -> fold index 0..3
    #
    # For inner training:
    #   INNER_TRAIN -> -1
    # --------------------------------------------------------

    subject_roles = (
        train_assignments[
            train_assignments["role"].isin(
                ["INNER_TRAIN", "INNER_VALIDATION"]
            )
        ]
        .sort_values(
            [
                "subject_id",
                "inner_fold",
                "role",
            ]
        )
    )

    # Every subject should occur exactly once as INNER_VALIDATION
    # and three times as INNER_TRAIN across the 4 inner folds.
    validation_rows = subject_roles[
        subject_roles["role"] == "INNER_VALIDATION"
    ]

    validation_count = (
        validation_rows
        .groupby("subject_id")
        .size()
    )

    if not np.all(validation_count.values == 1):
        raise ValueError(
            "Inner validation assignment is not exactly once "
            "per subject."
        )

    # Use the unique inner validation assignment for each subject.
    validation_map = (
        validation_rows
        .drop_duplicates("subject_id")
        .set_index("subject_id")["inner_fold"]
        .to_dict()
    )

    ordered_ids = sorted(outer_train_ids)

    inner_vector = []

    for subject_id in ordered_ids:

        if subject_id not in validation_map:
            raise ValueError(
                f"Missing inner validation assignment for "
                f"{subject_id}"
            )

        inner_vector.append(
            int(validation_map[subject_id])
        )

    inner_vector = np.asarray(
        inner_vector,
        dtype=int,
    )

    # PredefinedSplit expects:
    #   -1 = training
    #   0..3 = validation fold
    #
    # But each subject's "validation fold" assignment identifies
    # the fold where that subject is validation.
    #
    # This means subjects assigned to inner fold k are validation
    # for k and training for the other folds.
    #
    # GridSearchCV with PredefinedSplit uses one fixed split,
    # not four rotating splits. Therefore we construct a list of
    # explicit inner splits below instead of relying on a single
    # PredefinedSplit vector.

    return inner_vector, ordered_ids


# ============================================================
# CUSTOM INNER CV SPLIT
# ============================================================

class CanonicalInnerCV:
    """
    Four explicit inner folds from the canonical assignments.

    For each inner fold:
        validation = INNER_VALIDATION
        training   = INNER_TRAIN
    """

    def __init__(
        self,
        outer_assignments: pd.DataFrame,
        outer_train_ids: list[str],
    ):
        self.assignments = outer_assignments
        self.outer_train_ids = set(outer_train_ids)

    def split(
        self,
        X=None,
        y=None,
        groups=None,
    ):
        df = self.assignments[
            self.assignments["subject_id"].isin(
                self.outer_train_ids
            )
        ].copy()

        for inner_fold in sorted(
            df["inner_fold"].unique()
        ):

            train_ids = set(
                df[
                    (df["inner_fold"] == inner_fold)
                    & (df["role"] == "INNER_TRAIN")
                ]["subject_id"]
            )

            valid_ids = set(
                df[
                    (df["inner_fold"] == inner_fold)
                    & (df["role"] == "INNER_VALIDATION")
                ]["subject_id"]
            )

            if not train_ids:
                raise ValueError(
                    f"Inner fold {inner_fold}: "
                    "no training subjects."
                )

            if not valid_ids:
                raise ValueError(
                    f"Inner fold {inner_fold}: "
                    "no validation subjects."
                )

            train_indices = [
                i
                for i, sid in enumerate(
                    self.outer_train_ids_ordered
                )
                if sid in train_ids
            ]

            valid_indices = [
                i
                for i, sid in enumerate(
                    self.outer_train_ids_ordered
                )
                if sid in valid_ids
            ]

            yield (
                np.asarray(train_indices, dtype=int),
                np.asarray(valid_indices, dtype=int),
            )

    @property
    def outer_train_ids_ordered(self):
        return sorted(self.outer_train_ids)

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ):
        return 4


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    log("=" * 70)
    log("NEUROFUSION AI - EXP14")
    log("VBM + RBF SVM")
    log("STRICT OUTER-TRAINING SPATIAL MASK")
    log("=" * 70)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OOF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load canonical assignments
    # --------------------------------------------------------

    assignments = load_assignments()

    subject_table = (
        assignments[
            ["subject_id", "label"]
        ]
        .drop_duplicates("subject_id")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    subject_ids = subject_table["subject_id"].tolist()

    y = subject_table["label"].astype(int).to_numpy()

    if len(subject_ids) != N_SUBJECTS_EXPECTED:
        raise ValueError(
            "Unexpected subject count."
        )

    if int((y == 0).sum()) != N_CONTROLS_EXPECTED:
        raise ValueError(
            "Unexpected control count."
        )

    if int((y == 1).sum()) != N_SCHIZOPHRENIA_EXPECTED:
        raise ValueError(
            "Unexpected schizophrenia count."
        )

    # --------------------------------------------------------
    # Validate VBM
    # --------------------------------------------------------

    validate_vbm_files(subject_ids)

    # --------------------------------------------------------
    # Load all VBM maps
    # --------------------------------------------------------

    X = load_vbm_matrix(subject_ids)

    subject_to_index = {
        sid: i
        for i, sid in enumerate(subject_ids)
    }

    # --------------------------------------------------------
    # OOF storage
    # --------------------------------------------------------

    oof_rows = []

    outer_summary = []

    # --------------------------------------------------------
    # Outer folds
    # --------------------------------------------------------

    outer_folds = sorted(
        assignments["outer_fold"].unique()
    )

    if len(outer_folds) != 5:
        raise ValueError(
            f"Expected 5 outer folds, found {outer_folds}"
        )

    for outer_fold in outer_folds:

        fold_start = time.time()

        log("")
        log("=" * 70)
        log(
            f"OUTER FOLD "
            f"{outer_fold}/5"
        )
        log("=" * 70)

        fold_df = assignments[
            assignments["outer_fold"] == outer_fold
        ].copy()

        # ----------------------------------------------------
        # Outer test
        # ----------------------------------------------------

        outer_test_ids = sorted(
            fold_df[
                fold_df["role"] == "OUTER_TEST"
            ]["subject_id"]
            .unique()
        )

        # ----------------------------------------------------
        # Outer train
        # ----------------------------------------------------

        outer_train_ids = sorted(
            fold_df[
                fold_df["role"] != "OUTER_TEST"
            ]["subject_id"]
            .unique()
        )

        # ----------------------------------------------------
        # Validate partition
        # ----------------------------------------------------

        overlap = set(outer_train_ids) & set(
            outer_test_ids
        )

        if overlap:
            raise ValueError(
                f"Outer train/test overlap: {overlap}"
            )

        log(
            f"Outer train: {len(outer_train_ids)} "
            f"(C={sum(y[subject_to_index[s]] == 0 for s in outer_train_ids)}, "
            f"S={sum(y[subject_to_index[s]] == 1 for s in outer_train_ids)})"
        )

        log(
            f"Outer test:  {len(outer_test_ids)} "
            f"(C={sum(y[subject_to_index[s]] == 0 for s in outer_test_ids)}, "
            f"S={sum(y[subject_to_index[s]] == 1 for s in outer_test_ids)})"
        )

        # ----------------------------------------------------
        # Build spatial support ONLY from outer training.
        # ----------------------------------------------------

        outer_train_indices = [
            subject_to_index[sid]
            for sid in outer_train_ids
        ]

        outer_test_indices = [
            subject_to_index[sid]
            for sid in outer_test_ids
        ]

        X_outer_train_full = X[
            outer_train_indices
        ]

        X_outer_test_full = X[
            outer_test_indices
        ]

        support_mask = build_spatial_support_mask(
            X_outer_train_full
        )

        X_outer_train = (
            X_outer_train_full[:, support_mask]
        )

        X_outer_test = (
            X_outer_test_full[:, support_mask]
        )

        log(
            f"Outer-train masked matrix: "
            f"{X_outer_train.shape}"
        )

        log(
            f"Outer-test masked matrix: "
            f"{X_outer_test.shape}"
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        y_outer_train = np.array(
            [
                y[subject_to_index[sid]]
                for sid in outer_train_ids
            ],
            dtype=int,
        )

        y_outer_test = np.array(
            [
                y[subject_to_index[sid]]
                for sid in outer_test_ids
            ],
            dtype=int,
        )

        # ----------------------------------------------------
        # Inner fold information
        # ----------------------------------------------------

        inner_assignment_df = fold_df[
            fold_df["role"].isin(
                [
                    "INNER_TRAIN",
                    "INNER_VALIDATION",
                ]
            )
        ].copy()

        inner_counts = []

        for inner_fold in sorted(
            inner_assignment_df["inner_fold"].unique()
        ):

            count = len(
                inner_assignment_df[
                    (inner_assignment_df["inner_fold"] == inner_fold)
                    & (
                        inner_assignment_df["role"]
                        == "INNER_VALIDATION"
                    )
                ]["subject_id"].unique()
            )

            inner_counts.append(count)

        log(
            f"Inner validation counts: {inner_counts}"
        )

        # ----------------------------------------------------
        # Build explicit canonical inner CV.
        # ----------------------------------------------------

        inner_cv = CanonicalInnerCV(
            outer_assignments=fold_df,
            outer_train_ids=outer_train_ids,
        )

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        pipeline = build_pipeline()

        parameter_grid = {
            "svm__C": C_GRID,
            "svm__gamma": GAMMA_GRID,
        }

        log(
            "Inner hyperparameter search:"
        )

        log(
            f"C={C_GRID}"
        )

        log(
            f"gamma={GAMMA_GRID}"
        )

        # ----------------------------------------------------
        # Grid search
        # ----------------------------------------------------

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            verbose=0,
            return_train_score=False,
        )

        search.fit(
            X_outer_train,
            y_outer_train,
        )

        best_C = search.best_params_[
            "svm__C"
        ]

        best_gamma = search.best_params_[
            "svm__gamma"
        ]

        best_inner_auc = float(
            search.best_score_
        )

        log(
            f"Best inner C: {best_C}"
        )

        log(
            f"Best inner gamma: {best_gamma}"
        )

        log(
            f"Best inner ROC-AUC: "
            f"{best_inner_auc:.6f}"
        )

        # ----------------------------------------------------
        # Outer prediction
        # ----------------------------------------------------

        decision_scores = search.decision_function(
            X_outer_test
        )

        outer_auc = roc_auc_score(
            y_outer_test,
            decision_scores,
        )

        log(
            f"Outer fold {outer_fold} "
            f"ROC-AUC: {outer_auc:.6f}"
        )

        # ----------------------------------------------------
        # Store OOF predictions
        # ----------------------------------------------------

        for sid, true_label, score in zip(
            outer_test_ids,
            y_outer_test,
            decision_scores,
        ):

            oof_rows.append(
                {
                    "subject_id": sid,
                    "label": int(true_label),
                    "decision_score": float(score),
                    "outer_fold": int(outer_fold),
                    "best_C": best_C,
                    "best_gamma": best_gamma,
                }
            )

        # ----------------------------------------------------
        # Outer summary
        # ----------------------------------------------------

        outer_summary.append(
            {
                "outer_fold": int(outer_fold),
                "n_train": len(outer_train_ids),
                "n_test": len(outer_test_ids),
                "train_controls": int(
                    (y_outer_train == 0).sum()
                ),
                "train_schizophrenia": int(
                    (y_outer_train == 1).sum()
                ),
                "test_controls": int(
                    (y_outer_test == 0).sum()
                ),
                "test_schizophrenia": int(
                    (y_outer_test == 1).sum()
                ),
                "support_voxels": int(
                    support_mask.sum()
                ),
                "selected_voxels": N_SELECTED_VOXELS,
                "best_C": best_C,
                "best_gamma": best_gamma,
                "best_inner_auc": best_inner_auc,
                "outer_auc": float(outer_auc),
                "runtime_minutes": (
                    (time.time() - fold_start) / 60.0
                ),
            }
        )

        log(
            f"Outer fold {outer_fold} completed in "
            f"{(time.time() - fold_start) / 60.0:.2f} minutes."
        )

    # ========================================================
    # FINAL OOF RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("EXP14 FINAL RESULTS")
    log("=" * 70)

    oof = pd.DataFrame(oof_rows)

    # --------------------------------------------------------
    # OOF integrity
    # --------------------------------------------------------

    if len(oof) != N_SUBJECTS_EXPECTED:
        raise ValueError(
            f"Expected {N_SUBJECTS_EXPECTED} OOF rows, "
            f"found {len(oof)}"
        )

    unique_oof = oof["subject_id"].nunique()

    if unique_oof != N_SUBJECTS_EXPECTED:
        raise ValueError(
            f"Expected {N_SUBJECTS_EXPECTED} unique OOF subjects, "
            f"found {unique_oof}"
        )

    if not np.all(
        np.isfinite(oof["decision_score"].to_numpy())
    ):
        raise ValueError(
            "OOF decision scores contain non-finite values."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    y_true = oof["label"].to_numpy(dtype=int)

    scores = oof[
        "decision_score"
    ].to_numpy(dtype=float)

    y_pred = (
        scores >= 0
    ).astype(int)

    oof_auc = roc_auc_score(
        y_true,
        scores,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    specificity = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    outer_auc_values = np.array(
        [
            row["outer_auc"]
            for row in outer_summary
        ],
        dtype=float,
    )

    mean_fold_auc = float(
        np.mean(outer_auc_values)
    )

    fold_auc_sd = float(
        np.std(
            outer_auc_values,
            ddof=0,
        )
    )

    log(
        f"Subjects:              {len(oof)}"
    )

    log(
        f"Controls:              "
        f"{int((y_true == 0).sum())}"
    )

    log(
        f"Schizophrenia:         "
        f"{int((y_true == 1).sum())}"
    )

    log(
        f"Selected voxels/fold:  "
        f"{N_SELECTED_VOXELS}"
    )

    log(
        f"Outer fold AUCs:       "
        f"{outer_auc_values.tolist()}"
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
        f"{oof_auc:.6f}"
    )

    log(
        f"OOF Accuracy:          "
        f"{accuracy:.6f}"
    )

    log(
        f"OOF Balanced Accuracy: "
        f"{balanced_accuracy:.6f}"
    )

    log(
        f"OOF Sensitivity:       "
        f"{sensitivity:.6f}"
    )

    log(
        f"OOF Specificity:       "
        f"{specificity:.6f}"
    )

    log(
        f"OOF Precision:         "
        f"{precision:.6f}"
    )

    log(
        f"OOF F1:                "
        f"{f1:.6f}"
    )

    log(
        f"OOF rows:              "
        f"{len(oof)}"
    )

    log(
        f"Unique OOF subjects:   "
        f"{unique_oof}"
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    oof_path = (
        OOF_DIR
        / "exp14_vbm_rbf_svm_oof.csv"
    )

    outer_summary_path = (
        OUTPUT_ROOT
        / "outer_fold_summary.csv"
    )

    summary_json_path = (
        OUTPUT_ROOT
        / "exp14_summary.json"
    )

    oof = oof.sort_values(
        "subject_id"
    ).reset_index(drop=True)

    oof.to_csv(
        oof_path,
        index=False,
    )

    outer_summary_df = pd.DataFrame(
        outer_summary
    )

    outer_summary_df.to_csv(
        outer_summary_path,
        index=False,
    )

    summary = {
        "experiment": "Exp14",
        "description": (
            "VBM + RBF SVM with strict "
            "outer-training spatial support mask"
        ),
        "subjects": N_SUBJECTS_EXPECTED,
        "controls": N_CONTROLS_EXPECTED,
        "schizophrenia": N_SCHIZOPHRENIA_EXPECTED,
        "nested_cv": "5x4",
        "random_state": RANDOM_STATE,
        "vbm_shape": list(VBM_SHAPE),
        "selected_voxels": N_SELECTED_VOXELS,
        "spatial_support_threshold": (
            SPATIAL_SUPPORT_THRESHOLD
        ),
        "C_grid": C_GRID,
        "gamma_grid": [
            str(value)
            for value in GAMMA_GRID
        ],
        "outer_fold_auc": [
            float(value)
            for value in outer_auc_values
        ],
        "mean_fold_auc": mean_fold_auc,
        "fold_auc_sd": fold_auc_sd,
        "oof_roc_auc": float(oof_auc),
        "oof_accuracy": float(accuracy),
        "oof_balanced_accuracy": float(
            balanced_accuracy
        ),
        "oof_sensitivity": float(
            sensitivity
        ),
        "oof_specificity": float(
            specificity
        ),
        "oof_precision": float(
            precision
        ),
        "oof_f1": float(f1),
        "oof_rows": len(oof),
        "unique_oof_subjects": unique_oof,
        "cv_integrity": "PASS",
        "spatial_support_leakage_control": (
            "PASS - outer training only"
        ),
    }

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    log("")
    log(
        f"Output: {OUTPUT_ROOT}"
    )

    log(
        f"OOF: {oof_path}"
    )

    log(
        f"Outer summary: {outer_summary_path}"
    )

    log(
        f"Summary JSON: {summary_json_path}"
    )

    log("")
    log(
        f"Total runtime: "
        f"{(time.time() - total_start) / 60.0:.2f} minutes"
    )

    log("")
    log("=" * 70)
    log("EXP14 COMPLETED")
    log("=" * 70)


if __name__ == "__main__":
    main()