"""
NeuroFusion AI
Experiment 16
Regional GM Volume + Linear SVM
Strict Nested 5x4 Cross-Validation

Scientific question:
Can regional gray-matter volume derived from SPM VBM provide
more robust schizophrenia discrimination than voxel-wise VBM?

Input:
    SPM modulated GM maps:
    mwc1<SUBJECT_ID>_rms_n4.nii

Representation:
    116 AAL regional GM-volume features

Model:
    SelectKBest(f_classif) + StandardScaler + LinearSVC

Leakage control:
    - Canonical 172-subject nested CV assignments
    - RMS subset only
    - Feature selection inside inner CV
    - Scaling inside inner CV
    - SVM hyperparameter selection inside inner CV
    - Outer test subjects never used for fitting/model selection
    - AAL atlas is fixed and label-free

Primary metric:
    OOF ROC-AUC

Cohort:
    119 RMS subjects
    40 controls
    79 schizophrenia
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.ndimage import affine_transform

from sklearn.exceptions import FitFailedWarning
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
from sklearn.svm import LinearSVC


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

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

ATLAS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "atlas"
    / "aal_mni"
    / "AAL_MNI152_2mm.nii.gz"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "vbm_spm"
    / "experiments"
    / "exp16_regional_gm_svm"
)

OOF_DIR = OUTPUT_ROOT / "oof"

OOF_PATH = (
    OOF_DIR
    / "exp16_regional_gm_svm_oof.csv"
)

OUTER_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "outer_fold_summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_ROOT
    / "exp16_summary.json"
)

CONFIG_PATH = (
    OUTPUT_ROOT
    / "experiment_config.json"
)


# =====================================================================
# EXPERIMENT SETTINGS
# =====================================================================

SEED = 42

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

EXPECTED_SUBJECTS = 119
EXPECTED_CONTROLS = 40
EXPECTED_SCHIZOPHRENIA = 79

EXPECTED_AAL_ROIS = 116

# Number of regional features retained inside each inner split.
K_VALUES = [
    20,
    40,
    60,
    80,
    100,
    116,
]

# Linear SVM regularization values.
C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

MAX_ITER = 10000

# =====================================================================
# LOGGING
# =====================================================================


def log(message: str = "") -> None:
    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(
        f"[{timestamp}] {message}",
        flush=True,
    )


# =====================================================================
# DIRECTORY SETUP
# =====================================================================


def prepare_directories() -> None:
    OOF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# MANIFEST LOADING
# =====================================================================


def load_rms_cohort() -> pd.DataFrame:

    log("Loading labeled manifest...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Labeled manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "subject_id",
        "label",
        "acquisition_source",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Manifest is missing required columns:\n"
            f"{sorted(missing)}"
        )

    manifest["subject_id"] = (
        manifest["subject_id"]
        .astype(str)
        .str.strip()
    )

    manifest["label"] = (
        manifest["label"]
        .astype(int)
    )

    manifest["acquisition_source"] = (
        manifest["acquisition_source"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------------
    # RMS only
    # ---------------------------------------------------------------

    rms = manifest[
        manifest["acquisition_source"]
        .str.upper()
        .eq("RMS")
    ].copy()

    rms = (
        rms
        .drop_duplicates(
            subset=["subject_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Cohort validation
    # ---------------------------------------------------------------

    if len(rms) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} RMS subjects, "
            f"found {len(rms)}"
        )

    if rms["subject_id"].duplicated().any():
        raise ValueError(
            "Duplicate RMS subject IDs found."
        )

    labels = (
        rms["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_labels = {
        0: EXPECTED_CONTROLS,
        1: EXPECTED_SCHIZOPHRENIA,
    }

    if labels != expected_labels:
        raise ValueError(
            "Unexpected RMS class counts: "
            f"{labels}; expected {expected_labels}"
        )

    if not set(
        rms["label"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "RMS cohort contains non-binary labels."
        )

    log(
        f"RMS subjects:       {len(rms)}"
    )
    log(
        f"Controls:           "
        f"{(rms['label'] == 0).sum()}"
    )
    log(
        f"Schizophrenia:      "
        f"{(rms['label'] == 1).sum()}"
    )

    return rms


# =====================================================================
# NESTED CV ASSIGNMENTS
# =====================================================================


def load_assignments(
    rms_df: pd.DataFrame,
) -> pd.DataFrame:

    log("Loading canonical nested CV assignments...")

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            "Canonical nested CV assignments not found:\n"
            f"{ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH
    )

    required = {
        "subject_id",
        "label",
        "outer_fold",
        "inner_fold",
        "role",
    }

    missing = (
        required
        - set(assignments.columns)
    )

    if missing:
        raise ValueError(
            "Assignment file is missing columns:\n"
            f"{sorted(missing)}"
        )

    assignments["subject_id"] = (
        assignments["subject_id"]
        .astype(str)
        .str.strip()
    )

    assignments["label"] = (
        assignments["label"]
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Filter to RMS subjects.
    # ---------------------------------------------------------------

    rms_ids = set(
        rms_df["subject_id"]
    )

    assignments = assignments[
        assignments["subject_id"]
        .isin(rms_ids)
    ].copy()

    assignments = (
        assignments
        .reset_index(drop=True)
    )

    expected_rows = (
        EXPECTED_SUBJECTS
        * N_OUTER_FOLDS
        * N_INNER_FOLDS
    )

    if len(assignments) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} RMS assignment rows, "
            f"found {len(assignments)}"
        )

    # ---------------------------------------------------------------
    # Validate assignment labels.
    # ---------------------------------------------------------------

    manifest_labels = dict(
        zip(
            rms_df["subject_id"],
            rms_df["label"],
        )
    )

    mismatches = []

    for _, row in assignments.iterrows():

        sid = str(
            row["subject_id"]
        )

        if int(row["label"]) != int(
            manifest_labels[sid]
        ):
            mismatches.append(sid)

    if mismatches:
        raise ValueError(
            "Assignment labels disagree with manifest for:\n"
            + "\n".join(
                sorted(set(mismatches))
            )
        )

    # ---------------------------------------------------------------
    # Validate roles.
    # ---------------------------------------------------------------

    valid_roles = {
        "INNER_TRAIN",
        "INNER_VALIDATION",
        "OUTER_TEST",
    }

    actual_roles = set(
        assignments["role"]
        .astype(str)
    )

    if not actual_roles.issubset(
        valid_roles
    ):
        raise ValueError(
            "Unexpected CV roles:\n"
            f"{sorted(actual_roles - valid_roles)}"
        )

    # ---------------------------------------------------------------
    # Validate outer folds.
    # ---------------------------------------------------------------

    outer_folds = sorted(
        assignments["outer_fold"]
        .astype(int)
        .unique()
    )

    if outer_folds != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Unexpected outer folds: {outer_folds}"
        )

    # ---------------------------------------------------------------
    # Validate duplicate assignment rows.
    # ---------------------------------------------------------------

    duplicate_mask = assignments.duplicated(
        subset=[
            "subject_id",
            "outer_fold",
            "inner_fold",
            "role",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate nested-CV assignment rows detected."
        )

    log(
        f"Canonical assignments: "
        f"{len(assignments)} RMS rows"
    )

    log(
        "Nested CV assignment validation: PASS"
    )

    return assignments


# =====================================================================
# VBM PATH
# =====================================================================


def get_vbm_path(
    subject_id: str,
) -> Path:

    return (
        VBM_ROOT
        / subject_id
        / "mwc1"
        / f"mwc1{subject_id}_rms_n4.nii"
    )


# =====================================================================
# FIND VBM FILE ROBUSTLY
# =====================================================================


def find_modulated_gm(
    subject_id: str,
) -> Path:
    """
    Return the UNSMOOTHED SPM modulated GM map.

    Exp16 uses the modulated GM image (mwc1) as the source
    for regional GM-volume extraction.

    The smoothed image under:
        <subject>/smooth/

    is deliberately excluded here because it is a downstream
    smoothing product and should not be used as the raw
    regional-volume input.
    """

    subject_root = (
        VBM_ROOT
        / subject_id
    )

    # Primary expected location.
    direct_path = (
        subject_root
        / f"mwc1{subject_id}_rms_n4.nii"
    )

    if direct_path.exists():
        return direct_path

    # Fallback: search only outside the smooth directory.
    candidates = []

    for path in subject_root.rglob(
        f"mwc1{subject_id}_rms_n4.nii"
    ):
        try:
            relative_parts = path.relative_to(
                subject_root
            ).parts
        except ValueError:
            continue

        # Exclude any file inside the smooth directory.
        if "smooth" in {
            part.lower()
            for part in relative_parts[:-1]
        }:
            continue

        candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "Unsmooth modulated GM file not found for "
            f"{subject_id}.\n"
            f"Expected: {direct_path}"
        )

    raise RuntimeError(
        "Multiple UNSMOOTHED modulated GM files found for "
        f"{subject_id}:\n"
        + "\n".join(
            str(p)
            for p in candidates
        )
    )

    expected = get_vbm_path(
        subject_id
    )

    if expected.exists():
        return expected

    subject_root = (
        VBM_ROOT
        / subject_id
    )

    candidates = list(
        subject_root.rglob(
            f"mwc1{subject_id}_rms_n4.nii"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "Modulated GM file not found for "
            f"{subject_id}:\n"
            f"Expected: {expected}"
        )

    raise RuntimeError(
        "Multiple modulated GM files found for "
        f"{subject_id}:\n"
        + "\n".join(
            str(p)
            for p in candidates
        )
    )


# =====================================================================
# ATLAS LOADING
# =====================================================================


def load_atlas() -> nib.Nifti1Image:

    log("Loading verified AAL atlas...")

    if not ATLAS_PATH.exists():
        raise FileNotFoundError(
            "AAL atlas not found:\n"
            f"{ATLAS_PATH}"
        )

    atlas_img = nib.load(
        str(ATLAS_PATH)
    )

    atlas_data = np.asanyarray(
        atlas_img.dataobj
    )

    finite = np.isfinite(
        atlas_data
    ).all()

    if not finite:
        raise ValueError(
            "AAL atlas contains non-finite values."
        )

    labels = np.unique(
        atlas_data
    )

    labels = labels[
        labels > 0
    ]

    if len(labels) != EXPECTED_AAL_ROIS:
        raise ValueError(
            f"Expected {EXPECTED_AAL_ROIS} "
            f"non-zero AAL labels, found {len(labels)}"
        )

    log(
        f"AAL non-zero ROI labels: "
        f"{len(labels)}"
    )

    log(
        f"AAL label range: "
        f"{int(labels.min())} - "
        f"{int(labels.max())}"
    )

    return atlas_img


# =====================================================================
# ATLAS RESAMPLING
# =====================================================================


def resample_atlas_to_image(
    atlas_img: nib.Nifti1Image,
    target_img: nib.Nifti1Image,
) -> np.ndarray:
    """
    Resample discrete AAL labels into the VBM image grid.

    Nearest-neighbor interpolation is mandatory for atlas labels.
    """

    atlas_data = np.asanyarray(
        atlas_img.dataobj
    )

    target_shape = (
        target_img.shape[:3]
    )

    # target voxel coordinates -> atlas voxel coordinates
    transform = (
        np.linalg.inv(
            atlas_img.affine
        )
        @ target_img.affine
    )

    matrix = transform[:3, :3]
    offset = transform[:3, 3]

    resampled = affine_transform(
        atlas_data,
        matrix=matrix,
        offset=offset,
        output_shape=target_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    return resampled


# =====================================================================
# REGIONAL GM VOLUME EXTRACTION
# =====================================================================


def extract_regional_gm_volume(
    subject_id: str,
    atlas_img: nib.Nifti1Image,
) -> tuple[np.ndarray, list[str]]:

    vbm_path = find_modulated_gm(
        subject_id
    )

    img = nib.load(
        str(vbm_path)
    )

    data = np.asarray(
        img.get_fdata(
            dtype=np.float32
        )
    )

    if data.ndim != 3:
        raise ValueError(
            f"{subject_id}: VBM image is not 3D: "
            f"{data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"{subject_id}: VBM contains non-finite values."
        )

    atlas = resample_atlas_to_image(
        atlas_img,
        img,
    )

    atlas_labels = np.unique(
        atlas
    )

    atlas_labels = atlas_labels[
        atlas_labels > 0
    ]

    if len(atlas_labels) != EXPECTED_AAL_ROIS:
        raise ValueError(
            f"{subject_id}: resampled atlas has "
            f"{len(atlas_labels)} ROIs; expected "
            f"{EXPECTED_AAL_ROIS}"
        )

    # ---------------------------------------------------------------
    # Voxel volume in mm^3.
    #
    # For modulated VBM data, summing the modulated GM values over
    # each ROI and multiplying by voxel volume gives a regional
    # GM-volume-like measure in the normalized space.
    # ---------------------------------------------------------------

    voxel_volume_mm3 = abs(
        np.linalg.det(
            img.affine[:3, :3]
        )
    )

    if not np.isfinite(
        voxel_volume_mm3
    ) or voxel_volume_mm3 <= 0:
        raise ValueError(
            f"{subject_id}: invalid voxel volume "
            f"{voxel_volume_mm3}"
        )

    features = []

    feature_names = []

    for label_value in sorted(
        atlas_labels
    ):

        mask = (
            atlas == label_value
        )

        values = data[mask]

        if values.size == 0:
            raise ValueError(
                f"{subject_id}: ROI "
                f"{int(label_value)} contains no voxels."
            )

        # Numerical safety.
        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            raise ValueError(
                f"{subject_id}: ROI "
                f"{int(label_value)} has no finite voxels."
            )

        regional_volume = (
            float(
                np.sum(values)
            )
            * voxel_volume_mm3
        )

        if not np.isfinite(
            regional_volume
        ):
            raise ValueError(
                f"{subject_id}: ROI "
                f"{int(label_value)} produced "
                "non-finite volume."
            )

        features.append(
            regional_volume
        )

        feature_names.append(
            f"GMVOL_AAL_{int(label_value)}"
        )

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    return (
        features,
        feature_names,
    )


# =====================================================================
# BUILD COMPLETE FEATURE MATRIX
# =====================================================================


def build_feature_matrix(
    rms_df: pd.DataFrame,
    atlas_img: nib.Nifti1Image,
) -> tuple[np.ndarray, list[str]]:

    log("")
    log(
        "Extracting regional GM-volume features..."
    )

    feature_rows = []
    feature_names = None

    start = time.time()

    for index, subject_id in enumerate(
        rms_df["subject_id"],
        start=1,
    ):

        subject_id = str(
            subject_id
        )

        features, names = (
            extract_regional_gm_volume(
                subject_id,
                atlas_img,
            )
        )

        if feature_names is None:
            feature_names = names

        elif names != feature_names:
            raise ValueError(
                f"{subject_id}: feature-column mismatch."
            )

        feature_rows.append(
            features
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(rms_df)
        ):
            log(
                f"Feature extraction: "
                f"{index}/{len(rms_df)}"
            )

    X = np.vstack(
        feature_rows
    )

    if X.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_AAL_ROIS,
    ):
        raise ValueError(
            "Unexpected regional GM matrix shape: "
            f"{X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Regional GM matrix contains non-finite values."
        )

    log(
        f"Regional GM matrix: {X.shape}"
    )

    log(
        f"Extraction runtime: "
        f"{(time.time() - start) / 60:.2f} minutes"
    )

    return X, feature_names


# =====================================================================
# MODEL CONSTRUCTION
# =====================================================================


def make_pipeline() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "select",
                SelectKBest(
                    score_func=f_classif
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=MAX_ITER,
                ),
            ),
        ]
    )


# =====================================================================
# INNER CV VECTOR
# =====================================================================


def make_inner_predefined_split(
    outer_train_ids: set[str],
    outer_fold: int,
    assignments: pd.DataFrame,
    subject_order: list[str],
) -> np.ndarray:

    inner_vector = []

    for subject_id in subject_order:

        subject_id = str(
            subject_id
        )

        if subject_id not in outer_train_ids:
            raise ValueError(
                "Subject outside outer training set "
                "appeared in inner CV vector."
            )

        rows = assignments[
            (
                assignments["subject_id"]
                == subject_id
            )
            & (
                assignments["outer_fold"]
                == outer_fold
            )
        ]

        if len(rows) != N_INNER_FOLDS:
            raise ValueError(
                f"{subject_id}: expected "
                f"{N_INNER_FOLDS} inner assignment rows "
                f"for outer fold {outer_fold}, "
                f"found {len(rows)}"
            )

        roles = (
            rows["role"]
            .astype(str)
            .tolist()
        )

        validation_rows = rows[
            rows["role"]
            == "INNER_VALIDATION"
        ]

        train_rows = rows[
            rows["role"]
            == "INNER_TRAIN"
        ]

        # A subject has one row as INNER_VALIDATION
        # and three rows as INNER_TRAIN across the
        # four inner folds.
        if len(validation_rows) != 1:
            raise ValueError(
                f"{subject_id}: expected exactly one "
                "INNER_VALIDATION row."
            )

        # inner_fold identifies which validation
        # partition this subject belongs to.
        validation_inner_fold = int(
            validation_rows.iloc[0][
                "inner_fold"
            ]
        )

        if validation_inner_fold < 1:
            raise ValueError(
                f"{subject_id}: invalid inner fold."
            )

        inner_vector.append(
            validation_inner_fold - 1
        )

    return np.asarray(
        inner_vector,
        dtype=int,
    )


# =====================================================================
# OUTER FOLD
# =====================================================================


def run_outer_fold(
    outer_fold: int,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    assignments: pd.DataFrame,
    feature_names: list[str],
) -> dict:

    log("")
    log("=" * 70)
    log(
        f"OUTER FOLD {outer_fold}"
    )
    log("=" * 70)

    fold_start = time.time()

    # ---------------------------------------------------------------
    # Determine outer train/test subjects.
    # ---------------------------------------------------------------

    outer_rows = assignments[
        assignments["outer_fold"]
        == outer_fold
    ]

    outer_test_ids = set(
        outer_rows.loc[
            outer_rows["role"]
            == "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
    )

    outer_train_ids = set(
        outer_rows.loc[
            outer_rows["role"]
            != "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
    )

    if (
        outer_train_ids
        & outer_test_ids
    ):
        raise ValueError(
            f"Outer fold {outer_fold}: "
            "train/test subject overlap."
        )

    if (
        len(outer_train_ids)
        + len(outer_test_ids)
        != EXPECTED_SUBJECTS
    ):
        raise ValueError(
            f"Outer fold {outer_fold}: "
            "unexpected train/test subject count."
        )

    train_indices = [
        i
        for i, sid in enumerate(
            subject_ids
        )
        if sid in outer_train_ids
    ]

    test_indices = [
        i
        for i, sid in enumerate(
            subject_ids
        )
        if sid in outer_test_ids
    ]

    if len(train_indices) == 0:
        raise ValueError(
            f"Outer fold {outer_fold}: empty training set."
        )

    if len(test_indices) == 0:
        raise ValueError(
            f"Outer fold {outer_fold}: empty test set."
        )

    X_train = X[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    X_test = X[
        test_indices
    ]

    y_test = y[
        test_indices
    ]

    train_subject_order = [
        subject_ids[i]
        for i in train_indices
    ]

    test_subject_order = [
        subject_ids[i]
        for i in test_indices
    ]

    log(
        f"Outer train subjects: "
        f"{len(train_indices)}"
    )

    log(
        f"Outer test subjects:  "
        f"{len(test_indices)}"
    )

    log(
        f"Outer train controls: "
        f"{int((y_train == 0).sum())}"
    )

    log(
        f"Outer train schizophrenia: "
        f"{int((y_train == 1).sum())}"
    )

    log(
        f"Outer test controls: "
        f"{int((y_test == 0).sum())}"
    )

    log(
        f"Outer test schizophrenia: "
        f"{int((y_test == 1).sum())}"
    )

    # ---------------------------------------------------------------
    # Inner CV.
    #
    # IMPORTANT:
    # SelectKBest and StandardScaler are inside the Pipeline.
    # Therefore they are fitted independently inside each
    # inner training split.
    # ---------------------------------------------------------------

    inner_vector = make_inner_predefined_split(
        outer_train_ids=outer_train_ids,
        outer_fold=outer_fold,
        assignments=assignments,
        subject_order=train_subject_order,
    )

    inner_cv = PredefinedSplit(
        test_fold=inner_vector
    )

    pipeline = make_pipeline()

    parameter_grid = {
        "select__k": K_VALUES,
        "svm__C": C_VALUES,
    }

    # ---------------------------------------------------------------
    # Fit grid search using ONLY outer-training data.
    # ---------------------------------------------------------------

    log(
        "Running inner nested grid search..."
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="sklearn.feature_selection",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sklearn.feature_selection",
        )
        warnings.filterwarnings(
            "ignore",
            category=FitFailedWarning,
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            error_score=np.nan,
        )

        grid.fit(
            X_train,
            y_train,
        )

    if not np.isfinite(
        grid.best_score_
    ):
        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "inner CV produced no finite best AUC."
        )

    best_params = grid.best_params_

    best_k = int(
        best_params[
            "select__k"
        ]
    )

    best_C = float(
        best_params[
            "svm__C"
        ]
    )

    log(
        f"Best inner k:       {best_k}"
    )

    log(
        f"Best inner C:       {best_C}"
    )

    log(
        f"Best inner ROC-AUC: "
        f"{grid.best_score_:.6f}"
    )

    # ---------------------------------------------------------------
    # Outer test prediction.
    #
    # grid.best_estimator_ has been refit on ALL outer-training
    # subjects using only the selected hyperparameters.
    # ---------------------------------------------------------------

    decision_scores = (
        grid.best_estimator_
        .decision_function(
            X_test
        )
    )

    if not np.isfinite(
        decision_scores
    ).all():
        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "non-finite decision scores."
        )

    outer_auc = roc_auc_score(
        y_test,
        decision_scores,
    )

    log(
        f"Outer fold {outer_fold} ROC-AUC: "
        f"{outer_auc:.6f}"
    )

    # ---------------------------------------------------------------
    # Selected features from the outer-trained final pipeline.
    # ---------------------------------------------------------------

    selector = (
        grid.best_estimator_
        .named_steps["select"]
    )

    selected_mask = (
        selector.get_support()
    )

    selected_names = [
        feature_names[i]
        for i, selected in enumerate(
            selected_mask
        )
        if selected
    ]

    # ---------------------------------------------------------------
    # Save selected regional features for audit.
    # ---------------------------------------------------------------

    fold_selected_path = (
        OUTPUT_ROOT
        / f"outer_fold_{outer_fold}_selected_features.json"
    )

    with open(
        fold_selected_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "outer_fold": outer_fold,
                "selected_k": best_k,
                "selected_features": selected_names,
                "n_selected": len(
                    selected_names
                ),
            },
            f,
            indent=2,
        )

    log(
        f"Outer fold {outer_fold} completed in "
        f"{(time.time() - fold_start) / 60:.2f} minutes."
    )

    return {
        "outer_fold": outer_fold,
        "outer_train_n": len(
            train_indices
        ),
        "outer_test_n": len(
            test_indices
        ),
        "outer_test_controls": int(
            (y_test == 0).sum()
        ),
        "outer_test_schizophrenia": int(
            (y_test == 1).sum()
        ),
        "best_k": best_k,
        "best_C": best_C,
        "best_inner_auc": float(
            grid.best_score_
        ),
        "outer_auc": float(
            outer_auc
        ),
        "test_subject_ids": test_subject_order,
        "test_labels": y_test.astype(int).tolist(),
        "test_scores": decision_scores.astype(
            float
        ).tolist(),
        "selected_features": selected_names,
    }


# =====================================================================
# OOF METRICS
# =====================================================================


def calculate_oof_metrics(
    oof: pd.DataFrame,
) -> dict:

    y_true = (
        oof["label"]
        .astype(int)
        .to_numpy()
    )

    scores = (
        oof["decision_score"]
        .astype(float)
        .to_numpy()
    )

    # Default threshold only for secondary metrics.
    predictions = (
        scores >= 0
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        scores,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            predictions,
        )
    )

    sensitivity = recall_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    specificity = recall_score(
        y_true,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    return {
        "oof_roc_auc": float(
            roc_auc
        ),
        "oof_accuracy": float(
            accuracy
        ),
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
        "oof_f1": float(
            f1
        ),
    }


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:

    overall_start = time.time()

    prepare_directories()

    log("=" * 70)
    log("NEUROFUSION AI — EXP16")
    log("REGIONAL GM VOLUME + LINEAR SVM")
    log("=" * 70)

    log(
        f"Project root: {PROJECT_ROOT}"
    )

    # ---------------------------------------------------------------
    # Configuration record.
    # ---------------------------------------------------------------

    config = {
        "experiment": "Exp16",
        "description": (
            "Regional GM Volume + Linear SVM"
        ),
        "scientific_question": (
            "Can regional gray-matter volume derived "
            "from SPM VBM provide robust schizophrenia "
            "discrimination?"
        ),
        "cohort": "RMS-only",
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "representation": (
            "AAL regional volume from modulated GM VBM maps"
        ),
        "input": (
            "SPM mwc1 modulated GM maps"
        ),
        "atlas": str(
            ATLAS_PATH
        ),
        "expected_aal_rois": EXPECTED_AAL_ROIS,
        "feature_definition": (
            "sum of modulated GM values within each AAL ROI "
            "multiplied by voxel volume in mm^3"
        ),
        "feature_selection": (
            "SelectKBest(f_classif), fitted inside inner CV"
        ),
        "k_values": K_VALUES,
        "scaling": (
            "StandardScaler, fitted inside inner CV"
        ),
        "classifier": "LinearSVC",
        "C_values": C_VALUES,
        "class_weight": "balanced",
        "nested_cv": "5x4",
        "seed": SEED,
        "primary_metric": "OOF ROC-AUC",
        "outer_test_untouched": True,
        "acquisition_source": "RMS",
        "source_variation_within_cohort": False,
        "nuisance_covariates": (
            "Not included in this representation experiment; "
            "the existing subject-level manifest does not provide "
            "a reliable one-row-per-subject demographic mapping "
            "for this cohort."
        ),
        "note": (
            "Final clinical model selection requires later "
            "nuisance/confound and robustness analyses."
        ),
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Load cohort.
    # ---------------------------------------------------------------

    rms_df = load_rms_cohort()

    # ---------------------------------------------------------------
    # Load CV.
    # ---------------------------------------------------------------

    assignments = load_assignments(
        rms_df
    )

    # ---------------------------------------------------------------
    # Load atlas.
    # ---------------------------------------------------------------

    atlas_img = load_atlas()

    # ---------------------------------------------------------------
    # Build feature matrix.
    # ---------------------------------------------------------------

    X, feature_names = (
        build_feature_matrix(
            rms_df,
            atlas_img,
        )
    )

    subject_ids = (
        rms_df["subject_id"]
        .astype(str)
        .tolist()
    )

    y = (
        rms_df["label"]
        .astype(int)
        .to_numpy()
    )

    # ---------------------------------------------------------------
    # Save feature matrix for audit.
    #
    # This is a label-free derived representation. The labels are
    # stored separately in the experiment OOF output.
    # ---------------------------------------------------------------

    feature_matrix_path = (
        OUTPUT_ROOT
        / "regional_gm_volume_features.csv"
    )

    feature_df = pd.DataFrame(
        X,
        columns=feature_names,
    )

    feature_df.insert(
        0,
        "subject_id",
        subject_ids,
    )

    feature_df.to_csv(
        feature_matrix_path,
        index=False,
    )

    log(
        f"Feature matrix saved:\n"
        f"{feature_matrix_path}"
    )

    # ---------------------------------------------------------------
    # Nested outer CV.
    # ---------------------------------------------------------------

    outer_results = []

    oof_records = []

    for outer_fold in range(
        1,
        N_OUTER_FOLDS + 1,
    ):

        result = run_outer_fold(
            outer_fold=outer_fold,
            X=X,
            y=y,
            subject_ids=subject_ids,
            assignments=assignments,
            feature_names=feature_names,
        )

        outer_results.append(
            result
        )

        for (
            sid,
            label,
            score,
        ) in zip(
            result["test_subject_ids"],
            result["test_labels"],
            result["test_scores"],
        ):

            oof_records.append(
                {
                    "subject_id": sid,
                    "label": int(label),
                    "decision_score": float(
                        score
                    ),
                    "outer_fold": int(
                        outer_fold
                    ),
                }
            )

    # ---------------------------------------------------------------
    # OOF dataframe.
    # ---------------------------------------------------------------

    oof = pd.DataFrame(
        oof_records
    )

    # ---------------------------------------------------------------
    # OOF integrity.
    # ---------------------------------------------------------------

    if len(oof) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} OOF rows, "
            f"found {len(oof)}"
        )

    if (
        oof["subject_id"]
        .nunique()
        != EXPECTED_SUBJECTS
    ):
        raise ValueError(
            "OOF subject IDs are not unique."
        )

    if oof["subject_id"].duplicated().any():
        raise ValueError(
            "Duplicate OOF subjects detected."
        )

    if not np.isfinite(
        oof["decision_score"]
        .to_numpy()
    ).all():
        raise ValueError(
            "OOF contains non-finite decision scores."
        )

    # Ensure all canonical RMS subjects appear.
    if set(
        oof["subject_id"]
    ) != set(subject_ids):
        missing = (
            set(subject_ids)
            - set(oof["subject_id"])
        )

        extra = (
            set(oof["subject_id"])
            - set(subject_ids)
        )

        raise ValueError(
            "OOF subject mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    # ---------------------------------------------------------------
    # Save OOF.
    # ---------------------------------------------------------------

    oof = oof.sort_values(
        "subject_id"
    ).reset_index(
        drop=True
    )

    oof.to_csv(
        OOF_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Outer summary.
    # ---------------------------------------------------------------

    outer_summary_rows = []

    for result in outer_results:

        outer_summary_rows.append(
            {
                "outer_fold": result[
                    "outer_fold"
                ],
                "outer_train_n": result[
                    "outer_train_n"
                ],
                "outer_test_n": result[
                    "outer_test_n"
                ],
                "outer_test_controls": result[
                    "outer_test_controls"
                ],
                "outer_test_schizophrenia": result[
                    "outer_test_schizophrenia"
                ],
                "best_k": result[
                    "best_k"
                ],
                "best_C": result[
                    "best_C"
                ],
                "best_inner_auc": result[
                    "best_inner_auc"
                ],
                "outer_auc": result[
                    "outer_auc"
                ],
            }
        )

    outer_summary = pd.DataFrame(
        outer_summary_rows
    )

    outer_summary.to_csv(
        OUTER_SUMMARY_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Final metrics.
    # ---------------------------------------------------------------

    metrics = calculate_oof_metrics(
        oof
    )

    fold_aucs = [
        float(
            result["outer_auc"]
        )
        for result in outer_results
    ]

    mean_fold_auc = float(
        np.mean(fold_aucs)
    )

    fold_auc_sd = float(
        np.std(
            fold_aucs,
            ddof=0,
        )
    )

    # ---------------------------------------------------------------
    # Final summary JSON.
    # ---------------------------------------------------------------

    summary = {
        "experiment": "Exp16",
        "description": (
            "Regional GM Volume + Linear SVM"
        ),
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "aal_features": EXPECTED_AAL_ROIS,
        "feature_definition": (
            "Regional sum of modulated GM values "
            "times voxel volume"
        ),
        "outer_fold_aucs": fold_aucs,
        "mean_fold_auc": mean_fold_auc,
        "fold_auc_sd": fold_auc_sd,
        **metrics,
        "oof_rows": int(len(oof)),
        "unique_oof_subjects": int(
            oof["subject_id"].nunique()
        ),
        "oof_integrity": "PASS",
        "nested_cv": "5x4",
        "outer_test_untouched": True,
        "feature_selection_inside_cv": True,
        "scaling_inside_cv": True,
        "hyperparameter_selection_inside_cv": True,
        "seed": SEED,
        "feature_matrix": str(
            feature_matrix_path
        ),
        "oof_path": str(
            OOF_PATH
        ),
        "outer_summary_path": str(
            OUTER_SUMMARY_PATH
        ),
        "runtime_minutes": (
            (time.time() - overall_start)
            / 60.0
        ),
    }

    with open(
        SUMMARY_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # FINAL REPORT.
    # ---------------------------------------------------------------

    log("")
    log("=" * 70)
    log("EXP16 FINAL RESULTS")
    log("=" * 70)

    log(
        f"Subjects:              {EXPECTED_SUBJECTS}"
    )

    log(
        f"Controls:              {EXPECTED_CONTROLS}"
    )

    log(
        f"Schizophrenia:         {EXPECTED_SCHIZOPHRENIA}"
    )

    log(
        f"Regional GM features:  {EXPECTED_AAL_ROIS}"
    )

    log(
        "Outer fold AUCs:       "
        + str(
            [
                round(
                    x,
                    6,
                )
                for x in fold_aucs
            ]
        )
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
        f"{metrics['oof_roc_auc']:.6f}"
    )

    log(
        f"OOF Accuracy:          "
        f"{metrics['oof_accuracy']:.6f}"
    )

    log(
        f"OOF Balanced Accuracy: "
        f"{metrics['oof_balanced_accuracy']:.6f}"
    )

    log(
        f"OOF Sensitivity:       "
        f"{metrics['oof_sensitivity']:.6f}"
    )

    log(
        f"OOF Specificity:       "
        f"{metrics['oof_specificity']:.6f}"
    )

    log(
        f"OOF Precision:         "
        f"{metrics['oof_precision']:.6f}"
    )

    log(
        f"OOF F1:                "
        f"{metrics['oof_f1']:.6f}"
    )

    log(
        f"OOF rows:              "
        f"{len(oof)}"
    )

    log(
        f"Unique OOF subjects:   "
        f"{oof['subject_id'].nunique()}"
    )

    log(
        "OOF integrity:         PASS"
    )

    log("")
    log(
        f"Output: {OUTPUT_ROOT}"
    )

    log(
        f"Feature matrix: {feature_matrix_path}"
    )

    log(
        f"OOF: {OOF_PATH}"
    )

    log(
        f"Outer summary: {OUTER_SUMMARY_PATH}"
    )

    log(
        f"Summary JSON: {SUMMARY_JSON_PATH}"
    )

    log("")
    log(
        f"Total runtime: "
        f"{(time.time() - overall_start) / 60:.2f} minutes"
    )

    log("")
    log("=" * 70)
    log("EXP16 COMPLETED")
    log("=" * 70)


if __name__ == "__main__":
    main()