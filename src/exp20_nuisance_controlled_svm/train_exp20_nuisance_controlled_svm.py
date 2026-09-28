"""
NeuroFusion AI
Experiment 20
Nuisance-Controlled Regional GM Volume + Linear SVM

Purpose
-------
Robustness/control analysis of Exp16.

Scientific question
-------------------
Does the regional GM-volume signal observed in Exp16 remain
useful after controlling for available demographic nuisance
variables (age and gender) without allowing nuisance estimation
to use validation/test subjects?

Cohort
------
119 RMS subjects
40 controls
79 schizophrenia

Representation
--------------
116 AAL regional GM-volume features extracted from SPM
modulated GM VBM maps.

Model
-----
Training-fold nuisance residualization
+
SelectKBest(f_classif)
+
StandardScaler
+
LinearSVC(class_weight="balanced")

Nested CV
---------
Canonical 5 outer x 4 inner CV.

Leakage control
---------------
1. RMS-only cohort.
2. Canonical CV assignments.
3. Age/gender nuisance coefficients are fitted only on the
   relevant training partition.
4. Feature selection is inside the pipeline.
5. Scaling is inside the pipeline.
6. SVM C selection is inside inner CV.
7. Outer-test subjects are never used for fitting or selection.

Important demographic limitation
--------------------------------
The available demographic table contains 143 records for the
119 RMS subjects, with some subjects having multiple visit
records. The MRI manifest has no visit identifier.

Therefore:
- age is aggregated by subject using the median across available
  demographic records;
- gender must be consistent within a subject; conflicting gender
  records cause the experiment to stop rather than guessing;
- ICV is NOT included because no validated one-row-per-subject ICV
  mapping is currently available.

Primary metric
--------------
OOF ROC-AUC

This experiment does NOT replace Exp16.
It tests whether Exp16's signal is robust to available
demographic nuisance control.
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

from sklearn.base import BaseEstimator, TransformerMixin
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

DEMOGRAPHICS_PATH = (
    PROJECT_ROOT
    / "temp_folder"
    / "1139_Demographics_20260804.csv"
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
    / "exp20_nuisance_controlled_svm"
)

OOF_DIR = OUTPUT_ROOT / "oof"

OOF_PATH = (
    OOF_DIR
    / "exp20_nuisance_controlled_svm_oof.csv"
)

OUTER_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "outer_fold_summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_ROOT
    / "exp20_summary.json"
)

CONFIG_PATH = (
    OUTPUT_ROOT
    / "experiment_config.json"
)

DEMOGRAPHIC_AUDIT_PATH = (
    OUTPUT_ROOT
    / "demographic_mapping_audit.csv"
)


# =====================================================================
# SETTINGS
# =====================================================================

SEED = 42

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

EXPECTED_SUBJECTS = 119
EXPECTED_CONTROLS = 40
EXPECTED_SCHIZOPHRENIA = 79
EXPECTED_AAL_ROIS = 116

K_VALUES = [
    20,
    40,
    60,
    80,
    100,
    116,
]

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
# DIRECTORIES
# =====================================================================

def prepare_directories() -> None:

    OOF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# MANIFEST
# =====================================================================

def load_rms_cohort() -> pd.DataFrame:

    log("Loading labeled manifest...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
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
            f"Manifest missing columns: {sorted(missing)}"
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

    if len(rms) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} RMS subjects, "
            f"found {len(rms)}"
        )

    if rms["subject_id"].duplicated().any():
        raise ValueError(
            "Duplicate RMS subject IDs detected."
        )

    labels = (
        rms["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected = {
        0: EXPECTED_CONTROLS,
        1: EXPECTED_SCHIZOPHRENIA,
    }

    if labels != expected:
        raise ValueError(
            f"Unexpected class counts: {labels}; "
            f"expected {expected}"
        )

    log(
        f"RMS subjects: {len(rms)}"
    )

    log(
        f"Controls: {(rms['label'] == 0).sum()}"
    )

    log(
        f"Schizophrenia: {(rms['label'] == 1).sum()}"
    )

    return rms


# =====================================================================
# CANONICAL CV ASSIGNMENTS
# =====================================================================

def load_assignments(
    rms_df: pd.DataFrame,
) -> pd.DataFrame:

    log("Loading canonical nested CV assignments...")

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Assignments not found:\n{ASSIGNMENTS_PATH}"
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
            f"Assignment file missing columns: {sorted(missing)}"
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

    manifest_labels = dict(
        zip(
            rms_df["subject_id"],
            rms_df["label"],
        )
    )

    for _, row in assignments.iterrows():

        sid = str(
            row["subject_id"]
        )

        if int(row["label"]) != int(
            manifest_labels[sid]
        ):
            raise ValueError(
                f"Assignment label mismatch for {sid}"
            )

    valid_roles = {
        "INNER_TRAIN",
        "INNER_VALIDATION",
        "OUTER_TEST",
    }

    roles = set(
        assignments["role"]
        .astype(str)
    )

    if not roles.issubset(
        valid_roles
    ):
        raise ValueError(
            f"Unexpected roles: {roles - valid_roles}"
        )

    outer_folds = sorted(
        assignments["outer_fold"]
        .astype(int)
        .unique()
    )

    if outer_folds != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Unexpected outer folds: {outer_folds}"
        )

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
            "Duplicate canonical assignment rows detected."
        )

    log(
        f"Canonical RMS assignments: {len(assignments)} rows"
    )

    log(
        "Nested CV assignment validation: PASS"
    )

    return assignments


# =====================================================================
# DEMOGRAPHIC COLUMN DETECTION
# =====================================================================

def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:

    normalized = {
        str(c).strip().lower().replace(" ", "").replace("_", ""):
        c
        for c in columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if key in normalized:
            return normalized[key]

    return None


# =====================================================================
# DEMOGRAPHIC LOADING
# =====================================================================

def load_demographics(
    rms_df: pd.DataFrame,
) -> pd.DataFrame:

    log("")
    log("Loading demographic nuisance variables...")

    if not DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(
            "Demographic file not found:\n"
            f"{DEMOGRAPHICS_PATH}"
        )

    demo = pd.read_csv(
        DEMOGRAPHICS_PATH
    )

    log(
        f"Demographic rows: {len(demo)}"
    )

    log(
        f"Demographic columns: {list(demo.columns)}"
    )

    id_col = find_column(
        list(demo.columns),
        [
            "Anonymized ID",
            "ID",
            "Subject ID",
            "Subject_ID",
            "subject_id",
        ],
    )

    age_col = find_column(
        list(demo.columns),
        [
            "Current Age",
            "Age",
            "age",
        ],
    )

    gender_col = find_column(
        list(demo.columns),
        [
            "Gender",
            "gender",
            "Sex",
            "sex",
        ],
    )

    if id_col is None:
        raise ValueError(
            "Could not identify demographic subject-ID column.\n"
            f"Columns: {list(demo.columns)}"
        )

    if age_col is None:
        raise ValueError(
            "Could not identify demographic age column.\n"
            f"Columns: {list(demo.columns)}"
        )

    if gender_col is None:
        raise ValueError(
            "Could not identify demographic gender/sex column.\n"
            f"Columns: {list(demo.columns)}"
        )

    log(
        f"Demographic ID column: {id_col}"
    )

    log(
        f"Age column: {age_col}"
    )

    log(
        f"Gender column: {gender_col}"
    )

    demo = demo.copy()

    demo["subject_id"] = (
        demo[id_col]
        .astype(str)
        .str.strip()
    )

    demo["age_numeric"] = pd.to_numeric(
        demo[age_col],
        errors="coerce",
    )

    demo["gender_raw"] = (
        demo[gender_col]
        .astype(str)
        .str.strip()
    )

    rms_ids = set(
        rms_df["subject_id"]
    )

    demo = demo[
        demo["subject_id"]
        .isin(rms_ids)
    ].copy()

    if len(demo) == 0:
        raise ValueError(
            "No RMS subjects matched the demographic table."
        )

    # ---------------------------------------------------------------
    # Aggregate repeated demographic rows by subject.
    #
    # Age:
    #   median across available records.
    #
    # Gender:
    #   must be consistent. If conflicting values occur, stop.
    # ---------------------------------------------------------------

    records = []

    for sid, group in demo.groupby(
        "subject_id",
        sort=False,
    ):

        age_values = (
            group["age_numeric"]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(age_values) == 0:
            raise ValueError(
                f"{sid}: no valid age value."
            )

        age_value = float(
            np.median(age_values)
        )

        gender_values = (
            group["gender_raw"]
            .replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "None": np.nan,
                }
            )
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_gender = sorted(
            set(
                gender_values
            )
        )

        if len(unique_gender) == 0:
            raise ValueError(
                f"{sid}: no valid gender value."
            )

        if len(unique_gender) > 1:
            raise ValueError(
                f"{sid}: conflicting gender values "
                f"across demographic records: "
                f"{unique_gender}\n"
                "The experiment will not guess which visit is correct."
            )

        records.append(
            {
                "subject_id": sid,
                "age": age_value,
                "gender": unique_gender[0],
                "demographic_records": int(
                    len(group)
                ),
            }
        )

    nuisance = pd.DataFrame(
        records
    )

    # ---------------------------------------------------------------
    # Require exactly 119 subjects.
    # ---------------------------------------------------------------

    if len(nuisance) != EXPECTED_SUBJECTS:
        missing = (
            rms_ids
            - set(nuisance["subject_id"])
        )

        extra = (
            set(nuisance["subject_id"])
            - rms_ids
        )

        raise ValueError(
            "Demographic subject mismatch.\n"
            f"Expected: {EXPECTED_SUBJECTS}\n"
            f"Found: {len(nuisance)}\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    if nuisance["subject_id"].duplicated().any():
        raise ValueError(
            "Duplicate subject IDs remain after demographic aggregation."
        )

    if not np.isfinite(
        nuisance["age"]
        .to_numpy()
    ).all():
        raise ValueError(
            "Non-finite age values remain."
        )

    # ---------------------------------------------------------------
    # Save demographic audit.
    # ---------------------------------------------------------------

    audit = rms_df[
        [
            "subject_id",
            "label",
            "acquisition_source",
        ]
    ].merge(
        nuisance,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    audit.to_csv(
        DEMOGRAPHIC_AUDIT_PATH,
        index=False,
    )

    log(
        f"Matched demographic subjects: {len(audit)}"
    )

    log(
        f"Subjects with >1 demographic record: "
        f"{int((audit['demographic_records'] > 1).sum())}"
    )

    log(
        f"Age range: "
        f"{audit['age'].min():.2f} - "
        f"{audit['age'].max():.2f}"
    )

    log(
        "Gender values: "
        + str(
            sorted(
                audit["gender"]
                .astype(str)
                .unique()
                .tolist()
            )
        )
    )

    log(
        "Demographic mapping validation: PASS"
    )

    return nuisance


# =====================================================================
# NUISANCE MATRIX
# =====================================================================

def build_nuisance_matrix(
    subject_ids: list[str],
    nuisance_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:

    nuisance = nuisance_df.set_index(
        "subject_id"
    ).loc[
        subject_ids
    ].copy()

    if nuisance.isnull().any().any():
        raise ValueError(
            "Missing nuisance values after subject alignment."
        )

    # ---------------------------------------------------------------
    # Age is continuous.
    #
    # Gender is categorical and one-hot encoded.
    #
    # We intentionally do NOT interpret the biological meaning of
    # the source gender codes here.
    # ---------------------------------------------------------------

    gender_dummies = pd.get_dummies(
        nuisance["gender"].astype(str),
        prefix="gender",
        drop_first=True,
        dtype=float,
    )

    age = (
        nuisance[["age"]]
        .astype(float)
    )

    nuisance_encoded = pd.concat(
        [
            age,
            gender_dummies,
        ],
        axis=1,
    )

    nuisance_encoded = nuisance_encoded.astype(
        float
    )

    Z = nuisance_encoded.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(Z).all():
        raise ValueError(
            "Nuisance matrix contains non-finite values."
        )

    return (
        Z,
        nuisance_encoded.columns.tolist(),
    )


# =====================================================================
# VBM FILE
# =====================================================================

def find_modulated_gm(
    subject_id: str,
) -> Path:

    subject_root = (
        VBM_ROOT
        / subject_id
    )

    direct_path = (
        subject_root
        / f"mwc1{subject_id}_rms_n4.nii"
    )

    if direct_path.exists():
        return direct_path

    candidates = []

    for path in subject_root.rglob(
        f"mwc1{subject_id}_rms_n4.nii"
    ):

        relative_parts = (
            path
            .relative_to(subject_root)
            .parts
        )

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
            f"Unsmooth modulated GM file not found for {subject_id}.\n"
            f"Expected: {direct_path}"
        )

    raise RuntimeError(
        f"Multiple unsmoothed GM files found for {subject_id}:\n"
        + "\n".join(
            str(p)
            for p in candidates
        )
    )


# =====================================================================
# ATLAS
# =====================================================================

def load_atlas() -> nib.Nifti1Image:

    log("Loading AAL atlas...")

    if not ATLAS_PATH.exists():
        raise FileNotFoundError(
            f"AAL atlas not found:\n{ATLAS_PATH}"
        )

    atlas_img = nib.load(
        str(ATLAS_PATH)
    )

    atlas_data = np.asanyarray(
        atlas_img.dataobj
    )

    if not np.isfinite(
        atlas_data
    ).all():
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
            f"Expected {EXPECTED_AAL_ROIS} AAL ROIs, "
            f"found {len(labels)}"
        )

    log(
        f"AAL ROIs: {len(labels)}"
    )

    return atlas_img


# =====================================================================
# ATLAS RESAMPLING
# =====================================================================

def resample_atlas_to_image(
    atlas_img: nib.Nifti1Image,
    target_img: nib.Nifti1Image,
) -> np.ndarray:

    atlas_data = np.asanyarray(
        atlas_img.dataobj
    )

    transform = (
        np.linalg.inv(
            atlas_img.affine
        )
        @ target_img.affine
    )

    matrix = transform[:3, :3]
    offset = transform[:3, 3]

    return affine_transform(
        atlas_data,
        matrix=matrix,
        offset=offset,
        output_shape=target_img.shape[:3],
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )


# =====================================================================
# REGIONAL GM FEATURES
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
            f"{subject_id}: expected 3D image, "
            f"got {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"{subject_id}: non-finite VBM values."
        )

    atlas = resample_atlas_to_image(
        atlas_img,
        img,
    )

    labels = np.unique(
        atlas
    )

    labels = labels[
        labels > 0
    ]

    if len(labels) != EXPECTED_AAL_ROIS:
        raise ValueError(
            f"{subject_id}: resampled atlas has "
            f"{len(labels)} ROIs; expected {EXPECTED_AAL_ROIS}"
        )

    voxel_volume_mm3 = abs(
        np.linalg.det(
            img.affine[:3, :3]
        )
    )

    if not np.isfinite(
        voxel_volume_mm3
    ) or voxel_volume_mm3 <= 0:
        raise ValueError(
            f"{subject_id}: invalid voxel volume."
        )

    features = []
    names = []

    for label_value in sorted(labels):

        mask = (
            atlas == label_value
        )

        values = data[mask]

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            raise ValueError(
                f"{subject_id}: empty ROI {int(label_value)}"
            )

        volume = (
            float(np.sum(values))
            * voxel_volume_mm3
        )

        if not np.isfinite(volume):
            raise ValueError(
                f"{subject_id}: non-finite ROI volume."
            )

        features.append(
            volume
        )

        names.append(
            f"GMVOL_AAL_{int(label_value)}"
        )

    return (
        np.asarray(
            features,
            dtype=np.float64,
        ),
        names,
    )


# =====================================================================
# FEATURE MATRIX
# =====================================================================

def build_feature_matrix(
    rms_df: pd.DataFrame,
    atlas_img: nib.Nifti1Image,
) -> tuple[np.ndarray, list[str]]:

    log("")
    log("Extracting regional GM-volume features...")

    rows = []
    feature_names = None

    start = time.time()

    for index, sid in enumerate(
        rms_df["subject_id"],
        start=1,
    ):

        features, names = (
            extract_regional_gm_volume(
                str(sid),
                atlas_img,
            )
        )

        if feature_names is None:
            feature_names = names

        elif names != feature_names:
            raise ValueError(
                f"{sid}: feature-name mismatch."
            )

        rows.append(
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

    X = np.vstack(rows)

    if X.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_AAL_ROIS,
    ):
        raise ValueError(
            f"Unexpected feature shape: {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains non-finite values."
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
# NUISANCE RESIDUALIZER
# =====================================================================

class NuisanceResidualizer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Residualize the first n_feature columns against nuisance
    covariates stored in the remaining columns.

    Input format:
        [regional_features | nuisance_variables]

    The nuisance regression is fitted ONLY on the current
    training partition.

    Model:
        X = intercept + Z * beta + residual

    The intercept and beta are estimated using least squares.

    The nuisance variables themselves are never passed onward to
    SelectKBest or the SVM.
    """

    def __init__(
        self,
        n_features: int,
    ):
        self.n_features = n_features

    def fit(
        self,
        X,
        y=None,
    ):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "Residualizer input must be 2D."
            )

        if X.shape[1] <= self.n_features:
            raise ValueError(
                "Residualizer did not receive nuisance columns."
            )

        self.n_nuisance_ = (
            X.shape[1]
            - self.n_features
        )

        features = X[
            :,
            :self.n_features,
        ]

        nuisance = X[
            :,
            self.n_features:,
        ]

        if not np.isfinite(features).all():
            raise ValueError(
                "Feature values are non-finite."
            )

        if not np.isfinite(nuisance).all():
            raise ValueError(
                "Nuisance values are non-finite."
            )

        # Add intercept.
        design = np.column_stack(
            [
                np.ones(
                    len(X),
                    dtype=np.float64,
                ),
                nuisance,
            ]
        )

        # Least-squares coefficients.
        self.coef_ = np.linalg.lstsq(
            design,
            features,
            rcond=None,
        )[0]

        self.n_features_in_ = (
            X.shape[1]
        )

        return self

    def transform(
        self,
        X,
    ):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "Residualizer input must be 2D."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Residualizer input dimension changed."
            )

        features = X[
            :,
            :self.n_features,
        ]

        nuisance = X[
            :,
            self.n_features:,
        ]

        design = np.column_stack(
            [
                np.ones(
                    len(X),
                    dtype=np.float64,
                ),
                nuisance,
            ]
        )

        fitted = (
            design
            @ self.coef_
        )

        residuals = (
            features
            - fitted
        )

        if not np.isfinite(
            residuals
        ).all():
            raise ValueError(
                "Residualized features contain non-finite values."
            )

        return residuals


# =====================================================================
# PIPELINE
# =====================================================================

def make_pipeline(
    n_features: int,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "nuisance",
                NuisanceResidualizer(
                    n_features=n_features
                ),
            ),
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
# INNER CV
# =====================================================================

def make_inner_predefined_split(
    outer_train_ids: set[str],
    outer_fold: int,
    assignments: pd.DataFrame,
    subject_order: list[str],
) -> np.ndarray:

    inner_vector = []

    for sid in subject_order:

        sid = str(sid)

        if sid not in outer_train_ids:
            raise ValueError(
                f"{sid}: outside outer training set."
            )

        rows = assignments[
            (
                assignments["subject_id"]
                == sid
            )
            & (
                assignments["outer_fold"]
                == outer_fold
            )
        ]

        if len(rows) != N_INNER_FOLDS:
            raise ValueError(
                f"{sid}: expected {N_INNER_FOLDS} "
                f"inner rows, found {len(rows)}"
            )

        validation_rows = rows[
            rows["role"]
            == "INNER_VALIDATION"
        ]

        if len(validation_rows) != 1:
            raise ValueError(
                f"{sid}: expected exactly one "
                "INNER_VALIDATION row."
            )

        inner_fold = int(
            validation_rows.iloc[0][
                "inner_fold"
            ]
        )

        if inner_fold not in {
            1,
            2,
            3,
            4,
        }:
            raise ValueError(
                f"{sid}: invalid inner fold {inner_fold}"
            )

        # Canonical assignment uses 1..4.
        # sklearn PredefinedSplit uses 0..3.
        inner_vector.append(
            inner_fold - 1
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
    X_features: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    assignments: pd.DataFrame,
    feature_names: list[str],
    nuisance_names: list[str],
) -> dict:

    log("")
    log("=" * 70)
    log(
        f"EXP20 â€” OUTER FOLD {outer_fold}"
    )
    log("=" * 70)

    fold_start = time.time()

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

    if outer_train_ids & outer_test_ids:
        raise ValueError(
            "Outer train/test overlap detected."
        )

    if (
        len(outer_train_ids)
        + len(outer_test_ids)
        != EXPECTED_SUBJECTS
    ):
        raise ValueError(
            f"Outer fold {outer_fold}: invalid subject count."
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

    X_train_features = X_features[
        train_indices
    ]

    X_test_features = X_features[
        test_indices
    ]

    Z_train = Z[
        train_indices
    ]

    Z_test = Z[
        test_indices
    ]

    y_train = y[
        train_indices
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
        f"Outer train subjects: {len(train_indices)}"
    )

    log(
        f"Outer test subjects:  {len(test_indices)}"
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # The nuisance variables are concatenated to the feature matrix.
    # The NuisanceResidualizer is the FIRST pipeline step.
    #
    # During inner CV, GridSearchCV fits that residualizer separately
    # on every inner training partition.
    #
    # After selecting the best parameters, GridSearchCV refits the
    # whole pipeline on all outer-training subjects.
    #
    # Therefore outer-test nuisance information is only used during
    # transform(), never during fitting.
    # ---------------------------------------------------------------

    X_train_combined = np.hstack(
        [
            X_train_features,
            Z_train,
        ]
    )

    X_test_combined = np.hstack(
        [
            X_test_features,
            Z_test,
        ]
    )

    inner_vector = (
        make_inner_predefined_split(
            outer_train_ids=outer_train_ids,
            outer_fold=outer_fold,
            assignments=assignments,
            subject_order=train_subject_order,
        )
    )

    inner_cv = PredefinedSplit(
        test_fold=inner_vector
    )

    pipeline = make_pipeline(
        n_features=EXPECTED_AAL_ROIS
    )

    parameter_grid = {
        "select__k": K_VALUES,
        "svm__C": C_VALUES,
    }

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
            X_train_combined,
            y_train,
        )

    if not np.isfinite(
        grid.best_score_
    ):
        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "no finite inner-CV result."
        )

    best_k = int(
        grid.best_params_[
            "select__k"
        ]
    )

    best_C = float(
        grid.best_params_[
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
    # Outer-test prediction.
    # ---------------------------------------------------------------

    decision_scores = (
        grid.best_estimator_
        .decision_function(
            X_test_combined
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
        f"Outer fold ROC-AUC: "
        f"{outer_auc:.6f}"
    )

    # ---------------------------------------------------------------
    # Selected features.
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

    selected_path = (
        OUTPUT_ROOT
        / f"outer_fold_{outer_fold}_selected_features.json"
    )

    with open(
        selected_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "outer_fold": outer_fold,
                "selected_k": best_k,
                "selected_features": selected_names,
                "n_selected": len(selected_names),
                "nuisance_variables": nuisance_names,
            },
            f,
            indent=2,
        )

    log(
        f"Fold runtime: "
        f"{(time.time() - fold_start) / 60:.2f} minutes"
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

    predictions = (
        scores >= 0
    ).astype(int)

    return {
        "oof_roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "oof_accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "oof_balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "oof_sensitivity": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "oof_specificity": float(
            recall_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "oof_precision": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "oof_f1": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    overall_start = time.time()

    prepare_directories()

    log("=" * 70)
    log("NEUROFUSION AI â€” EXP20")
    log("NUISANCE-CONTROLLED REGIONAL GM VOLUME + LINEAR SVM")
    log("=" * 70)

    log(
        f"Project root: {PROJECT_ROOT}"
    )

    # ---------------------------------------------------------------
    # Configuration.
    # ---------------------------------------------------------------

    config = {
        "experiment": "Exp20",
        "description": (
            "Nuisance-controlled Regional GM Volume + Linear SVM"
        ),
        "scientific_question": (
            "Does the Exp16 regional GM-volume signal remain useful "
            "after controlling for available demographic nuisance "
            "variables?"
        ),
        "cohort": "RMS-only",
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "representation": (
            "116 AAL regional GM-volume features from SPM mwc1 maps"
        ),
        "nuisance_variables": [
            "age",
            "gender",
        ],
        "nuisance_handling": (
            "Linear residualization fitted separately inside each "
            "training partition of nested CV"
        ),
        "icv_included": False,
        "icv_exclusion_reason": (
            "No validated one-row-per-subject ICV mapping currently available"
        ),
        "demographic_age_aggregation": "subject-level median",
        "demographic_gender_rule": (
            "must be consistent across available records; "
            "conflicting values cause a hard stop"
        ),
        "feature_selection": (
            "SelectKBest(f_classif), inside inner CV"
        ),
        "k_values": K_VALUES,
        "scaling": "StandardScaler, inside inner CV",
        "classifier": "LinearSVC",
        "C_values": C_VALUES,
        "class_weight": "balanced",
        "nested_cv": "5x4",
        "seed": SEED,
        "primary_metric": "OOF ROC-AUC",
        "outer_test_untouched": True,
        "exp16_comparison": {
            "oof_auc": 0.692722,
            "description": (
                "Reference Exp16 result"
            ),
        },
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
    # Cohort.
    # ---------------------------------------------------------------

    rms_df = load_rms_cohort()

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
    # CV.
    # ---------------------------------------------------------------

    assignments = load_assignments(
        rms_df
    )

    # ---------------------------------------------------------------
    # Demographics.
    # ---------------------------------------------------------------

    nuisance_df = load_demographics(
        rms_df
    )

    Z, nuisance_names = (
        build_nuisance_matrix(
            subject_ids,
            nuisance_df,
        )
    )

    log(
        f"Nuisance matrix: {Z.shape}"
    )

    log(
        f"Nuisance variables: {nuisance_names}"
    )

    # ---------------------------------------------------------------
    # Atlas.
    # ---------------------------------------------------------------

    atlas_img = load_atlas()

    # ---------------------------------------------------------------
    # Regional GM features.
    # ---------------------------------------------------------------

    X, feature_names = (
        build_feature_matrix(
            rms_df,
            atlas_img,
        )
    )

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

    # ---------------------------------------------------------------
    # Nested CV.
    # ---------------------------------------------------------------

    outer_results = []
    oof_records = []

    for outer_fold in range(
        1,
        N_OUTER_FOLDS + 1,
    ):

        result = run_outer_fold(
            outer_fold=outer_fold,
            X_features=X,
            Z=Z,
            y=y,
            subject_ids=subject_ids,
            assignments=assignments,
            feature_names=feature_names,
            nuisance_names=nuisance_names,
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
                    "decision_score": float(score),
                    "outer_fold": int(
                        outer_fold
                    ),
                }
            )

    # ---------------------------------------------------------------
    # OOF.
    # ---------------------------------------------------------------

    oof = pd.DataFrame(
        oof_records
    )

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
            "OOF decision scores contain non-finite values."
        )

    if set(
        oof["subject_id"]
    ) != set(subject_ids):
        raise ValueError(
            "OOF subject set differs from RMS cohort."
        )

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
    # Metrics.
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
    # Difference from Exp16.
    # ---------------------------------------------------------------

    exp16_auc = 0.692722

    auc_difference = (
        metrics["oof_roc_auc"]
        - exp16_auc
    )

    # ---------------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------------

    summary = {
        "experiment": "Exp20",
        "description": (
            "Nuisance-controlled Regional GM Volume + Linear SVM"
        ),
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "regional_features": EXPECTED_AAL_ROIS,
        "nuisance_variables": nuisance_names,
        "icv_included": False,
        "outer_fold_aucs": fold_aucs,
        "mean_fold_auc": mean_fold_auc,
        "fold_auc_sd": fold_auc_sd,
        **metrics,
        "exp16_reference_oof_auc": exp16_auc,
        "oof_auc_difference_vs_exp16": float(
            auc_difference
        ),
        "oof_rows": int(
            len(oof)
        ),
        "unique_oof_subjects": int(
            oof["subject_id"].nunique()
        ),
        "oof_integrity": "PASS",
        "nested_cv": "5x4",
        "outer_test_untouched": True,
        "nuisance_residualization_inside_cv": True,
        "feature_selection_inside_cv": True,
        "scaling_inside_cv": True,
        "hyperparameter_selection_inside_cv": True,
        "demographic_mapping_audit": str(
            DEMOGRAPHIC_AUDIT_PATH
        ),
        "feature_matrix": str(
            feature_matrix_path
        ),
        "oof_path": str(
            OOF_PATH
        ),
        "outer_summary_path": str(
            OUTER_SUMMARY_PATH
        ),
        "seed": SEED,
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
    # Final report.
    # ---------------------------------------------------------------

    log("")
    log("=" * 70)
    log("EXP20 FINAL RESULTS")
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
        f"Nuisance variables:    {nuisance_names}"
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
        f"Exp16 reference AUC:   "
        f"{exp16_auc:.6f}"
    )

    log(
        f"Difference vs Exp16:   "
        f"{auc_difference:+.6f}"
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
        f"OOF rows:              {len(oof)}"
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
        f"Output directory:\n{OUTPUT_ROOT}"
    )

    log("")
    log(
        f"Total runtime: "
        f"{(time.time() - overall_start) / 60:.2f} minutes"
    )

    log("")
    log("=" * 70)
    log("EXP20 COMPLETED")
    log("=" * 70)


if __name__ == "__main__":
    main()

