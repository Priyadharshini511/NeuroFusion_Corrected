"""
NeuroFusion AI
Experiment 18
Combined sMRI Morphometry + AAL Intensity + Linear SVM

Strict Nested 5x4 Cross-Validation

Scientific question:
Does combining regional gray-matter volume with regional sMRI
mean-intensity information improve subject-level schizophrenia
discrimination compared with either representation alone?

Representations:
    1. Regional GM volume from Exp16
       116 AAL regional features derived from SPM modulated GM maps.

    2. Existing validated AAL mean-intensity representation
       116 sMRI_AAL_* regional mean-intensity features.

Combined:
    232 features

Model:
    SelectKBest(f_classif)
    + StandardScaler
    + LinearSVC(class_weight="balanced")

Leakage control:
    - Canonical 172-subject nested CV assignments
    - RMS subset only
    - Feature selection inside inner CV
    - Scaling inside inner CV
    - SVM hyperparameter selection inside inner CV
    - Outer test subjects never used for model selection

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
from pathlib import Path

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

EXP16_FEATURE_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "vbm_spm"
    / "experiments"
    / "exp16_regional_gm_svm"
    / "regional_gm_volume_features.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "vbm_spm"
    / "experiments"
    / "exp18_combined_smri_svm"
)

OOF_DIR = OUTPUT_ROOT / "oof"

OOF_PATH = (
    OOF_DIR
    / "exp18_combined_smri_svm_oof.csv"
)

OUTER_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "outer_fold_summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_ROOT
    / "exp18_summary.json"
)

CONFIG_PATH = (
    OUTPUT_ROOT
    / "experiment_config.json"
)

COMBINED_FEATURE_PATH = (
    OUTPUT_ROOT
    / "combined_smri_features.csv"
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

EXPECTED_GM_FEATURES = 116
EXPECTED_AAL_FEATURES = 116
EXPECTED_COMBINED_FEATURES = 232

# Feature-selection values.
#
# These are selected inside each inner-CV split.
K_VALUES = [
    20,
    40,
    60,
    80,
    100,
    120,
    160,
    200,
    232,
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
# LOAD RMS COHORT
# =====================================================================


def load_rms_cohort() -> pd.DataFrame:

    log("Loading labeled manifest...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "subject_id",
        "label",
        "acquisition_source",
        "aal_csv",
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
            "Duplicate RMS subject IDs found."
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
            f"Unexpected RMS labels: {labels}; "
            f"expected {expected}"
        )

    log(
        f"RMS subjects:       {len(rms)}"
    )

    log(
        f"Controls:           "
        f"{int((rms['label'] == 0).sum())}"
    )

    log(
        f"Schizophrenia:      "
        f"{int((rms['label'] == 1).sum())}"
    )

    return rms


# =====================================================================
# LOAD CANONICAL NESTED CV
# =====================================================================


def load_assignments(
    rms_df: pd.DataFrame,
) -> pd.DataFrame:

    log(
        "Loading canonical nested CV assignments..."
    )

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            "Canonical assignment file not found:\n"
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
            "Assignment file is missing:\n"
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

    rms_ids = set(
        rms_df["subject_id"]
    )

    assignments = assignments[
        assignments["subject_id"]
        .isin(rms_ids)
    ].copy()

    assignments.reset_index(
        drop=True,
        inplace=True,
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
                f"Assignment label mismatch: {sid}"
            )

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

    if (
        sorted(
            assignments[
                "outer_fold"
            ]
            .astype(int)
            .unique()
        )
        != [1, 2, 3, 4, 5]
    ):
        raise ValueError(
            "Expected outer folds 1-5."
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
            "Duplicate nested-CV assignment rows found."
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
# RESOLVE AAL CSV
# =====================================================================


def resolve_aal_path(
    value: str,
) -> Path:

    path = Path(
        str(value)
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


# =====================================================================
# LOAD AAL FEATURES
# =====================================================================


def load_aal_features(
    rms_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:

    log("")
    log("Loading existing validated AAL features...")

    feature_rows = []
    feature_names = None

    for index, row in rms_df.iterrows():

        subject_id = str(row["subject_id"])

        aal_path = resolve_aal_path(row["aal_csv"])

        if not aal_path.exists():
            raise FileNotFoundError(
                f"AAL feature CSV not found for {subject_id}:\n{aal_path}"
            )

        aal_df = pd.read_csv(aal_path)

        if len(aal_df) != 1:
            raise ValueError(
                f"{subject_id}: expected exactly one row in AAL CSV, "
                f"found {len(aal_df)}"
            )

        # -----------------------------------------------------------
        # Select ONLY canonical AAL intensity ROI columns.
        # The validated files contain metadata such as subject_id,
        # acquisition_type and volume in addition to the 116 ROIs.
        # -----------------------------------------------------------
        current_columns = [
            c for c in aal_df.columns
            if str(c).startswith("sMRI_AAL_")
        ]

        if len(current_columns) != EXPECTED_AAL_FEATURES:
            raise ValueError(
                f"{subject_id}: expected {EXPECTED_AAL_FEATURES} AAL features "
                f"with prefix 'sMRI_AAL_', found {len(current_columns)}\n"
                f"File: {aal_path}"
            )

        # Keep the canonical ROI ordering identical for every subject.
        current_columns = sorted(
            current_columns,
            key=lambda c: int(str(c).replace("sMRI_AAL_", ""))
        )

        if feature_names is None:
            feature_names = list(current_columns)

        elif list(current_columns) != feature_names:
            raise ValueError(
                f"{subject_id}: AAL feature-column order does not match previous subjects."
            )

        values = (
            aal_df.iloc[0][feature_names]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{subject_id}: AAL features contain non-finite values."
            )

        feature_rows.append(values)

        if (
            index == 0
            or (index + 1) % 10 == 0
            or (index + 1) == len(rms_df)
        ):
            log(f"AAL loading: {index + 1}/{len(rms_df)}")

    X_aal = np.vstack(feature_rows)

    if X_aal.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_AAL_FEATURES,
    ):
        raise ValueError(
            f"Unexpected AAL matrix shape: {X_aal.shape}"
        )

    if not np.isfinite(X_aal).all():
        raise ValueError(
            "AAL feature matrix contains non-finite values."
        )

    log(f"AAL matrix: {X_aal.shape}")

    return X_aal, feature_names

# =====================================================================
# LOAD EXP16 GM VOLUME FEATURES
# =====================================================================


def load_gm_volume_features(
    subject_ids: list[str],
) -> tuple[np.ndarray, list[str]]:

    log("")
    log(
        "Loading completed Exp16 regional GM-volume features..."
    )

    if not EXP16_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Exp16 feature matrix not found:\n"
            f"{EXP16_FEATURE_PATH}"
        )

    gm_df = pd.read_csv(
        EXP16_FEATURE_PATH
    )

    if "subject_id" not in gm_df.columns:
        raise ValueError(
            "Exp16 feature matrix does not contain "
            "'subject_id'."
        )

    gm_df["subject_id"] = (
        gm_df["subject_id"]
        .astype(str)
        .str.strip()
    )

    if gm_df["subject_id"].duplicated().any():
        raise ValueError(
            "Exp16 feature matrix contains duplicate subjects."
        )

    gm_feature_names = [
        c
        for c in gm_df.columns
        if c != "subject_id"
    ]

    if len(gm_feature_names) != EXPECTED_GM_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_GM_FEATURES} GM-volume features, "
            f"found {len(gm_feature_names)}"
        )

    available_ids = set(
        gm_df["subject_id"]
    )

    requested_ids = set(
        subject_ids
    )

    missing = (
        requested_ids
        - available_ids
    )

    extra = (
        available_ids
        - requested_ids
    )

    if missing:
        raise ValueError(
            "Exp16 feature matrix is missing RMS subjects:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    # Extra subjects are not a problem, but they are excluded.
    if extra:
        log(
            f"Exp16 feature matrix contains "
            f"{len(extra)} extra subjects; "
            "excluding them."
        )

    gm_df = (
        gm_df[
            gm_df["subject_id"]
            .isin(requested_ids)
        ]
        .set_index("subject_id")
        .loc[subject_ids]
        .reset_index()
    )

    values = (
        gm_df[
            gm_feature_names
        ]
        .astype(float)
        .to_numpy()
    )

    if values.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_GM_FEATURES,
    ):
        raise ValueError(
            "Unexpected GM-volume matrix shape: "
            f"{values.shape}"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "GM-volume matrix contains non-finite values."
        )

    log(
        f"GM-volume matrix: {values.shape}"
    )

    return (
        values,
        gm_feature_names,
    )


# =====================================================================
# BUILD COMBINED MATRIX
# =====================================================================


def build_combined_matrix(
    X_gm: np.ndarray,
    gm_names: list[str],
    X_aal: np.ndarray,
    aal_names: list[str],
    subject_ids: list[str],
) -> tuple[np.ndarray, list[str]]:

    if X_gm.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_GM_FEATURES,
    ):
        raise ValueError(
            f"Unexpected GM matrix shape: {X_gm.shape}"
        )

    if X_aal.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_AAL_FEATURES,
    ):
        raise ValueError(
            f"Unexpected AAL matrix shape: {X_aal.shape}"
        )

    if len(
        set(gm_names)
        & set(aal_names)
    ) > 0:

        # Prefixing prevents accidental duplicate names.
        gm_names = [
            (
                name
                if name.startswith("GMVOL_")
                else f"GMVOL_{name}"
            )
            for name in gm_names
        ]

        aal_names = [
            (
                name
                if name.startswith("AALINT_")
                else f"AALINT_{name}"
            )
            for name in aal_names
        ]

    else:

        gm_names = [
            (
                name
                if name.startswith("GMVOL_")
                else f"GMVOL_{name}"
            )
            for name in gm_names
        ]

        aal_names = [
            (
                name
                if name.startswith("AALINT_")
                else f"AALINT_{name}"
            )
            for name in aal_names
        ]

    combined_names = (
        gm_names
        + aal_names
    )

    if len(combined_names) != EXPECTED_COMBINED_FEATURES:
        raise ValueError(
            "Unexpected combined feature count: "
            f"{len(combined_names)}"
        )

    X_combined = np.hstack(
        [
            X_gm,
            X_aal,
        ]
    )

    if X_combined.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_COMBINED_FEATURES,
    ):
        raise ValueError(
            "Unexpected combined matrix shape: "
            f"{X_combined.shape}"
        )

    if not np.isfinite(
        X_combined
    ).all():
        raise ValueError(
            "Combined feature matrix contains non-finite values."
        )

    # ---------------------------------------------------------------
    # Save audit copy.
    # ---------------------------------------------------------------

    combined_df = pd.DataFrame(
        X_combined,
        columns=combined_names,
    )

    combined_df.insert(
        0,
        "subject_id",
        subject_ids,
    )

    combined_df.to_csv(
        COMBINED_FEATURE_PATH,
        index=False,
    )

    log(
        f"Combined feature matrix: "
        f"{X_combined.shape}"
    )

    log(
        f"Combined feature matrix saved:\n"
        f"{COMBINED_FEATURE_PATH}"
    )

    return (
        X_combined,
        combined_names,
    )


# =====================================================================
# INNER CV VECTOR
# =====================================================================


def make_inner_vector(
    outer_train_ids: set[str],
    outer_fold: int,
    assignments: pd.DataFrame,
    subject_order: list[str],
) -> np.ndarray:

    vector = []

    for subject_id in subject_order:

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
                f"{N_INNER_FOLDS} inner rows for "
                f"outer fold {outer_fold}; "
                f"found {len(rows)}"
            )

        validation_rows = rows[
            rows["role"]
            == "INNER_VALIDATION"
        ]

        if len(validation_rows) != 1:
            raise ValueError(
                f"{subject_id}: expected exactly "
                "one INNER_VALIDATION row."
            )

        inner_fold = int(
            validation_rows.iloc[0][
                "inner_fold"
            ]
        )

        vector.append(
            inner_fold - 1
        )

    vector = np.asarray(
        vector,
        dtype=int,
    )

    if set(vector.tolist()) != {
        0,
        1,
        2,
        3,
    }:

        raise ValueError(
            f"Invalid inner CV vector: "
            f"{sorted(set(vector.tolist()))}"
        )

    return vector


# =====================================================================
# PIPELINE
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

    start = time.time()

    fold_assignments = assignments[
        assignments["outer_fold"]
        == outer_fold
    ]

    outer_test_ids = set(
        fold_assignments.loc[
            fold_assignments["role"]
            == "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
    )

    outer_train_ids = set(
        fold_assignments.loc[
            fold_assignments["role"]
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
            "train/test overlap."
        )

    if (
        len(outer_train_ids)
        + len(outer_test_ids)
        != EXPECTED_SUBJECTS
    ):

        raise ValueError(
            f"Outer fold {outer_fold}: "
            "unexpected subject count."
        )

    train_idx = [
        i
        for i, sid in enumerate(
            subject_ids
        )
        if sid in outer_train_ids
    ]

    test_idx = [
        i
        for i, sid in enumerate(
            subject_ids
        )
        if sid in outer_test_ids
    ]

    X_train = X[
        train_idx
    ]

    y_train = y[
        train_idx
    ]

    X_test = X[
        test_idx
    ]

    y_test = y[
        test_idx
    ]

    train_subject_order = [
        subject_ids[i]
        for i in train_idx
    ]

    test_subject_order = [
        subject_ids[i]
        for i in test_idx
    ]

    log(
        f"Outer train subjects: {len(train_idx)}"
    )

    log(
        f"Outer test subjects:  {len(test_idx)}"
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
    # ---------------------------------------------------------------

    inner_vector = make_inner_vector(
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

    log(
        "Running inner nested grid search..."
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    grid.fit(
        X_train,
        y_train,
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

    best_inner_auc = float(
        grid.best_score_
    )

    log(
        f"Best inner k:       {best_k}"
    )

    log(
        f"Best inner C:       {best_C}"
    )

    log(
        f"Best inner ROC-AUC: "
        f"{best_inner_auc:.6f}"
    )

    # ---------------------------------------------------------------
    # Outer prediction.
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
                "best_k": best_k,
                "best_C": best_C,
                "selected_features": selected_names,
            },
            f,
            indent=2,
        )

    log(
        f"Outer fold {outer_fold} completed in "
        f"{(time.time() - start) / 60:.2f} minutes."
    )

    return {
        "outer_fold": outer_fold,
        "outer_train_n": len(
            train_idx
        ),
        "outer_test_n": len(
            test_idx
        ),
        "outer_test_controls": int(
            (y_test == 0).sum()
        ),
        "outer_test_schizophrenia": int(
            (y_test == 1).sum()
        ),
        "best_k": best_k,
        "best_C": best_C,
        "best_inner_auc": best_inner_auc,
        "outer_auc": float(
            outer_auc
        ),
        "test_subject_ids": test_subject_order,
        "test_labels": y_test.astype(
            int
        ).tolist(),
        "test_scores": decision_scores.astype(
            float
        ).tolist(),
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
    log(
        "NEUROFUSION AI — EXP18"
    )
    log(
        "COMBINED sMRI MORPHOMETRY + AAL INTENSITY + LINEAR SVM"
    )
    log("=" * 70)

    # ---------------------------------------------------------------
    # Configuration.
    # ---------------------------------------------------------------

    config = {
        "experiment": "Exp18",
        "description": (
            "Combined regional GM volume + AAL mean intensity "
            "+ Linear SVM"
        ),
        "scientific_question": (
            "Does combining regional GM volume and regional "
            "sMRI intensity improve subject-level discrimination?"
        ),
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "gm_features": EXPECTED_GM_FEATURES,
        "aal_intensity_features": EXPECTED_AAL_FEATURES,
        "combined_features": EXPECTED_COMBINED_FEATURES,
        "nested_cv": "5x4",
        "seed": SEED,
        "feature_selection": (
            "SelectKBest(f_classif) inside inner CV"
        ),
        "k_values": K_VALUES,
        "scaling": (
            "StandardScaler inside inner CV"
        ),
        "classifier": "LinearSVC",
        "C_values": C_VALUES,
        "class_weight": "balanced",
        "primary_metric": "OOF ROC-AUC",
        "outer_test_untouched": True,
        "feature_selection_inside_cv": True,
        "scaling_inside_cv": True,
        "hyperparameter_selection_inside_cv": True,
        "gm_source": str(
            EXP16_FEATURE_PATH
        ),
        "aal_source": (
            "validated subject-level aal_csv "
            "paths from labeled manifest"
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
    # Canonical nested CV.
    # ---------------------------------------------------------------

    assignments = load_assignments(
        rms_df
    )

    # ---------------------------------------------------------------
    # Existing AAL intensity features.
    # ---------------------------------------------------------------

    X_aal, aal_names = (
        load_aal_features(
            rms_df
        )
    )

    # ---------------------------------------------------------------
    # Exp16 GM-volume features.
    # ---------------------------------------------------------------

    X_gm, gm_names = (
        load_gm_volume_features(
            subject_ids
        )
    )

    # ---------------------------------------------------------------
    # Combine.
    # ---------------------------------------------------------------

    (
        X,
        feature_names,
    ) = build_combined_matrix(
        X_gm=X_gm,
        gm_names=gm_names,
        X_aal=X_aal,
        aal_names=aal_names,
        subject_ids=subject_ids,
    )

    # ---------------------------------------------------------------
    # Final matrix checks.
    # ---------------------------------------------------------------

    if X.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_COMBINED_FEATURES,
    ):

        raise ValueError(
            f"Combined matrix has unexpected shape: {X.shape}"
        )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            "Combined matrix contains non-finite values."
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
            "OOF contains non-finite scores."
        )

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

    oof = (
        oof
        .sort_values("subject_id")
        .reset_index(drop=True)
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
    # Final OOF metrics.
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
        np.mean(
            fold_aucs
        )
    )

    fold_auc_sd = float(
        np.std(
            fold_aucs,
            ddof=0,
        )
    )

    # ---------------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------------

    summary = {
        "experiment": "Exp18",
        "description": (
            "Combined regional GM volume + AAL intensity "
            "+ Linear SVM"
        ),
        "subjects": EXPECTED_SUBJECTS,
        "controls": EXPECTED_CONTROLS,
        "schizophrenia": EXPECTED_SCHIZOPHRENIA,
        "gm_features": EXPECTED_GM_FEATURES,
        "aal_intensity_features": EXPECTED_AAL_FEATURES,
        "combined_features": EXPECTED_COMBINED_FEATURES,
        "outer_fold_aucs": fold_aucs,
        "mean_fold_auc": mean_fold_auc,
        "fold_auc_sd": fold_auc_sd,
        **metrics,
        "oof_rows": int(
            len(oof)
        ),
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
        "runtime_minutes": (
            (time.time() - overall_start)
            / 60.0
        ),
        "combined_feature_matrix": str(
            COMBINED_FEATURE_PATH
        ),
        "oof_path": str(
            OOF_PATH
        ),
        "outer_summary_path": str(
            OUTER_SUMMARY_PATH
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
    # FINAL RESULTS.
    # ---------------------------------------------------------------

    log("")
    log("=" * 70)
    log("EXP18 FINAL RESULTS")
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
        f"GM-volume features:    {EXPECTED_GM_FEATURES}"
    )

    log(
        f"AAL intensity features:{EXPECTED_AAL_FEATURES}"
    )

    log(
        f"Combined features:     {EXPECTED_COMBINED_FEATURES}"
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
        f"Combined features: "
        f"{COMBINED_FEATURE_PATH}"
    )

    log(
        f"OOF: {OOF_PATH}"
    )

    log(
        f"Outer summary: "
        f"{OUTER_SUMMARY_PATH}"
    )

    log(
        f"Summary JSON: "
        f"{SUMMARY_JSON_PATH}"
    )

    log("")
    log(
        f"Total runtime: "
        f"{(time.time() - overall_start) / 60:.2f} minutes"
    )

    log("")
    log("=" * 70)
    log("EXP18 COMPLETED")
    log("=" * 70)


if __name__ == "__main__":
    main()