from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

NESTED_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp06_rms_cnn_aal"
    / "cross_validation"
    / "nested_cv_assignments.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp08_aal_svm_rbf"
    / "training_nested_cv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INNER_FOLDS = 4
OUTER_FOLDS = 5

# ------------------------------------------------------------
# Predeclared hyperparameter search
# ------------------------------------------------------------

K_VALUES = [
    "all",
    20,
    40,
    60,
    80,
]

C_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0,
]

GAMMA_VALUES = [
    "scale",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# LOAD AAL MATRIX
# ============================================================

def load_aal_matrix(manifest: pd.DataFrame):

    rows = []
    subject_ids = []
    feature_columns = None

    for _, row in manifest.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        aal_path = Path(
            row["aal_csv"]
        )

        if not aal_path.is_absolute():

            aal_path = (
                PROJECT_ROOT
                / aal_path
            )

        if not aal_path.exists():

            raise FileNotFoundError(
                f"AAL file not found: {aal_path}"
            )

        df = pd.read_csv(
            aal_path
        )

        if len(df) != 1:

            raise ValueError(
                f"{subject_id}: expected "
                f"one AAL row, found {len(df)}"
            )

        feature_df = df.drop(
            columns=[
                "subject_id",
                "acquisition_type",
                "volume",
            ],
            errors="ignore",
        )

        current_columns = list(
            feature_df.columns
        )

        if feature_columns is None:

            feature_columns = current_columns

        if current_columns != feature_columns:

            raise ValueError(
                f"AAL feature-column mismatch "
                f"for {subject_id}"
            )

        values = (
            feature_df.iloc[0]
            .to_numpy(
                dtype=np.float64
            )
        )

        if values.shape != (116,):

            raise ValueError(
                f"{subject_id}: expected 116 "
                f"features, got {values.shape}"
            )

        if not np.isfinite(values).all():

            raise ValueError(
                f"{subject_id}: non-finite "
                f"AAL features"
            )

        rows.append(values)
        subject_ids.append(subject_id)

    X = np.vstack(rows)

    if X.shape != (
        len(manifest),
        116,
    ):

        raise ValueError(
            f"Unexpected AAL matrix shape: "
            f"{X.shape}"
        )

    return (
        X,
        subject_ids,
        feature_columns,
    )


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(
    k_value,
    c_value,
    gamma_value,
):

    if k_value == "all":

        selector = SelectKBest(
            score_func=f_classif,
            k="all",
        )

    else:

        selector = SelectKBest(
            score_func=f_classif,
            k=int(k_value),
        )

    classifier = SVC(
        C=float(c_value),
        kernel="rbf",
        gamma=gamma_value,
        probability=True,
        random_state=SEED,
    )

    # IMPORTANT:
    #
    # Scaling and feature selection remain inside
    # the Pipeline.
    #
    # Therefore both are fitted only on the training
    # data supplied to pipeline.fit().
    #

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "selector",
                selector,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = (
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
        / np.sum(
            y_true == 1
        )
    )

    specificity = (
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
        / np.sum(
            y_true == 0
        )
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    (
        tn,
        fp,
        fn,
        tp,
    ) = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    return {
        "auc":
            float(auc),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "precision":
            float(precision),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# INNER CV MODEL SELECTION
# ============================================================

def select_hyperparameters(
    outer_fold,
    assignments,
    manifest,
    X,
    subject_to_index,
):

    inner_results = []

    candidate_scores = {}

    inner_fold_values = sorted(
        assignments.loc[
            (
                assignments[
                    "outer_fold"
                ]
                == outer_fold
            )
            & (
                assignments[
                    "role"
                ]
                == "INNER_TRAIN"
            ),
            "inner_fold",
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if inner_fold_values != [
        1,
        2,
        3,
        4,
    ]:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"unexpected inner folds: "
            f"{inner_fold_values}"
        )

    for k_value in K_VALUES:

        for c_value in C_VALUES:

            for gamma_value in GAMMA_VALUES:

                fold_aucs = []

                for inner_fold in inner_fold_values:

                    inner_assignments = assignments[
                        (
                            assignments[
                                "outer_fold"
                            ]
                            == outer_fold
                        )
                        & (
                            assignments[
                                "inner_fold"
                            ]
                            == inner_fold
                        )
                    ]

                    train_ids = (
                        inner_assignments.loc[
                            inner_assignments[
                                "role"
                            ]
                            == "INNER_TRAIN",
                            "subject_id",
                        ]
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    val_ids = (
                        inner_assignments.loc[
                            inner_assignments[
                                "role"
                            ]
                            == "INNER_VALIDATION",
                            "subject_id",
                        ]
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if (
                        set(train_ids)
                        & set(val_ids)
                    ):

                        raise RuntimeError(
                            f"Outer {outer_fold}, "
                            f"inner {inner_fold}: "
                            f"train/validation overlap."
                        )

                    train_indices = [
                        subject_to_index[
                            sid
                        ]
                        for sid in train_ids
                    ]

                    val_indices = [
                        subject_to_index[
                            sid
                        ]
                        for sid in val_ids
                    ]

                    X_train = X[
                        train_indices
                    ]

                    y_train = manifest[
                        manifest[
                            "subject_id"
                        ]
                        .astype(str)
                        .isin(train_ids)
                    ].set_index(
                        "subject_id"
                    ).loc[
                        train_ids,
                        "label",
                    ].to_numpy(
                        dtype=int
                    )

                    X_val = X[
                        val_indices
                    ]

                    y_val = manifest[
                        manifest[
                            "subject_id"
                        ]
                        .astype(str)
                        .isin(val_ids)
                    ].set_index(
                        "subject_id"
                    ).loc[
                        val_ids,
                        "label",
                    ].to_numpy(
                        dtype=int
                    )

                    pipeline = build_pipeline(
                        k_value,
                        c_value,
                        gamma_value,
                    )

                    pipeline.fit(
                        X_train,
                        y_train,
                    )

                    val_prob = (
                        pipeline.predict_proba(
                            X_val
                        )[:, 1]
                    )

                    val_auc = roc_auc_score(
                        y_val,
                        val_prob,
                    )

                    fold_aucs.append(
                        float(val_auc)
                    )

                mean_auc = float(
                    np.mean(
                        fold_aucs
                    )
                )

                candidate_key = (
                    f"k={k_value}"
                    f"|C={c_value}"
                    f"|gamma={gamma_value}"
                )

                candidate_scores[
                    candidate_key
                ] = {
                    "k":
                        k_value,

                    "C":
                        float(c_value),

                    "gamma":
                        gamma_value,

                    "inner_fold_aucs":
                        fold_aucs,

                    "mean_inner_auc":
                        mean_auc,
                }

                inner_results.append(
                    {
                        "outer_fold":
                            int(outer_fold),

                        "k":
                            k_value,

                        "C":
                            float(c_value),

                        "gamma":
                            gamma_value,

                        "fold_1_auc":
                            fold_aucs[0],

                        "fold_2_auc":
                            fold_aucs[1],

                        "fold_3_auc":
                            fold_aucs[2],

                        "fold_4_auc":
                            fold_aucs[3],

                        "mean_inner_auc":
                            mean_auc,
                    }
                )

    # ========================================================
    # SELECT BEST INNER-CV CONFIGURATION
    #
    # Tie-breaking:
    # 1. Higher mean inner AUC
    # 2. Smaller k
    # 3. Smaller C
    #
    # Outer-test information is never used.
    # ========================================================

    def tie_key(item):

        result = item[1]

        k = result["k"]

        if k == "all":

            k_rank = 10**9

        else:

            k_rank = int(k)

        return (
            result["mean_inner_auc"],
            -k_rank,
            -float(result["C"]),
        )

    best_key, best_result = max(
        candidate_scores.items(),
        key=tie_key,
    )

    selected_k = best_result[
        "k"
    ]

    selected_c = best_result[
        "C"
    ]

    selected_gamma = best_result[
        "gamma"
    ]

    return (
        selected_k,
        selected_c,
        selected_gamma,
        inner_results,
        candidate_scores,
    )


# ============================================================
# OUTER FOLD
# ============================================================

def run_outer_fold(
    outer_fold,
    assignments,
    manifest,
    X,
    subject_to_index,
):

    print()
    print(
        "=" * 70
    )
    print(
        f"OUTER FOLD {outer_fold}/5"
    )
    print(
        "=" * 70
    )

    (
        selected_k,
        selected_c,
        selected_gamma,
        inner_results,
        candidate_scores,
    ) = select_hyperparameters(
        outer_fold=outer_fold,
        assignments=assignments,
        manifest=manifest,
        X=X,
        subject_to_index=subject_to_index,
    )

    best_inner_auc = max(
        result[
            "mean_inner_auc"
        ]
        for result
        in candidate_scores.values()
    )

    print(
        "Selected k:",
        selected_k,
    )

    print(
        "Selected C:",
        selected_c,
    )

    print(
        "Selected gamma:",
        selected_gamma,
    )

    print(
        "Best inner-CV AUC:",
        f"{best_inner_auc:.6f}",
    )

    outer_assignments = assignments[
        assignments[
            "outer_fold"
        ]
        == outer_fold
    ]

    train_ids = (
        outer_assignments.loc[
            outer_assignments[
                "role"
            ]
            != "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    test_ids = (
        outer_assignments.loc[
            outer_assignments[
                "role"
            ]
            == "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if (
        set(train_ids)
        & set(test_ids)
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"train/test overlap."
        )

    train_indices = [
        subject_to_index[
            sid
        ]
        for sid in train_ids
    ]

    test_indices = [
        subject_to_index[
            sid
        ]
        for sid in test_ids
    ]

    X_train = X[
        train_indices
    ]

    X_test = X[
        test_indices
    ]

    y_train = manifest[
        manifest[
            "subject_id"
        ]
        .astype(str)
        .isin(train_ids)
    ].set_index(
        "subject_id"
    ).loc[
        train_ids,
        "label",
    ].to_numpy(
        dtype=int
    )

    y_test = manifest[
        manifest[
            "subject_id"
        ]
        .astype(str)
        .isin(test_ids)
    ].set_index(
        "subject_id"
    ).loc[
        test_ids,
        "label",
    ].to_numpy(
        dtype=int
    )

    # ========================================================
    # REFIT FINAL PIPELINE ON ALL OUTER TRAINING DATA
    # ========================================================

    final_pipeline = build_pipeline(
        selected_k,
        selected_c,
        selected_gamma,
    )

    final_pipeline.fit(
        X_train,
        y_train,
    )

    test_prob = (
        final_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_test,
        test_prob,
    )

    print(
        "Outer test subjects:",
        len(test_ids),
    )

    print(
        "Outer test AUC:",
        f"{metrics['auc']:.6f}",
    )

    # ========================================================
    # SELECTED FEATURE INFORMATION
    # ========================================================

    selector = (
        final_pipeline.named_steps[
            "selector"
        ]
    )

    support_mask = (
        selector.get_support()
    )

    selected_features = [
        feature
        for feature, selected
        in zip(
            AAL_FEATURE_NAMES,
            support_mask,
        )
        if selected
    ]

    fold_dir = (
        OUTPUT_DIR
        / f"outer_fold_{outer_fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_pipeline,
        fold_dir
        / "final_pipeline.joblib",
    )

    with open(
        fold_dir
        / "selected_features.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "selected_k":
                    selected_k,

                "selected_C":
                    selected_c,

                "selected_gamma":
                    selected_gamma,

                "selected_features":
                    selected_features,
            },
            file,
            indent=2,
        )

    pd.DataFrame(
        inner_results
    ).to_csv(
        fold_dir
        / "inner_model_selection.csv",
        index=False,
    )

    predictions = pd.DataFrame(
        {
            "subject_id":
                test_ids,

            "label":
                y_test.astype(int),

            "probability":
                test_prob,

            "outer_fold":
                int(outer_fold),
        }
    )

    predictions.to_csv(
        fold_dir
        / "outer_test_predictions.csv",
        index=False,
    )

    fold_summary = {
        "outer_fold":
            int(outer_fold),

        "selected_k":
            selected_k,

        "selected_C":
            float(selected_c),

        "selected_gamma":
            selected_gamma,

        "best_inner_auc":
            float(best_inner_auc),

        "selected_feature_count":
            int(
                len(
                    selected_features
                )
            ),

        "train_subjects":
            int(len(train_ids)),

        "test_subjects":
            int(len(test_ids)),

        "test_auc":
            metrics["auc"],

        "test_accuracy":
            metrics["accuracy"],

        "test_balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],

        "test_precision":
            metrics["precision"],

        "test_sensitivity":
            metrics["sensitivity"],

        "test_specificity":
            metrics["specificity"],

        "test_f1":
            metrics["f1"],

        "tn":
            metrics["tn"],

        "fp":
            metrics["fp"],

        "fn":
            metrics["fn"],

        "tp":
            metrics["tp"],
    }

    return (
        predictions,
        fold_summary,
        inner_results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print("=" * 70)
    print(
        "EXPERIMENT 08 — AAL SVM-RBF"
    )
    print(
        "LEAKAGE-SAFE 5×4 NESTED CV"
    )
    print("=" * 70)

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest["subject_id"] = (
        manifest[
            "subject_id"
        ]
        .astype(str)
    )

    # RMS-only controlled cohort
    manifest = manifest[
        manifest[
            "acquisition_source"
        ]
        .astype(str)
        .str.upper()
        == "RMS"
    ].copy()

    manifest = (
        manifest
        .sort_values(
            "subject_id"
        )
        .reset_index(drop=True)
    )

    if len(manifest) != 119:

        raise ValueError(
            f"Expected 119 RMS subjects, "
            f"found {len(manifest)}"
        )

    if manifest[
        "subject_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate RMS subjects."
        )

    class_counts = (
        manifest[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if class_counts != {
        0: 40,
        1: 79,
    }:

        raise ValueError(
            f"Expected class counts "
            f"{{0:40, 1:79}}, "
            f"found {class_counts}"
        )

    print(
        "RMS subjects:",
        len(manifest),
    )

    print(
        "Controls:",
        int(
            np.sum(
                manifest["label"]
                == 0
            )
        ),
    )

    print(
        "Schizophrenia:",
        int(
            np.sum(
                manifest["label"]
                == 1
            )
        ),
    )

    # ========================================================
    # LOAD NESTED ASSIGNMENTS
    # ========================================================

    assignments = pd.read_csv(
        NESTED_ASSIGNMENTS_PATH
    )

    assignments["subject_id"] = (
        assignments[
            "subject_id"
        ]
        .astype(str)
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
        - set(
            assignments.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing assignment columns: "
            f"{sorted(missing)}"
        )

    assignment_subjects = set(
        assignments[
            "subject_id"
        ].unique()
    )

    manifest_subjects = set(
        manifest[
            "subject_id"
        ].unique()
    )

    if (
        assignment_subjects
        != manifest_subjects
    ):

        raise RuntimeError(
            "Nested assignment subject set "
            "does not match RMS manifest."
        )

    if set(
        assignments[
            "acquisition_source"
        ]
        .astype(str)
        .str.upper()
        .unique()
    ) != {"RMS"}:

        raise RuntimeError(
            "Experiment 08 assignments "
            "must contain RMS only."
        )

    # ========================================================
    # LOAD AAL
    # ========================================================

    print()
    print(
        "Loading AAL features..."
    )

    (
        X,
        aal_subject_ids,
        feature_names,
    ) = load_aal_matrix(
        manifest
    )

    if X.shape != (
        119,
        116,
    ):

        raise RuntimeError(
            f"Expected AAL matrix "
            f"(119,116), found {X.shape}"
        )

    AAL_FEATURE_NAMES = feature_names

    globals()[
        "AAL_FEATURE_NAMES"
    ] = AAL_FEATURE_NAMES

    subject_to_index = {
        subject_id: index
        for index, subject_id
        in enumerate(
            aal_subject_ids
        )
    }

    print(
        "AAL matrix:",
        X.shape,
    )

    print(
        "AAL finite:",
        bool(
            np.isfinite(X).all()
        ),
    )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "AAL matrix contains "
            "non-finite values."
        )

    # ========================================================
    # NESTED CV
    # ========================================================

    all_predictions = []
    all_fold_summaries = []
    all_inner_results = []

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        (
            predictions,
            fold_summary,
            inner_results,
        ) = run_outer_fold(
            outer_fold=outer_fold,
            assignments=assignments,
            manifest=manifest,
            X=X,
            subject_to_index=subject_to_index,
        )

        all_predictions.append(
            predictions
        )

        all_fold_summaries.append(
            fold_summary
        )

        all_inner_results.extend(
            inner_results
        )

    # ========================================================
    # OOF
    # ========================================================

    oof = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    duplicate_count = int(
        oof[
            "subject_id"
        ].duplicated()
        .sum()
    )

    print()
    print(
        "=" * 70
    )
    print(
        "OOF INTEGRITY"
    )
    print(
        "=" * 70
    )

    print(
        "OOF rows:",
        len(oof),
    )

    print(
        "Unique subjects:",
        oof[
            "subject_id"
        ].nunique(),
    )

    print(
        "Duplicate subjects:",
        duplicate_count,
    )

    if len(oof) != 119:

        raise RuntimeError(
            f"Expected 119 OOF rows, "
            f"found {len(oof)}"
        )

    if (
        oof[
            "subject_id"
        ].nunique()
        != 119
    ):

        raise RuntimeError(
            "OOF subjects are not unique."
        )

    if duplicate_count != 0:

        raise RuntimeError(
            "Duplicate OOF subjects."
        )

    if set(
        oof[
            "subject_id"
        ]
    ) != manifest_subjects:

        raise RuntimeError(
            "OOF subject set does not "
            "match RMS cohort."
        )

    if not np.isfinite(
        oof[
            "probability"
        ]
    ).all():

        raise RuntimeError(
            "Non-finite OOF probabilities."
        )

    print(
        "OOF integrity: PASS"
    )

    # ========================================================
    # OOF METRICS
    # ========================================================

    oof_metrics = calculate_metrics(
        oof[
            "label"
        ].to_numpy(),
        oof[
            "probability"
        ].to_numpy(),
    )

    fold_auc = [
        summary[
            "test_auc"
        ]
        for summary
        in all_fold_summaries
    ]

    mean_fold_auc = float(
        np.mean(
            fold_auc
        )
    )

    fold_auc_std = float(
        np.std(
            fold_auc,
            ddof=0,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    oof.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        all_fold_summaries
    ).to_csv(
        OUTPUT_DIR
        / "outer_fold_results.csv",
        index=False,
    )

    pd.DataFrame(
        all_inner_results
    ).to_csv(
        OUTPUT_DIR
        / "all_inner_model_selection.csv",
        index=False,
    )

    summary = {
        "experiment":
            "Experiment 08 — "
            "AAL SVM-RBF",

        "cohort":
            "RMS only",

        "subjects":
            119,

        "controls":
            40,

        "schizophrenia":
            79,

        "features":
            116,

        "outer_folds":
            5,

        "inner_folds":
            4,

        "feature_selection_inside_cv":
            True,

        "scaling_inside_cv":
            True,

        "outer_test_used_for_tuning":
            False,

        "kernel":
            "rbf",

        "hyperparameters":
            {
                "k_values":
                    K_VALUES,

                "C_values":
                    C_VALUES,

                "gamma_values":
                    GAMMA_VALUES,
            },

        "oof_rows":
            int(len(oof)),

        "oof_unique_subjects":
            int(
                oof[
                    "subject_id"
                ].nunique()
            ),

        "oof_duplicate_subjects":
            duplicate_count,

        "outer_fold_auc":
            fold_auc,

        "mean_outer_fold_auc":
            mean_fold_auc,

        "outer_fold_auc_population_std":
            fold_auc_std,

        **{
            f"oof_{key}":
                value
            for key, value
            in oof_metrics.items()
        },

        "status":
            "PASS",
    }

    with open(
        OUTPUT_DIR
        / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================
    # HUMAN-READABLE REPORT
    # ========================================================

    report = [
        "=" * 70,
        "EXPERIMENT 08 — AAL SVM-RBF",
        "LEAKAGE-SAFE 5×4 NESTED CROSS-VALIDATION",
        "=" * 70,
        "",
        "Cohort: RMS only",
        "Subjects: 119",
        "Controls: 40",
        "Schizophrenia: 79",
        "AAL features: 116",
        "",
        "Classifier:",
        "  Kernel: RBF",
        "",
        "Pipeline:",
        "  StandardScaler: inside CV",
        "  SelectKBest: inside CV",
        "  Hyperparameter selection: inner CV",
        "  Outer-test tuning: NO",
        "",
        "Outer fold AUCs:",
    ]

    for index, auc in enumerate(
        fold_auc,
        start=1,
    ):

        report.append(
            f"  Fold {index}: "
            f"{auc:.6f}"
        )

    report.extend(
        [
            "",
            (
                "Mean fold AUC: "
                f"{mean_fold_auc:.6f}"
            ),
            (
                "Fold AUC population SD: "
                f"{fold_auc_std:.6f}"
            ),
            "",
            (
                "OOF ROC-AUC: "
                f"{oof_metrics['auc']:.6f}"
            ),
            (
                "OOF Accuracy: "
                f"{oof_metrics['accuracy']:.6f}"
            ),
            (
                "OOF Balanced Accuracy: "
                f"{oof_metrics['balanced_accuracy']:.6f}"
            ),
            (
                "OOF Precision: "
                f"{oof_metrics['precision']:.6f}"
            ),
            (
                "OOF Sensitivity: "
                f"{oof_metrics['sensitivity']:.6f}"
            ),
            (
                "OOF Specificity: "
                f"{oof_metrics['specificity']:.6f}"
            ),
            (
                "OOF F1: "
                f"{oof_metrics['f1']:.6f}"
            ),
            "",
            "OOF integrity:",
            f"  Rows: {len(oof)}",
            (
                "  Unique subjects: "
                f"{oof['subject_id'].nunique()}"
            ),
            "  Duplicates: 0",
            "",
            "FINAL STATUS: PASS",
        ]
    )

    with open(
        OUTPUT_DIR
        / "training_report.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(report)
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 08 COMPLETE"
    )

    print(
        "Outer fold AUCs:",
        [
            round(
                x,
                6,
            )
            for x in fold_auc
        ],
    )

    print(
        "Mean fold AUC:",
        f"{mean_fold_auc:.6f}",
    )

    print(
        "Fold AUC population SD:",
        f"{fold_auc_std:.6f}",
    )

    print(
        "OOF ROC-AUC:",
        f"{oof_metrics['auc']:.6f}",
    )

    print(
        "OOF Accuracy:",
        f"{oof_metrics['accuracy']:.6f}",
    )

    print(
        "OOF Balanced Accuracy:",
        f"{oof_metrics['balanced_accuracy']:.6f}",
    )

    print(
        "OOF Precision:",
        f"{oof_metrics['precision']:.6f}",
    )

    print(
        "OOF Sensitivity:",
        f"{oof_metrics['sensitivity']:.6f}",
    )

    print(
        "OOF Specificity:",
        f"{oof_metrics['specificity']:.6f}",
    )

    print(
        "OOF F1:",
        f"{oof_metrics['f1']:.6f}",
    )

    print(
        "OOF rows:",
        len(oof),
    )

    print(
        "Unique OOF subjects:",
        oof[
            "subject_id"
        ].nunique(),
    )

    print(
        "Duplicate OOF subjects:",
        duplicate_count,
    )

    print(
        "Results:",
        OUTPUT_DIR,
    )

    print(
        "FINAL STATUS: "
        "EXPERIMENT 08 NESTED CV PASS"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()