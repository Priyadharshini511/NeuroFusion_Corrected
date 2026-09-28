from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score


# ============================================================
# EXP19 — Exp16 Regional GM Volume + Linear SVM
#           Label Permutation Test
# ============================================================

SEED = 42
N_PERMUTATIONS = 1000
EXPECTED_AUC = 0.692722

ROOT = Path(".")

MANIFEST = (
    ROOT / "processed_data" / "smri" / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

ASSIGNMENTS = (
    ROOT / "processed_data" / "smri" / "cnn_dataset"
    / "cross_validation" / "nested"
    / "nested_cv_assignments.csv"
)

FEATURES = (
    ROOT / "processed_data" / "smri" / "vbm_spm"
    / "experiments" / "exp16_regional_gm_svm"
    / "regional_gm_volume_features.csv"
)

OUTDIR = (
    ROOT / "processed_data" / "smri" / "vbm_spm"
    / "experiments" / "exp19_exp16_permutation_test"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

NULL_CSV = OUTDIR / "permutation_null_auc.csv"
SUMMARY_JSON = OUTDIR / "exp19_permutation_summary.json"
CONFIG_JSON = OUTDIR / "exp19_config.json"


# ============================================================
# COHORT VALIDATION
# ============================================================

def validate_cohort(features, manifest, assignments):

    rms = manifest[
        manifest["has_rms_representation"].astype(bool)
    ].copy()

    ids_manifest = set(
        rms["subject_id"].astype(str)
    )

    ids_features = set(
        features["subject_id"].astype(str)
    )

    rms_assign = assignments[
        assignments["subject_id"].astype(str).isin(ids_manifest)
    ].copy()

    ids_assign = set(
        rms_assign["subject_id"].astype(str)
    )

    print("\n=== COHORT VALIDATION ===")
    print("RMS manifest subjects :", len(ids_manifest))
    print("Feature subjects      :", len(ids_features))
    print("Assignment subjects   :", len(ids_assign))

    if ids_features != ids_manifest:
        raise RuntimeError(
            "Feature subject IDs do not exactly match RMS manifest."
        )

    if ids_assign != ids_manifest:
        raise RuntimeError(
            "Canonical assignment subject IDs do not exactly match RMS cohort."
        )

    if len(rms_assign) != 119 * 20:
        raise RuntimeError(
            f"Expected 2380 RMS assignment rows, "
            f"got {len(rms_assign)}."
        )

    print("Cohort validation PASS.")

    return rms_assign


# ============================================================
# INNER PREDEFINED SPLIT
# ============================================================

def make_inner_split(train_subject_ids, assignments_outer):
    """
    Build the Exp16-style PredefinedSplit.

    IMPORTANT:
    The split array must contain exactly one entry per
    training subject.

    -1 = inner training
     0 = inner validation fold 0
     1 = inner validation fold 1
     2 = inner validation fold 2
     3 = inner validation fold 3
    """

    train_subject_ids = np.asarray(
        train_subject_ids
    ).astype(str)

    val_rows = assignments_outer[
        assignments_outer["role"] == "INNER_VALIDATION"
    ].copy()

    val_rows["subject_id"] = (
        val_rows["subject_id"].astype(str)
    )

    if val_rows["subject_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate INNER_VALIDATION subject rows detected."
        )

    inner_map = dict(
        zip(
            val_rows["subject_id"],
            val_rows["inner_fold"].astype(int)
        )
    )

    split = np.array(
        [
            inner_map.get(sid, -1)
            for sid in train_subject_ids
        ],
        dtype=int
    )

    if len(split) != len(train_subject_ids):
        raise RuntimeError(
            "PredefinedSplit length does not match "
            "number of training subjects."
        )

    validation_values = set(
        np.unique(split[split >= 0])
    )

    if validation_values != {1, 2, 3, 4}:
        raise RuntimeError(
            "Expected inner folds {0,1,2,3}, "
            f"found {validation_values}."
        )

    print(
        f"    Inner split: "
        f"{len(train_subject_ids)} training subjects | "
        f"validation folds = {sorted(validation_values)}"
    )

    split = np.where(split >= 0, split - 1, -1)
    return PredefinedSplit(split)


# ============================================================
# MODEL
# ============================================================

def build_pipeline():

    return Pipeline([
        (
            "select",
            SelectKBest(
                score_func=f_classif
            )
        ),

        (
            "scale",
            StandardScaler()
        ),

        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=SEED,
                max_iter=10000
            )
        )
    ])


PARAM_GRID = {
    "select__k": [
        20,
        40,
        60,
        80,
        100,
        116
    ],

    "clf__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


# ============================================================
# NESTED CV
# ============================================================

def run_nested_cv(X, y, assignments):

    subject_ids = np.asarray(
        X.index.astype(str)
    )

    y_arr = np.asarray(y).astype(int)

    oof_rows = []
    outer_rows = []

    for outer_fold in range(1, 6):

        print(
            f"\n  ---------- OUTER FOLD {outer_fold} ----------"
        )

        outer = assignments[
            assignments["outer_fold"].astype(int)
            == outer_fold
        ].copy()

        # ----------------------------------------------------
        # Outer training subjects
        # ----------------------------------------------------

        train_ids = set(
            outer.loc[
                outer["role"] != "OUTER_TEST",
                "subject_id"
            ].astype(str)
        )

        # ----------------------------------------------------
        # Outer test subjects
        # ----------------------------------------------------

        test_ids = set(
            outer.loc[
                outer["role"] == "OUTER_TEST",
                "subject_id"
            ].astype(str)
        )

        train_mask = np.array([
            sid in train_ids
            for sid in subject_ids
        ])

        test_mask = np.array([
            sid in test_ids
            for sid in subject_ids
        ])

        if train_mask.sum() == 0:
            raise RuntimeError(
                f"Outer fold {outer_fold} has empty training set."
            )

        if test_mask.sum() == 0:
            raise RuntimeError(
                f"Outer fold {outer_fold} has empty test set."
            )

        train_subjects = subject_ids[
            train_mask
        ]

        test_subjects = subject_ids[
            test_mask
        ]

        print(
            f"    Outer train subjects : {len(train_subjects)}"
        )

        print(
            f"    Outer test subjects  : {len(test_subjects)}"
        )

        # ----------------------------------------------------
        # Inner assignments
        # ----------------------------------------------------

        inner_assign = outer[
            outer["subject_id"].astype(str).isin(
                train_subjects
            )
        ].copy()

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # Build PredefinedSplit using exactly the training
        # subject order supplied to GridSearchCV.
        # ----------------------------------------------------

        inner_split = make_inner_split(
            train_subjects,
            inner_assign
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        pipe = build_pipeline()

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=PARAM_GRID,
            scoring="roc_auc",
            cv=inner_split,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        # ----------------------------------------------------
        # Inner model selection
        # ----------------------------------------------------

        grid.fit(
            X.loc[train_subjects],
            y_arr[train_mask]
        )

        # ----------------------------------------------------
        # Outer test prediction
        # ----------------------------------------------------

        test_scores = grid.decision_function(
            X.loc[test_subjects]
        )

        fold_auc = roc_auc_score(
            y_arr[test_mask],
            test_scores
        )

        best_k = grid.best_params_[
            "select__k"
        ]

        best_c = grid.best_params_[
            "clf__C"
        ]

        print(
            f"    Outer AUC          : {fold_auc:.6f}"
        )

        print(
            f"    Best inner AUC     : "
            f"{grid.best_score_:.6f}"
        )

        print(
            f"    Best k             : {best_k}"
        )

        print(
            f"    Best C             : {best_c}"
        )

        # ----------------------------------------------------
        # Store OOF predictions
        # ----------------------------------------------------

        for sid, label, score in zip(
            test_subjects,
            y_arr[test_mask],
            test_scores
        ):

            oof_rows.append({
                "subject_id": sid,
                "label": int(label),
                "decision_score": float(score),
                "outer_fold": int(outer_fold)
            })

        # ----------------------------------------------------
        # Outer summary
        # ----------------------------------------------------

        outer_rows.append({
            "outer_fold": int(outer_fold),
            "auc": float(fold_auc),
            "inner_best_auc": float(
                grid.best_score_
            ),
            "best_k": int(best_k),
            "best_C": float(best_c),
            "n_train": int(
                train_mask.sum()
            ),
            "n_test": int(
                test_mask.sum()
            )
        })

    # ========================================================
    # OOF integrity
    # ========================================================

    oof = pd.DataFrame(
        oof_rows
    )

    if len(oof) != len(X):

        raise RuntimeError(
            f"Expected {len(X)} OOF rows, "
            f"got {len(oof)}."
        )

    if oof["subject_id"].nunique() != len(X):

        raise RuntimeError(
            "OOF subject duplication detected."
        )

    if oof["subject_id"].duplicated().any():

        raise RuntimeError(
            "Duplicate OOF subject IDs detected."
        )

    # ========================================================
    # OOF AUC
    # ========================================================

    auc = roc_auc_score(
        oof["label"],
        oof["decision_score"]
    )

    return (
        float(auc),
        oof,
        pd.DataFrame(outer_rows)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXP19 — EXP16 LABEL PERMUTATION TEST")
print("=" * 70)

print("\nLoading files...")

manifest = pd.read_csv(
    MANIFEST
)

assignments = pd.read_csv(
    ASSIGNMENTS
)

features = pd.read_csv(
    FEATURES
)

# ------------------------------------------------------------
# Normalize subject IDs
# ------------------------------------------------------------

features["subject_id"] = (
    features["subject_id"].astype(str)
)

manifest["subject_id"] = (
    manifest["subject_id"].astype(str)
)

assignments["subject_id"] = (
    assignments["subject_id"].astype(str)
)

# ------------------------------------------------------------
# Cohort validation
# ------------------------------------------------------------

rms_assignments = validate_cohort(
    features,
    manifest,
    assignments
)

# ============================================================
# RMS SUBJECTS
# ============================================================

rms_ids = set(
    manifest.loc[
        manifest["has_rms_representation"].astype(bool),
        "subject_id"
    ].astype(str)
)

features = features[
    features["subject_id"].isin(rms_ids)
].copy()

# ============================================================
# LABEL MERGE
# ============================================================

label_map = (
    manifest[
        ["subject_id", "label"]
    ]
    .drop_duplicates(
        "subject_id"
    )
)

features = features.merge(
    label_map,
    on="subject_id",
    how="left",
    validate="one_to_one"
)

if features["label"].isna().any():

    raise RuntimeError(
        "Missing labels after merge."
    )

features["label"] = (
    features["label"].astype(int)
)

features = (
    features
    .sort_values("subject_id")
    .reset_index(drop=True)
)

# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_cols = [
    c
    for c in features.columns
    if c.startswith("GMVOL_AAL_")
]

if len(feature_cols) != 116:

    raise RuntimeError(
        f"Expected 116 GM volume features, "
        f"got {len(feature_cols)}."
    )

X = (
    features
    .set_index("subject_id")
    [feature_cols]
    .copy()
)

y = (
    features
    .set_index("subject_id")
    ["label"]
    .copy()
)

# ------------------------------------------------------------
# Restrict canonical assignments to RMS subjects
# ------------------------------------------------------------

rms_assignments = rms_assignments[
    rms_assignments["subject_id"].isin(
        X.index
    )
].copy()

# ============================================================
# DATA VALIDATION
# ============================================================

print("\n=== DATA ===")

print(
    "Subjects :",
    len(X)
)

print(
    "Controls :",
    int((y == 0).sum())
)

print(
    "SCZ      :",
    int((y == 1).sum())
)

print(
    "Features :",
    X.shape[1]
)

if len(X) != 119:

    raise RuntimeError(
        "Expected exactly 119 RMS subjects."
    )

label_counts = (
    y.value_counts()
    .to_dict()
)

if label_counts != {
    1: 79,
    0: 40
}:

    raise RuntimeError(
        "Unexpected label distribution: "
        f"{label_counts}"
    )

if not np.isfinite(
    X.to_numpy()
).all():

    raise RuntimeError(
        "Feature matrix contains NaN/Inf."
    )

# ============================================================
# ASSIGNMENT VALIDATION
# ============================================================

expected_roles = {
    "INNER_TRAIN",
    "INNER_VALIDATION",
    "OUTER_TEST"
}

actual_roles = set(
    rms_assignments["role"].astype(str)
)

if actual_roles != expected_roles:

    raise RuntimeError(
        "Unexpected CV role values: "
        f"{actual_roles}"
    )

expected_outer_folds = {
    1, 2, 3, 4, 5
}

actual_outer_folds = set(
    rms_assignments[
        "outer_fold"
    ].astype(int)
)

if actual_outer_folds != expected_outer_folds:

    raise RuntimeError(
        "Unexpected outer folds: "
        f"{actual_outer_folds}"
    )

print(
    "CV assignment validation PASS."
)

# ============================================================
# REFERENCE REPRODUCTION
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE REPRODUCTION")
print("=" * 70)

reference_auc, reference_oof, reference_outer = (
    run_nested_cv(
        X,
        y,
        rms_assignments
    )
)

print(
    f"\nExpected Exp16 OOF AUC : "
    f"{EXPECTED_AUC:.6f}"
)

print(
    f"Recomputed OOF AUC     : "
    f"{reference_auc:.6f}"
)

difference = abs(
    reference_auc - EXPECTED_AUC
)

print(
    f"Absolute difference    : "
    f"{difference:.12f}"
)

# ============================================================
# HARD SAFETY CHECK
# ============================================================

if difference > 1e-6:

    raise RuntimeError(
        "\nREFERENCE REPRODUCTION FAILED.\n"
        "The recomputed Exp16 AUC does not match "
        "0.692722.\n"
        "Permutation testing has been stopped to prevent "
        "testing the wrong pipeline."
    )

print(
    "REFERENCE REPRODUCTION PASS."
)

# ============================================================
# SAVE REFERENCE RESULTS
# ============================================================

reference_oof.to_csv(
    OUTDIR / "reference_exp16_oof.csv",
    index=False
)

reference_outer.to_csv(
    OUTDIR / "reference_exp16_outer_summary.csv",
    index=False
)

# ============================================================
# PERMUTATION TEST
# ============================================================

print("\n" + "=" * 70)
print("STARTING PERMUTATION TEST")
print("=" * 70)

print(
    f"Number of permutations : "
    f"{N_PERMUTATIONS}"
)

print(
    f"Random seed             : "
    f"{SEED}"
)

print(
    f"Observed OOF AUC        : "
    f"{reference_auc:.6f}"
)

# ------------------------------------------------------------
# Fixed random generator
# ------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

# Original diagnosis labels.
# Each permutation preserves the original class counts.
y_original = y.to_numpy().copy()

null_results = []

# ============================================================
# 1000 GLOBAL LABEL PERMUTATIONS
# ============================================================

for perm_idx in range(
    1,
    N_PERMUTATIONS + 1
):

    # --------------------------------------------------------
    # Globally permute diagnosis labels.
    # Subject IDs, features and CV assignments remain fixed.
    # --------------------------------------------------------

    y_perm = rng.permutation(
        y_original
    )

    y_perm_series = pd.Series(
        y_perm,
        index=X.index,
        name="label"
    )

    # --------------------------------------------------------
    # Run the exact same nested CV/model-selection pipeline.
    # --------------------------------------------------------

    perm_auc, _, _ = run_nested_cv(
        X,
        y_perm_series,
        rms_assignments
    )

    null_results.append({
        "permutation": int(
            perm_idx
        ),
        "null_auc": float(
            perm_auc
        )
    })

    # --------------------------------------------------------
    # Progress report every 25 permutations.
    # --------------------------------------------------------

    if (
        perm_idx % 25 == 0
        or perm_idx == 1
    ):

        current = np.array([
            r["null_auc"]
            for r in null_results
        ])

        exceed = int(
            np.sum(
                current >= reference_auc
            )
        )

        p_current = (
            1 + exceed
        ) / (
            1 + perm_idx
        )

        print(
            f"\nPermutation "
            f"{perm_idx:4d}/{N_PERMUTATIONS}"
        )

        print(
            f"    Current null mean : "
            f"{np.mean(current):.6f}"
        )

        print(
            f"    Current null max  : "
            f"{np.max(current):.6f}"
        )

        print(
            f"    Null >= observed  : "
            f"{exceed}"
        )

        print(
            f"    Current empirical p ≈ "
            f"{p_current:.6f}"
        )

# ============================================================
# FINAL NULL DISTRIBUTION
# ============================================================

null_df = pd.DataFrame(
    null_results
)

if len(null_df) != N_PERMUTATIONS:

    raise RuntimeError(
        f"Expected {N_PERMUTATIONS} "
        f"permutations, got {len(null_df)}."
    )

null_df.to_csv(
    NULL_CSV,
    index=False
)

null_auc = (
    null_df["null_auc"]
    .to_numpy()
)

# ============================================================
# EMPIRICAL P-VALUE
# ============================================================

n_exceed = int(
    np.sum(
        null_auc >= reference_auc
    )
)

empirical_p = (
    1 + n_exceed
) / (
    1 + N_PERMUTATIONS
)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

null_mean = float(
    np.mean(null_auc)
)

null_std = float(
    np.std(null_auc)
)

null_median = float(
    np.median(null_auc)
)

null_min = float(
    np.min(null_auc)
)

null_max = float(
    np.max(null_auc)
)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "experiment":
        "Exp19",

    "parent_experiment":
        "Exp16",

    "description":
        (
            "Global diagnosis-label permutation test "
            "using the exact Exp16 Regional GM Volume "
            "+ Linear SVM nested-CV pipeline."
        ),

    "seed":
        SEED,

    "n_permutations":
        N_PERMUTATIONS,

    "n_subjects":
        int(len(X)),

    "n_controls":
        int(
            (y_original == 0).sum()
        ),

    "n_schizophrenia":
        int(
            (y_original == 1).sum()
        ),

    "n_features":
        int(X.shape[1]),

    "observed_auc":
        float(reference_auc),

    "expected_exp16_auc":
        float(EXPECTED_AUC),

    "reference_reproduction_difference":
        float(difference),

    "null_mean_auc":
        null_mean,

    "null_std_auc":
        null_std,

    "null_median_auc":
        null_median,

    "null_min_auc":
        null_min,

    "null_max_auc":
        null_max,

    "n_null_auc_ge_observed":
        n_exceed,

    "empirical_p_value":
        float(empirical_p),

    "p_formula":
        (
            "(1 + count(null_auc >= observed_auc)) "
            "/ (1 + n_permutations)"
        ),

    "reference_reproduction":
        "PASS",

    "reference_expected_auc":
        EXPECTED_AUC,

    "interpretation_rule":
        (
            "A small empirical p-value indicates that "
            "the observed OOF AUC is unusually high "
            "relative to the label-permuted null "
            "distribution under the same nested-CV "
            "pipeline."
        )
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "seed":
        SEED,

    "n_permutations":
        N_PERMUTATIONS,

    "expected_auc":
        EXPECTED_AUC,

    "feature_selection":
        "SelectKBest(f_classif)",

    "k_values":
        [
            20,
            40,
            60,
            80,
            100,
            116
        ],

    "scaler":
        "StandardScaler",

    "classifier":
        "LinearSVC",

    "C_values":
        [
            0.001,
            0.01,
            0.1,
            1,
            10,
            100
        ],

    "class_weight":
        "balanced",

    "max_iter":
        10000,

    "cv":
        "canonical nested 5x4 CV",

    "permutation":
        "global diagnosis-label permutation",

    "cohort":
        "119 RMS subjects",

    "controls":
        40,

    "schizophrenia":
        79,

    "features":
        116,

    "feature_type":
        "Regional GM Volume AAL features",

    "model_selection":
        "Nested inner CV",

    "primary_metric":
        "OOF ROC-AUC"
}

with open(
    CONFIG_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EXP19 COMPLETE")
print("=" * 70)

print(
    f"Observed OOF AUC       : "
    f"{reference_auc:.6f}"
)

print(
    f"Null mean AUC          : "
    f"{null_mean:.6f}"
)

print(
    f"Null SD                : "
    f"{null_std:.6f}"
)

print(
    f"Null median AUC        : "
    f"{null_median:.6f}"
)

print(
    f"Null minimum AUC       : "
    f"{null_min:.6f}"
)

print(
    f"Null maximum AUC       : "
    f"{null_max:.6f}"
)

print(
    f"Null >= observed       : "
    f"{n_exceed}"
)

print(
    f"Empirical p-value      : "
    f"{empirical_p:.6f}"
)

print("\nSaved:")
print(
    NULL_CSV
)

print(
    SUMMARY_JSON
)

print(
    CONFIG_JSON
)

print(
    OUTDIR / "reference_exp16_oof.csv"
)

print(
    OUTDIR / "reference_exp16_outer_summary.csv"
)

print("\nAll Exp19 checks completed successfully.")