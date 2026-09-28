from pathlib import Path
import json
import joblib
import numpy as np
from deployment_components import NuisanceResidualizer
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ================================================================
# NeuroFusion AI
# FINAL Exp20 DEPLOYMENT MODEL
# ================================================================

SEED = 42
N_FEATURES = 116
N_SUBJECTS = 119
N_SPLITS = 4

PROJECT_ROOT = Path(".")

FEATURE_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "vbm_spm"
    / "experiments"
    / "exp20_nuisance_controlled_svm"
    / "regional_gm_volume_features.csv"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

DEMOGRAPHIC_FILE = (
    PROJECT_ROOT
    / "temp_folder"
    / "1139_Demographics_20260804.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "final_model"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "exp20_final_deployment_model.joblib"
)

CONFIG_FILE = (
    MODEL_DIR
    / "exp20_final_deployment_config.json"
)

FEATURE_ORDER_FILE = (
    MODEL_DIR
    / "exp20_final_feature_order.json"
)


# ================================================================
# NUISANCE RESIDUALIZER
# ================================================================

def fit_model(
    X_train,
    y_train,
    nuisance_train,
    k,
    C
):

    residualizer = NuisanceResidualizer()

    X_residual = residualizer.fit(
        X_train,
        nuisance_train
    ).transform(
        X_train,
        nuisance_train
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_selected = selector.fit_transform(
        X_residual,
        y_train
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_selected
    )

    svm = LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=SEED,
        max_iter=10000
    )

    svm.fit(
        X_scaled,
        y_train
    )

    return {
        "residualizer": residualizer,
        "selector": selector,
        "scaler": scaler,
        "svm": svm
    }


# ================================================================
# PREDICT WITH ONE MODEL
# ================================================================

def predict_scores(
    model,
    X_data,
    nuisance_data
):

    X_residual = (
        model["residualizer"]
        .transform(
            X_data,
            nuisance_data
        )
    )

    X_selected = (
        model["selector"]
        .transform(
            X_residual
        )
    )

    X_scaled = (
        model["scaler"]
        .transform(
            X_selected
        )
    )

    return (
        model["svm"]
        .decision_function(
            X_scaled
        )
    )


# ================================================================
# LOAD FEATURES
# ================================================================

print("=" * 80)
print("NeuroFusion AI - FINAL Exp20 DEPLOYMENT MODEL")
print("=" * 80)

print("\n[1/9] Loading regional GM-volume features...")

features = pd.read_csv(
    FEATURE_FILE
)

features["subject_id"] = (
    features["subject_id"]
    .astype(str)
    .str.strip()
)

feature_columns = [
    c
    for c in features.columns
    if c.startswith("GMVOL_AAL_")
]

if len(feature_columns) != N_FEATURES:
    raise ValueError(
        f"Expected {N_FEATURES} features, "
        f"found {len(feature_columns)}."
    )

features = features.sort_values(
    "subject_id"
).reset_index(
    drop=True
)

print(
    f"  Subjects: {len(features)}"
)

print(
    f"  Features: {len(feature_columns)}"
)


# ================================================================
# LOAD LABELS
# ================================================================

print("\n[2/9] Loading canonical labels...")

manifest = pd.read_csv(
    MANIFEST_FILE
)

manifest["subject_id"] = (
    manifest["subject_id"]
    .astype(str)
    .str.strip()
)

label_table = (
    manifest[
        ["subject_id", "label"]
    ]
    .drop_duplicates()
)

if label_table["subject_id"].duplicated().any():
    raise ValueError(
        "Duplicate subject labels found."
    )

features = features.merge(
    label_table,
    on="subject_id",
    how="left",
    validate="one_to_one"
)

if features["label"].isna().any():
    missing = features.loc[
        features["label"].isna(),
        "subject_id"
    ].tolist()

    raise ValueError(
        f"Missing labels: {missing}"
    )

features["label"] = (
    features["label"]
    .astype(int)
)

print(
    f"  Controls: {(features['label'] == 0).sum()}"
)

print(
    f"  Schizophrenia: {(features['label'] == 1).sum()}"
)


# ================================================================
# VERIFY RMS
# ================================================================

print("\n[3/9] Verifying RMS cohort...")

rms_ids = set(
    manifest.loc[
        manifest["acquisition_source"]
        .astype(str)
        .str.upper()
        .eq("RMS"),
        "subject_id"
    ]
)

feature_ids = set(
    features["subject_id"]
)

if feature_ids != rms_ids:
    raise ValueError(
        "Feature cohort does not exactly match RMS cohort."
    )

if len(features) != N_SUBJECTS:
    raise ValueError(
        f"Expected {N_SUBJECTS} RMS subjects, "
        f"found {len(features)}."
    )

print("  RMS cohort: PASS")
print("  119 subjects: PASS")


# ================================================================
# LOAD DEMOGRAPHICS
# ================================================================

print("\n[4/9] Loading demographics...")

demo = pd.read_csv(
    DEMOGRAPHIC_FILE
)

demo["Anonymized ID"] = (
    demo["Anonymized ID"]
    .astype(str)
    .str.strip()
)

demo["Current Age"] = pd.to_numeric(
    demo["Current Age"],
    errors="coerce"
)

subject_ids = (
    features["subject_id"]
    .tolist()
)

ages = []
genders = []

for sid in subject_ids:

    rows = demo[
        demo["Anonymized ID"] == sid
    ]

    if rows.empty:
        raise ValueError(
            f"No demographic record for {sid}"
        )

    valid_age = rows[
        "Current Age"
    ].dropna()

    if valid_age.empty:
        raise ValueError(
            f"No valid age for {sid}"
        )

    ages.append(
        float(valid_age.median())
    )

    gender_values = (
        rows["Gender"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(gender_values) != 1:
        raise ValueError(
            f"Gender missing/inconsistent for {sid}"
        )

    genders.append(
        gender_values[0]
    )

ages = np.asarray(
    ages,
    dtype=float
)

genders = np.asarray(
    genders,
    dtype=str
)

print(
    f"  Age range: "
    f"{ages.min():.3f} - {ages.max():.3f}"
)

print(
    f"  Gender categories: "
    f"{sorted(set(genders))}"
)


# ================================================================
# BUILD NUISANCE MATRIX
# ================================================================

print("\n[5/9] Building nuisance matrix...")

gender_series = pd.Series(
    genders,
    dtype="string"
)

gender_dummies = pd.get_dummies(
    gender_series,
    prefix="gender",
    drop_first=True,
    dtype=float
)

nuisance_df = pd.concat(
    [
        pd.DataFrame(
            {
                "age": ages
            }
        ),
        gender_dummies
    ],
    axis=1
)

nuisance_columns = (
    nuisance_df.columns.tolist()
)

nuisance_matrix = (
    nuisance_df.to_numpy(
        dtype=float
    )
)

print(
    f"  Nuisance columns: "
    f"{nuisance_columns}"
)


# ================================================================
# X / y
# ================================================================

X = features[
    feature_columns
].to_numpy(
    dtype=float
)

y = features[
    "label"
].to_numpy(
    dtype=int
)

if not np.all(
    np.isfinite(X)
):
    raise ValueError(
        "Non-finite feature values detected."
    )


# ================================================================
# MANUAL 4-FOLD HYPERPARAMETER SELECTION
# ================================================================

print("\n[6/9] Selecting final k and C using 4-fold CV...")

K_VALUES = [
    20,
    40,
    60,
    80,
    100,
    116
]

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED
)

results = []

splits = list(
    cv.split(X, y)
)

total_combinations = (
    len(K_VALUES)
    * len(C_VALUES)
)

combination_number = 0

for k in K_VALUES:

    for C in C_VALUES:

        combination_number += 1

        fold_auc = []

        for (
            train_idx,
            valid_idx
        ) in splits:

            model = fit_model(
                X[
                    train_idx
                ],
                y[
                    train_idx
                ],
                nuisance_matrix[
                    train_idx
                ],
                k,
                C
            )

            valid_scores = predict_scores(
                model,
                X[
                    valid_idx
                ],
                nuisance_matrix[
                    valid_idx
                ]
            )

            auc = roc_auc_score(
                y[
                    valid_idx
                ],
                valid_scores
            )

            fold_auc.append(
                auc
            )

        mean_auc = float(
            np.mean(fold_auc)
        )

        results.append(
            {
                "k": k,
                "C": C,
                "mean_auc": mean_auc,
                "fold_auc": fold_auc
            }
        )

        print(
            f"  [{combination_number:02d}/{total_combinations}] "
            f"k={k:3d}, C={C:<7g}, "
            f"mean AUC={mean_auc:.6f}"
        )


results_sorted = sorted(
    results,
    key=lambda r: (
        -r["mean_auc"],
        r["k"],
        r["C"]
    )
)

best = results_sorted[0]

best_k = int(
    best["k"]
)

best_C = float(
    best["C"]
)

best_cv_auc = float(
    best["mean_auc"]
)

print()
print(
    f"Selected k = {best_k}"
)

print(
    f"Selected C = {best_C}"
)

print(
    f"Best 4-fold CV ROC-AUC = "
    f"{best_cv_auc:.6f}"
)


# ================================================================
# FINAL FIT ON ALL 119
# ================================================================

print("\n[7/9] Fitting final model on all 119 RMS subjects...")

final_model = fit_model(
    X,
    y,
    nuisance_matrix,
    best_k,
    best_C
)

print("  Final fit: PASS")


# ================================================================
# CROSS-FITTED CALIBRATION SCORES
# ================================================================

print("\n[8/9] Building leakage-safe probability calibration...")

cross_fitted_scores = np.full(
    len(y),
    np.nan,
    dtype=float
)

for fold, (
    train_idx,
    valid_idx
) in enumerate(
    splits,
    start=1
):

    fold_model = fit_model(
        X[
            train_idx
        ],
        y[
            train_idx
        ],
        nuisance_matrix[
            train_idx
        ],
        best_k,
        best_C
    )

    valid_scores = predict_scores(
        fold_model,
        X[
            valid_idx
        ],
        nuisance_matrix[
            valid_idx
        ]
    )

    cross_fitted_scores[
        valid_idx
    ] = valid_scores

    print(
        f"  Fold {fold}: "
        f"{len(valid_idx)} validation subjects"
    )

if not np.all(
    np.isfinite(
        cross_fitted_scores
    )
):
    raise ValueError(
        "Cross-fitted scores are invalid."
    )

calibrator = LogisticRegression(
    random_state=SEED,
    max_iter=10000
)

calibrator.fit(
    cross_fitted_scores.reshape(
        -1,
        1
    ),
    y
)

print(
    "  Platt calibrator: PASS"
)


# ================================================================
# DEPLOYMENT PACKAGE
# ================================================================

deployment_package = {

    "model_type":
        "NeuroFusion AI Exp20 final deployment model",

    "model":
        final_model,

    "calibrator":
        calibrator,

    "feature_order":
        feature_columns,

    "nuisance_columns":
        nuisance_columns,

    "n_features":
        N_FEATURES,

    "training_subject_count":
        N_SUBJECTS,

    "training_controls":
        int((y == 0).sum()),

    "training_schizophrenia":
        int((y == 1).sum()),

    "selected_k":
        best_k,

    "selected_C":
        best_C,

    "hyperparameter_selection":
        "manual 4-fold StratifiedKFold",

    "hyperparameter_selection_auc":
        best_cv_auc,

    "feature_selection":
        "SelectKBest(f_classif)",

    "scaler":
        "StandardScaler",

    "classifier":
        "LinearSVC",

    "class_weight":
        "balanced",

    "calibration":
        "LogisticRegression sigmoid / Platt scaling",

    "seed":
        SEED,

    "reference_nested_cv":
        {
            "outer_folds": 5,
            "inner_folds": 4,
            "cohort": "RMS-only",
            "subjects": 119,
            "oof_roc_auc": 0.756646
        },

    "warning":
        "This model is fitted on all 119 RMS subjects. "
        "Training predictions must not be reported as "
        "independent validation performance. The previously "
        "completed nested-CV OOF ROC-AUC is the validation result."
}


# ================================================================
# SAVE JOBLIB
# ================================================================

joblib.dump(
    deployment_package,
    MODEL_FILE,
    compress=3
)

with FEATURE_ORDER_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_columns,
        f,
        indent=2
    )

config = {

    "model":
        "Exp20 final deployment model",

    "cohort":
        "RMS-only",

    "subjects":
        119,

    "controls":
        40,

    "schizophrenia":
        79,

    "features":
        116,

    "feature_order":
        str(FEATURE_ORDER_FILE),

    "nuisance_variables":
        nuisance_columns,

    "selected_k":
        best_k,

    "selected_C":
        best_C,

    "selection_method":
        "4-fold StratifiedKFold",

    "selection_cv_auc":
        best_cv_auc,

    "classifier":
        "LinearSVC",

    "calibration":
        "LogisticRegression sigmoid / Platt scaling",

    "reference_nested_cv_oof_auc":
        0.756646,

    "model_file":
        str(MODEL_FILE)
}

with CONFIG_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )


# ================================================================
# RELOAD TEST
# ================================================================

print("\n[9/9] Reloading saved .joblib...")

loaded = joblib.load(
    MODEL_FILE
)

loaded_model = loaded[
    "model"
]

loaded_calibrator = loaded[
    "calibrator"
]

loaded_scores = predict_scores(
    loaded_model,
    X,
    nuisance_matrix
)

loaded_probabilities = (
    loaded_calibrator
    .predict_proba(
        loaded_scores.reshape(
            -1,
            1
        )
    )[:, 1]
)

if not np.all(
    np.isfinite(
        loaded_scores
    )
):
    raise ValueError(
        "Reloaded model produced invalid scores."
    )

if not np.all(
    np.isfinite(
        loaded_probabilities
    )
):
    raise ValueError(
        "Reloaded calibrator produced invalid probabilities."
    )

if not np.all(
    (
        loaded_probabilities >= 0
    )
    &
    (
        loaded_probabilities <= 1
    )
):
    raise ValueError(
        "Reloaded probabilities outside [0,1]."
    )

print(
    "  Reload: PASS"
)

print(
    "  Decision scores: PASS"
)

print(
    "  Probability output: PASS"
)


# ================================================================
# FINAL STATUS
# ================================================================

print()
print("=" * 80)
print("FINAL EXP20 DEPLOYMENT MODEL CREATED SUCCESSFULLY")
print("=" * 80)

print()
print("MODEL:")
print(
    MODEL_FILE.resolve()
)

print()
print("FEATURE ORDER:")
print(
    FEATURE_ORDER_FILE.resolve()
)

print()
print("CONFIG:")
print(
    CONFIG_FILE.resolve()
)

print()
print(
    f"Selected k: {best_k}"
)

print(
    f"Selected C: {best_C}"
)

print(
    f"4-fold selection ROC-AUC: "
    f"{best_cv_auc:.6f}"
)

print()
print(
    "Reference nested-CV OOF ROC-AUC: 0.756646"
)

print()
print(
    "STATUS: READY FOR UNSEEN-DATA INFERENCE"
)

print()
print(
    "IMPORTANT: New subjects must go through the same "
    "sMRI preprocessing and AAL GM-volume feature extraction "
    "pipeline before this model is used."
)

print()
print("=" * 80)
