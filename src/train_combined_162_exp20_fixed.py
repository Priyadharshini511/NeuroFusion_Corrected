# train_combined_162_exp20.py
# NeuroFusion sMRI Exp20:
# Combine the already-prepared 119 RMS subjects + 43 OpenNeuro subjects
# and train one schizophrenia-vs-control model on all 162 subjects.
#
# 119 cohort:
#   - 116 GMVOL_AAL_* features: existing Exp16 feature file
#   - labels: existing RMS subject-level audit
#   - age/gender: temp_folder/1139_Demographics_20260804.csv
#
# 43 cohort:
#   - 116 GMVOL_AAL_* + age/gender: existing Exp20 inference-input file
#   - labels: OpenNeuro ds000115 participants.tsv
#       CON    -> 0 (control)
#       SCZ    -> 1 (schizophrenia)
#       SCZ-SIB / CON-SIB -> excluded
#
# IMPORTANT:
# - The 43 subjects are NOT treated as an external test set anymore.
# - All 162 subjects are used for model development.
# - Nested CV gives the unbiased development-performance estimate.
# - After CV, a final model is fitted on all 162 subjects and saved.
# - Source/dataset is NOT used as a predictive feature.
# - Only the 116 GM-volume features enter the classifier.
# - Age and gender are nuisance variables and are residualized inside
#   each training partition to avoid leakage.

from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

ROOT = Path.cwd()

F119 = ROOT / "processed_data/smri/vbm_spm/experiments/exp16_regional_gm_svm/regional_gm_volume_features.csv"
AUDIT119 = ROOT / "processed_data/smri/audit/rms_input_audit/rms_subject_level_audit.csv"
DEMO119 = ROOT / "temp_folder/1139_Demographics_20260804.csv"

F43 = ROOT / "processed_data/smri/openneuro_ds000115/features/openneuro_43_subjects_exp20_inference_input.csv"

# participants.tsv is searched automatically because its exact location may differ.
OPENNEURO_ROOT = ROOT / "processed_data/smri/openneuro_ds000115"

OUT = ROOT / "processed_data/smri/combined_162_exp20"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 4

EXPECTED_FEATURES = 116


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def norm_id(x):
    """
    Normalize subject IDs across RMS and OpenNeuro.

    OpenNeuro may represent the same subject as sub-1 or sub-01.
    Both are normalized to SUB-1 so feature rows and participants.tsv
    match reliably. RMS IDs such as A00001243 are preserved.
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()
    s = re.sub(r"^participant[-_]", "", s, flags=re.I)

    # Canonicalize OpenNeuro IDs: sub-01, sub-1, SUB-001 -> SUB-1.
    m = re.fullmatch(r"sub[-_](\d+)", s, flags=re.I)
    if m:
        return f"SUB-{int(m.group(1))}"

    # Keep other IDs stable and case-insensitive.
    return s.upper()


def feature_contract(df, name):
    cols = [c for c in df.columns if c.startswith("GMVOL_AAL_")]
    if len(cols) != EXPECTED_FEATURES:
        raise ValueError(
            f"{name}: found {len(cols)} GMVOL_AAL features; expected {EXPECTED_FEATURES}."
        )
    return cols


def find_col(df, candidates, required=True):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def load_rms_labels():
    """Use the already-validated 119-subject audit as the label source."""
    if not AUDIT119.exists():
        raise FileNotFoundError(f"Missing RMS audit: {AUDIT119}")

    a = pd.read_csv(AUDIT119)

    id_col = find_col(a, ["subject_id"])
    label_col = find_col(a, ["label"])

    a = a[[id_col, label_col]].copy()
    a["subject_id_norm"] = a[id_col].map(norm_id)

    if a["subject_id_norm"].duplicated().any():
        dup = a.loc[a["subject_id_norm"].duplicated(), "subject_id_norm"].tolist()
        raise ValueError(f"Duplicate RMS subject IDs in audit: {dup}")

    a[label_col] = pd.to_numeric(a[label_col], errors="coerce")
    if a[label_col].isna().any():
        raise ValueError("Missing RMS labels in rms_subject_level_audit.csv")

    a["label"] = a[label_col].astype(int)
    if not set(a["label"].unique()).issubset({0, 1}):
        raise ValueError("RMS audit labels are not binary 0/1.")

    return a[["subject_id_norm", "label"]].copy()


def load_rms_demographics():
    """
    Load the actual demographic CSV found by the user.
    Row 2 contains variable-code labels, so skip it.
    """
    if not DEMO119.exists():
        raise FileNotFoundError(f"Missing RMS demographics CSV: {DEMO119}")

    d = pd.read_csv(DEMO119, skiprows=[1])

    id_col = find_col(d, ["Anonymized ID", "subject_id", "participant_id"])
    age_col = find_col(d, ["Current Age", "age"])
    gender_col = find_col(d, ["Gender", "gender", "sex"])

    out = d[[id_col, age_col, gender_col]].copy()
    out.columns = ["subject_id_norm", "age", "gender"]
    out["subject_id_norm"] = out["subject_id_norm"].map(norm_id)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["gender"] = pd.to_numeric(out["gender"], errors="coerce")

    # The demographics file has repeated records. Keep one subject-level
    # record: median age and a consistency-checked gender.
    rows = []
    for sid, g in out.groupby("subject_id_norm", sort=False):
        if not sid:
            continue

        ages = g["age"].dropna()
        genders = g["gender"].dropna().unique()

        if len(genders) > 1:
            raise ValueError(f"Gender conflict for RMS subject {sid}: {genders}")

        rows.append({
            "subject_id_norm": sid,
            "age": float(ages.median()) if len(ages) else np.nan,
            "gender": float(genders[0]) if len(genders) else np.nan,
        })

    return pd.DataFrame(rows)


def locate_participants_tsv():
    candidates = [
        OPENNEURO_ROOT / "participants.tsv",
        OPENNEURO_ROOT / "ds000115" / "participants.tsv",
        ROOT / "participants.tsv",
        ROOT / "temp_folder" / "participants.tsv",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(OPENNEURO_ROOT.rglob("participants.tsv")) if OPENNEURO_ROOT.exists() else []
    if found:
        return found[0]

    raise FileNotFoundError(
        "Could not find OpenNeuro participants.tsv under "
        f"{OPENNEURO_ROOT}"
    )


def load_openneuro_labels():
    """
    Load OpenNeuro ds000115 labels from participants.tsv.

    IMPORTANT:
    - SCZ = schizophrenia -> label 1
    - CON = control -> label 0
    - SCZ-SIB and CON-SIB are sibling/control-sibling categories and
      are intentionally EXCLUDED.

    The existing 43-subject feature CSV is the cohort to be trained on.
    Therefore this function returns labels only for the true SCZ/CON
    participants and does not silently convert sibling groups into cases
    or controls.
    """
    p = locate_participants_tsv()
    d = pd.read_csv(p, sep="\t")

    id_col = find_col(d, ["participant_id", "subject_id"])

    # ds000115 uses 'condit', not 'group'.
    condition_col = find_col(
        d, ["condit", "condition", "diagnosis", "group"]
    )

    out = d[[id_col, condition_col]].copy()
    out.columns = ["subject_id_norm", "condition"]
    out["subject_id_norm"] = out["subject_id_norm"].map(norm_id)
    out["condition"] = (
        out["condition"].astype(str).str.strip().str.upper()
    )

    if out["subject_id_norm"].duplicated().any():
        dup = out.loc[
            out["subject_id_norm"].duplicated(keep=False),
            "subject_id_norm"
        ].tolist()
        raise ValueError(
            "Duplicate OpenNeuro participant IDs in participants.tsv: "
            f"{dup}"
        )

    print("\nOpenNeuro condition counts in participants.tsv:")
    print(out["condition"].value_counts(dropna=False))

    # Only genuine disease/control groups are eligible for the 43-subject
    # classifier. Sibling groups are deliberately not mapped to labels.
    valid = out[out["condition"].isin(["SCZ", "CON"])].copy()

    # Explicitly report what is being excluded.
    excluded = out[~out["condition"].isin(["SCZ", "CON"])].copy()
    if not excluded.empty:
        print("\nExcluded OpenNeuro sibling/non-target groups:")
        print(excluded["condition"].value_counts())

    valid["label"] = valid["condition"].map({
        "CON": 0,
        "SCZ": 1,
    }).astype(int)

    print("\nOpenNeuro usable disease/control labels:")
    print(valid["label"].value_counts().sort_index())
    print("SCZ subjects:", int((valid["label"] == 1).sum()))
    print("CON subjects:", int((valid["label"] == 0).sum()))
    print("Sibling/non-target subjects excluded:", len(excluded))

    return valid[["subject_id_norm", "label", "condition"]].copy(), p


def find_age_gender_columns_43(df):
    age_col = find_col(df, ["age", "Age"], required=False)
    gender_col = find_col(
        df, ["gender", "gender_code", "sex", "Sex"], required=False
    )

    if age_col is None or gender_col is None:
        raise ValueError(
            "The 43-subject feature file must already contain age and gender. "
            f"Found age={age_col}, gender={gender_col}. Columns={list(df.columns)}"
        )

    return age_col, gender_col


# ---------------------------------------------------------------------
# Nuisance residualization
# ---------------------------------------------------------------------

class NuisanceResidualizer(BaseEstimator, TransformerMixin):
    """
    For each imaging feature, fit:
        feature = beta0 + beta1*age + beta2*gender + residual

    Fit only on training subjects, then apply the learned coefficients
    to validation/test subjects.
    """

    def __init__(self):
        self.models_ = None
        self.medians_ = None

    def fit_with_nuisance(self, X, Z):
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)

        if Z.ndim != 2 or Z.shape[1] != 2:
            raise ValueError("Expected nuisance matrix with age + gender.")

        self.n_features_in_ = X.shape[1]
        self.medians_ = np.nanmedian(Z, axis=0)

        Z2 = Z.copy()
        inds = np.where(np.isnan(Z2))
        if len(inds[0]):
            Z2[inds] = np.take(self.medians_, inds[1])

        # LinearRegression already includes an intercept.
        self.models_ = []
        for j in range(X.shape[1]):
            model = LinearRegression()
            model.fit(Z2, X[:, j])
            self.models_.append(model)

        return self

    def transform_with_nuisance(self, X, Z):
        if self.models_ is None:
            raise RuntimeError("Residualizer has not been fitted.")

        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)

        Z2 = Z.copy()
        inds = np.where(np.isnan(Z2))
        if len(inds[0]):
            Z2[inds] = np.take(self.medians_, inds[1])

        R = np.empty_like(X, dtype=float)
        for j, model in enumerate(self.models_):
            R[:, j] = X[:, j] - model.predict(Z2)

        return R


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def safe_metrics(y, score, pred):
    y = np.asarray(y)
    score = np.asarray(score)
    pred = np.asarray(pred)

    tp = np.sum((y == 1) & (pred == 1))
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))

    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------
# Load the already-prepared datasets
# ---------------------------------------------------------------------

for p in [F119, AUDIT119, DEMO119, F43]:
    if not p.exists():
        raise FileNotFoundError(p)

d119 = pd.read_csv(F119)
d43 = pd.read_csv(F43)

features119 = feature_contract(d119, "119 feature file")
features43 = feature_contract(d43, "43 feature file")

# EXACT feature compatibility check.
if features43 != features119:
    missing = [c for c in features119 if c not in features43]
    extra = [c for c in features43 if c not in features119]
    raise ValueError(
        "The 119 and 43 feature contracts do not match.\n"
        f"Missing in 43: {missing}\n"
        f"Extra in 43: {extra}"
    )

feature_cols = features119

# ---------------- 119: labels + demographics ----------------

rms_labels = load_rms_labels()
rms_demo = load_rms_demographics()

d119["subject_id_norm"] = d119["subject_id"].map(norm_id)

a = d119[["subject_id_norm"] + feature_cols].copy()
a = a.merge(rms_labels, on="subject_id_norm", how="left", validate="one_to_one")
a = a.merge(rms_demo, on="subject_id_norm", how="left", validate="one_to_one")
a["source"] = "RMS119"

if len(a) != 119:
    raise ValueError(f"Expected 119 RMS rows, got {len(a)}")

if a["label"].isna().any():
    bad = a.loc[a["label"].isna(), "subject_id_norm"].tolist()
    raise ValueError(f"Missing RMS labels after ID matching: {bad}")

if a["age"].isna().any() or a["gender"].isna().any():
    bad = a.loc[a["age"].isna() | a["gender"].isna(),
                ["subject_id_norm", "age", "gender"]]
    raise ValueError(
        "Missing RMS age/gender after matching the demographics CSV:\n"
        f"{bad.to_string(index=False)}"
    )

# ---------------- 43: existing age/gender + participants labels ----------------

open_labels, participants_path = load_openneuro_labels()

d43["subject_id_norm"] = d43["subject_id"].map(norm_id)
age43, gender43 = find_age_gender_columns_43(d43)

# The feature CSV is already the validated 43-subject cohort.
# Match those 43 rows ONLY against SCZ/CON participants.
b = d43[["subject_id_norm", age43, gender43] + feature_cols].copy()
b = b.rename(columns={age43: "age", gender43: "gender"})
b = b.merge(
    open_labels,
    on="subject_id_norm",
    how="left",
    validate="one_to_one"
)
b["source"] = "OpenNeuro43"

if len(b) != 43:
    raise ValueError(f"Expected 43 OpenNeuro rows, got {len(b)}")

if b["label"].isna().any():
    bad = b.loc[b["label"].isna(), "subject_id_norm"].tolist()
    raise ValueError(
        "Some of the 43 feature subjects are not SCZ or CON in "
        "participants.tsv. These rows cannot be used for the 162-subject "
        "SCZ-vs-CON training set:\n"
        f"{bad}"
    )

# Verify the requested 43-subject composition before training.
open_counts = b["label"].value_counts().sort_index()
expected_open_counts = {0: 20, 1: 23}
actual_open_counts = {int(k): int(v) for k, v in open_counts.items()}
if actual_open_counts != expected_open_counts:
    raise ValueError(
        "Unexpected OpenNeuro 43-subject label composition. Expected "
        "20 CON (label 0) + 23 SCZ (label 1), got "
        f"{actual_open_counts}. "
        "Check the feature CSV subject list and participants.tsv."
    )

# Keep the original condition for an audit artifact, but do NOT use it
# as a model feature.
b = b.rename(columns={"condition": "openneuro_condition"}) if "condition" in b.columns else b

if b["age"].isna().any() or b["gender"].isna().any():
    bad = b.loc[b["age"].isna() | b["gender"].isna(),
                ["subject_id_norm", "age", "gender"]]
    raise ValueError(
        "Missing OpenNeuro age/gender in the existing 43-subject feature file:\n"
        f"{bad.to_string(index=False)}"
    )

# ---------------- exact final schema ----------------

keep = ["subject_id_norm", "label", "age", "gender", "source"] + feature_cols
a = a[keep].copy()
b = b[keep].copy()

combined = pd.concat([a, b], ignore_index=True)

if len(combined) != 162:
    raise ValueError(f"Expected 162 combined subjects, got {len(combined)}")

if combined["subject_id_norm"].duplicated().any():
    dup = combined.loc[
        combined["subject_id_norm"].duplicated(keep=False),
        "subject_id_norm"
    ].tolist()
    raise ValueError(
        "Duplicate subject IDs across the 119 + 43 cohorts. "
        f"Cannot train until resolved: {dup}"
    )

combined["label"] = pd.to_numeric(combined["label"], errors="raise").astype(int)

if not set(combined["label"].unique()).issubset({0, 1}):
    raise ValueError("Final labels are not binary 0/1.")

if combined[feature_cols].isna().any().any():
    raise ValueError("Missing GM-volume features found.")

if not np.isfinite(combined[feature_cols].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite GM-volume features found.")

combined.to_csv(
    OUT / "combined_162_subjects_116_GMVOL_features.csv",
    index=False
)

print("\n============================================================")
print("COMBINED 162-SUBJECT DATASET")
print("============================================================")
print("119 RMS + 43 OpenNeuro =", len(combined))
print("Features               =", len(feature_cols))
print("\nSource counts:")
print(combined["source"].value_counts())

print("\nLabel counts:")
print(combined["label"].value_counts().sort_index())

print("\nSource x label:")
print(combined.groupby(["source", "label"]).size())

print("\nRMS demographics source:")
print(DEMO119)

print("\nOpenNeuro labels source:")
print(participants_path)

# ---------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------

X = combined[feature_cols].to_numpy(dtype=float)
y = combined["label"].to_numpy(dtype=int)
Z = combined[["age", "gender"]].to_numpy(dtype=float)

source = combined["source"].to_numpy(dtype=str)

# This controls fold composition only. It is NOT an input feature.
strata = np.array([
    f"{src}_{yy}" for src, yy in zip(source, y)
])

outer = StratifiedKFold(
    n_splits=N_SPLITS_OUTER,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_score = np.full(len(combined), np.nan)
oof_pred = np.full(len(combined), -1)
fold_rows = []

for fold, (tr, te) in enumerate(outer.split(X, strata), start=1):
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    Ztr, Zte = Z[tr], Z[te]
    strat_tr = strata[tr]

    # Nuisance control is fitted ONLY on outer-training subjects.
    residualizer = NuisanceResidualizer().fit_with_nuisance(Xtr, Ztr)
    Xtr_r = residualizer.transform_with_nuisance(Xtr, Ztr)
    Xte_r = residualizer.transform_with_nuisance(Xte, Zte)

    inner = StratifiedKFold(
        n_splits=N_SPLITS_INNER,
        shuffle=True,
        random_state=RANDOM_STATE + fold
    )

    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_classif)),
        ("scale", StandardScaler()),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={
            "select__k": [20, 40, 60, 80, 100, 116],
            "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
        scoring="roc_auc",
        cv=inner.split(Xtr_r, strat_tr),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(Xtr_r, ytr)

    score = grid.best_estimator_.decision_function(Xte_r)
    pred = (score >= 0).astype(int)

    oof_score[te] = score
    oof_pred[te] = pred

    met = safe_metrics(yte, score, pred)
    met.update({
        "fold": fold,
        "best_k": int(grid.best_params_["select__k"]),
        "best_C": float(grid.best_params_["clf__C"]),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    })
    fold_rows.append(met)

    print(
        f"\nFold {fold}: "
        f"AUC={met['roc_auc']:.6f} | "
        f"ACC={met['accuracy']:.6f} | "
        f"BA={met['balanced_accuracy']:.6f} | "
        f"Sens={met['sensitivity']:.6f} | "
        f"Spec={met['specificity']:.6f} | "
        f"k={met['best_k']} | C={met['best_C']}"
    )

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(
    OUT / "combined_162_nested_outer_fold_summary.csv",
    index=False
)

overall = safe_metrics(y, oof_score, oof_pred)

print("\n============================================================")
print("162-SUBJECT NESTED-CV RESULT")
print("============================================================")
for k, v in overall.items():
    print(f"{k}: {v}")

print(f"\nMean outer AUC: {fold_df['roc_auc'].mean():.6f}")
print(f"Outer AUC SD:   {fold_df['roc_auc'].std(ddof=1):.6f}")

oof = combined[["subject_id_norm", "label", "source"]].copy()
oof["oof_decision_score"] = oof_score
oof["oof_prediction"] = oof_pred
oof.to_csv(OUT / "combined_162_nested_oof.csv", index=False)

# ---------------------------------------------------------------------
# Final model: fit on ALL 162 subjects
# ---------------------------------------------------------------------
#
# This is the deployable development model.
# The 43 subjects are no longer an independent external test set.

print("\n============================================================")
print("TRAINING FINAL MODEL ON ALL 162 SUBJECTS")
print("============================================================")

final_residualizer = NuisanceResidualizer().fit_with_nuisance(X, Z)
X_final_r = final_residualizer.transform_with_nuisance(X, Z)

final_inner = StratifiedKFold(
    n_splits=N_SPLITS_INNER,
    shuffle=True,
    random_state=RANDOM_STATE
)

final_pipe = Pipeline([
    ("select", SelectKBest(score_func=f_classif)),
    ("scale", StandardScaler()),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )),
])

final_grid = GridSearchCV(
    final_pipe,
    param_grid={
        "select__k": [20, 40, 60, 80, 100, 116],
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    scoring="roc_auc",
    cv=final_inner.split(X_final_r, strata),
    refit=True,
    n_jobs=-1,
)

final_grid.fit(X_final_r, y)

print("Final best parameters:", final_grid.best_params_)
print("Final inner-CV best AUC:", final_grid.best_score_)

bundle = {
    "experiment": "Exp20_combined_162",
    "feature_columns": feature_cols,
    "nuisance_columns": ["age", "gender"],
    "label_definition": {
        "0": "control",
        "1": "schizophrenia"
    },
    "source_definition": {
        "RMS119": "RMS cohort",
        "OpenNeuro43": "OpenNeuro ds000115"
    },
    "openneuro_label_rule": {
        "CON": 0,
        "SCZ": 1,
        "SCZ-SIB": "excluded",
        "CON-SIB": "excluded"
    },
    "residualizer": final_residualizer,
    "model": final_grid.best_estimator_,
    "best_params": final_grid.best_params_,
    "training_subjects": 162,
}

joblib.dump(
    bundle,
    OUT / "exp20_combined_162_final_model.joblib"
)

summary = {
    "experiment": "Exp20_combined_162",
    "subjects": 162,
    "RMS_subjects": 119,
    "OpenNeuro_subjects": 43,
    "features": 116,
    "feature_representation": "regional GM volume from SPM mwc1 maps",
    "nuisance_variables": ["age", "gender"],
    "classifier": "LinearSVC(class_weight='balanced')",
    "feature_selection": "SelectKBest(f_classif) inside inner CV",
    "scaling": "StandardScaler inside inner CV",
    "outer_cv": "5-fold stratified by source+diagnosis",
    "inner_cv": "4-fold stratified by source+diagnosis",
    "primary_metric": "OOF ROC-AUC",
    "nested_cv_metrics": overall,
    "mean_outer_auc": float(fold_df["roc_auc"].mean()),
    "outer_auc_sd": float(fold_df["roc_auc"].std(ddof=1)),
    "final_model_best_params": final_grid.best_params_,
    "final_model_inner_cv_best_auc": float(final_grid.best_score_),
    "external_test_status": "The 43-subject cohort is no longer an independent external test set.",
    "purpose": "Train a unified sMRI schizophrenia-vs-control classifier on all 162 subjects."
}

(OUT / "combined_162_nested_cv_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n============================================================")
print("DONE")
print("============================================================")
print("Combined dataset:")
print(OUT / "combined_162_subjects_116_GMVOL_features.csv")
print("\nNested CV:")
print(OUT / "combined_162_nested_outer_fold_summary.csv")
print(OUT / "combined_162_nested_oof.csv")
print(OUT / "combined_162_nested_cv_summary.json")
print("\nFINAL DEPLOYABLE MODEL:")
print(OUT / "exp20_combined_162_final_model.joblib")
