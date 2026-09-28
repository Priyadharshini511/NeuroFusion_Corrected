
# train_combined_162_exp20.py
# NeuroFusion sMRI: merge the already-preprocessed 119 + 43 cohorts and
# retrain/evaluate an Exp20-style model on the combined 162 subjects.
#
# IMPORTANT:
# - Does NOT inspect raw MRI, MPRAGE, RMS, preprocessing, or re-run preprocessing.
# - Uses the existing model-ready 116 GMVOL_AAL_* features.
# - The original 43-subject cohort is no longer an external test set after merging.
# - Creates NEW artifacts; it does not overwrite the validated 119-subject Exp20 files.

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

ROOT = Path.cwd()

F119 = ROOT / "processed_data/smri/vbm_spm/experiments/exp16_regional_gm_svm/regional_gm_volume_features.csv"
F43 = ROOT / "processed_data/smri/openneuro_ds000115/features/openneuro_43_subjects_exp20_inference_input.csv"

M119 = ROOT / "processed_data/smri/dataset/smri_subject_manifest_labeled.csv"
M43 = ROOT / "processed_data/smri/openneuro_ds000115/openneuro_smri_manifest.csv"

OUT = ROOT / "processed_data/smri/combined_162_exp20"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 4

# ---------- helpers ----------

def norm_id(x):
    s = str(x).strip()
    if s.startswith("sub-"):
        return "sub-" + str(int(s[4:]))
    return s

def find_label_col(df):
    for c in ["label", "diagnosis", "target", "class"]:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")

def find_age_col(df):
    for c in ["age", "Age", "AGE"]:
        if c in df.columns:
            return c
    return None

def find_gender_col(df):
    for c in ["gender", "gender_code", "sex", "Sex", "GENDER"]:
        if c in df.columns:
            return c
    return None

def aggregate_demographics(manifest, id_col="subject_id"):
    """One row per subject. Median age; gender must be consistent."""
    m = manifest.copy()
    m["subject_id_norm"] = m[id_col].map(norm_id)

    label_col = find_label_col(m)
    age_col = find_age_col(m)
    gender_col = find_gender_col(m)

    rows = []
    for sid, g in m.groupby("subject_id_norm", sort=False):
        labels = pd.to_numeric(g[label_col], errors="coerce").dropna().unique()
        if len(labels) != 1:
            raise ValueError(f"Label conflict/missing for {sid}: {labels}")
        row = {
            "subject_id_norm": sid,
            "label": int(labels[0]),
        }

        if age_col:
            age = pd.to_numeric(g[age_col], errors="coerce").dropna()
            row["age"] = float(age.median()) if len(age) else np.nan
        else:
            row["age"] = np.nan

        if gender_col:
            gv = pd.to_numeric(g[gender_col], errors="coerce").dropna().unique()
            if len(gv) > 1:
                raise ValueError(f"Gender conflict for {sid}: {gv}")
            row["gender"] = float(gv[0]) if len(gv) else np.nan
        else:
            row["gender"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

class NuisanceResidualizer(BaseEstimator, TransformerMixin):
    """
    Residualize each imaging feature against age + gender.
    Fit ONLY on the training partition.
    """
    def __init__(self):
        self.models_ = None
        self.medians_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self.n_features_in_ = X.shape[1]

        # X_nuisance is supplied separately through fit/transform below
        return self

    def fit_with_nuisance(self, X, Z):
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        self.n_features_in_ = X.shape[1]

        Z = self._prepare_Z(Z, fit=True)
        self.models_ = []
        for j in range(X.shape[1]):
            model = LinearRegression()
            model.fit(Z, X[:, j])
            self.models_.append(model)
        return self

    def transform_with_nuisance(self, X, Z):
        if self.models_ is None:
            raise RuntimeError("Residualizer not fitted")
        X = np.asarray(X, dtype=float)
        Z = self._prepare_Z(Z, fit=False)
        R = np.empty_like(X, dtype=float)
        for j, model in enumerate(self.models_):
            R[:, j] = X[:, j] - model.predict(Z)
        return R

    def _prepare_Z(self, Z, fit):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != 2:
            raise ValueError("Expected nuisance matrix with age + gender")
        if fit:
            self.medians_ = np.nanmedian(Z, axis=0)
        Z = Z.copy()
        inds = np.where(np.isnan(Z))
        if len(inds[0]):
            Z[inds] = np.take(self.medians_, inds[1])
        # include intercept via LinearRegression; standardize nuisance not needed
        return np.column_stack([np.ones(len(Z)), Z])

def safe_metrics(y, score, pred):
    out = {}
    out["roc_auc"] = float(roc_auc_score(y, score))
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    tp = np.sum((y == 1) & (pred == 1))
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out["precision"] = float(tp / (tp + fp)) if (tp + fp) else np.nan
    out["f1"] = float(2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) else np.nan
    out["tn"], out["fp"], out["fn"], out["tp"] = map(int, [tn, fp, fn, tp])
    return out

# ---------- load ----------

if not F119.exists():
    raise FileNotFoundError(F119)
if not F43.exists():
    raise FileNotFoundError(F43)
if not M119.exists():
    raise FileNotFoundError(M119)
if not M43.exists():
    raise FileNotFoundError(M43)

d119 = pd.read_csv(F119)
d43 = pd.read_csv(F43)
m119 = pd.read_csv(M119)
m43 = pd.read_csv(M43)

d119["subject_id_norm"] = d119["subject_id"].map(norm_id)
d43["subject_id_norm"] = d43["subject_id"].map(norm_id)

demo119 = aggregate_demographics(m119)
demo43 = aggregate_demographics(m43)

# ---------- exact common 116-feature contract ----------

feature_cols = [c for c in d119.columns if c.startswith("GMVOL_AAL_")]
feature43 = [c for c in d43.columns if c.startswith("GMVOL_AAL_")]

if len(feature_cols) != 116:
    raise ValueError(f"119 file has {len(feature_cols)} GM features, expected 116")
if feature43 != feature_cols:
    missing = [c for c in feature_cols if c not in feature43]
    extra = [c for c in feature43 if c not in feature_cols]
    raise ValueError(f"Feature mismatch. missing={missing}, extra={extra}")

# Labels/demographics by subject ID
d119 = d119.merge(demo119, on="subject_id_norm", how="left", validate="one_to_one")
d43 = d43.merge(demo43, on="subject_id_norm", how="left", validate="one_to_one")

d119["source"] = "RMS119"
d43["source"] = "OpenNeuro43"

keep = ["subject_id_norm", "label", "age", "gender", "source"] + feature_cols
a = d119[keep].copy()
b = d43[keep].copy()

combined = pd.concat([a, b], ignore_index=True)

# ---------- hard validation, NOT raw-data audit ----------

if len(a) != 119 or len(b) != 43:
    raise ValueError(f"Unexpected rows: 119={len(a)}, 43={len(b)}")

if combined["subject_id_norm"].duplicated().any():
    dup = combined.loc[combined["subject_id_norm"].duplicated(), "subject_id_norm"].tolist()
    raise ValueError(f"Duplicate subject IDs after merge: {dup}")

if len(combined) != 162:
    raise ValueError(f"Expected 162 combined subjects, got {len(combined)}")

if combined[feature_cols].isna().any().any():
    raise ValueError("Missing imaging feature values found")

if not np.isfinite(combined[feature_cols].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite imaging feature values found")

if combined["label"].isna().any():
    raise ValueError("Missing labels found")

combined["label"] = combined["label"].astype(int)
if not set(combined["label"].unique()).issubset({0, 1}):
    raise ValueError("Labels are not binary 0/1")

print("\n=== COMBINED DATASET ===")
print("Rows:", len(combined))
print("Features:", len(feature_cols))
print("Label counts:")
print(combined.groupby(["source", "label"]).size())
print("\nSource counts:")
print(combined["source"].value_counts())

combined.to_csv(OUT / "combined_162_subjects_116_GMVOL_features.csv", index=False)

# ---------- nested CV ----------
#
# To keep source and diagnosis represented in every fold, stratify on
# source + diagnosis. This is NOT a source-as-feature model; source is
# only used to construct balanced CV folds.

X = combined[feature_cols].to_numpy(float)
y = combined["label"].to_numpy(int)
Z = combined[["age", "gender"]].to_numpy(float)
source = combined["source"].to_numpy(str)
strata = np.array([f"{s}_{yy}" for s, yy in zip(source, y)])

outer = StratifiedKFold(
    n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE
)

oof_score = np.full(len(combined), np.nan)
oof_pred = np.full(len(combined), -1)
fold_rows = []

for fold, (tr, te) in enumerate(outer.split(X, strata), start=1):
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    Ztr, Zte = Z[tr], Z[te]
    strat_tr = strata[tr]

    # Nuisance residualization is fitted only on outer-training data.
    res = NuisanceResidualizer().fit_with_nuisance(Xtr, Ztr)
    Xtr_r = res.transform_with_nuisance(Xtr, Ztr)
    Xte_r = res.transform_with_nuisance(Xte, Zte)

    # Inner CV chooses k and C using only outer-training subjects.
    inner = StratifiedKFold(
        n_splits=N_SPLITS_INNER, shuffle=True,
        random_state=RANDOM_STATE + fold
    )

    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_classif)),
        ("scale", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))
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
        "n_train": len(tr),
        "n_test": len(te),
    })
    fold_rows.append(met)

    print(
        f"Fold {fold}: AUC={met['roc_auc']:.6f} "
        f"BA={met['balanced_accuracy']:.6f} "
        f"Sens={met['sensitivity']:.6f} "
        f"Spec={met['specificity']:.6f} "
        f"k={met['best_k']} C={met['best_C']}"
    )

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT / "combined_162_nested_outer_fold_summary.csv", index=False)

overall = safe_metrics(y, oof_score, oof_pred)

print("\n=== 162-SUBJECT NESTED-CV RESULT ===")
for k, v in overall.items():
    print(f"{k}: {v}")

print("\nMean outer AUC:", fold_df["roc_auc"].mean())
print("Outer AUC SD:", fold_df["roc_auc"].std(ddof=1))

oof = combined[["subject_id_norm", "label", "source"]].copy()
oof["oof_decision_score"] = oof_score
oof["oof_prediction"] = oof_pred
oof.to_csv(OUT / "combined_162_nested_oof.csv", index=False)

summary = {
    "experiment": "Exp20_combined_162",
    "cohort": {
        "total": 162,
        "previous_RMS_training": 119,
        "previous_OpenNeuro_test": 43,
        "controls": int((y == 0).sum()),
        "schizophrenia": int((y == 1).sum()),
    },
    "representation": "116 AAL regional GM-volume features",
    "nuisance": ["age", "gender"],
    "classifier": "LinearSVC(class_weight='balanced')",
    "feature_selection": "SelectKBest(f_classif) inside inner CV",
    "scaling": "StandardScaler inside inner CV",
    "outer_cv": "5-fold stratified by source+diagnosis",
    "inner_cv": "4-fold stratified by source+diagnosis",
    "primary_metric": "OOF ROC-AUC",
    "metrics": overall,
    "mean_outer_auc": float(fold_df["roc_auc"].mean()),
    "outer_auc_sd": float(fold_df["roc_auc"].std(ddof=1)),
    "original_119_exp20_oof_auc": 0.756646,
    "note": "The original 43-subject cohort is now included in development and is no longer an independent external test set.",
}
(OUT / "combined_162_nested_cv_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\nSaved:")
print(OUT / "combined_162_subjects_116_GMVOL_features.csv")
print(OUT / "combined_162_nested_outer_fold_summary.csv")
print(OUT / "combined_162_nested_oof.csv")
print(OUT / "combined_162_nested_cv_summary.json")
