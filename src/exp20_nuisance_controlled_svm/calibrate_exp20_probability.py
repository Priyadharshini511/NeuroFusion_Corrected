"""
NeuroFusion AI — Exp20 leakage-safe probability calibration.

Creates OOF P(schizophrenia | sMRI) from the already validated Exp20
nuisance-controlled regional-GM-volume + LinearSVC pipeline.

Existing Exp20 outputs are NOT modified.

Calibration:
- sigmoid/Platt scaling
- calibrator fitted only on cross-fitted inner-training scores
- outer-test labels are never used for calibration
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Reuse the exact validated Exp20 implementation.
EXP20_DIR = Path(__file__).resolve().parent
if str(EXP20_DIR) not in sys.path:
    sys.path.insert(0, str(EXP20_DIR))

import train_exp20_nuisance_controlled_svm as exp20


CALIBRATION_ROOT = exp20.OUTPUT_ROOT / "calibration"
OOF_DIR = CALIBRATION_ROOT / "oof"

PROBABILITY_PATH = OOF_DIR / "exp20_smri_probability_oof.csv"
FOLD_SUMMARY_PATH = CALIBRATION_ROOT / "calibration_outer_fold_summary.csv"
SUMMARY_PATH = CALIBRATION_ROOT / "calibration_summary.json"
CONFIG_PATH = CALIBRATION_ROOT / "calibration_config.json"


def fit_model(X, y, k, C):
    model = exp20.make_pipeline(
        n_features=exp20.EXPECTED_AAL_ROIS
    )
    model.set_params(
        select__k=k,
        svm__C=C,
    )
    model.fit(X, y)
    return model


def select_parameters(X, y, inner_vector):
    cv = PredefinedSplit(test_fold=inner_vector)

    pipeline = exp20.make_pipeline(
        n_features=exp20.EXPECTED_AAL_ROIS
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "select__k": exp20.K_VALUES,
            "svm__C": exp20.C_VALUES,
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=False,
        return_train_score=False,
        error_score=np.nan,
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
        grid.fit(X, y)

    if not np.isfinite(grid.best_score_):
        raise RuntimeError("No finite inner-CV result.")

    return (
        int(grid.best_params_["select__k"]),
        float(grid.best_params_["svm__C"]),
        float(grid.best_score_),
    )


def cross_fitted_scores(X, y, inner_vector, k, C):
    scores = np.full(len(y), np.nan, dtype=float)

    if sorted(np.unique(inner_vector).tolist()) != [0, 1, 2, 3]:
        raise ValueError(
            "Expected mapped inner folds [0,1,2,3]."
        )

    for fold in [0, 1, 2, 3]:
        train_mask = inner_vector != fold
        valid_mask = inner_vector == fold

        model = fit_model(
            X[train_mask],
            y[train_mask],
            k,
            C,
        )

        fold_scores = model.decision_function(
            X[valid_mask]
        )

        if not np.isfinite(fold_scores).all():
            raise RuntimeError(
                f"Non-finite inner scores in fold {fold}."
            )

        scores[valid_mask] = fold_scores

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Cross-fitted inner scores are incomplete."
        )

    return scores


def main():
    start = time.time()

    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP20 — LEAKAGE-SAFE SIGMOID PROBABILITY CALIBRATION")
    print("=" * 72)

    # Exact Exp20 data construction.
    rms_df = exp20.load_rms_cohort()
    assignments = exp20.load_assignments(rms_df)
    nuisance_df = exp20.load_demographics(rms_df)
    atlas_img = exp20.load_atlas()

    X_features, _ = exp20.build_feature_matrix(
        rms_df,
        atlas_img,
    )

    subject_ids = (
        rms_df["subject_id"].astype(str).tolist()
    )

    y = (
        rms_df["label"].astype(int).to_numpy()
    )

    Z, nuisance_names = exp20.build_nuisance_matrix(
        subject_ids,
        nuisance_df,
    )

    if len(subject_ids) != exp20.EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {exp20.EXPECTED_SUBJECTS} subjects; "
            f"found {len(subject_ids)}."
        )

    if X_features.shape[0] != len(subject_ids):
        raise RuntimeError("Feature/subject mismatch.")

    if Z.shape[0] != len(subject_ids):
        raise RuntimeError("Nuisance/subject mismatch.")

    oof_parts = []
    fold_summary = []

    for outer_fold in range(1, exp20.N_OUTER_FOLDS + 1):
        fold_start = time.time()

        outer_rows = assignments[
            assignments["outer_fold"] == outer_fold
        ]

        test_ids = set(
            outer_rows.loc[
                outer_rows["role"] == "OUTER_TEST",
                "subject_id",
            ].astype(str)
        )

        train_ids = set(
            outer_rows.loc[
                outer_rows["role"] != "OUTER_TEST",
                "subject_id",
            ].astype(str)
        )

        train_idx = [
            i for i, sid in enumerate(subject_ids)
            if sid in train_ids
        ]
        test_idx = [
            i for i, sid in enumerate(subject_ids)
            if sid in test_ids
        ]

        train_order = [
            subject_ids[i] for i in train_idx
        ]
        test_order = [
            subject_ids[i] for i in test_idx
        ]

        X_train = np.hstack([
            X_features[train_idx],
            Z[train_idx],
        ])
        X_test = np.hstack([
            X_features[test_idx],
            Z[test_idx],
        ])

        y_train = y[train_idx]
        y_test = y[test_idx]

        inner_vector = exp20.make_inner_predefined_split(
            outer_train_ids=train_ids,
            outer_fold=outer_fold,
            assignments=assignments,
            subject_order=train_order,
        )

        # Exact Exp20 hyperparameter selection.
        k, C, inner_auc = select_parameters(
            X_train,
            y_train,
            inner_vector,
        )

        # Cross-fitted scores: every calibration-training score comes
        # from a model that did not train on that subject.
        inner_scores = cross_fitted_scores(
            X_train,
            y_train,
            inner_vector,
            k,
            C,
        )

        # Platt/sigmoid calibration.
        calibrator = LogisticRegression(
            solver="lbfgs",
            max_iter=10000,
            random_state=exp20.SEED,
        )
        calibrator.fit(
            inner_scores.reshape(-1, 1),
            y_train,
        )

        # Refit selected Exp20 model on complete outer training set.
        final_model = fit_model(
            X_train,
            y_train,
            k,
            C,
        )

        outer_scores = final_model.decision_function(X_test)

        probabilities = calibrator.predict_proba(
            outer_scores.reshape(-1, 1)
        )[:, 1]

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                f"Outer fold {outer_fold}: non-finite probabilities."
            )

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError(
                f"Outer fold {outer_fold}: probability outside [0,1]."
            )

        auc = roc_auc_score(y_test, probabilities)
        brier = brier_score_loss(y_test, probabilities)
        loss = log_loss(
            y_test,
            probabilities,
            labels=[0, 1],
        )

        part = pd.DataFrame({
            "subject_id": test_order,
            "label": y_test.astype(int),
            "smri_probability": probabilities.astype(float),
            "outer_fold": outer_fold,
        })

        oof_parts.append(part)

        fold_summary.append({
            "outer_fold": outer_fold,
            "outer_train_n": len(train_idx),
            "outer_test_n": len(test_idx),
            "best_k": k,
            "best_C": C,
            "best_inner_auc": inner_auc,
            "outer_calibrated_auc": float(auc),
            "outer_brier_score": float(brier),
            "outer_log_loss": float(loss),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "runtime_minutes": (
                (time.time() - fold_start) / 60
            ),
        })

        print(
            f"Fold {outer_fold}: "
            f"k={k}, C={C}, "
            f"inner AUC={inner_auc:.6f}, "
            f"outer AUC={auc:.6f}, "
            f"Brier={brier:.6f}, "
            f"log-loss={loss:.6f}"
        )

    oof = pd.concat(
        oof_parts,
        ignore_index=True,
    )

    oof = oof.sort_values(
        "subject_id"
    ).reset_index(drop=True)

    if len(oof) != exp20.EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {exp20.EXPECTED_SUBJECTS} OOF rows; "
            f"got {len(oof)}."
        )

    if oof["subject_id"].nunique() != len(oof):
        raise RuntimeError(
            "Duplicate subject IDs in OOF probabilities."
        )

    p = oof["smri_probability"].to_numpy(float)
    y_true = oof["label"].to_numpy(int)

    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite OOF probabilities.")

    if ((p < 0) | (p > 1)).any():
        raise RuntimeError("OOF probabilities outside [0,1].")

    oof_auc = roc_auc_score(y_true, p)
    oof_brier = brier_score_loss(y_true, p)
    oof_logloss = log_loss(
        y_true,
        p,
        labels=[0, 1],
    )

    oof.to_csv(
        PROBABILITY_PATH,
        index=False,
    )

    pd.DataFrame(fold_summary).to_csv(
        FOLD_SUMMARY_PATH,
        index=False,
    )

    summary = {
        "experiment": "Exp20",
        "calibration_method": "Sigmoid/Platt scaling",
        "calibrator": (
            "LogisticRegression on cross-fitted inner decision scores"
        ),
        "cohort": "RMS-only",
        "subjects": int(len(oof)),
        "controls": int((y_true == 0).sum()),
        "schizophrenia": int((y_true == 1).sum()),
        "nested_cv": "5x4",
        "seed": int(exp20.SEED),
        "probability_column": "smri_probability",
        "oof_roc_auc": float(oof_auc),
        "oof_brier_score": float(oof_brier),
        "oof_log_loss": float(oof_logloss),
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
        "oof_rows": int(len(oof)),
        "unique_subjects": int(oof["subject_id"].nunique()),
        "integrity": "PASS",
        "outer_test_labels_used_for_calibration": False,
        "existing_exp20_outputs_modified": False,
        "nuisance_variables": nuisance_names,
        "runtime_minutes": (
            (time.time() - start) / 60
        ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    config = {
        "base_model": (
            "Exp20 Nuisance-Controlled Regional GM Volume + Linear SVM"
        ),
        "calibration": "Sigmoid/Platt scaling",
        "calibration_training_data": (
            "cross-fitted inner-validation decision scores "
            "inside each outer-training partition"
        ),
        "outer_test_used_for_calibration": False,
        "nested_cv": "canonical 5x4",
        "cohort": "119 RMS subjects: 40 controls, 79 schizophrenia",
        "purpose": "Fusion-compatible sMRI probability output",
        "note": (
            "Calibration changes output scale; it is not intended "
            "to artificially increase ROC-AUC."
        ),
        "original_exp20_outputs_modified": False,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    print("")
    print("=" * 72)
    print("CALIBRATION COMPLETE")
    print("=" * 72)
    print(f"OOF ROC-AUC:       {oof_auc:.6f}")
    print(f"OOF Brier score:   {oof_brier:.6f}")
    print(f"OOF log-loss:      {oof_logloss:.6f}")
    print(f"Probability range: {p.min():.6f} to {p.max():.6f}")
    print(f"OOF rows:          {len(oof)}")
    print(f"Unique subjects:   {oof['subject_id'].nunique()}")
    print("Integrity:         PASS")
    print("")
    print(f"Output: {PROBABILITY_PATH}")


if __name__ == "__main__":
    main()
