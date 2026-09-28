# """
# NeuroFusion AI
# sMRI Branch - Nested CV Fold / Source / Class Composition Analysis

# READ-ONLY diagnostic analysis.
# Does not retrain models or modify datasets/splits.
# """

# from pathlib import Path
# import numpy as np
# import pandas as pd


# # ============================================================
# # PATHS
# # ============================================================

# ROOT = Path("processed_data/smri/cnn_dataset")
# CV_DIR = ROOT / "cross_validation"
# TRAINING_DIR = CV_DIR / "training_full_5x4"
# AUDIT_DIR = CV_DIR / "audit"

# OOF_FILE = TRAINING_DIR / "nested_cv_oof_predictions.csv"
# MANIFEST_FILE = ROOT / "smri_subject_manifest_labeled.csv"
# ASSIGNMENT_FILE = CV_DIR / "cv_assignments.csv"

# OUTPUT_DIR = AUDIT_DIR / "fold_composition"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# REPORT_FILE = OUTPUT_DIR / "fold_composition_report.txt"
# FOLD_FILE = OUTPUT_DIR / "fold_composition_summary.csv"
# SUBJECT_FILE = OUTPUT_DIR / "fold_subject_composition.csv"


# # ============================================================
# # HELPERS
# # ============================================================

# def find_column(df, candidates):
#     lower_map = {c.lower(): c for c in df.columns}

#     for candidate in candidates:
#         if candidate.lower() in lower_map:
#             return lower_map[candidate.lower()]

#     for c in df.columns:
#         cl = c.lower()
#         for candidate in candidates:
#             if candidate.lower() in cl:
#                 return c

#     raise KeyError(
#         f"Could not find {candidates} in columns: {list(df.columns)}"
#     )


# def normalize_source(value):
#     if pd.isna(value):
#         return "UNKNOWN"

#     text = str(value).strip().upper()

#     if "LEGACY" in text:
#         return "LEGACY"

#     if "RMS" in text:
#         return "RMS"

#     return text


# # ============================================================
# # LOAD
# # ============================================================

# print("=" * 72)
# print("NeuroFusion AI - Nested CV Fold/Source/Class Composition")
# print("=" * 72)

# for path in [
#     OOF_FILE,
#     MANIFEST_FILE,
#     ASSIGNMENT_FILE,
# ]:
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Required file not found: {path}"
#         )

# oof = pd.read_csv(OOF_FILE)
# manifest = pd.read_csv(MANIFEST_FILE)
# assignments = pd.read_csv(ASSIGNMENT_FILE)

# # OOF columns
# subject_oof = find_column(
#     oof,
#     ["subject_id", "subject", "id"]
# )

# label_oof = find_column(
#     oof,
#     ["label", "y_true", "target"]
# )

# prob_oof = find_column(
#     oof,
#     ["probability", "prob", "prediction_probability", "y_prob"]
# )

# fold_oof = find_column(
#     oof,
#     ["outer_fold", "fold"]
# )

# # Manifest columns
# subject_manifest = find_column(
#     manifest,
#     ["subject_id", "subject", "id"]
# )

# label_manifest = find_column(
#     manifest,
#     ["label", "target"]
# )

# # Source column
# source_candidates = [
#     c for c in manifest.columns
#     if any(
#         token in c.lower()
#         for token in [
#             "source",
#             "lineage",
#             "acquisition_source",
#             "preprocessing_source",
#         ]
#     )
# ]

# if not source_candidates:
#     raise KeyError(
#         "Could not identify source/lineage column in manifest."
#     )

# source_manifest = source_candidates[0]

# print("\nDetected columns:")
# print(f"  OOF subject:   {subject_oof}")
# print(f"  OOF label:     {label_oof}")
# print(f"  OOF probability: {prob_oof}")
# print(f"  OOF outer fold: {fold_oof}")
# print(f"  Manifest subject: {subject_manifest}")
# print(f"  Manifest label:   {label_manifest}")
# print(f"  Manifest source:  {source_manifest}")


# # ============================================================
# # NORMALIZE
# # ============================================================

# oof[subject_oof] = oof[subject_oof].astype(str)
# oof[label_oof] = pd.to_numeric(
#     oof[label_oof],
#     errors="coerce"
# ).astype(int)
# oof[prob_oof] = pd.to_numeric(
#     oof[prob_oof],
#     errors="coerce"
# )
# oof[fold_oof] = pd.to_numeric(
#     oof[fold_oof],
#     errors="coerce"
# ).astype(int)

# manifest[subject_manifest] = (
#     manifest[subject_manifest].astype(str)
# )

# manifest[label_manifest] = pd.to_numeric(
#     manifest[label_manifest],
#     errors="coerce"
# ).astype(int)

# manifest["_source_normalized"] = (
#     manifest[source_manifest]
#     .apply(normalize_source)
# )


# # ============================================================
# # MERGE
# # ============================================================

# print("\n[1] Joining OOF predictions with subject metadata...")

# meta = manifest[
#     [
#         subject_manifest,
#         label_manifest,
#         source_manifest,
#         "_source_normalized",
#     ]
# ].drop_duplicates(
#     subset=[subject_manifest]
# )

# meta = meta.rename(
#     columns={
#         subject_manifest: subject_oof,
#         label_manifest: "_manifest_label",
#         source_manifest: "_manifest_source",
#     }
# )

# merged = oof.merge(
#     meta,
#     on=subject_oof,
#     how="left",
# )

# print(f"OOF subjects: {oof[subject_oof].nunique()}")
# print(
#     f"Subjects missing manifest metadata: "
#     f"{merged['_source_normalized'].isna().sum()}"
# )

# label_mismatch = (
#     merged[label_oof] != merged["_manifest_label"]
# )

# print(
#     f"OOF/manifest label mismatches: "
#     f"{label_mismatch.sum()}"
# )


# # ============================================================
# # SUBJECT CATEGORIES
# # ============================================================

# def make_category(row):
#     source = row["_source_normalized"]
#     label = int(row[label_oof])

#     if source == "LEGACY" and label == 0:
#         return "CONTROL_LEGACY"

#     if source == "RMS" and label == 0:
#         return "CONTROL_RMS"

#     if source == "RMS" and label == 1:
#         return "SCHIZOPHRENIA_RMS"

#     return "OTHER"


# merged["source_class"] = merged.apply(
#     make_category,
#     axis=1
# )

# print("\nOverall source/class composition:")
# print(
#     merged["source_class"]
#     .value_counts()
#     .sort_index()
#     .to_string()
# )


# # ============================================================
# # FOLD COMPOSITION
# # ============================================================

# print("\n[2] Fold composition")

# fold_rows = []

# for fold in sorted(
#     merged[fold_oof].unique()
# ):

#     fold_df = merged[
#         merged[fold_oof] == fold
#     ]

#     n = len(fold_df)

#     control_legacy = int(
#         np.sum(
#             fold_df["source_class"]
#             == "CONTROL_LEGACY"
#         )
#     )

#     control_rms = int(
#         np.sum(
#             fold_df["source_class"]
#             == "CONTROL_RMS"
#         )
#     )

#     schizophrenia_rms = int(
#         np.sum(
#             fold_df["source_class"]
#             == "SCHIZOPHRENIA_RMS"
#         )
#     )

#     total_controls = (
#         control_legacy + control_rms
#     )

#     total_cases = schizophrenia_rms

#     fold_rows.append(
#         {
#             "outer_fold": int(fold),
#             "n": n,
#             "controls": total_controls,
#             "schizophrenia": total_cases,
#             "control_fraction": (
#                 total_controls / n
#                 if n else np.nan
#             ),
#             "schizophrenia_fraction": (
#                 total_cases / n
#                 if n else np.nan
#             ),
#             "legacy_controls": control_legacy,
#             "rms_controls": control_rms,
#             "rms_schizophrenia": schizophrenia_rms,
#             "legacy_fraction": (
#                 control_legacy / n
#                 if n else np.nan
#             ),
#             "rms_fraction": (
#                 (control_rms + schizophrenia_rms) / n
#                 if n else np.nan
#             ),
#         }
#     )

# fold_summary = pd.DataFrame(
#     fold_rows
# )

# print(
#     fold_summary.to_string(index=False)
# )


# # ============================================================
# # FOLD PREDICTION BEHAVIOR
# # ============================================================

# print("\n[3] Prediction behavior by fold and source/class")

# prediction_rows = []

# for fold in sorted(
#     merged[fold_oof].unique()
# ):

#     fold_df = merged[
#         merged[fold_oof] == fold
#     ]

#     for category in [
#         "CONTROL_LEGACY",
#         "CONTROL_RMS",
#         "SCHIZOPHRENIA_RMS",
#     ]:

#         subset = fold_df[
#             fold_df["source_class"] == category
#         ]

#         if len(subset) == 0:
#             continue

#         prediction_rows.append(
#             {
#                 "outer_fold": int(fold),
#                 "source_class": category,
#                 "n": len(subset),
#                 "prob_mean": float(
#                     subset[prob_oof].mean()
#                 ),
#                 "prob_std": float(
#                     subset[prob_oof].std(ddof=0)
#                 ),
#                 "prob_min": float(
#                     subset[prob_oof].min()
#                 ),
#                 "prob_max": float(
#                     subset[prob_oof].max()
#                 ),
#                 "predicted_positive_at_0.5": int(
#                     np.sum(
#                         subset[prob_oof] >= 0.5
#                     )
#                 ),
#             }
#         )

# prediction_summary = pd.DataFrame(
#     prediction_rows
# )

# print(
#     prediction_summary.to_string(index=False)
# )


# # ============================================================
# # CLASS / SOURCE BALANCE ACROSS FOLDS
# # ============================================================

# print("\n[4] Balance diagnostics")

# overall_counts = (
#     merged["source_class"]
#     .value_counts()
# )

# balance_rows = []

# for fold in sorted(
#     merged[fold_oof].unique()
# ):

#     fold_df = merged[
#         merged[fold_oof] == fold
#     ]

#     for category in [
#         "CONTROL_LEGACY",
#         "CONTROL_RMS",
#         "SCHIZOPHRENIA_RMS",
#     ]:

#         observed = int(
#             np.sum(
#                 fold_df["source_class"]
#                 == category
#             )
#         )

#         total = int(
#             overall_counts.get(
#                 category,
#                 0
#             )
#         )

#         expected = (
#             total * len(fold_df) / len(merged)
#         )

#         balance_rows.append(
#             {
#                 "outer_fold": int(fold),
#                 "source_class": category,
#                 "observed": observed,
#                 "expected_if_proportional": expected,
#                 "difference": (
#                     observed - expected
#                 ),
#             }
#         )

# balance_df = pd.DataFrame(
#     balance_rows
# )

# print(
#     balance_df.to_string(index=False)
# )


# # ============================================================
# # SUBJECT COVERAGE
# # ============================================================

# print("\n[5] Subject coverage")

# subject_fold_counts = (
#     merged
#     .groupby(subject_oof)[fold_oof]
#     .nunique()
# )

# multi_fold = subject_fold_counts[
#     subject_fold_counts > 1
# ]

# fold_subject_table = (
#     merged[
#         [
#             subject_oof,
#             label_oof,
#             "_source_normalized",
#             "source_class",
#             fold_oof,
#             prob_oof,
#         ]
#     ]
#     .sort_values(
#         [fold_oof, subject_oof]
#     )
# )

# fold_subject_table.to_csv(
#     SUBJECT_FILE,
#     index=False
# )

# print(
#     f"Subjects: {merged[subject_oof].nunique()}"
# )

# print(
#     f"Subjects in multiple folds: "
#     f"{len(multi_fold)}"
# )


# # ============================================================
# # SOURCE COMPOSITION OF TRAINING SETS
# # ============================================================

# print("\n[6] Outer training-set composition")

# training_rows = []

# all_subjects = set(
#     merged[subject_oof]
# )

# for fold in sorted(
#     merged[fold_oof].unique()
# ):

#     test_subjects = set(
#         merged.loc[
#             merged[fold_oof] == fold,
#             subject_oof,
#         ]
#     )

#     train_subjects = (
#         all_subjects - test_subjects
#     )

#     train_df = merged[
#         merged[subject_oof].isin(
#             train_subjects
#         )
#     ]

#     for category in [
#         "CONTROL_LEGACY",
#         "CONTROL_RMS",
#         "SCHIZOPHRENIA_RMS",
#     ]:

#         n = int(
#             np.sum(
#                 train_df["source_class"]
#                 == category
#             )
#         )

#         training_rows.append(
#             {
#                 "outer_fold": int(fold),
#                 "training_subjects": len(
#                     train_df
#                 ),
#                 "source_class": category,
#                 "n": n,
#                 "fraction": (
#                     n / len(train_df)
#                     if len(train_df)
#                     else np.nan
#                 ),
#             }
#         )

# training_summary = pd.DataFrame(
#     training_rows
# )


# # ============================================================
# # SAVE
# # ============================================================

# fold_summary.to_csv(
#     FOLD_FILE,
#     index=False
# )


# # ============================================================
# # REPORT
# # ============================================================

# with open(
#     REPORT_FILE,
#     "w",
#     encoding="utf-8",
# ) as f:

#     f.write(
#         "NeuroFusion AI - Nested CV Fold/Source/Class Composition\n"
#     )
#     f.write("=" * 72 + "\n\n")

#     f.write(
#         "READ-ONLY diagnostic analysis.\n"
#         "No training, model, prediction, dataset, or split files "
#         "were modified.\n\n"
#     )

#     f.write("1. OVERALL COMPOSITION\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         merged["source_class"]
#         .value_counts()
#         .sort_index()
#         .to_string()
#     )
#     f.write("\n\n")

#     f.write("2. OUTER TEST-FOLD COMPOSITION\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         fold_summary.to_string(index=False)
#     )
#     f.write("\n\n")

#     f.write("3. PREDICTION BEHAVIOR\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         prediction_summary.to_string(index=False)
#     )
#     f.write("\n\n")

#     f.write("4. PROPORTIONAL BALANCE\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         balance_df.to_string(index=False)
#     )
#     f.write("\n\n")

#     f.write("5. SUBJECT COVERAGE\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         f"Unique subjects: "
#         f"{merged[subject_oof].nunique()}\n"
#     )

#     f.write(
#         f"Subjects in multiple folds: "
#         f"{len(multi_fold)}\n"
#     )

#     f.write(
#         f"OOF/manifest label mismatches: "
#         f"{label_mismatch.sum()}\n\n"
#     )

#     f.write("6. OUTER TRAINING COMPOSITION\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         training_summary.to_string(index=False)
#     )
#     f.write("\n\n")

#     f.write("7. INTERPRETATION NOTES\n")
#     f.write("-" * 40 + "\n")

#     f.write(
#         "The cohort contains three relevant source/class groups:\n"
#         "  CONTROL_LEGACY = legacy preprocessing controls\n"
#         "  CONTROL_RMS = RMS preprocessing controls\n"
#         "  SCHIZOPHRENIA_RMS = RMS preprocessing schizophrenia subjects\n\n"
#     )

#     f.write(
#         "Because LEGACY occurs only among controls, source and diagnosis "
#         "are structurally confounded in this cohort. This analysis is "
#         "therefore descriptive and is intended to determine whether "
#         "outer-fold composition could contribute to variation in model "
#         "behavior. It does not establish causality.\n\n"
#     )

#     f.write(
#         "No threshold or model-selection decisions should be made from "
#         "this descriptive analysis alone.\n"
#     )


# # ============================================================
# # FINAL VALIDATION
# # ============================================================

# validation_checks = {
#     "172 OOF subjects": (
#         merged[subject_oof].nunique() == 172
#     ),
#     "No duplicate OOF subjects": (
#         merged[subject_oof].duplicated().sum() == 0
#     ),
#     "No missing source metadata": (
#         merged["_source_normalized"].notna().all()
#     ),
#     "No label mismatches": (
#         label_mismatch.sum() == 0
#     ),
#     "No subject in multiple folds": (
#         len(multi_fold) == 0
#     ),
#     "All 5 outer folds present": (
#         set(merged[fold_oof].unique())
#         == {1, 2, 3, 4, 5}
#     ),
# }

# print("\n[7] Final validation")

# for name, passed in validation_checks.items():
#     print(
#         f"{name}: "
#         f"{'PASS' if passed else 'FAIL'}"
#     )

# final_pass = all(
#     validation_checks.values()
# )

# print(
#     f"\nFINAL STATUS: "
#     f"{'PASS' if final_pass else 'FAIL'}"
# )

# print("\nOutputs:")
# print(REPORT_FILE)
# print(FOLD_FILE)
# print(SUBJECT_FILE)

# print("\nNo training or dataset files were modified.")
# print("=" * 72)



"""
NeuroFusion AI
sMRI Branch - Nested CV Fold / Source / Class Composition Analysis

READ-ONLY diagnostic analysis.

Purpose:
    Analyze whether outer-fold source/class composition may help explain
    variation in nested-CV model performance.

This script:
    1. Loads completed 5x4 nested-CV OOF predictions.
    2. Loads the labeled sMRI subject manifest.
    3. Joins prediction results with source/lineage information.
    4. Audits source/class composition of every outer fold.
    5. Audits prediction behavior by source/class and fold.
    6. Audits subject coverage and leakage.
    7. Audits outer training-set composition.
    8. Saves descriptive reports and CSV files.

IMPORTANT:
    - No model training.
    - No model modification.
    - No prediction modification.
    - No dataset modification.
    - No CV split modification.
    - No threshold/model-selection decisions are made.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data/smri/cnn_dataset")

CV_DIR = ROOT / "cross_validation"

TRAINING_DIR = (
    CV_DIR / "training_full_5x4"
)

AUDIT_DIR = (
    CV_DIR / "audit"
)

# Completed nested-CV OOF predictions
OOF_FILE = (
    TRAINING_DIR /
    "nested_cv_oof_predictions.csv"
)

# IMPORTANT:
# The labeled manifest is under processed_data/smri/dataset,
# NOT under processed_data/smri/cnn_dataset.
MANIFEST_FILE = Path(
    "processed_data/smri/dataset/"
    "smri_subject_manifest_labeled.csv"
)

# Official outer CV assignment
ASSIGNMENT_FILE = (
    CV_DIR /
    "cv_assignments.csv"
)

# Output directory
OUTPUT_DIR = (
    AUDIT_DIR /
    "fold_composition"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    OUTPUT_DIR /
    "fold_composition_report.txt"
)

FOLD_FILE = (
    OUTPUT_DIR /
    "fold_composition_summary.csv"
)

SUBJECT_FILE = (
    OUTPUT_DIR /
    "fold_subject_composition.csv"
)

PREDICTION_FILE = (
    OUTPUT_DIR /
    "fold_prediction_behavior.csv"
)

BALANCE_FILE = (
    OUTPUT_DIR /
    "fold_proportional_balance.csv"
)

TRAINING_COMPOSITION_FILE = (
    OUTPUT_DIR /
    "outer_training_composition.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find a column by exact case-insensitive name first,
    then by partial case-insensitive match.
    """

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    # Exact match
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    # Partial match
    for column in df.columns:

        column_lower = str(column).lower()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    raise KeyError(
        f"Could not find any of {candidates} "
        f"in columns: {list(df.columns)}"
    )


def normalize_source(value):
    """
    Normalize source/lineage names to:
        LEGACY
        RMS
        UNKNOWN
        or original uppercase text
    """

    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    if "LEGACY" in text:
        return "LEGACY"

    if "RMS" in text:
        return "RMS"

    return text


def require_file(path):
    """
    Stop clearly if a required input file is missing.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 72)
print(
    "NeuroFusion AI - Nested CV "
    "Fold/Source/Class Composition"
)
print("=" * 72)

print("\nInput files:")

print(
    f"  OOF predictions : {OOF_FILE}"
)

print(
    f"  Labeled manifest: {MANIFEST_FILE}"
)

print(
    f"  CV assignments  : {ASSIGNMENT_FILE}"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

require_file(OOF_FILE)
require_file(MANIFEST_FILE)
require_file(ASSIGNMENT_FILE)


# ============================================================
# LOAD FILES
# ============================================================

print(
    "\n[1] Loading completed nested-CV outputs..."
)

oof = pd.read_csv(
    OOF_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

assignments = pd.read_csv(
    ASSIGNMENT_FILE
)

print(
    f"OOF rows: {len(oof)}"
)

print(
    f"Manifest rows: {len(manifest)}"
)

print(
    f"Assignment rows: {len(assignments)}"
)


# ============================================================
# IDENTIFY OOF COLUMNS
# ============================================================

subject_oof = find_column(
    oof,
    [
        "subject_id",
        "subject",
        "id",
    ]
)

label_oof = find_column(
    oof,
    [
        "label",
        "y_true",
        "target",
    ]
)

prob_oof = find_column(
    oof,
    [
        "probability",
        "prob",
        "prediction_probability",
        "y_prob",
    ]
)

fold_oof = find_column(
    oof,
    [
        "outer_fold",
        "fold",
    ]
)


# ============================================================
# IDENTIFY MANIFEST COLUMNS
# ============================================================

subject_manifest = find_column(
    manifest,
    [
        "subject_id",
        "subject",
        "id",
    ]
)

label_manifest = find_column(
    manifest,
    [
        "label",
        "target",
    ]
)


# ============================================================
# IDENTIFY SOURCE COLUMN
# ============================================================

source_candidates = []

for column in manifest.columns:

    column_lower = str(
        column
    ).lower()

    if any(
        token in column_lower
        for token in [
            "source",
            "lineage",
            "acquisition_source",
            "preprocessing_source",
        ]
    ):

        source_candidates.append(
            column
        )


if not source_candidates:

    raise KeyError(
        "Could not identify a source/lineage "
        "column in the labeled manifest.\n"
        f"Manifest columns:\n{list(manifest.columns)}"
    )


source_manifest = source_candidates[0]


print("\nDetected columns:")

print(
    f"  OOF subject       : {subject_oof}"
)

print(
    f"  OOF label         : {label_oof}"
)

print(
    f"  OOF probability   : {prob_oof}"
)

print(
    f"  OOF outer fold    : {fold_oof}"
)

print(
    f"  Manifest subject  : {subject_manifest}"
)

print(
    f"  Manifest label    : {label_manifest}"
)

print(
    f"  Manifest source   : {source_manifest}"
)


# ============================================================
# NORMALIZE OOF DATA TYPES
# ============================================================

print(
    "\n[2] Normalizing prediction data..."
)

oof[subject_oof] = (
    oof[subject_oof]
    .astype(str)
)

oof[label_oof] = pd.to_numeric(
    oof[label_oof],
    errors="coerce"
)

oof[prob_oof] = pd.to_numeric(
    oof[prob_oof],
    errors="coerce"
)

oof[fold_oof] = pd.to_numeric(
    oof[fold_oof],
    errors="coerce"
)

# Remove rows that cannot be analyzed.
oof = oof.dropna(
    subset=[
        subject_oof,
        label_oof,
        prob_oof,
        fold_oof,
    ]
).copy()

oof[label_oof] = (
    oof[label_oof]
    .astype(int)
)

oof[fold_oof] = (
    oof[fold_oof]
    .astype(int)
)


# ============================================================
# NORMALIZE MANIFEST DATA TYPES
# ============================================================

manifest[subject_manifest] = (
    manifest[subject_manifest]
    .astype(str)
)

manifest[label_manifest] = pd.to_numeric(
    manifest[label_manifest],
    errors="coerce"
)

manifest = manifest.dropna(
    subset=[
        subject_manifest,
        label_manifest,
    ]
).copy()

manifest[label_manifest] = (
    manifest[label_manifest]
    .astype(int)
)

manifest["_source_normalized"] = (
    manifest[source_manifest]
    .apply(normalize_source)
)


# ============================================================
# CHECK MANIFEST UNIQUENESS
# ============================================================

print(
    "\n[3] Checking manifest subject uniqueness..."
)

manifest_duplicate_count = int(
    manifest[subject_manifest]
    .duplicated()
    .sum()
)

print(
    f"Manifest duplicate subject rows: "
    f"{manifest_duplicate_count}"
)

if manifest_duplicate_count > 0:

    print(
        "WARNING: Manifest contains duplicate "
        "subject rows."
    )

    # For the merge, retain the first unique row.
    manifest_unique = (
        manifest
        .drop_duplicates(
            subset=[subject_manifest],
            keep="first",
        )
        .copy()
    )

else:

    manifest_unique = manifest.copy()


# ============================================================
# BUILD METADATA TABLE
# ============================================================

metadata = manifest_unique[
    [
        subject_manifest,
        label_manifest,
        source_manifest,
        "_source_normalized",
    ]
].copy()

metadata = metadata.rename(
    columns={
        subject_manifest: subject_oof,
        label_manifest: "_manifest_label",
        source_manifest: "_manifest_source",
    }
)


# ============================================================
# MERGE OOF + MANIFEST
# ============================================================

print(
    "\n[4] Joining OOF predictions "
    "with source/class metadata..."
)

merged = oof.merge(
    metadata,
    on=subject_oof,
    how="left",
    validate="one_to_one",
)

missing_metadata = int(
    merged["_source_normalized"]
    .isna()
    .sum()
)

print(
    f"OOF subjects: "
    f"{merged[subject_oof].nunique()}"
)

print(
    f"Missing source metadata: "
    f"{missing_metadata}"
)


# ============================================================
# LABEL CONSISTENCY
# ============================================================

label_mismatch_mask = (
    merged[label_oof]
    !=
    merged["_manifest_label"]
)

label_mismatch_count = int(
    label_mismatch_mask.sum()
)

print(
    f"OOF/manifest label mismatches: "
    f"{label_mismatch_count}"
)


# ============================================================
# SOURCE/CLASS CATEGORY
# ============================================================

def make_source_class(row):

    source = row[
        "_source_normalized"
    ]

    label = int(
        row[label_oof]
    )

    if (
        source == "LEGACY"
        and label == 0
    ):

        return "CONTROL_LEGACY"

    if (
        source == "RMS"
        and label == 0
    ):

        return "CONTROL_RMS"

    if (
        source == "RMS"
        and label == 1
    ):

        return "SCHIZOPHRENIA_RMS"

    return "OTHER"


merged["source_class"] = (
    merged.apply(
        make_source_class,
        axis=1,
    )
)


# ============================================================
# OVERALL COMPOSITION
# ============================================================

print(
    "\n[5] Overall source/class composition..."
)

overall_composition = (
    merged[
        "source_class"
    ]
    .value_counts()
    .sort_index()
)

print(
    overall_composition.to_string()
)


# ============================================================
# OUTER TEST-FOLD COMPOSITION
# ============================================================

print(
    "\n[6] Outer test-fold composition..."
)

fold_rows = []

for fold in sorted(
    merged[fold_oof].unique()
):

    fold_df = merged[
        merged[fold_oof] == fold
    ]

    n = len(fold_df)

    control_legacy = int(
        np.sum(
            fold_df["source_class"]
            == "CONTROL_LEGACY"
        )
    )

    control_rms = int(
        np.sum(
            fold_df["source_class"]
            == "CONTROL_RMS"
        )
    )

    schizophrenia_rms = int(
        np.sum(
            fold_df["source_class"]
            == "SCHIZOPHRENIA_RMS"
        )
    )

    controls = (
        control_legacy
        +
        control_rms
    )

    cases = schizophrenia_rms

    fold_rows.append(
        {
            "outer_fold": int(fold),
            "n": int(n),
            "controls": int(controls),
            "schizophrenia": int(cases),
            "control_fraction": (
                controls / n
                if n > 0
                else np.nan
            ),
            "schizophrenia_fraction": (
                cases / n
                if n > 0
                else np.nan
            ),
            "legacy_controls": int(
                control_legacy
            ),
            "rms_controls": int(
                control_rms
            ),
            "rms_schizophrenia": int(
                schizophrenia_rms
            ),
            "legacy_fraction": (
                control_legacy / n
                if n > 0
                else np.nan
            ),
            "rms_fraction": (
                (
                    control_rms
                    +
                    schizophrenia_rms
                )
                / n
                if n > 0
                else np.nan
            ),
        }
    )


fold_summary = pd.DataFrame(
    fold_rows
)

print(
    fold_summary.to_string(
        index=False
    )
)


# ============================================================
# PREDICTION BEHAVIOR BY FOLD + SOURCE/CLASS
# ============================================================

print(
    "\n[7] Prediction behavior "
    "by fold and source/class..."
)

prediction_rows = []

categories = [
    "CONTROL_LEGACY",
    "CONTROL_RMS",
    "SCHIZOPHRENIA_RMS",
]

for fold in sorted(
    merged[fold_oof].unique()
):

    fold_df = merged[
        merged[fold_oof] == fold
    ]

    for category in categories:

        subset = fold_df[
            fold_df["source_class"]
            == category
        ]

        if len(subset) == 0:
            continue

        probabilities = (
            subset[prob_oof]
            .to_numpy(
                dtype=float
            )
        )

        prediction_rows.append(
            {
                "outer_fold": int(fold),
                "source_class": category,
                "n": int(len(subset)),
                "prob_mean": float(
                    np.mean(
                        probabilities
                    )
                ),
                "prob_std": float(
                    np.std(
                        probabilities
                    )
                ),
                "prob_min": float(
                    np.min(
                        probabilities
                    )
                ),
                "prob_max": float(
                    np.max(
                        probabilities
                    )
                ),
                "predicted_positive_at_0.5": int(
                    np.sum(
                        probabilities >= 0.5
                    )
                ),
                "predicted_positive_fraction_at_0.5": float(
                    np.mean(
                        probabilities >= 0.5
                    )
                ),
            }
        )


prediction_summary = pd.DataFrame(
    prediction_rows
)

print(
    prediction_summary.to_string(
        index=False
    )
)


# ============================================================
# PROPORTIONAL BALANCE
# ============================================================

print(
    "\n[8] Proportional balance diagnostics..."
)

overall_total = len(
    merged
)

balance_rows = []

for fold in sorted(
    merged[fold_oof].unique()
):

    fold_df = merged[
        merged[fold_oof] == fold
    ]

    fold_n = len(
        fold_df
    )

    for category in categories:

        total_category = int(
            overall_composition.get(
                category,
                0
            )
        )

        observed = int(
            np.sum(
                fold_df["source_class"]
                == category
            )
        )

        expected = (
            total_category
            *
            fold_n
            /
            overall_total
        )

        balance_rows.append(
            {
                "outer_fold": int(fold),
                "source_class": category,
                "observed": observed,
                "expected_if_proportional": float(
                    expected
                ),
                "difference": float(
                    observed - expected
                ),
            }
        )


balance_df = pd.DataFrame(
    balance_rows
)

print(
    balance_df.to_string(
        index=False
    )
)


# ============================================================
# SUBJECT COVERAGE
# ============================================================

print(
    "\n[9] Subject coverage and fold leakage..."
)

unique_subjects = (
    merged[subject_oof]
    .nunique()
)

duplicate_oof_subjects = int(
    merged[subject_oof]
    .duplicated()
    .sum()
)

subject_fold_counts = (
    merged
    .groupby(
        subject_oof
    )[fold_oof]
    .nunique()
)

multi_fold_subjects = (
    subject_fold_counts[
        subject_fold_counts > 1
    ]
)

print(
    f"Unique subjects: "
    f"{unique_subjects}"
)

print(
    f"Duplicate OOF subject rows: "
    f"{duplicate_oof_subjects}"
)

print(
    f"Subjects appearing in multiple "
    f"outer folds: {len(multi_fold_subjects)}"
)


# ============================================================
# OFFICIAL OUTER ASSIGNMENT CHECK
# ============================================================

print(
    "\n[10] Comparing OOF folds "
    "with official CV assignments..."
)

assignment_subject = find_column(
    assignments,
    [
        "subject_id",
        "subject",
        "id",
    ]
)

assignment_fold = find_column(
    assignments,
    [
        "outer_fold",
        "fold",
    ]
)

official_map = assignments[
    [
        assignment_subject,
        assignment_fold,
    ]
].copy()

official_map[assignment_subject] = (
    official_map[assignment_subject]
    .astype(str)
)

official_map = (
    official_map
    .drop_duplicates(
        subset=[assignment_subject]
    )
    .rename(
        columns={
            assignment_subject:
                subject_oof,
            assignment_fold:
                "_official_outer_fold",
        }
    )
)

comparison = merged.merge(
    official_map[
        [
            subject_oof,
            "_official_outer_fold",
        ]
    ],
    on=subject_oof,
    how="left",
    validate="one_to_one",
)

official_fold_missing = int(
    comparison[
        "_official_outer_fold"
    ]
    .isna()
    .sum()
)

official_fold_mismatch = int(
    np.sum(
        comparison[fold_oof]
        !=
        comparison[
            "_official_outer_fold"
        ]
    )
)

print(
    f"Missing official fold assignments: "
    f"{official_fold_missing}"
)

print(
    f"OOF/official fold mismatches: "
    f"{official_fold_mismatch}"
)


# ============================================================
# OUTER TRAINING-SET COMPOSITION
# ============================================================

print(
    "\n[11] Outer training-set composition..."
)

all_subjects = set(
    merged[subject_oof]
)

training_rows = []

for fold in sorted(
    merged[fold_oof].unique()
):

    test_subjects = set(
        merged.loc[
            merged[fold_oof] == fold,
            subject_oof,
        ]
    )

    training_subjects = (
        all_subjects
        -
        test_subjects
    )

    training_df = merged[
        merged[subject_oof]
        .isin(
            training_subjects
        )
    ]

    for category in categories:

        category_count = int(
            np.sum(
                training_df[
                    "source_class"
                ]
                == category
            )
        )

        training_rows.append(
            {
                "outer_fold": int(fold),
                "training_subjects": int(
                    len(training_df)
                ),
                "source_class": category,
                "n": category_count,
                "fraction": (
                    category_count
                    /
                    len(training_df)
                    if len(training_df) > 0
                    else np.nan
                ),
            }
        )


training_summary = pd.DataFrame(
    training_rows
)

print(
    training_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE OUTPUT TABLES
# ============================================================

print(
    "\n[12] Saving audit tables..."
)

fold_summary.to_csv(
    FOLD_FILE,
    index=False
)

prediction_summary.to_csv(
    PREDICTION_FILE,
    index=False
)

balance_df.to_csv(
    BALANCE_FILE,
    index=False
)

training_summary.to_csv(
    TRAINING_COMPOSITION_FILE,
    index=False
)

subject_table = merged[
    [
        subject_oof,
        label_oof,
        "_source_normalized",
        "source_class",
        fold_oof,
        prob_oof,
    ]
].copy()

subject_table = subject_table.rename(
    columns={
        "_source_normalized":
            "source",
        fold_oof:
            "outer_fold",
        prob_oof:
            "probability",
    }
)

subject_table = subject_table.sort_values(
    [
        "outer_fold",
        subject_oof,
    ]
)

subject_table.to_csv(
    SUBJECT_FILE,
    index=False
)


# ============================================================
# FINAL VALIDATION CHECKS
# ============================================================

print(
    "\n[13] Final validation..."
)

validation_checks = {
    "172 OOF subjects": (
        unique_subjects == 172
    ),

    "No duplicate OOF subjects": (
        duplicate_oof_subjects == 0
    ),

    "No missing source metadata": (
        missing_metadata == 0
    ),

    "No OOF/manifest label mismatches": (
        label_mismatch_count == 0
    ),

    "No subject in multiple folds": (
        len(multi_fold_subjects) == 0
    ),

    "All 5 outer folds present": (
        set(
            merged[fold_oof].unique()
        )
        ==
        {1, 2, 3, 4, 5}
    ),

    "No official-fold mismatches": (
        official_fold_mismatch == 0
    ),

    "No missing official-fold assignments": (
        official_fold_missing == 0
    ),
}

for name, passed in validation_checks.items():

    print(
        f"{name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )

final_pass = all(
    validation_checks.values()
)


# ============================================================
# WRITE REPORT
# ============================================================

print(
    "\n[14] Writing report..."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "NeuroFusion AI - sMRI Nested CV "
        "Fold/Source/Class Composition Analysis\n"
    )

    f.write(
        "=" * 72
        +
        "\n\n"
    )

    f.write(
        "READ-ONLY diagnostic analysis.\n"
    )

    f.write(
        "No model training, checkpoint, "
        "prediction, dataset, or CV split "
        "files were modified.\n\n"
    )

    f.write(
        "1. INPUTS\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        f"OOF predictions: {OOF_FILE}\n"
    )

    f.write(
        f"Labeled manifest: {MANIFEST_FILE}\n"
    )

    f.write(
        f"CV assignments: {ASSIGNMENT_FILE}\n\n"
    )

    f.write(
        "2. OVERALL SOURCE/CLASS COMPOSITION\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        overall_composition
        .to_string()
    )

    f.write(
        "\n\n"
    )

    f.write(
        "3. OUTER TEST-FOLD COMPOSITION\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        fold_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "4. PREDICTION BEHAVIOR "
        "BY SOURCE/CLASS\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        prediction_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "5. PROPORTIONAL BALANCE\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        balance_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "6. SUBJECT/FOLD COVERAGE\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        f"Unique OOF subjects: "
        f"{unique_subjects}\n"
    )

    f.write(
        f"Duplicate OOF subjects: "
        f"{duplicate_oof_subjects}\n"
    )

    f.write(
        f"Subjects in multiple folds: "
        f"{len(multi_fold_subjects)}\n"
    )

    f.write(
        f"OOF/manifest label mismatches: "
        f"{label_mismatch_count}\n"
    )

    f.write(
        f"Official fold mismatches: "
        f"{official_fold_mismatch}\n"
    )

    f.write(
        f"Missing official fold assignments: "
        f"{official_fold_missing}\n\n"
    )

    f.write(
        "7. OUTER TRAINING-SET COMPOSITION\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        training_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "8. INTERPRETATION NOTES\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    f.write(
        "The cohort contains three relevant "
        "source/class groups:\n"
    )

    f.write(
        "  CONTROL_LEGACY = legacy preprocessing controls\n"
    )

    f.write(
        "  CONTROL_RMS = RMS preprocessing controls\n"
    )

    f.write(
        "  SCHIZOPHRENIA_RMS = RMS preprocessing "
        "schizophrenia subjects\n\n"
    )

    f.write(
        "LEGACY occurs only among controls, so "
        "source and diagnosis are structurally "
        "confounded in this cohort.\n\n"
    )

    f.write(
        "This analysis is descriptive. It is "
        "intended to determine whether outer-fold "
        "composition may contribute to variation "
        "in model behavior. It does not establish "
        "causality.\n\n"
    )

    f.write(
        "No threshold, feature-selection, or "
        "model-selection decision should be made "
        "from this analysis alone.\n\n"
    )

    f.write(
        "9. VALIDATION\n"
    )

    f.write(
        "-" * 40
        +
        "\n"
    )

    for name, passed in validation_checks.items():

        f.write(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "FINAL STATUS: "
        f"{'PASS' if final_pass else 'FAIL'}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    f"FINAL STATUS: "
    f"{'PASS' if final_pass else 'FAIL'}"
)

print(
    "=" * 72
)

print(
    "\nOutput files:"
)

print(
    f"  {REPORT_FILE}"
)

print(
    f"  {FOLD_FILE}"
)

print(
    f"  {SUBJECT_FILE}"
)

print(
    f"  {PREDICTION_FILE}"
)

print(
    f"  {BALANCE_FILE}"
)

print(
    f"  {TRAINING_COMPOSITION_FILE}"
)

print(
    "\nNo training or dataset files were modified."
)