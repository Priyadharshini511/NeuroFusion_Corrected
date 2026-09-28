# src\check_aal_features.py

import os
import numpy as np
import pandas as pd


# ================================================================
# AAL FEATURE MATRIX INTEGRITY QC
#
# Generic, manifest-driven QC for the final 5-echo AAL feature
# matrix.
#
# Checks:
#   1. Current subjects come from selection manifest
#   2. Exactly 5 echoes per subject
#   3. Exactly 116 AAL ROI features
#   4. Expected N x 5 rows
#   5. Exactly one row per subject/echo
#   6. No duplicate subject/echo pairs
#   7. No missing subject/echo pairs
#   8. No unexpected subjects
#   9. No unexpected echoes
#  10. No missing feature values
#  11. No NaN / infinite feature values
#  12. Feature names are unique and correctly prefixed
#  13. Subject/echo identifiers are valid
#
# This script is QC ONLY.
# It does not modify the feature matrix.
# ================================================================


print("=" * 70)
print("AAL FEATURE MATRIX INTEGRITY QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

FEATURE_MATRIX = (
    r"processed_data\smri"
    r"\features\aal_5e"
    r"\COBRE_MPRAGE_5E_AAL_sMRI_features.csv"
)

MANIFEST_PATH = (
    r"processed_data\smri"
    r"\mprage_selection"
    r"\mprage_selection_manifest.csv"
)

QC_DIR = (
    r"processed_data\smri"
    r"\qc_aal_features"
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "aal_features_qc_summary.csv"
)

FEATURE_STATS_CSV = os.path.join(
    QC_DIR,
    "aal_features_descriptive_statistics.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ECHOES = [1, 2, 3, 4, 5]

EXPECTED_ROIS = 116

FEATURE_PREFIX = "sMRI_AAL_"

SUBJECT_COLUMN = "subject_id"

ECHO_COLUMN = "echo"


# ================================================================
# HELPERS
# ================================================================

def get_selected_subjects():

    if not os.path.isfile(
        MANIFEST_PATH
    ):

        raise FileNotFoundError(
            f"Selection manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "subject_id",
        "selection_status"
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:

        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    selected = manifest[
        manifest["selection_status"]
        .astype(str)
        .str.upper()
        .eq("SELECTED")
    ].copy()

    if selected.empty:

        raise ValueError(
            "No SELECTED subjects found in manifest."
        )

    subjects = (
        selected["subject_id"]
        .astype(str)
        .str.strip()
    )

    if subjects.eq("").any():

        raise ValueError(
            "Manifest contains empty subject IDs."
        )

    if subjects.duplicated().any():

        duplicates = sorted(
            subjects[
                subjects.duplicated()
            ].unique()
        )

        raise ValueError(
            "Duplicate SELECTED subjects found: "
            f"{duplicates}"
        )

    return sorted(
        subjects.unique()
    )


def get_feature_columns(
    dataframe
):

    return [
        column
        for column in dataframe.columns
        if str(column).startswith(
            FEATURE_PREFIX
        )
    ]


# ================================================================
# PREPARE QC DIRECTORY
# ================================================================

os.makedirs(
    QC_DIR,
    exist_ok=True
)


# ================================================================
# LOAD MANIFEST
# ================================================================

print("\n" + "=" * 70)
print("LOADING CURRENT SUBJECT COHORT")
print("=" * 70)

subjects = get_selected_subjects()

expected_subject_count = len(
    subjects
)

expected_rows = (
    expected_subject_count
    * len(EXPECTED_ECHOES)
)

print(
    "Selected subjects:",
    expected_subject_count
)

print(
    "Expected rows:",
    expected_rows
)


# ================================================================
# LOAD FEATURE MATRIX
# ================================================================

print("\n" + "=" * 70)
print("LOADING FEATURE MATRIX")
print("=" * 70)

if not os.path.isfile(
    FEATURE_MATRIX
):

    raise FileNotFoundError(
        f"Feature matrix not found:\n{FEATURE_MATRIX}"
    )

df = pd.read_csv(
    FEATURE_MATRIX
)

print(
    "Matrix shape:",
    df.shape
)


# ================================================================
# REQUIRED COLUMNS
# ================================================================

print("\n" + "=" * 70)
print("REQUIRED COLUMN VALIDATION")
print("=" * 70)

required_columns = {
    SUBJECT_COLUMN,
    ECHO_COLUMN
}

missing_required = (
    required_columns
    - set(df.columns)
)

if missing_required:

    raise RuntimeError(
        "Missing required columns: "
        f"{sorted(missing_required)}"
    )

print(
    "Subject column: PASS"
)

print(
    "Echo column: PASS"
)


# ================================================================
# FEATURE COLUMN DISCOVERY
# ================================================================

feature_columns = get_feature_columns(
    df
)

print(
    "Detected AAL feature columns:",
    len(feature_columns)
)

if len(feature_columns) != EXPECTED_ROIS:

    raise RuntimeError(
        "Incorrect number of AAL features.\n"
        f"Expected: {EXPECTED_ROIS}\n"
        f"Found   : {len(feature_columns)}"
    )

print(
    "AAL feature count: PASS"
)


# ================================================================
# FEATURE NAME VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("FEATURE NAME VALIDATION")
print("=" * 70)

if len(
    set(feature_columns)
) != len(feature_columns):

    raise RuntimeError(
        "Duplicate feature names detected."
    )

invalid_prefix = [
    column
    for column in feature_columns
    if not str(column).startswith(
        FEATURE_PREFIX
    )
]

if invalid_prefix:

    raise RuntimeError(
        "Feature names with invalid prefix: "
        f"{invalid_prefix}"
    )

print(
    "Feature names unique: PASS"
)

print(
    "Feature prefix validation: PASS"
)


# ================================================================
# ROW COUNT
# ================================================================

print("\n" + "=" * 70)
print("ROW COUNT VALIDATION")
print("=" * 70)

actual_rows = len(df)

print(
    "Expected rows:",
    expected_rows
)

print(
    "Actual rows:",
    actual_rows
)

if actual_rows != expected_rows:

    raise RuntimeError(
        "Incorrect number of rows.\n"
        f"Expected: {expected_rows}\n"
        f"Found   : {actual_rows}"
    )

print(
    "Row count: PASS"
)


# ================================================================
# SUBJECT ID VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("SUBJECT ID VALIDATION")
print("=" * 70)

df[SUBJECT_COLUMN] = (
    df[SUBJECT_COLUMN]
    .astype(str)
    .str.strip()
)

if df[
    SUBJECT_COLUMN
].eq("").any():

    raise RuntimeError(
        "Empty subject IDs found."
    )


matrix_subjects = set(
    df[
        SUBJECT_COLUMN
    ]
)

manifest_subjects = set(
    subjects
)

missing_subjects = sorted(
    manifest_subjects
    - matrix_subjects
)

unexpected_subjects = sorted(
    matrix_subjects
    - manifest_subjects
)


print(
    "Subjects in manifest:",
    len(manifest_subjects)
)

print(
    "Subjects in matrix:",
    len(matrix_subjects)
)


if missing_subjects:

    raise RuntimeError(
        "Subjects missing from feature matrix: "
        f"{missing_subjects}"
    )


if unexpected_subjects:

    raise RuntimeError(
        "Unexpected subjects in feature matrix: "
        f"{unexpected_subjects}"
    )


print(
    "Manifest ↔ matrix subject matching: PASS"
)


# ================================================================
# ECHO VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("ECHO VALIDATION")
print("=" * 70)

try:

    df[ECHO_COLUMN] = (
        pd.to_numeric(
            df[ECHO_COLUMN],
            errors="raise"
        )
        .astype(int)
    )

except Exception as e:

    raise RuntimeError(
        f"Invalid echo column: {e}"
    )


observed_echoes = sorted(
    df[ECHO_COLUMN].unique()
)

print(
    "Observed echoes:",
    observed_echoes
)

print(
    "Expected echoes:",
    EXPECTED_ECHOES
)

if observed_echoes != EXPECTED_ECHOES:

    raise RuntimeError(
        "Feature matrix does not contain exactly "
        "echoes 1-5."
    )

print(
    "Echo set: PASS"
)


# ================================================================
# SUBJECT/ECHO PAIR VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("SUBJECT/ECHO PAIR VALIDATION")
print("=" * 70)

expected_pairs = {
    (
        subject,
        echo
    )
    for subject in subjects
    for echo in EXPECTED_ECHOES
}

actual_pairs = set(
    zip(
        df[SUBJECT_COLUMN],
        df[ECHO_COLUMN]
    )
)

missing_pairs = sorted(
    expected_pairs
    - actual_pairs
)

unexpected_pairs = sorted(
    actual_pairs
    - expected_pairs
)

duplicate_pairs = int(
    df.duplicated(
        subset=[
            SUBJECT_COLUMN,
            ECHO_COLUMN
        ]
    ).sum()
)


if missing_pairs:

    raise RuntimeError(
        "Missing subject/echo pairs: "
        f"{missing_pairs}"
    )


if unexpected_pairs:

    raise RuntimeError(
        "Unexpected subject/echo pairs: "
        f"{unexpected_pairs}"
    )


if duplicate_pairs != 0:

    raise RuntimeError(
        "Duplicate subject/echo pairs: "
        f"{duplicate_pairs}"
    )


print(
    "Missing subject/echo pairs: 0"
)

print(
    "Unexpected subject/echo pairs: 0"
)

print(
    "Duplicate subject/echo pairs: 0"
)


# ================================================================
# MISSING VALUE VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUE VALIDATION")
print("=" * 70)

missing_feature_values = int(
    df[
        feature_columns
    ].isna().sum().sum()
)

print(
    "Missing feature values:",
    missing_feature_values
)

if missing_feature_values != 0:

    raise RuntimeError(
        "Missing feature values detected."
    )

print(
    "Missing feature values: PASS"
)


# ================================================================
# NUMERIC / FINITE VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("NUMERIC / FINITE FEATURE VALIDATION")
print("=" * 70)

try:

    feature_matrix = df[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

except Exception as e:

    raise RuntimeError(
        "AAL feature columns contain non-numeric values: "
        f"{e}"
    )


nonfinite_count = int(
    np.sum(
        ~np.isfinite(
            feature_matrix
        )
    )
)

print(
    "Non-finite values:",
    nonfinite_count
)

if nonfinite_count != 0:

    raise RuntimeError(
        "Non-finite feature values detected."
    )

print(
    "Numeric/finite features: PASS"
)


# ================================================================
# SUBJECT-WISE ECHO COMPLETENESS
# ================================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE ECHO COMPLETENESS")
print("=" * 70)

subject_echo_counts = (
    df.groupby(
        SUBJECT_COLUMN
    )[ECHO_COLUMN]
    .nunique()
)

incomplete_subjects = sorted(
    subject_echo_counts[
        subject_echo_counts
        != len(EXPECTED_ECHOES)
    ].index.astype(str)
    .tolist()
)

if incomplete_subjects:

    raise RuntimeError(
        "Subjects without all five echoes: "
        f"{incomplete_subjects}"
    )

print(
    "Every selected subject has echoes 1-5: PASS"
)


# ================================================================
# ECHO-WISE ROW COUNTS
# ================================================================

print("\n" + "=" * 70)
print("ECHO-WISE ROW COUNTS")
print("=" * 70)

echo_counts = (
    df[ECHO_COLUMN]
    .value_counts()
    .sort_index()
)

print(
    echo_counts
)

invalid_echo_counts = [
    echo
    for echo in EXPECTED_ECHOES
    if int(
        echo_counts.get(
            echo,
            0
        )
    ) != expected_subject_count
]

if invalid_echo_counts:

    raise RuntimeError(
        "Incorrect row count for echo(s): "
        f"{invalid_echo_counts}"
    )

print(
    "Echo-wise row counts: PASS"
)


# ================================================================
# DESCRIPTIVE FEATURE STATISTICS
# ================================================================
#
# These statistics are descriptive only.
# They are NOT rejection criteria.
# ================================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE FEATURE STATISTICS")
print("=" * 70)

feature_stats = pd.DataFrame(
    {
        "feature_name": feature_columns,
        "mean": np.mean(
            feature_matrix,
            axis=0
        ),
        "std": np.std(
            feature_matrix,
            axis=0
        ),
        "min": np.min(
            feature_matrix,
            axis=0
        ),
        "max": np.max(
            feature_matrix,
            axis=0
        ),
        "median": np.median(
            feature_matrix,
            axis=0
        )
    }
)

feature_stats.to_csv(
    FEATURE_STATS_CSV,
    index=False
)

print(
    "Feature statistics saved:"
)

print(
    FEATURE_STATS_CSV
)


# ================================================================
# FINAL MATRIX REORDER CHECK
# ================================================================

print("\n" + "=" * 70)
print("FINAL STRUCTURE CHECK")
print("=" * 70)

expected_columns = (
    [
        SUBJECT_COLUMN,
        ECHO_COLUMN
    ]
    + feature_columns
)

if list(
    df.columns
) != expected_columns:

    raise RuntimeError(
        "Final column ordering is incorrect."
    )

print(
    "Column structure: PASS"
)


# ================================================================
# SAVE QC SUMMARY
# ================================================================

summary = pd.DataFrame(
    [
        {
            "check": "selected_subjects",
            "value": expected_subject_count
        },
        {
            "check": "expected_echoes",
            "value": "1,2,3,4,5"
        },
        {
            "check": "expected_aal_rois",
            "value": EXPECTED_ROIS
        },
        {
            "check": "expected_rows",
            "value": expected_rows
        },
        {
            "check": "actual_rows",
            "value": actual_rows
        },
        {
            "check": "actual_subjects",
            "value": len(matrix_subjects)
        },
        {
            "check": "actual_aal_features",
            "value": len(feature_columns)
        },
        {
            "check": "duplicate_subject_echo_pairs",
            "value": duplicate_pairs
        },
        {
            "check": "missing_subject_echo_pairs",
            "value": len(missing_pairs)
        },
        {
            "check": "unexpected_subject_echo_pairs",
            "value": len(unexpected_pairs)
        },
        {
            "check": "missing_feature_values",
            "value": missing_feature_values
        },
        {
            "check": "nonfinite_feature_values",
            "value": nonfinite_count
        }
    ]
)

summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("AAL FEATURE MATRIX QC SUMMARY")
print("=" * 70)

print(
    "Selected subjects:",
    expected_subject_count
)

print(
    "Rows:",
    actual_rows,
    "/",
    expected_rows
)

print(
    "AAL features:",
    len(feature_columns),
    "/",
    EXPECTED_ROIS
)

print(
    "Echoes:",
    observed_echoes
)

print(
    "Duplicate subject/echo pairs:",
    duplicate_pairs
)

print(
    "Missing subject/echo pairs:",
    len(missing_pairs)
)

print(
    "Unexpected subject/echo pairs:",
    len(unexpected_pairs)
)

print(
    "Missing feature values:",
    missing_feature_values
)

print(
    "Non-finite feature values:",
    nonfinite_count
)

print(
    "\nQC summary:",
    SUMMARY_CSV
)

print(
    "Feature statistics:",
    FEATURE_STATS_CSV
)


if (
    duplicate_pairs == 0
    and len(missing_pairs) == 0
    and len(unexpected_pairs) == 0
    and missing_feature_values == 0
    and nonfinite_count == 0
    and len(feature_columns) == EXPECTED_ROIS
    and actual_rows == expected_rows
    and matrix_subjects == manifest_subjects
    and observed_echoes == EXPECTED_ECHOES
):

    print(
        "\nSTATUS: AUTOMATED QC PASS"
    )

else:

    print(
        "\nSTATUS: QC FAILED"
    )


print("=" * 70)