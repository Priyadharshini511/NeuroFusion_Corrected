# src\check_aal_5e_echo_consistency.py

import os
import numpy as np
import pandas as pd


# ================================================================
# AAL 5-ECHO FEATURE CONSISTENCY QC
#
# Generic, manifest-driven analysis of the extracted AAL feature
# matrix.
#
# Checks:
#   1. Current subjects exactly match the selection manifest
#   2. Exactly 5 echoes per subject
#   3. Exactly 116 AAL ROI features
#   4. One row per subject/echo
#   5. No duplicate subject/echo pairs
#   6. No missing subject/echo pairs
#   7. No unexpected subjects/echoes
#   8. No missing/non-finite feature values
#
# Descriptive echo analysis:
#   - Mean / SD by echo
#   - Subject-wise echo variability
#   - ROI-wise echo variability
#   - Coefficient of variation
#   - Relative range
#   - Echo correlations
#   - Mean absolute differences
#
# IMPORTANT:
#   Echo variability/correlation is DESCRIPTIVE.
#   It is NOT used as an automatic rejection threshold.
#
# This script does not modify the feature matrix.
# ================================================================


print("=" * 70)
print("AAL 5-ECHO FEATURE CONSISTENCY QC")
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
    r"\qc_aal_5e"
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "aal_5e_echo_consistency_summary.csv"
)

SUBJECT_STATS_CSV = os.path.join(
    QC_DIR,
    "aal_5e_subject_echo_variability.csv"
)

ROI_STATS_CSV = os.path.join(
    QC_DIR,
    "aal_5e_roi_echo_variability.csv"
)

CORRELATION_CSV = os.path.join(
    QC_DIR,
    "aal_5e_echo_correlations.csv"
)

ECHO_MEAN_DIFF_CSV = os.path.join(
    QC_DIR,
    "aal_5e_echo_mean_differences.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ECHOES = [1, 2, 3, 4, 5]

EXPECTED_ROIS = 116

ID_COLUMN = "subject_id"
ECHO_COLUMN = "echo"

FEATURE_PREFIX = "sMRI_AAL_"


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

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:

        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
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


def numeric_feature_columns(
    dataframe
):

    feature_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith(
            FEATURE_PREFIX
        )
    ]

    return feature_columns


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
print("LOADING AAL FEATURE MATRIX")
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
# BASIC COLUMN VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("COLUMN VALIDATION")
print("=" * 70)

required_columns = {
    ID_COLUMN,
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


feature_columns = numeric_feature_columns(
    df
)

print(
    "AAL feature columns:",
    len(feature_columns)
)

if len(feature_columns) != EXPECTED_ROIS:

    raise RuntimeError(
        "Incorrect number of AAL feature columns.\n"
        f"Expected: {EXPECTED_ROIS}\n"
        f"Found   : {len(feature_columns)}"
    )


# ================================================================
# FEATURE NAME VALIDATION
# ================================================================

if len(
    set(feature_columns)
) != len(feature_columns):

    raise RuntimeError(
        "Duplicate feature names detected."
    )


invalid_feature_names = [
    column
    for column in feature_columns
    if not str(column).startswith(
        FEATURE_PREFIX
    )
]

if invalid_feature_names:

    raise RuntimeError(
        "Invalid feature names detected: "
        f"{invalid_feature_names}"
    )


# ================================================================
# ROW COUNT
# ================================================================

print("\n" + "=" * 70)
print("ROW COUNT VALIDATION")
print("=" * 70)

if len(df) != expected_rows:

    raise RuntimeError(
        "Incorrect feature matrix row count.\n"
        f"Expected: {expected_rows}\n"
        f"Found   : {len(df)}"
    )

print(
    "Row count: PASS"
)


# ================================================================
# SUBJECT ID VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("SUBJECT VALIDATION")
print("=" * 70)

df[ID_COLUMN] = (
    df[ID_COLUMN]
    .astype(str)
    .str.strip()
)

if df[ID_COLUMN].eq("").any():

    raise RuntimeError(
        "Empty subject IDs found in feature matrix."
    )


matrix_subjects = set(
    df[ID_COLUMN]
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
    "Subjects in matrix:",
    len(matrix_subjects)
)

print(
    "Subjects in manifest:",
    len(manifest_subjects)
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
        f"Echo column is not valid integer data: {e}"
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
        df[ID_COLUMN],
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

duplicate_count = int(
    df.duplicated(
        subset=[
            ID_COLUMN,
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


if duplicate_count != 0:

    raise RuntimeError(
        "Duplicate subject/echo rows detected: "
        f"{duplicate_count}"
    )


print(
    "All subject/echo pairs present: PASS"
)

print(
    "Duplicate subject/echo rows: 0"
)


# ================================================================
# MISSING / NON-FINITE FEATURE VALUES
# ================================================================

print("\n" + "=" * 70)
print("FEATURE VALUE VALIDATION")
print("=" * 70)

missing_feature_values = int(
    df[
        feature_columns
    ].isna().sum().sum()
)

if missing_feature_values != 0:

    raise RuntimeError(
        "Missing feature values detected: "
        f"{missing_feature_values}"
    )


feature_matrix = df[
    feature_columns
].to_numpy(
    dtype=np.float64
)

nonfinite_count = int(
    np.sum(
        ~np.isfinite(
            feature_matrix
        )
    )
)

if nonfinite_count != 0:

    raise RuntimeError(
        "Non-finite feature values detected: "
        f"{nonfinite_count}"
    )


print(
    "Missing feature values: 0"
)

print(
    "Non-finite feature values: 0"
)


# ================================================================
# ECHO-WISE GLOBAL DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("GLOBAL ECHO DESCRIPTIVE STATISTICS")
print("=" * 70)

global_echo_rows = []

for echo in EXPECTED_ECHOES:

    subset = df[
        df[ECHO_COLUMN] == echo
    ]

    values = subset[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    global_echo_rows.append(
        {
            "echo": echo,
            "rows": len(subset),
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            )
        }
    )

global_echo_df = pd.DataFrame(
    global_echo_rows
)

print(
    global_echo_df.to_string(
        index=False
    )
)


# ================================================================
# SUBJECT-WISE ECHO VARIABILITY
# ================================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE ECHO VARIABILITY")
print("=" * 70)

subject_rows = []

for subject in subjects:

    subject_df = df[
        df[ID_COLUMN] == subject
    ].sort_values(
        ECHO_COLUMN
    )

    values = subject_df[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    if values.shape != (
        len(EXPECTED_ECHOES),
        EXPECTED_ROIS
    ):

        raise RuntimeError(
            f"{subject}: unexpected echo/ROI shape "
            f"{values.shape}"
        )

    echo_means = np.mean(
        values,
        axis=1
    )

    overall_mean = float(
        np.mean(echo_means)
    )

    overall_std = float(
        np.std(echo_means)
    )

    if abs(overall_mean) > 1e-12:

        cv_percent = (
            overall_std
            / abs(overall_mean)
            * 100.0
        )

    else:

        cv_percent = np.nan

    relative_range_percent = (
        (
            np.max(echo_means)
            - np.min(echo_means)
        )
        / max(
            abs(overall_mean),
            1e-12
        )
        * 100.0
    )

    subject_rows.append(
        {
            "subject_id": subject,
            "echo_mean_mean": overall_mean,
            "echo_mean_std": overall_std,
            "cv_percent": cv_percent,
            "relative_range_percent":
                relative_range_percent
        }
    )

subject_stats_df = pd.DataFrame(
    subject_rows
)

subject_stats_df.to_csv(
    SUBJECT_STATS_CSV,
    index=False
)


print(
    "Saved:",
    SUBJECT_STATS_CSV
)


# ================================================================
# ROI-WISE ECHO VARIABILITY
# ================================================================

print("\n" + "=" * 70)
print("ROI-WISE ECHO VARIABILITY")
print("=" * 70)

roi_rows = []

for roi_index, roi_name in enumerate(
    feature_columns
):

    values = df[
        roi_name
    ].to_numpy(
        dtype=np.float64
    )

    echo_means = []

    for echo in EXPECTED_ECHOES:

        echo_values = df.loc[
            df[ECHO_COLUMN] == echo,
            roi_name
        ].to_numpy(
            dtype=np.float64
        )

        echo_means.append(
            float(
                np.mean(
                    echo_values
                )
            )
        )

    echo_means = np.asarray(
        echo_means,
        dtype=np.float64
    )

    mean_value = float(
        np.mean(echo_means)
    )

    std_value = float(
        np.std(echo_means)
    )

    if abs(mean_value) > 1e-12:

        cv_percent = (
            std_value
            / abs(mean_value)
            * 100.0
        )

    else:

        cv_percent = np.nan

    relative_range_percent = (
        (
            np.max(echo_means)
            - np.min(echo_means)
        )
        / max(
            abs(mean_value),
            1e-12
        )
        * 100.0
    )

    roi_rows.append(
        {
            "roi_index": roi_index + 1,
            "feature_name": roi_name,
            "mean_across_echoes":
                mean_value,
            "std_across_echoes":
                std_value,
            "cv_percent":
                cv_percent,
            "relative_range_percent":
                relative_range_percent
        }
    )

roi_stats_df = pd.DataFrame(
    roi_rows
)

roi_stats_df.to_csv(
    ROI_STATS_CSV,
    index=False
)


print(
    "Saved:",
    ROI_STATS_CSV
)


# ================================================================
# ECHO CORRELATIONS
# ================================================================

print("\n" + "=" * 70)
print("ECHO CORRELATIONS")
print("=" * 70)

wide_echo_matrix = {}

for echo in EXPECTED_ECHOES:

    subset = df[
        df[ECHO_COLUMN] == echo
    ].copy()

    subset = subset.sort_values(
        ID_COLUMN
    )

    wide_echo_matrix[
        echo
    ] = subset[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    ).reshape(-1)


correlation_matrix = np.zeros(
    (
        len(EXPECTED_ECHOES),
        len(EXPECTED_ECHOES)
    ),
    dtype=np.float64
)


for i, echo_i in enumerate(
    EXPECTED_ECHOES
):

    for j, echo_j in enumerate(
        EXPECTED_ECHOES
    ):

        x = wide_echo_matrix[
            echo_i
        ]

        y = wide_echo_matrix[
            echo_j
        ]

        if i == j:

            correlation = 1.0

        else:

            if (
                np.std(x) == 0
                or np.std(y) == 0
            ):

                correlation = np.nan

            else:

                correlation = float(
                    np.corrcoef(
                        x,
                        y
                    )[0, 1]
                )

        correlation_matrix[
            i,
            j
        ] = correlation


correlation_df = pd.DataFrame(
    correlation_matrix,
    index=[
        f"echo{e}"
        for e in EXPECTED_ECHOES
    ],
    columns=[
        f"echo{e}"
        for e in EXPECTED_ECHOES
    ]
)

correlation_df.to_csv(
    CORRELATION_CSV
)


print(
    correlation_df
)

print(
    "\nSaved:",
    CORRELATION_CSV
)


# ================================================================
# ECHO MEAN ABSOLUTE DIFFERENCES
# ================================================================

print("\n" + "=" * 70)
print("ECHO MEAN ABSOLUTE DIFFERENCES")
print("=" * 70)

difference_rows = []

for i, echo_i in enumerate(
    EXPECTED_ECHOES
):

    for echo_j in EXPECTED_ECHOES[
        i + 1:
    ]:

        x = wide_echo_matrix[
            echo_i
        ]

        y = wide_echo_matrix[
            echo_j
        ]

        absolute_difference = (
            np.abs(
                x - y
            )
        )

        difference_rows.append(
            {
                "echo_a": echo_i,
                "echo_b": echo_j,
                "mean_absolute_difference":
                    float(
                        np.mean(
                            absolute_difference
                        )
                    ),
                "median_absolute_difference":
                    float(
                        np.median(
                            absolute_difference
                        )
                    ),
                "max_absolute_difference":
                    float(
                        np.max(
                            absolute_difference
                        )
                    )
            }
        )


difference_df = pd.DataFrame(
    difference_rows
)

difference_df.to_csv(
    ECHO_MEAN_DIFF_CSV,
    index=False
)


print(
    difference_df.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    ECHO_MEAN_DIFF_CSV
)


# ================================================================
# SUMMARY
# ================================================================

summary_rows = [
    {
        "check": "selected_subjects",
        "value": expected_subject_count
    },
    {
        "check": "expected_rows",
        "value": expected_rows
    },
    {
        "check": "actual_rows",
        "value": len(df)
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
        "check": "actual_aal_rois",
        "value": len(feature_columns)
    },
    {
        "check": "duplicate_subject_echo_rows",
        "value": duplicate_count
    },
    {
        "check": "missing_feature_values",
        "value": missing_feature_values
    },
    {
        "check": "nonfinite_feature_values",
        "value": nonfinite_count
    },
    {
        "check": "missing_subjects",
        "value": len(missing_subjects)
    },
    {
        "check": "unexpected_subjects",
        "value": len(unexpected_subjects)
    },
    {
        "check": "missing_subject_echo_pairs",
        "value": len(missing_pairs)
    },
    {
        "check": "unexpected_subject_echo_pairs",
        "value": len(unexpected_pairs)
    }
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("AAL 5-ECHO CONSISTENCY QC SUMMARY")
print("=" * 70)

print(
    "Subjects:",
    expected_subject_count
)

print(
    "Rows:",
    len(df),
    "/",
    expected_rows
)

print(
    "AAL ROIs:",
    len(feature_columns),
    "/",
    EXPECTED_ROIS
)

print(
    "Duplicate subject/echo rows:",
    duplicate_count
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
    "Missing subject/echo pairs:",
    len(missing_pairs)
)

print(
    "Unexpected subject/echo pairs:",
    len(unexpected_pairs)
)

print(
    "\nSummary:",
    SUMMARY_CSV
)


# ================================================================
# FINAL INTERPRETATION
# ================================================================
#
# The integrity checks above are pass/fail.
#
# Echo variability, CV, relative range and correlation are
# descriptive statistics only. They do not automatically reject
# subjects, echoes or ROIs.
# ================================================================

print(
    "\nSTATUS: AUTOMATED INTEGRITY QC PASS"
)

print(
    "Echo variability statistics are descriptive only."
)

print("=" * 70)