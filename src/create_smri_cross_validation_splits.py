"""
Create and validate subject-level 5-fold cross-validation splits
for the NeuroFusion AI sMRI branch.

Input
-----
processed_data/smri/dataset/smri_subject_manifest_labeled.csv

CV strategy
-----------
5-fold StratifiedKFold using the combined:

    diagnosis + acquisition_source

strata:

    Control + LEGACY
    Control + RMS
    Schizophrenia + RMS

This preserves the available acquisition-source/diagnosis composition
as consistently as possible across folds.

Important
---------
- Subject-level splitting only.
- Every subject appears in exactly one test fold.
- No subject appears in both train and test within a fold.
- Existing baseline train/validation/test split is NOT modified.
- No CNN training is performed.
- No model files are modified.
- No test-set tuning is performed.

Outputs
-------
processed_data/smri/cnn_dataset/cross_validation/
    cv_assignments.csv
    fold_1.csv
    fold_2.csv
    fold_3.csv
    fold_4.csv
    fold_5.csv
    cv_split_summary.csv
    cv_split_validation_report.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

MANIFEST = (
    ROOT
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

OUTPUT_DIR = (
    ROOT
    / "cnn_dataset"
    / "cross_validation"
)

ASSIGNMENTS_FILE = OUTPUT_DIR / "cv_assignments.csv"
SUMMARY_FILE = OUTPUT_DIR / "cv_split_summary.csv"
VALIDATION_REPORT = (
    OUTPUT_DIR
    / "cv_split_validation_report.txt"
)


# ============================================================
# CV CONFIGURATION
# ============================================================

N_SPLITS = 5
RANDOM_SEED = 42

EXPECTED_SUBJECTS = 172
EXPECTED_CONTROLS = 93
EXPECTED_SCHIZOPHRENIA = 79

EXPECTED_LEGACY = 53
EXPECTED_RMS = 119


# ============================================================
# HELPERS
# ============================================================

def make_stratum(
    label: int,
    acquisition_source: str,
) -> str:
    """
    Create the combined diagnosis/source stratum.
    """
    if label == 0 and acquisition_source == "LEGACY":
        return "CONTROL_LEGACY"

    if label == 0 and acquisition_source == "RMS":
        return "CONTROL_RMS"

    if label == 1 and acquisition_source == "RMS":
        return "SCHIZOPHRENIA_RMS"

    if label == 1 and acquisition_source == "LEGACY":
        return "SCHIZOPHRENIA_LEGACY"

    raise ValueError(
        "Unexpected label/source combination: "
        f"label={label}, "
        f"acquisition_source={acquisition_source}"
    )


def validate_manifest(df: pd.DataFrame) -> list[str]:
    """
    Validate the labeled manifest before generating CV splits.
    """
    errors = []

    required_columns = {
        "subject_id",
        "label",
        "acquisition_source",
        "mni_volume",
        "aal_csv",
        "mni_valid",
        "aal_valid",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        return errors

    # --------------------------------------------------------
    # Subject count
    # --------------------------------------------------------

    if len(df) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} subjects; "
            f"found {len(df)}"
        )

    # --------------------------------------------------------
    # Subject uniqueness
    # --------------------------------------------------------

    duplicate_ids = (
        df.loc[
            df["subject_id"].duplicated(keep=False),
            "subject_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if duplicate_ids:
        errors.append(
            "Duplicate subject IDs found: "
            + ", ".join(sorted(duplicate_ids))
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    if labels.isna().any():
        errors.append(
            "Missing or non-numeric labels found."
        )

    invalid_labels = labels[
        ~labels.isin([0, 1])
    ]

    if not invalid_labels.empty:
        errors.append(
            "Non-binary labels found."
        )

    control_count = int(
        (labels == 0).sum()
    )

    schizophrenia_count = int(
        (labels == 1).sum()
    )

    if control_count != EXPECTED_CONTROLS:
        errors.append(
            f"Expected {EXPECTED_CONTROLS} controls; "
            f"found {control_count}"
        )

    if schizophrenia_count != EXPECTED_SCHIZOPHRENIA:
        errors.append(
            f"Expected {EXPECTED_SCHIZOPHRENIA} "
            f"schizophrenia-spectrum subjects; "
            f"found {schizophrenia_count}"
        )

    # --------------------------------------------------------
    # Acquisition source
    # --------------------------------------------------------

    valid_sources = {"RMS", "LEGACY"}

    invalid_sources = set(
        df["acquisition_source"].astype(str)
    ).difference(valid_sources)

    if invalid_sources:
        errors.append(
            "Invalid acquisition source(s): "
            + ", ".join(sorted(invalid_sources))
        )

    legacy_count = int(
        (
            df["acquisition_source"]
            == "LEGACY"
        ).sum()
    )

    rms_count = int(
        (
            df["acquisition_source"]
            == "RMS"
        ).sum()
    )

    if legacy_count != EXPECTED_LEGACY:
        errors.append(
            f"Expected {EXPECTED_LEGACY} LEGACY subjects; "
            f"found {legacy_count}"
        )

    if rms_count != EXPECTED_RMS:
        errors.append(
            f"Expected {EXPECTED_RMS} RMS subjects; "
            f"found {rms_count}"
        )

    # --------------------------------------------------------
    # Input validity
    # --------------------------------------------------------

    invalid_mni = df[
        pd.to_numeric(
            df["mni_valid"],
            errors="coerce",
        ) != 1
    ]

    if not invalid_mni.empty:
        errors.append(
            "Subjects with invalid MNI inputs: "
            + ", ".join(
                invalid_mni["subject_id"]
                .astype(str)
                .tolist()
            )
        )

    invalid_aal = df[
        pd.to_numeric(
            df["aal_valid"],
            errors="coerce",
        ) != 1
    ]

    if not invalid_aal.empty:
        errors.append(
            "Subjects with invalid AAL inputs: "
            + ", ".join(
                invalid_aal["subject_id"]
                .astype(str)
                .tolist()
            )
        )

    return errors


def validate_cv_assignments(
    assignments: pd.DataFrame,
) -> list[str]:
    """
    Validate the complete 5-fold assignment table.
    """
    errors = []

    # --------------------------------------------------------
    # Required assignment columns
    # --------------------------------------------------------

    required = {
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "fold",
    }

    missing = required.difference(
        assignments.columns
    )

    if missing:
        errors.append(
            "Assignment file missing columns: "
            + ", ".join(sorted(missing))
        )
        return errors

    # --------------------------------------------------------
    # Subject count and uniqueness
    # --------------------------------------------------------

    if len(assignments) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} assignments; "
            f"found {len(assignments)}"
        )

    if assignments["subject_id"].duplicated().any():
        errors.append(
            "Duplicate subject IDs in CV assignments."
        )

    # --------------------------------------------------------
    # Fold validity
    # --------------------------------------------------------

    expected_folds = set(
        range(1, N_SPLITS + 1)
    )

    actual_folds = set(
        assignments["fold"].astype(int)
    )

    if actual_folds != expected_folds:
        errors.append(
            "Invalid fold numbers. "
            f"Expected {sorted(expected_folds)}, "
            f"found {sorted(actual_folds)}"
        )

    fold_counts = (
        assignments["fold"]
        .value_counts()
        .sort_index()
    )

    if len(fold_counts) != N_SPLITS:
        errors.append(
            "Not all CV folds contain subjects."
        )

    # --------------------------------------------------------
    # Every subject appears exactly once
    # --------------------------------------------------------

    subject_fold_counts = (
        assignments
        .groupby("subject_id")["fold"]
        .nunique()
    )

    if not (
        subject_fold_counts == 1
    ).all():
        errors.append(
            "Some subjects are assigned to multiple "
            "or zero folds."
        )

    # --------------------------------------------------------
    # Verify each fold against every other fold
    # --------------------------------------------------------

    for fold in range(1, N_SPLITS + 1):

        test_ids = set(
            assignments.loc[
                assignments["fold"] == fold,
                "subject_id",
            ]
            .astype(str)
        )

        train_ids = set(
            assignments.loc[
                assignments["fold"] != fold,
                "subject_id",
            ]
            .astype(str)
        )

        overlap = test_ids.intersection(
            train_ids
        )

        if overlap:
            errors.append(
                f"Fold {fold}: train/test subject overlap: "
                + ", ".join(sorted(overlap))
            )

        if len(test_ids) == 0:
            errors.append(
                f"Fold {fold}: empty test set."
            )

        if len(train_ids) == 0:
            errors.append(
                f"Fold {fold}: empty training set."
            )

    # --------------------------------------------------------
    # Verify expected stratum preservation
    # --------------------------------------------------------

    expected_strata = {
        "CONTROL_LEGACY",
        "CONTROL_RMS",
        "SCHIZOPHRENIA_RMS",
    }

    actual_strata = set(
        assignments["cv_stratum"].astype(str)
    )

    if actual_strata != expected_strata:
        errors.append(
            "Unexpected CV strata: "
            + ", ".join(sorted(actual_strata))
        )

    # Every fold must contain every available stratum.
    for fold in range(1, N_SPLITS + 1):

        fold_df = assignments[
            assignments["fold"] == fold
        ]

        fold_strata = set(
            fold_df["cv_stratum"].astype(str)
        )

        missing_strata = (
            expected_strata - fold_strata
        )

        if missing_strata:
            errors.append(
                f"Fold {fold} missing strata: "
                + ", ".join(
                    sorted(missing_strata)
                )
            )

    return errors


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "NeuroFusion AI - CREATE sMRI 5-FOLD "
        "CROSS-VALIDATION SPLITS"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Labeled manifest not found:\n{MANIFEST}"
        )

    print(f"Input manifest : {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    print(f"Rows           : {len(df)}")

    # --------------------------------------------------------
    # Validate source manifest
    # --------------------------------------------------------

    manifest_errors = validate_manifest(df)

    if manifest_errors:
        print()
        print("MANIFEST VALIDATION: FAIL")

        for error in manifest_errors:
            print(f"  - {error}")

        raise SystemExit(1)

    print("Manifest validation: PASS")
    print()

    # --------------------------------------------------------
    # Normalize types
    # --------------------------------------------------------

    df = df.copy()

    df["subject_id"] = (
        df["subject_id"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        pd.to_numeric(
            df["label"],
            errors="raise",
        )
        .astype(int)
    )

    df["acquisition_source"] = (
        df["acquisition_source"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Create combined stratification variable
    # --------------------------------------------------------

    df["cv_stratum"] = [
        make_stratum(
            label,
            source,
        )
        for label, source in zip(
            df["label"],
            df["acquisition_source"],
        )
    ]

    print("CV strata:")
    print(
        df["cv_stratum"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 5-fold stratified split
    # --------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    assignments = np.full(
        len(df),
        -1,
        dtype=np.int32,
    )

    indices = np.arange(len(df))

    for fold_number, (_, test_indices) in enumerate(
        splitter.split(
            indices,
            df["cv_stratum"],
        ),
        start=1,
    ):
        assignments[test_indices] = fold_number

    if np.any(assignments < 1):
        raise RuntimeError(
            "Some subjects were not assigned "
            "to a CV fold."
        )

    df["fold"] = assignments

    # --------------------------------------------------------
    # Sort for reproducibility
    # --------------------------------------------------------

    df = df.sort_values(
        ["fold", "subject_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validate assignments
    # --------------------------------------------------------

    assignment_errors = (
        validate_cv_assignments(df)
    )

    # --------------------------------------------------------
    # Save master assignment file
    # --------------------------------------------------------

    assignment_columns = [
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "fold",
        "mni_volume",
        "aal_csv",
    ]

    df[assignment_columns].to_csv(
        ASSIGNMENTS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Create individual fold files
    # --------------------------------------------------------

    summary_rows = []

    for fold in range(1, N_SPLITS + 1):

        test_df = df[
            df["fold"] == fold
        ].copy()

        train_df = df[
            df["fold"] != fold
        ].copy()

        # Add explicit role.
        test_df["split"] = "TEST"
        train_df["split"] = "TRAIN"

        fold_df = pd.concat(
            [train_df, test_df],
            axis=0,
            ignore_index=True,
        )

        fold_df = fold_df.sort_values(
            ["split", "subject_id"]
        ).reset_index(drop=True)

        fold_file = (
            OUTPUT_DIR
            / f"fold_{fold}.csv"
        )

        fold_df.to_csv(
            fold_file,
            index=False,
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Fold statistics
        # ----------------------------------------------------

        test_controls = int(
            (
                test_df["label"] == 0
            ).sum()
        )

        test_cases = int(
            (
                test_df["label"] == 1
            ).sum()
        )

        train_controls = int(
            (
                train_df["label"] == 0
            ).sum()
        )

        train_cases = int(
            (
                train_df["label"] == 1
            ).sum()
        )

        test_legacy = int(
            (
                test_df["acquisition_source"]
                == "LEGACY"
            ).sum()
        )

        test_rms = int(
            (
                test_df["acquisition_source"]
                == "RMS"
            ).sum()
        )

        train_legacy = int(
            (
                train_df["acquisition_source"]
                == "LEGACY"
            ).sum()
        )

        train_rms = int(
            (
                train_df["acquisition_source"]
                == "RMS"
            ).sum()
        )

        summary_rows.append(
            {
                "fold": fold,
                "train_subjects": len(train_df),
                "test_subjects": len(test_df),
                "train_controls": train_controls,
                "train_schizophrenia": train_cases,
                "test_controls": test_controls,
                "test_schizophrenia": test_cases,
                "train_legacy": train_legacy,
                "train_rms": train_rms,
                "test_legacy": test_legacy,
                "test_rms": test_rms,
                "test_control_legacy": int(
                    (
                        test_df["cv_stratum"]
                        == "CONTROL_LEGACY"
                    ).sum()
                ),
                "test_control_rms": int(
                    (
                        test_df["cv_stratum"]
                        == "CONTROL_RMS"
                    ).sum()
                ),
                "test_schizophrenia_rms": int(
                    (
                        test_df["cv_stratum"]
                        == "SCHIZOPHRENIA_RMS"
                    ).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final validation report
    # --------------------------------------------------------

    report_lines = [
        "NeuroFusion AI - sMRI 5-Fold CV Split Validation",
        "=" * 70,
        "",
        f"Input manifest          : {MANIFEST}",
        f"Number of subjects      : {len(df)}",
        f"Number of folds         : {N_SPLITS}",
        f"Random seed             : {RANDOM_SEED}",
        "",
        "Overall cohort:",
        f"  Controls              : "
        f"{int((df['label'] == 0).sum())}",
        f"  Schizophrenia         : "
        f"{int((df['label'] == 1).sum())}",
        f"  LEGACY                : "
        f"{int((df['acquisition_source'] == 'LEGACY').sum())}",
        f"  RMS                   : "
        f"{int((df['acquisition_source'] == 'RMS').sum())}",
        "",
        "Combined strata:",
    ]

    for stratum, count in (
        df["cv_stratum"]
        .value_counts()
        .sort_index()
        .items()
    ):
        report_lines.append(
            f"  {stratum}: {count}"
        )

    report_lines.extend(
        [
            "",
            "Fold distribution:",
        ]
    )

    for _, row in summary_df.iterrows():

        report_lines.append(
            f"  Fold {int(row['fold'])}: "
            f"train={int(row['train_subjects'])}, "
            f"test={int(row['test_subjects'])}, "
            f"test_controls={int(row['test_controls'])}, "
            f"test_schizophrenia="
            f"{int(row['test_schizophrenia'])}, "
            f"test_legacy={int(row['test_legacy'])}, "
            f"test_rms={int(row['test_rms'])}"
        )

    report_lines.extend(
        [
            "",
            "Validation errors:",
        ]
    )

    all_errors = (
        manifest_errors
        + assignment_errors
    )

    if all_errors:
        report_lines.extend(
            f"  - {error}"
            for error in all_errors
        )
        final_status = "FAIL"
    else:
        report_lines.append("  None")
        final_status = "PASS"

    report_lines.extend(
        [
            "",
            "Important:",
            "  The existing baseline 70/15/15 split was not modified.",
            "  No CNN training was performed.",
            "  No model files were modified.",
            "  Each subject belongs to exactly one test fold.",
            "  Each fold uses the combined diagnosis/source strata.",
            "",
            f"Assignments file : {ASSIGNMENTS_FILE}",
            f"Summary file     : {SUMMARY_FILE}",
            f"FINAL STATUS     : {final_status}",
        ]
    )

    VALIDATION_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("=" * 70)
    print("CV SPLIT SUMMARY")
    print("=" * 70)
    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(f"Assignments : {ASSIGNMENTS_FILE}")
    print(f"Summary     : {SUMMARY_FILE}")
    print(f"Report      : {VALIDATION_REPORT}")
    print()

    if all_errors:
        print(
            f"FINAL STATUS: FAIL "
            f"({len(all_errors)} issue(s))"
        )
        raise SystemExit(1)

    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()