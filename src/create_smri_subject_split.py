"""
Create a leakage-safe subject-level train/validation/test split
for the NeuroFusion AI sMRI 3D CNN branch.

Input
-----
processed_data/smri/cnn_dataset/labels.csv

Output
------
processed_data/smri/cnn_dataset/splits/
    train.csv
    validation.csv
    test.csv
    split_summary.csv
    split_validation_report.txt

Design
------
- Subject-level splitting
- Stratified by binary diagnosis label
- 70% train
- 15% validation
- 15% test
- Fixed random seed
- No subject overlap
- Every labeled subject appears exactly once

This script does NOT modify MRI tensors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

CNN_DATASET_DIR = ROOT / "cnn_dataset"

LABELS_FILE = (
    CNN_DATASET_DIR
    / "labels.csv"
)

SPLIT_DIR = (
    CNN_DATASET_DIR
    / "splits"
)

TRAIN_FILE = (
    SPLIT_DIR
    / "train.csv"
)

VALIDATION_FILE = (
    SPLIT_DIR
    / "validation.csv"
)

TEST_FILE = (
    SPLIT_DIR
    / "test.csv"
)

SUMMARY_FILE = (
    SPLIT_DIR
    / "split_summary.csv"
)

REPORT_FILE = (
    SPLIT_DIR
    / "split_validation_report.txt"
)


# ============================================================
# DATASET EXPECTATIONS
# ============================================================

EXPECTED_TOTAL = 172

EXPECTED_CONTROL = 93
EXPECTED_SCHIZOPHRENIA = 79

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def class_counts(
    df: pd.DataFrame,
) -> dict[int, int]:
    """Return binary class counts."""

    counts = (
        df["label"]
        .astype(int)
        .value_counts()
        .to_dict()
    )

    return {
        0: int(counts.get(0, 0)),
        1: int(counts.get(1, 0)),
    }


def validate_split_names(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:
    """Validate subject-level split integrity."""

    errors = []

    train_ids = set(
        train["subject_id"].astype(str)
    )

    validation_ids = set(
        validation["subject_id"].astype(str)
    )

    test_ids = set(
        test["subject_id"].astype(str)
    )

    # --------------------------------------------------------
    # Duplicate IDs within each split
    # --------------------------------------------------------

    for name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        duplicates = (
            dataframe[
                dataframe["subject_id"].duplicated(
                    keep=False
                )
            ]["subject_id"]
            .astype(str)
            .unique()
            .tolist()
        )

        if duplicates:
            errors.append(
                f"Duplicate subject IDs in {name}: "
                + ", ".join(sorted(duplicates))
            )

    # --------------------------------------------------------
    # Cross-split overlap
    # --------------------------------------------------------

    train_validation_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    validation_test_overlap = (
        validation_ids & test_ids
    )

    if train_validation_overlap:
        errors.append(
            "Train/validation overlap: "
            + ", ".join(
                sorted(train_validation_overlap)
            )
        )

    if train_test_overlap:
        errors.append(
            "Train/test overlap: "
            + ", ".join(
                sorted(train_test_overlap)
            )
        )

    if validation_test_overlap:
        errors.append(
            "Validation/test overlap: "
            + ", ".join(
                sorted(validation_test_overlap)
            )
        )

    # --------------------------------------------------------
    # Complete subject coverage
    # --------------------------------------------------------

    combined_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    original_ids = set(
        pd.concat(
            [
                train[["subject_id"]],
                validation[["subject_id"]],
                test[["subject_id"]],
            ]
        )["subject_id"].astype(str)
    )

    if len(combined_ids) != len(
        train
    ) + len(
        validation
    ) + len(
        test
    ):
        errors.append(
            "Duplicate subject IDs detected across "
            "the combined splits."
        )

    if len(combined_ids) != EXPECTED_TOTAL:
        errors.append(
            f"Combined split contains "
            f"{len(combined_ids)} unique subjects; "
            f"expected {EXPECTED_TOTAL}."
        )

    return errors


def create_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact split summary table."""

    rows = []

    total = len(train) + len(validation) + len(test)

    for split_name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        counts = class_counts(dataframe)

        rows.append(
            {
                "split": split_name,
                "subjects": len(dataframe),
                "percentage": (
                    100.0
                    * len(dataframe)
                    / total
                ),
                "control": counts[0],
                "schizophrenia": counts[1],
                "control_percentage": (
                    100.0
                    * counts[0]
                    / len(dataframe)
                ),
                "schizophrenia_percentage": (
                    100.0
                    * counts[1]
                    / len(dataframe)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print("NeuroFusion AI - CREATE sMRI SUBJECT-LEVEL SPLIT")
    print("=" * 72)
    print()

    errors: list[str] = []

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_FILE}"
        )

    labels = pd.read_csv(
        LABELS_FILE
    )

    print(
        f"Input subjects     : {len(labels)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "subject_id",
        "label",
    }

    missing_columns = (
        required_columns
        - set(labels.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Subject uniqueness
    # --------------------------------------------------------

    if labels["subject_id"].duplicated().any():

        duplicates = (
            labels.loc[
                labels["subject_id"].duplicated(
                    keep=False
                ),
                "subject_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        errors.append(
            "Duplicate subjects in input labels: "
            + ", ".join(
                sorted(duplicates)
            )
        )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    labels["label"] = pd.to_numeric(
        labels["label"],
        errors="coerce",
    )

    if labels["label"].isna().any():
        errors.append(
            "Missing or non-numeric labels detected."
        )

    if not set(
        labels["label"]
        .dropna()
        .astype(int)
        .unique()
    ).issubset({0, 1}):

        errors.append(
            "Non-binary labels detected."
        )

    labels["label"] = labels["label"].astype(int)

    counts = class_counts(labels)

    print(
        f"Control subjects   : {counts[0]}"
    )

    print(
        f"Schizophrenia      : {counts[1]}"
    )

    if len(labels) != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} subjects; "
            f"found {len(labels)}"
        )

    if counts[0] != EXPECTED_CONTROL:
        errors.append(
            f"Expected {EXPECTED_CONTROL} controls; "
            f"found {counts[0]}"
        )

    if counts[1] != EXPECTED_SCHIZOPHRENIA:
        errors.append(
            f"Expected {EXPECTED_SCHIZOPHRENIA} "
            f"schizophrenia-spectrum subjects; "
            f"found {counts[1]}"
        )

    if errors:
        print()
        print("Input validation failed.")

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    # --------------------------------------------------------
    # Split sizes
    # --------------------------------------------------------

    total_subjects = len(labels)

    train_size = int(
        round(
            total_subjects
            * TRAIN_FRACTION
        )
    )

    test_size = int(
        round(
            total_subjects
            * TEST_FRACTION
        )
    )

    validation_size = (
        total_subjects
        - train_size
        - test_size
    )

    print()
    print("Requested proportions:")
    print(
        f"  Train       : "
        f"{TRAIN_FRACTION * 100:.1f}%"
    )
    print(
        f"  Validation  : "
        f"{VALIDATION_FRACTION * 100:.1f}%"
    )
    print(
        f"  Test        : "
        f"{TEST_FRACTION * 100:.1f}%"
    )

    print()
    print("Calculated sizes:")
    print(
        f"  Train       : {train_size}"
    )
    print(
        f"  Validation  : {validation_size}"
    )
    print(
        f"  Test        : {test_size}"
    )

    if (
        train_size
        + validation_size
        + test_size
        != total_subjects
    ):
        raise RuntimeError(
            "Split sizes do not sum to total subjects."
        )

    # --------------------------------------------------------
    # First split:
    # Train vs temporary
    #
    # Temporary contains validation + test.
    # --------------------------------------------------------

    train, temporary = train_test_split(
        labels,
        test_size=(
            validation_size
            + test_size
        ),
        stratify=labels["label"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # --------------------------------------------------------
    # Second split:
    # Validation vs test
    # --------------------------------------------------------

    temporary_test_fraction = (
        test_size
        / (
            validation_size
            + test_size
        )
    )

    validation, test = train_test_split(
        temporary,
        test_size=temporary_test_fraction,
        stratify=temporary["label"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # --------------------------------------------------------
    # Sort each split by subject ID
    # --------------------------------------------------------

    train = (
        train
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    validation = (
        validation
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate split integrity
    # --------------------------------------------------------

    split_errors = validate_split_names(
        train,
        validation,
        test,
    )

    errors.extend(split_errors)

    # --------------------------------------------------------
    # Validate split class distributions
    # --------------------------------------------------------

    train_counts = class_counts(train)
    validation_counts = class_counts(validation)
    test_counts = class_counts(test)

    print()
    print("Final split distribution:")
    print(
        f"  Train      : {len(train)} "
        f"(0={train_counts[0]}, "
        f"1={train_counts[1]})"
    )

    print(
        f"  Validation : {len(validation)} "
        f"(0={validation_counts[0]}, "
        f"1={validation_counts[1]})"
    )

    print(
        f"  Test       : {len(test)} "
        f"(0={test_counts[0]}, "
        f"1={test_counts[1]})"
    )

    # --------------------------------------------------------
    # Verify total class counts
    # --------------------------------------------------------

    combined_control = (
        train_counts[0]
        + validation_counts[0]
        + test_counts[0]
    )

    combined_schizophrenia = (
        train_counts[1]
        + validation_counts[1]
        + test_counts[1]
    )

    if combined_control != EXPECTED_CONTROL:
        errors.append(
            "Combined control count does not match "
            f"{EXPECTED_CONTROL}."
        )

    if (
        combined_schizophrenia
        != EXPECTED_SCHIZOPHRENIA
    ):
        errors.append(
            "Combined schizophrenia count does not "
            f"match {EXPECTED_SCHIZOPHRENIA}."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save split files
    # --------------------------------------------------------

    train.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8",
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8",
    )

    test.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_summary(
        train,
        validation,
        test,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = [
        "NeuroFusion AI - sMRI Subject-Level Split",
        "Validation Report",
        "=" * 72,
        "",
        f"Input labels           : {LABELS_FILE}",
        f"Total subjects         : {total_subjects}",
        f"Random seed            : {RANDOM_SEED}",
        "",
        "Requested split:",
        f"  Train                : {TRAIN_FRACTION:.2f}",
        f"  Validation           : {VALIDATION_FRACTION:.2f}",
        f"  Test                 : {TEST_FRACTION:.2f}",
        "",
        "Actual split:",
        f"  Train                : {len(train)}",
        f"  Validation           : {len(validation)}",
        f"  Test                 : {len(test)}",
        f"  Total                : "
        f"{len(train) + len(validation) + len(test)}",
        "",
        "Class distribution:",
        f"  Train  -> 0={train_counts[0]}, "
        f"1={train_counts[1]}",
        f"  Valid  -> 0={validation_counts[0]}, "
        f"1={validation_counts[1]}",
        f"  Test   -> 0={test_counts[0]}, "
        f"1={test_counts[1]}",
        "",
        "Leakage checks:",
        "  Subject-level splitting : enforced",
        "  Train/validation overlap: none expected",
        "  Train/test overlap      : none expected",
        "  Validation/test overlap : none expected",
        "",
        "Validation errors:",
    ]

    if errors:
        report_lines.extend(
            f"  - {error}"
            for error in errors
        )

        final_status = "FAIL"

    else:
        report_lines.append(
            "  None"
        )

        final_status = "PASS"

    report_lines.extend(
        [
            "",
            f"FINAL STATUS: {final_status}",
            "",
            f"Train file       : {TRAIN_FILE}",
            f"Validation file  : {VALIDATION_FILE}",
            f"Test file        : {TEST_FILE}",
            f"Summary file     : {SUMMARY_FILE}",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("SPLIT CREATION SUMMARY")
    print("=" * 72)

    print(
        f"Total subjects : {total_subjects}"
    )

    print(
        f"Train         : {len(train)}"
    )

    print(
        f"Validation    : {len(validation)}"
    )

    print(
        f"Test          : {len(test)}"
    )

    print()
    print(
        f"Train file      : {TRAIN_FILE}"
    )

    print(
        f"Validation file : {VALIDATION_FILE}"
    )

    print(
        f"Test file       : {TEST_FILE}"
    )

    print(
        f"Summary file    : {SUMMARY_FILE}"
    )

    print(
        f"Report          : {REPORT_FILE}"
    )

    print()

    if errors:
        print(
            f"FINAL STATUS: FAIL "
            f"({len(errors)} issue(s))"
        )
        raise SystemExit(1)

    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()