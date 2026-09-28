from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp06_rms_cnn_aal"
    / "cross_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER
# ============================================================

def get_cv_stratum(label):
    """
    RMS-only diagnosis stratum.

    0 = Control
    1 = Schizophrenia
    """

    label = int(label)

    if label == 0:
        return "RMS_CONTROL"

    if label == 1:
        return "RMS_SCHIZOPHRENIA"

    raise ValueError(
        f"Unexpected label: {label}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 06")
    print("RMS-ONLY CNN + AAL")
    print("CREATING 5×4 NESTED CROSS-VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest["subject_id"] = (
        manifest["subject_id"]
        .astype(str)
    )

    # ========================================================
    # SELECT RMS SUBJECTS ONLY
    # ========================================================

    rms = manifest[
        manifest[
            "acquisition_source"
        ]
        .astype(str)
        .str.upper()
        == "RMS"
    ].copy()

    rms = (
        rms
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if len(rms) != 119:

        raise ValueError(
            f"Expected 119 RMS subjects, "
            f"found {len(rms)}"
        )

    if rms[
        "subject_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate RMS subject IDs detected."
        )

    class_counts = (
        rms["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_class_counts = {
        0: 40,
        1: 79,
    }

    if class_counts != expected_class_counts:

        raise ValueError(
            f"Expected RMS class counts "
            f"{expected_class_counts}, "
            f"found {class_counts}"
        )

    print()

    print(
        "RMS subjects:",
        len(rms),
    )

    print(
        "RMS controls:",
        int(
            (
                rms["label"] == 0
            ).sum()
        ),
    )

    print(
        "RMS schizophrenia:",
        int(
            (
                rms["label"] == 1
            ).sum()
        ),
    )

    # ========================================================
    # OUTER 5-FOLD STRATIFIED SPLIT
    # ========================================================

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    X_outer = np.zeros(
        (len(rms), 1),
        dtype=np.float32,
    )

    y_outer = (
        rms["label"]
        .to_numpy()
    )

    # Maps subject_id -> outer fold.
    outer_fold_assignment = {}

    for outer_fold, (
        outer_train_idx,
        outer_test_idx,
    ) in enumerate(
        outer_cv.split(
            X_outer,
            y_outer,
        ),
        start=1,
    ):

        for idx in outer_test_idx:

            subject_id = str(
                rms.iloc[idx][
                    "subject_id"
                ]
            )

            if subject_id in (
                outer_fold_assignment
            ):

                raise RuntimeError(
                    f"Subject {subject_id} "
                    f"assigned to multiple outer folds."
                )

            outer_fold_assignment[
                subject_id
            ] = int(
                outer_fold
            )

    # --------------------------------------------------------
    # Every subject must belong to exactly one outer fold.
    # --------------------------------------------------------

    if len(
        outer_fold_assignment
    ) != 119:

        raise RuntimeError(
            "Outer-fold assignment does not "
            "cover all 119 subjects."
        )

    # ========================================================
    # BUILD NESTED ASSIGNMENTS
    # ========================================================

    rows = []

    for outer_fold in range(
        1,
        6,
    ):

        print()

        print(
            "-" * 70
        )

        print(
            f"Preparing Outer Fold {outer_fold}"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # Outer test subjects
        # ----------------------------------------------------

        outer_test_ids = [
            subject_id
            for subject_id, assigned_fold
            in outer_fold_assignment.items()
            if int(assigned_fold)
            == int(outer_fold)
        ]

        outer_test_ids = sorted(
            outer_test_ids
        )

        # ----------------------------------------------------
        # Outer training subjects
        # ----------------------------------------------------

        outer_train_ids = [
            subject_id
            for subject_id in rms[
                "subject_id"
            ].astype(str)
            if subject_id
            not in set(
                outer_test_ids
            )
        ]

        outer_train_ids = sorted(
            outer_train_ids
        )

        # ----------------------------------------------------
        # Check outer partition
        # ----------------------------------------------------

        if (
            len(outer_train_ids)
            + len(outer_test_ids)
            != 119
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"train + test does not equal 119."
            )

        if (
            set(outer_train_ids)
            & set(outer_test_ids)
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"train/test overlap detected."
            )

        outer_train = (
            rms[
                rms[
                    "subject_id"
                ].isin(
                    outer_train_ids
                )
            ]
            .sort_values("subject_id")
            .reset_index(drop=True)
        )

        outer_test = (
            rms[
                rms[
                    "subject_id"
                ].isin(
                    outer_test_ids
                )
            ]
            .sort_values("subject_id")
            .reset_index(drop=True)
        )

        print(
            "Outer training subjects:",
            len(outer_train),
        )

        print(
            "Outer test subjects:",
            len(outer_test),
        )

        print(
            "Outer training controls:",
            int(
                (
                    outer_train["label"]
                    == 0
                ).sum()
            ),
        )

        print(
            "Outer training schizophrenia:",
            int(
                (
                    outer_train["label"]
                    == 1
                ).sum()
            ),
        )

        print(
            "Outer test controls:",
            int(
                (
                    outer_test["label"]
                    == 0
                ).sum()
            ),
        )

        print(
            "Outer test schizophrenia:",
            int(
                (
                    outer_test["label"]
                    == 1
                ).sum()
            ),
        )

        # ====================================================
        # INNER 4-FOLD SPLIT
        #
        # IMPORTANT:
        #
        # We generate each inner fold independently.
        # We DO NOT store one global subject -> inner fold
        # mapping.
        #
        # Each outer-training subject therefore receives:
        #
        #   INNER_VALIDATION in exactly one inner fold
        #   INNER_TRAIN in the other three inner folds
        #
        # This is the correct nested-CV structure.
        # ====================================================

        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=(
                SEED
                + int(outer_fold)
            ),
        )

        X_inner = np.zeros(
            (len(outer_train), 1),
            dtype=np.float32,
        )

        y_inner = (
            outer_train["label"]
            .to_numpy()
        )

        for inner_fold, (
            inner_train_idx,
            inner_validation_idx,
        ) in enumerate(
            inner_cv.split(
                X_inner,
                y_inner,
            ),
            start=1,
        ):

            inner_train = (
                outer_train
                .iloc[
                    inner_train_idx
                ]
                .copy()
            )

            inner_validation = (
                outer_train
                .iloc[
                    inner_validation_idx
                ]
                .copy()
            )

            inner_train_ids = (
                inner_train[
                    "subject_id"
                ]
                .astype(str)
                .tolist()
            )

            inner_validation_ids = (
                inner_validation[
                    "subject_id"
                ]
                .astype(str)
                .tolist()
            )

            # ------------------------------------------------
            # Inner partition integrity
            # ------------------------------------------------

            if (
                set(
                    inner_train_ids
                )
                & set(
                    inner_validation_ids
                )
            ):

                raise RuntimeError(
                    f"Outer fold {outer_fold}, "
                    f"inner fold {inner_fold}: "
                    f"train/validation overlap."
                )

            if (
                len(inner_train_ids)
                + len(inner_validation_ids)
                != len(outer_train)
            ):

                raise RuntimeError(
                    f"Outer fold {outer_fold}, "
                    f"inner fold {inner_fold}: "
                    f"inner partition does not cover "
                    f"outer training subjects."
                )

            # ------------------------------------------------
            # INNER TRAIN rows
            # ------------------------------------------------

            for _, subject_row in (
                inner_train.iterrows()
            ):

                subject_id = str(
                    subject_row[
                        "subject_id"
                    ]
                )

                rows.append(
                    {
                        "subject_id":
                            subject_id,

                        "label":
                            int(
                                subject_row[
                                    "label"
                                ]
                            ),

                        "acquisition_source":
                            str(
                                subject_row[
                                    "acquisition_source"
                                ]
                            ),

                        "cv_stratum":
                            get_cv_stratum(
                                subject_row[
                                    "label"
                                ]
                            ),

                        "outer_fold":
                            int(
                                outer_fold
                            ),

                        "inner_fold":
                            int(
                                inner_fold
                            ),

                        "role":
                            "INNER_TRAIN",
                    }
                )

            # ------------------------------------------------
            # INNER VALIDATION rows
            # ------------------------------------------------

            for _, subject_row in (
                inner_validation.iterrows()
            ):

                subject_id = str(
                    subject_row[
                        "subject_id"
                    ]
                )

                rows.append(
                    {
                        "subject_id":
                            subject_id,

                        "label":
                            int(
                                subject_row[
                                    "label"
                                ]
                            ),

                        "acquisition_source":
                            str(
                                subject_row[
                                    "acquisition_source"
                                ]
                            ),

                        "cv_stratum":
                            get_cv_stratum(
                                subject_row[
                                    "label"
                                ]
                            ),

                        "outer_fold":
                            int(
                                outer_fold
                            ),

                        "inner_fold":
                            int(
                                inner_fold
                            ),

                        "role":
                            "INNER_VALIDATION",
                    }
                )

            # ------------------------------------------------
            # OUTER TEST rows
            #
            # These are intentionally repeated for every
            # inner fold because the nested assignment table
            # represents the full outer/inner structure.
            #
            # The downstream training script MUST use
            # .unique() when extracting OUTER_TEST subjects.
            # ------------------------------------------------

            for _, subject_row in (
                outer_test.iterrows()
            ):

                subject_id = str(
                    subject_row[
                        "subject_id"
                    ]
                )

                rows.append(
                    {
                        "subject_id":
                            subject_id,

                        "label":
                            int(
                                subject_row[
                                    "label"
                                ]
                            ),

                        "acquisition_source":
                            str(
                                subject_row[
                                    "acquisition_source"
                                ]
                            ),

                        "cv_stratum":
                            get_cv_stratum(
                                subject_row[
                                    "label"
                                ]
                            ),

                        "outer_fold":
                            int(
                                outer_fold
                            ),

                        "inner_fold":
                            int(
                                inner_fold
                            ),

                        "role":
                            "OUTER_TEST",
                    }
                )

            print(
                f"  Inner fold {inner_fold}: "
                f"train={len(inner_train)}, "
                f"validation={len(inner_validation)}, "
                f"outer_test={len(outer_test)}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    assignments = pd.DataFrame(
        rows
    )

    # ========================================================
    # EXPECTED TOTAL ROWS
    # ========================================================

    expected_rows = (
        119
        * 5
        * 4
    )

    print()

    print(
        "=" * 70
    )

    print(
        "VALIDATING NESTED-CV ASSIGNMENTS"
    )

    print(
        "=" * 70
    )

    print(
        "Expected rows:",
        expected_rows,
    )

    print(
        "Actual rows:",
        len(assignments),
    )

    if len(assignments) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} "
            f"assignment rows, "
            f"found {len(assignments)}"
        )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    expected_columns = [
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "outer_fold",
        "inner_fold",
        "role",
    ]

    if list(
        assignments.columns
    ) != expected_columns:

        raise RuntimeError(
            "Unexpected assignment columns: "
            f"{list(assignments.columns)}"
        )

    # ========================================================
    # SUBJECT SET
    # ========================================================

    assignment_subjects = set(
        assignments[
            "subject_id"
        ]
        .astype(str)
        .unique()
    )

    rms_subjects = set(
        rms[
            "subject_id"
        ]
        .astype(str)
        .unique()
    )

    if (
        assignment_subjects
        != rms_subjects
    ):

        missing = (
            rms_subjects
            - assignment_subjects
        )

        extra = (
            assignment_subjects
            - rms_subjects
        )

        raise RuntimeError(
            "Assignment subject set mismatch. "
            f"Missing={sorted(missing)} "
            f"Extra={sorted(extra)}"
        )

    # ========================================================
    # EACH SUBJECT APPEARS 20 TIMES
    # ========================================================

    subject_counts = (
        assignments[
            "subject_id"
        ]
        .value_counts()
    )

    if not (
        subject_counts == 20
    ).all():

        bad_subjects = (
            subject_counts[
                subject_counts != 20
            ]
            .to_dict()
        )

        raise RuntimeError(
            "Every subject must appear exactly "
            f"20 times. Problems: {bad_subjects}"
        )

    print(
        "Unique subjects:",
        assignments[
            "subject_id"
        ].nunique(),
    )

    print(
        "Each subject appears exactly 20 times: PASS"
    )

    # ========================================================
    # ROLE COUNTS
    # ========================================================

    role_counts = (
        assignments[
            "role"
        ]
        .value_counts()
        .to_dict()
    )

    expected_role_counts = {
        "INNER_TRAIN": 1428,
        "INNER_VALIDATION": 476,
        "OUTER_TEST": 476,
    }

    print()

    print(
        "Role counts:",
        role_counts,
    )

    if role_counts != expected_role_counts:

        raise RuntimeError(
            "Unexpected role counts. "
            f"Expected={expected_role_counts}, "
            f"Found={role_counts}"
        )

    # ========================================================
    # OUTER FOLD VALIDATION
    # ========================================================

    outer_summary = []

    for outer_fold in range(
        1,
        6,
    ):

        outer = assignments[
            assignments[
                "outer_fold"
            ]
            == outer_fold
        ]

        outer_test = outer[
            outer[
                "role"
            ]
            == "OUTER_TEST"
        ]

        outer_test_unique = (
            outer_test[
                "subject_id"
            ]
            .astype(str)
            .nunique()
        )

        # Each outer test subject is repeated across
        # four inner folds.
        outer_test_rows = len(
            outer_test
        )

        if (
            outer_test_rows
            != outer_test_unique * 4
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"outer-test rows do not equal "
                f"unique test subjects × 4."
            )

        if outer_test_unique not in (
            23,
            24,
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"unexpected unique test size "
                f"{outer_test_unique}"
            )

        outer_summary.append(
            {
                "outer_fold":
                    outer_fold,

                "outer_test_subjects":
                    outer_test_unique,

                "outer_test_rows":
                    outer_test_rows,

                "outer_test_controls":
                    int(
                        outer_test[
                            "label"
                        ]
                        .eq(0)
                        .sum()
                        // 4
                    ),

                "outer_test_schizophrenia":
                    int(
                        outer_test[
                            "label"
                        ]
                        .eq(1)
                        .sum()
                        // 4
                    ),
            }
        )

    # ========================================================
    # OUTER TEST PARTITION COVERAGE
    # ========================================================

    outer_test_unique_all = (
        assignments.loc[
            assignments[
                "role"
            ]
            == "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
        .unique()
    )

    if len(
        outer_test_unique_all
    ) != 119:

        raise RuntimeError(
            "Across the five outer folds, "
            "the unique outer-test subjects "
            f"must cover all 119 subjects. "
            f"Found {len(outer_test_unique_all)}."
        )

    # Each subject must be OUTER_TEST in exactly
    # one outer fold.
    outer_test_membership = (
        assignments.loc[
            assignments[
                "role"
            ]
            == "OUTER_TEST"
        ]
        .groupby(
            "subject_id"
        )[
            "outer_fold"
        ]
        .nunique()
    )

    if not (
        outer_test_membership == 1
    ).all():

        raise RuntimeError(
            "A subject is assigned as OUTER_TEST "
            "in more than one outer fold."
        )

    # ========================================================
    # INNER FOLD VALIDATION
    # ========================================================

    for outer_fold in range(
        1,
        6,
    ):

        for inner_fold in range(
            1,
            5,
        ):

            block = assignments[
                (
                    assignments[
                        "outer_fold"
                    ]
                    == outer_fold
                )
                & (
                    assignments[
                        "inner_fold"
                    ]
                    == inner_fold
                )
            ]

            # Must contain every RMS subject once
            # for this outer/inner combination.
            if len(block) != 119:

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    f"expected 119 rows, "
                    f"found {len(block)}"
                )

            if (
                block[
                    "subject_id"
                ]
                .astype(str)
                .nunique()
                != 119
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    f"subject IDs are not unique."
                )

            role_counts_block = (
                block[
                    "role"
                ]
                .value_counts()
                .to_dict()
            )

            if (
                "OUTER_TEST"
                not in role_counts_block
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    f"missing OUTER_TEST role."
                )

            # Determine the actual outer-test subjects.
            outer_test_ids = set(
                assignments.loc[
                    (
                        assignments[
                            "outer_fold"
                        ]
                        == outer_fold
                    )
                    & (
                        assignments[
                            "role"
                        ]
                        == "OUTER_TEST"
                    ),
                    "subject_id",
                ]
                .astype(str)
                .unique()
            )

            block_test_ids = set(
                block.loc[
                    block[
                        "role"
                    ]
                    == "OUTER_TEST",
                    "subject_id",
                ]
                .astype(str)
            )

            if (
                block_test_ids
                != outer_test_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    f"OUTER_TEST subjects changed "
                    f"between inner folds."
                )

            # ------------------------------------------------
            # The inner train + validation subjects must equal
            # the outer training subjects.
            # ------------------------------------------------

            inner_non_test = block[
                block[
                    "role"
                ].isin(
                    [
                        "INNER_TRAIN",
                        "INNER_VALIDATION",
                    ]
                )
            ]

            if (
                len(
                    inner_non_test
                )
                + len(
                    block_test_ids
                )
                != 119
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    f"partition does not cover all subjects."
                )

            inner_train_count = int(
                (
                    block[
                        "role"
                    ]
                    == "INNER_TRAIN"
                ).sum()
            )

            inner_validation_count = int(
                (
                    block[
                        "role"
                    ]
                    == "INNER_VALIDATION"
                ).sum()
            )

            outer_test_count = int(
                (
                    block[
                        "role"
                    ]
                    == "OUTER_TEST"
                ).sum()
            )

            if (
                inner_train_count
                + inner_validation_count
                + outer_test_count
                != 119
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    f"role counts do not sum to 119."
                )

    # ========================================================
    # LABEL VALIDATION
    # ========================================================

    label_counts = (
        assignments[
            [
                "subject_id",
                "label",
            ]
        ]
        .drop_duplicates()
        [
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if label_counts != {
        0: 40,
        1: 79,
    }:

        raise RuntimeError(
            f"Unexpected label counts: "
            f"{label_counts}"
        )

    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    source_values = (
        assignments[
            "acquisition_source"
        ]
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    if source_values != [
        "RMS"
    ]:

        raise RuntimeError(
            "Experiment 06 must contain "
            f"RMS subjects only. Found: "
            f"{source_values}"
        )

    # ========================================================
    # SAVE ASSIGNMENTS
    # ========================================================

    assignments_path = (
        OUTPUT_DIR
        / "nested_cv_assignments.csv"
    )

    assignments.to_csv(
        assignments_path,
        index=False,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        outer_summary
    )

    summary_path = (
        OUTPUT_DIR
        / "nested_cv_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # VALIDATION REPORT
    # ========================================================

    report_lines = [
        "=" * 70,
        "EXPERIMENT 06",
        "RMS-ONLY CNN + AAL",
        "NESTED CROSS-VALIDATION VALIDATION REPORT",
        "=" * 70,
        "",
        "Cohort:",
        "  Total subjects: 119",
        "  Controls: 40",
        "  Schizophrenia: 79",
        "  Acquisition source: RMS only",
        "",
        "Nested CV:",
        "  Outer folds: 5",
        "  Inner folds: 4",
        "",
        "Assignment rows:",
        f"  Expected: {expected_rows}",
        f"  Actual: {len(assignments)}",
        "",
        "Role counts:",
        f"  INNER_TRAIN: "
        f"{role_counts['INNER_TRAIN']}",
        f"  INNER_VALIDATION: "
        f"{role_counts['INNER_VALIDATION']}",
        f"  OUTER_TEST: "
        f"{role_counts['OUTER_TEST']}",
        "",
        "Subject integrity:",
        f"  Unique subjects: "
        f"{assignments['subject_id'].nunique()}",
        "  Rows per subject: 20",
        "",
        "Outer folds:",
    ]

    for row in outer_summary:

        report_lines.append(
            f"  Fold {row['outer_fold']}: "
            f"{row['outer_test_subjects']} unique "
            f"test subjects "
            f"({row['outer_test_controls']} controls, "
            f"{row['outer_test_schizophrenia']} schizophrenia)"
        )

    report_lines.extend(
        [
            "",
            "Validation checks:",
            "  Subject set: PASS",
            "  Subject repetition: PASS",
            "  Outer partition: PASS",
            "  Inner partition: PASS",
            "  Label counts: PASS",
            "  RMS-only source: PASS",
            "",
            "FINAL STATUS: "
            "EXPERIMENT 06 NESTED CV SPLIT PASS",
        ]
    )

    report_path = (
        OUTPUT_DIR
        / "nested_cv_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "NESTED-CV VALIDATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Assignment rows:",
        len(assignments),
    )

    print(
        "Unique subjects:",
        assignments[
            "subject_id"
        ].nunique(),
    )

    print(
        "Rows per subject:",
        int(
            assignments[
                "subject_id"
            ]
            .value_counts()
            .iloc[0]
        ),
    )

    print(
        "Role counts:",
        role_counts,
    )

    print()

    print(
        "Outer fold sizes:"
    )

    for row in outer_summary:

        print(
            f"  Fold {row['outer_fold']}: "
            f"{row['outer_test_subjects']} "
            f"unique test subjects "
            f"("
            f"{row['outer_test_controls']} controls, "
            f"{row['outer_test_schizophrenia']} schizophrenia"
            f")"
        )

    print()

    print(
        "Assignments:",
        assignments_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Report:",
        report_path,
    )

    print()

    print(
        "FINAL STATUS: "
        "EXPERIMENT 06 NESTED CV SPLIT PASS"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()