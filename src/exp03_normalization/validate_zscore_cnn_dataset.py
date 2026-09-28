from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03 — Validate Z-score sMRI CNN Dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp03_normalization"
    / "cnn_dataset"
)

IMAGE_DIR = DATASET_DIR / "images"
LABELS_PATH = DATASET_DIR / "labels.csv"
METADATA_PATH = DATASET_DIR / "metadata.csv"
REPORT_PATH = DATASET_DIR / "validation_report.txt"

EXPECTED_SUBJECTS = 172
EXPECTED_CONTROLS = 93
EXPECTED_CASES = 79

EXPECTED_TENSOR_SHAPE = (1, 95, 117, 99)
EXPECTED_DTYPE = np.float32


def fail(message):
    print(f"ERROR: {message}")
    sys.exit(1)


def main():

    print("=" * 70)
    print("EXPERIMENT 03 — VALIDATE Z-SCORE CNN DATASET")
    print("=" * 70)

    errors = []
    checked_subjects = []

    # --------------------------------------------------------
    # Check required files/directories
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        fail(f"Dataset directory not found: {DATASET_DIR}")

    if not IMAGE_DIR.exists():
        fail(f"Image directory not found: {IMAGE_DIR}")

    if not LABELS_PATH.exists():
        fail(f"labels.csv not found: {LABELS_PATH}")

    if not METADATA_PATH.exists():
        fail(f"metadata.csv not found: {METADATA_PATH}")

    # --------------------------------------------------------
    # Load labels and metadata
    # --------------------------------------------------------

    labels_df = pd.read_csv(LABELS_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)

    required_label_columns = [
        "subject_id",
        "label",
    ]

    required_metadata_columns = [
        "subject_id",
        "label",
        "subject_type",
        "acquisition_source",
        "zscore_nifti",
        "tensor_path",
        "tensor_shape",
        "dtype",
        "brain_mean",
        "brain_std",
        "status",
    ]

    for column in required_label_columns:
        if column not in labels_df.columns:
            errors.append(
                f"labels.csv missing column: {column}"
            )

    for column in required_metadata_columns:
        if column not in metadata_df.columns:
            errors.append(
                f"metadata.csv missing column: {column}"
            )

    if errors:
        for error in errors:
            print(f"  FAIL: {error}")

        fail("CSV structure validation failed.")

    # --------------------------------------------------------
    # Validate row counts
    # --------------------------------------------------------

    print()
    print("CSV validation")

    print(f"  Labels rows   : {len(labels_df)}")
    print(f"  Metadata rows : {len(metadata_df)}")

    if len(labels_df) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} label rows, "
            f"found {len(labels_df)}"
        )

    if len(metadata_df) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} metadata rows, "
            f"found {len(metadata_df)}"
        )

    # --------------------------------------------------------
    # Validate subject uniqueness
    # --------------------------------------------------------

    label_duplicates = labels_df[
        labels_df["subject_id"].duplicated(keep=False)
    ]

    metadata_duplicates = metadata_df[
        metadata_df["subject_id"].duplicated(keep=False)
    ]

    print(
        f"  Label duplicate rows   : {len(label_duplicates)}"
    )

    print(
        f"  Metadata duplicate rows: {len(metadata_duplicates)}"
    )

    if not label_duplicates.empty:
        errors.append("Duplicate subject IDs found in labels.csv")

    if not metadata_duplicates.empty:
        errors.append("Duplicate subject IDs found in metadata.csv")

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    label_counts = labels_df["label"].value_counts().to_dict()

    controls = int(label_counts.get(0, 0))
    cases = int(label_counts.get(1, 0))

    print(f"  Controls              : {controls}")
    print(f"  Schizophrenia         : {cases}")

    if controls != EXPECTED_CONTROLS:
        errors.append(
            f"Expected {EXPECTED_CONTROLS} controls, found {controls}"
        )

    if cases != EXPECTED_CASES:
        errors.append(
            f"Expected {EXPECTED_CASES} schizophrenia subjects, "
            f"found {cases}"
        )

    invalid_labels = labels_df[
        ~labels_df["label"].isin([0, 1])
    ]

    if not invalid_labels.empty:
        errors.append(
            "Labels other than 0/1 detected"
        )

    # --------------------------------------------------------
    # Validate labels vs metadata
    # --------------------------------------------------------

    merged = labels_df.merge(
        metadata_df,
        on="subject_id",
        how="outer",
        suffixes=("_labels", "_metadata"),
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        errors.append(
            "Subject IDs differ between labels.csv and metadata.csv"
        )

    matched = merged[merged["_merge"] == "both"]

    if not matched.empty:

        label_mismatch = matched[
            matched["label_labels"] != matched["label_metadata"]
        ]

        if not label_mismatch.empty:
            errors.append(
                f"Label mismatch between CSV files for "
                f"{len(label_mismatch)} subjects"
            )

    # --------------------------------------------------------
    # Validate source composition
    # --------------------------------------------------------

    source_counts = (
        metadata_df["acquisition_source"]
        .value_counts()
        .to_dict()
    )

    legacy_count = int(source_counts.get("LEGACY", 0))
    rms_count = int(source_counts.get("RMS", 0))

    print()
    print("Acquisition source")
    print(f"  LEGACY: {legacy_count}")
    print(f"  RMS   : {rms_count}")

    if legacy_count != 53:
        errors.append(
            f"Expected 53 LEGACY subjects, found {legacy_count}"
        )

    if rms_count != 119:
        errors.append(
            f"Expected 119 RMS subjects, found {rms_count}"
        )

    # --------------------------------------------------------
    # Validate every tensor
    # --------------------------------------------------------

    print()
    print("Tensor validation")

    for index, row in metadata_df.iterrows():

        subject_id = str(row["subject_id"]).strip()

        tensor_path = Path(str(row["tensor_path"]))

        if not tensor_path.is_absolute():
            tensor_path = PROJECT_ROOT / tensor_path

        if not tensor_path.exists():

            errors.append(
                f"{subject_id}: tensor file missing"
            )

            print(
                f"[{index + 1:03d}/{len(metadata_df):03d}] "
                f"{subject_id} | FAIL | missing"
            )

            continue

        try:

            tensor = np.load(tensor_path)

            # Shape
            if tensor.shape != EXPECTED_TENSOR_SHAPE:
                raise ValueError(
                    f"shape {tensor.shape}, expected "
                    f"{EXPECTED_TENSOR_SHAPE}"
                )

            # Dtype
            if tensor.dtype != EXPECTED_DTYPE:
                raise ValueError(
                    f"dtype {tensor.dtype}, expected "
                    f"{EXPECTED_DTYPE}"
                )

            # Finite values
            if not np.isfinite(tensor).all():
                raise ValueError(
                    "contains NaN or infinite values"
                )

            # Channel dimension
            if tensor.shape[0] != 1:
                raise ValueError(
                    "channel dimension is not 1"
                )

            # Brain/background statistics
            volume = tensor[0]

            nonzero = volume[volume != 0]

            if nonzero.size == 0:
                raise ValueError(
                    "no nonzero brain voxels"
                )

            brain_mean = float(nonzero.mean())
            brain_std = float(nonzero.std())

            if not np.isfinite(brain_mean):
                raise ValueError(
                    "brain mean is not finite"
                )

            if not np.isfinite(brain_std):
                raise ValueError(
                    "brain standard deviation is not finite"
                )

            # The Experiment 03 normalization is brain-aware
            # z-score normalization. Allow a small numerical
            # tolerance around zero mean and unit variance.
            if abs(brain_mean) > 0.10:
                raise ValueError(
                    f"brain mean {brain_mean:.6f} "
                    f"is outside tolerance"
                )

            if brain_std < 0.90 or brain_std > 1.10:
                raise ValueError(
                    f"brain std {brain_std:.6f} "
                    f"is outside tolerance"
                )

            tensor_min = float(tensor.min())
            tensor_max = float(tensor.max())

            checked_subjects.append(
                {
                    "subject_id": subject_id,
                    "label": int(row["label"]),
                    "source": str(row["acquisition_source"]),
                    "shape": str(tuple(tensor.shape)),
                    "dtype": str(tensor.dtype),
                    "min": tensor_min,
                    "max": tensor_max,
                    "brain_mean": brain_mean,
                    "brain_std": brain_std,
                }
            )

            print(
                f"[{index + 1:03d}/{len(metadata_df):03d}] "
                f"{subject_id} | PASS | "
                f"mean={brain_mean:.4f} | "
                f"std={brain_std:.4f}"
            )

        except Exception as exc:

            errors.append(
                f"{subject_id}: {exc}"
            )

            print(
                f"[{index + 1:03d}/{len(metadata_df):03d}] "
                f"{subject_id} | FAIL | {exc}"
            )

    # --------------------------------------------------------
    # Validate tensor count
    # --------------------------------------------------------

    tensor_files = list(IMAGE_DIR.glob("*.npy"))

    print()
    print("Tensor inventory")
    print(f"  .npy files found : {len(tensor_files)}")
    print(f"  tensors checked  : {len(checked_subjects)}")

    if len(tensor_files) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} .npy files, "
            f"found {len(tensor_files)}"
        )

    if len(checked_subjects) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} valid tensors, "
            f"checked {len(checked_subjects)}"
        )

    # --------------------------------------------------------
    # Validate subject ↔ tensor correspondence
    # --------------------------------------------------------

    expected_subjects = set(
        metadata_df["subject_id"].astype(str)
    )

    actual_subjects = {
        path.stem
        for path in tensor_files
    }

    missing_subjects = expected_subjects - actual_subjects
    extra_subjects = actual_subjects - expected_subjects

    if missing_subjects:
        errors.append(
            f"Missing tensor subjects: "
            f"{sorted(missing_subjects)}"
        )

    if extra_subjects:
        errors.append(
            f"Unexpected tensor subjects: "
            f"{sorted(extra_subjects)}"
        )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    checked_df = pd.DataFrame(checked_subjects)

    report_lines = []

    report_lines.append(
        "EXPERIMENT 03 — Z-SCORE CNN DATASET VALIDATION"
    )
    report_lines.append("=" * 65)
    report_lines.append("")

    report_lines.append(
        f"Expected subjects: {EXPECTED_SUBJECTS}"
    )

    report_lines.append(
        f"Valid tensors checked: {len(checked_subjects)}"
    )

    report_lines.append(
        f"Tensor files found: {len(tensor_files)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Controls: {controls}"
    )

    report_lines.append(
        f"Schizophrenia: {cases}"
    )

    report_lines.append("")

    report_lines.append(
        f"LEGACY: {legacy_count}"
    )

    report_lines.append(
        f"RMS: {rms_count}"
    )

    report_lines.append("")

    report_lines.append(
        f"Expected tensor shape: {EXPECTED_TENSOR_SHAPE}"
    )

    report_lines.append(
        "Expected dtype: float32"
    )

    report_lines.append("")

    if not checked_df.empty:

        report_lines.append(
            f"Brain mean range: "
            f"{checked_df['brain_mean'].min():.8f} to "
            f"{checked_df['brain_mean'].max():.8f}"
        )

        report_lines.append(
            f"Brain std range: "
            f"{checked_df['brain_std'].min():.8f} to "
            f"{checked_df['brain_std'].max():.8f}"
        )

        report_lines.append(
            f"Tensor min range: "
            f"{checked_df['min'].min():.8f} to "
            f"{checked_df['min'].max():.8f}"
        )

        report_lines.append(
            f"Tensor max range: "
            f"{checked_df['max'].min():.8f} to "
            f"{checked_df['max'].max():.8f}"
        )

    report_lines.append("")

    report_lines.append(
        f"Total validation errors: {len(errors)}"
    )

    if errors:

        report_lines.append("")
        report_lines.append("ERRORS:")

        for error in errors:
            report_lines.append(
                f"- {error}"
            )

        final_status = (
            "FINAL STATUS: EXPERIMENT 03 CNN DATASET VALIDATION FAIL"
        )

    else:

        final_status = (
            "FINAL STATUS: EXPERIMENT 03 CNN DATASET VALIDATION PASS"
        )

    report_lines.append("")
    report_lines.append(final_status)

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT 03 VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Subjects expected : {EXPECTED_SUBJECTS}")
    print(f"Tensors checked   : {len(checked_subjects)}")
    print(f"Tensor files      : {len(tensor_files)}")
    print(f"Controls          : {controls}")
    print(f"Schizophrenia     : {cases}")
    print(f"LEGACY            : {legacy_count}")
    print(f"RMS               : {rms_count}")
    print(f"Errors            : {len(errors)}")
    print()

    if errors:
        print(
            "FINAL STATUS: EXPERIMENT 03 CNN DATASET "
            "VALIDATION FAIL"
        )
        print()
        print("First validation errors:")

        for error in errors[:10]:
            print(f"  - {error}")

        sys.exit(1)

    print(
        "FINAL STATUS: EXPERIMENT 03 CNN DATASET "
        "VALIDATION PASS"
    )

    print()
    print(f"Validation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()