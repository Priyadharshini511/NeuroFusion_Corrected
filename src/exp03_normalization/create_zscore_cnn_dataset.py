from pathlib import Path
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# Experiment 03 — Z-score Normalized sMRI CNN Dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

ZSCORE_IMAGE_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp03_normalization"
    / "zscore_images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp03_normalization"
    / "cnn_dataset"
)

IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

LABELS_PATH = OUTPUT_DIR / "labels.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
VALIDATION_REPORT_PATH = OUTPUT_DIR / "validation_report.txt"


EXPECTED_SUBJECTS = 172
EXPECTED_NIFTI_SHAPE = (99, 117, 95)
EXPECTED_TENSOR_SHAPE = (1, 95, 117, 99)

EXPECTED_ROI_COUNT = 116


def fail(message):
    print(f"ERROR: {message}")
    sys.exit(1)


def load_manifest():
    if not MANIFEST_PATH.exists():
        fail(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "subject_id",
        "mni_volume",
        "aal_csv",
        "acquisition_source",
        "subject_type",
        "label",
        "label_status",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        fail(f"Manifest missing required columns: {missing}")

    return df


def validate_manifest(df):
    errors = []

    if len(df) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} manifest rows, found {len(df)}"
        )

    if df["subject_id"].duplicated().any():
        duplicates = df.loc[
            df["subject_id"].duplicated(keep=False),
            "subject_id"
        ].tolist()

        errors.append(f"Duplicate subjects: {duplicates}")

    label_counts = df["label"].value_counts().to_dict()

    if label_counts.get(0, 0) != 93:
        errors.append(
            f"Expected 93 controls, found {label_counts.get(0, 0)}"
        )

    if label_counts.get(1, 0) != 79:
        errors.append(
            f"Expected 79 schizophrenia subjects, found {label_counts.get(1, 0)}"
        )

    if df["mni_volume"].isna().any():
        errors.append("Some subjects have missing mni_volume paths")

    if errors:
        for error in errors:
            print(f"  FAIL: {error}")

        fail("Manifest validation failed.")

    print("Manifest validation: PASS")
    print(f"  Subjects: {len(df)}")
    print(f"  Controls: {label_counts.get(0, 0)}")
    print(f"  Schizophrenia: {label_counts.get(1, 0)}")


def read_nifti_as_tensor(path):
    image = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(image).astype(np.float32)

    # SimpleITK returns arrays as:
    # (Z, Y, X)
    #
    # Expected NIfTI geometry:
    # (X, Y, Z) = (99, 117, 95)
    #
    # Therefore:
    # (Z, Y, X) = (95, 117, 99)
    if tuple(array.shape) != EXPECTED_TENSOR_SHAPE[1:]:
        raise ValueError(
            f"Unexpected voxel-array shape {array.shape}; "
            f"expected {EXPECTED_TENSOR_SHAPE[1:]}"
        )

    if not np.isfinite(array).all():
        raise ValueError("Volume contains NaN or infinite values")

    # Add channel dimension for PyTorch 3D CNN:
    # (95,117,99) -> (1,95,117,99)
    tensor = array[np.newaxis, ...]

    if tensor.shape != EXPECTED_TENSOR_SHAPE:
        raise ValueError(
            f"Unexpected tensor shape {tensor.shape}; "
            f"expected {EXPECTED_TENSOR_SHAPE}"
        )

    return tensor


def validate_zscore_tensor(tensor):
    if tensor.dtype != np.float32:
        raise ValueError(
            f"Expected float32, found {tensor.dtype}"
        )

    if tensor.shape != EXPECTED_TENSOR_SHAPE:
        raise ValueError(
            f"Expected tensor shape {EXPECTED_TENSOR_SHAPE}, "
            f"found {tensor.shape}"
        )

    if not np.isfinite(tensor).all():
        raise ValueError("Tensor contains NaN or infinite values")

    # Background is expected to remain exactly zero.
    background = tensor == 0

    if not background.any():
        raise ValueError("No zero-valued background voxels detected")

    nonzero = tensor[~background]

    if len(nonzero) == 0:
        raise ValueError("No nonzero brain voxels detected")

    # Z-score normalized brain values should be centered
    # approximately around zero with approximately unit variance.
    mean_value = float(nonzero.mean())
    std_value = float(nonzero.std())

    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        raise ValueError("Invalid brain statistics")

    if std_value < 0.5 or std_value > 1.5:
        raise ValueError(
            f"Unexpected z-score standard deviation: {std_value:.6f}"
        )

    if abs(mean_value) > 0.10:
        raise ValueError(
            f"Unexpected z-score brain mean: {mean_value:.6f}"
        )

    return mean_value, std_value


def main():
    print("=" * 70)
    print("EXPERIMENT 03 — CREATE Z-SCORE CNN DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load and validate manifest
    # --------------------------------------------------------
    df = load_manifest()
    validate_manifest(df)

    # --------------------------------------------------------
    # Create isolated output directories
    # --------------------------------------------------------
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Process subjects
    # --------------------------------------------------------
    records = []
    failures = []

    print()
    print("Processing z-score normalized volumes...")
    print()

    for index, row in df.iterrows():

        subject_id = str(row["subject_id"]).strip()
        label = int(row["label"])
        source = str(row["acquisition_source"]).strip()
        subject_type = str(row["subject_type"]).strip()

        zscore_path = ZSCORE_IMAGE_DIR / f"{subject_id}_zscore_mni.nii.gz"
        output_path = IMAGE_OUTPUT_DIR / f"{subject_id}.npy"

        try:
            if not zscore_path.exists():
                raise FileNotFoundError(
                    f"Z-score volume not found: {zscore_path}"
                )

            # ------------------------------------------------
            # Preserve existing valid tensor
            # ------------------------------------------------
            if output_path.exists():

                tensor = np.load(output_path)

                mean_value, std_value = validate_zscore_tensor(tensor)

                status = "existing_valid"

            else:

                tensor = read_nifti_as_tensor(zscore_path)

                mean_value, std_value = validate_zscore_tensor(tensor)

                np.save(output_path, tensor)

                status = "created"

            records.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "subject_type": subject_type,
                    "acquisition_source": source,
                    "zscore_nifti": str(zscore_path),
                    "tensor_path": str(output_path),
                    "tensor_shape": str(tuple(tensor.shape)),
                    "dtype": str(tensor.dtype),
                    "brain_mean": mean_value,
                    "brain_std": std_value,
                    "status": status,
                }
            )

            print(
                f"[{index + 1:03d}/{len(df):03d}] "
                f"{subject_id} | label={label} | "
                f"source={source} | {status}"
            )

        except Exception as exc:

            failures.append(
                {
                    "subject_id": subject_id,
                    "error": str(exc),
                }
            )

            print(
                f"[{index + 1:03d}/{len(df):03d}] "
                f"{subject_id} | FAIL | {exc}"
            )

    # --------------------------------------------------------
    # Save labels and metadata
    # --------------------------------------------------------
    records_df = pd.DataFrame(records)

    labels_df = records_df[
        [
            "subject_id",
            "label",
        ]
    ].copy()

    metadata_df = records_df[
        [
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
    ].copy()

    labels_df.to_csv(LABELS_PATH, index=False)
    metadata_df.to_csv(METADATA_PATH, index=False)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    validation_lines = []

    validation_lines.append(
        "EXPERIMENT 03 — Z-SCORE CNN DATASET VALIDATION"
    )
    validation_lines.append("=" * 60)
    validation_lines.append("")

    validation_lines.append(
        f"Manifest subjects: {len(df)}"
    )
    validation_lines.append(
        f"Created/validated tensors: {len(records)}"
    )
    validation_lines.append(
        f"Failures: {len(failures)}"
    )
    validation_lines.append("")

    if len(records) > 0:

        final_df = pd.DataFrame(records)

        duplicate_count = int(
            final_df["subject_id"].duplicated().sum()
        )

        class_counts = final_df["label"].value_counts().to_dict()

        source_counts = (
            final_df["acquisition_source"]
            .value_counts()
            .to_dict()
        )

        validation_lines.append(
            f"Duplicate subject rows: {duplicate_count}"
        )
        validation_lines.append(
            f"Controls: {class_counts.get(0, 0)}"
        )
        validation_lines.append(
            f"Schizophrenia: {class_counts.get(1, 0)}"
        )
        validation_lines.append(
            f"LEGACY source: {source_counts.get('LEGACY', 0)}"
        )
        validation_lines.append(
            f"RMS source: {source_counts.get('RMS', 0)}"
        )
        validation_lines.append("")

    validation_pass = (
        len(records) == EXPECTED_SUBJECTS
        and len(failures) == 0
        and len(records_df) == EXPECTED_SUBJECTS
        and not records_df["subject_id"].duplicated().any()
        and set(records_df["label"].unique()) == {0, 1}
    )

    if validation_pass:
        validation_lines.append(
            "FINAL STATUS: EXPERIMENT 03 CNN DATASET PASS"
        )
    else:
        validation_lines.append(
            "FINAL STATUS: EXPERIMENT 03 CNN DATASET FAIL"
        )

    if failures:
        validation_lines.append("")
        validation_lines.append("FAILURES:")
        for failure in failures:
            validation_lines.append(
                f"{failure['subject_id']}: {failure['error']}"
            )

    VALIDATION_REPORT_PATH.write_text(
        "\n".join(validation_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("EXPERIMENT 03 CNN DATASET SUMMARY")
    print("=" * 70)

    print(f"Manifest subjects : {len(df)}")
    print(f"Tensors processed : {len(records)}")
    print(f"Failures          : {len(failures)}")

    if len(records) > 0:
        print(
            f"Controls          : "
            f"{int((records_df['label'] == 0).sum())}"
        )

        print(
            f"Schizophrenia     : "
            f"{int((records_df['label'] == 1).sum())}"
        )

        print(
            f"Tensor shape      : "
            f"{EXPECTED_TENSOR_SHAPE}"
        )

        print("Data type         : float32")

    print()

    if validation_pass:
        print("FINAL STATUS: EXPERIMENT 03 CNN DATASET PASS")
    else:
        print("FINAL STATUS: EXPERIMENT 03 CNN DATASET FAIL")

    print()
    print(f"Images directory  : {IMAGE_OUTPUT_DIR}")
    print(f"Labels file       : {LABELS_PATH}")
    print(f"Metadata file     : {METADATA_PATH}")
    print(f"Validation report : {VALIDATION_REPORT_PATH}")

    if not validation_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()