
"""
Create model-ready 3D CNN tensors for the NeuroFusion AI sMRI branch.

Input
-----
processed_data/smri/dataset/smri_subject_manifest_labeled.csv

For every labeled subject:
    MNI NIfTI
        -> float32 NumPy array
        -> channel dimension
        -> (1, 99, 117, 95)
        -> individual .npy tensor

Outputs
-------
processed_data/smri/cnn_dataset/
    images/
        AXXXXXXXX.npy
        ...
    labels.csv
    metadata.csv
    smri_cnn_dataset_validation_report.txt

Important
---------
- No MRI preprocessing is performed here.
- Original NIfTI files are never modified.
- Subject IDs remain the primary identifiers.
- One tensor is created per subject.
- This script does NOT perform train/validation/test splitting.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

MANIFEST = (
    ROOT
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

OUTPUT_DIR = ROOT / "cnn_dataset"
IMAGE_DIR = OUTPUT_DIR / "images"

LABELS_FILE = OUTPUT_DIR / "labels.csv"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

REPORT_FILE = (
    OUTPUT_DIR
    / "smri_cnn_dataset_validation_report.txt"
)

MNI_TEMPLATE = (
    ROOT
    / "mni"
    / "mni152_template_2mm.nii.gz"
)


# ============================================================
# EXPECTED DATASET PROPERTIES
# ============================================================

EXPECTED_SUBJECTS = 172
EXPECTED_CONTROL = 93
EXPECTED_SCHIZOPHRENIA = 79

EXPECTED_SIZE = (99, 117, 95)
EXPECTED_SPACING = (2.0, 2.0, 2.0)

EXPECTED_TENSOR_SHAPE = (
    1,
    95,
    117,
    99,
)

GEOMETRY_ATOL = 1e-6

INTENSITY_MIN = 0.0
INTENSITY_MAX = 1.0
INTENSITY_ATOL = 1e-5

SUBJECT_PATTERN = re.compile(
    r"^A\d{8}$"
)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_geometry(
    image: sitk.Image,
    template: sitk.Image,
) -> tuple[bool, str]:
    """Validate MRI geometry against the common MNI template."""

    if image.GetDimension() != 3:
        return False, "not_3d"

    if image.GetSize() != template.GetSize():
        return False, f"size:{image.GetSize()}"

    if not np.allclose(
        image.GetSpacing(),
        template.GetSpacing(),
        atol=GEOMETRY_ATOL,
        rtol=0.0,
    ):
        return False, "spacing"

    if not np.allclose(
        image.GetOrigin(),
        template.GetOrigin(),
        atol=GEOMETRY_ATOL,
        rtol=0.0,
    ):
        return False, "origin"

    if not np.allclose(
        image.GetDirection(),
        template.GetDirection(),
        atol=GEOMETRY_ATOL,
        rtol=0.0,
    ):
        return False, "direction"

    return True, "valid"


def validate_tensor(
    tensor: np.ndarray,
) -> tuple[bool, str]:
    """Validate one saved CNN tensor."""

    if tensor.shape != EXPECTED_TENSOR_SHAPE:
        return (
            False,
            f"shape:{tensor.shape}",
        )

    if tensor.dtype != np.float32:
        return (
            False,
            f"dtype:{tensor.dtype}",
        )

    if not np.isfinite(tensor).all():
        return False, "nonfinite"

    minimum = float(np.min(tensor))
    maximum = float(np.max(tensor))

    if (
        minimum < INTENSITY_MIN - INTENSITY_ATOL
        or maximum > INTENSITY_MAX + INTENSITY_ATOL
    ):
        return (
            False,
            f"range:{minimum:.6f}..{maximum:.6f}",
        )

    return True, "valid"


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print("NeuroFusion AI - CREATE sMRI 3D CNN DATASET")
    print("=" * 72)
    print()

    errors: list[str] = []

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    manifest = pd.read_csv(MANIFEST)

    print(
        f"Manifest subjects : {len(manifest)}"
    )

    if len(manifest) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} subjects; "
            f"found {len(manifest)}"
        )

    required_columns = {
        "subject_id",
        "mni_volume",
        "acquisition_source",
        "label",
        "mni_valid",
        "aal_valid",
    }

    missing = required_columns.difference(
        manifest.columns
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Subject uniqueness
    # --------------------------------------------------------

    if manifest["subject_id"].duplicated().any():
        duplicates = (
            manifest.loc[
                manifest["subject_id"].duplicated(
                    keep=False
                ),
                "subject_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        errors.append(
            "Duplicate subject IDs: "
            + ", ".join(sorted(duplicates))
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = pd.to_numeric(
        manifest["label"],
        errors="coerce",
    )

    if labels.isna().any():
        errors.append(
            "Missing/non-numeric labels detected."
        )

    if not set(labels.dropna().unique()).issubset(
        {0, 1}
    ):
        errors.append(
            "Non-binary labels detected."
        )

    control_count = int(
        (labels == 0).sum()
    )

    schizophrenia_count = int(
        (labels == 1).sum()
    )

    print(
        f"Control subjects  : {control_count}"
    )

    print(
        f"Schizophrenia     : {schizophrenia_count}"
    )

    if control_count != EXPECTED_CONTROL:
        errors.append(
            f"Expected {EXPECTED_CONTROL} controls; "
            f"found {control_count}"
        )

    if schizophrenia_count != EXPECTED_SCHIZOPHRENIA:
        errors.append(
            f"Expected {EXPECTED_SCHIZOPHRENIA} "
            f"schizophrenia-spectrum subjects; "
            f"found {schizophrenia_count}"
        )

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    if not MNI_TEMPLATE.exists():
        raise FileNotFoundError(
            f"MNI template not found: {MNI_TEMPLATE}"
        )

    template = sitk.ReadImage(
        str(MNI_TEMPLATE),
        sitk.sitkFloat32,
    )

    if template.GetSize() != EXPECTED_SIZE:
        errors.append(
            f"Unexpected template size: "
            f"{template.GetSize()}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    metadata_rows = []

    for index, row in manifest.iterrows():

        subject_id = str(
            row["subject_id"]
        ).strip()

        label = int(row["label"])

        if not SUBJECT_PATTERN.fullmatch(
            subject_id
        ):
            errors.append(
                f"{subject_id}: invalid subject ID"
            )
            continue

        source = str(
            row["acquisition_source"]
        ).strip()

        nifti_path = Path(
            str(row["mni_volume"])
        )

        output_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        print(
            f"[{index + 1:03d}/{len(manifest):03d}] "
            f"{subject_id} | "
            f"{source:<6} | "
            f"label={label}",
            end="",
        )

        # ----------------------------------------------------
        # Load MRI
        # ----------------------------------------------------

        if not nifti_path.exists():
            print(" | FAIL")
            errors.append(
                f"{subject_id}: MRI file missing"
            )
            continue

        try:
            image = sitk.ReadImage(
                str(nifti_path),
                sitk.sitkFloat32,
            )
        except Exception as exc:
            print(" | FAIL")
            errors.append(
                f"{subject_id}: MRI read error "
                f"{type(exc).__name__}"
            )
            continue

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        geometry_valid, geometry_message = (
            validate_geometry(
                image,
                template,
            )
        )

        if not geometry_valid:
            print(" | FAIL")
            errors.append(
                f"{subject_id}: geometry invalid "
                f"{geometry_message}"
            )
            continue

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        data = sitk.GetArrayFromImage(
            image
        ).astype(
            np.float32,
            copy=False,
        )

        # SimpleITK -> NumPy order is:
        # (Z, Y, X) = (95, 117, 99)
        #
        # Add channel dimension:
        # (95, 117, 99)
        #     ->
        # (1, 95, 117, 99)
        #
        # This is the PyTorch Conv3D convention:
        # (C, D, H, W)

        expected_numpy_shape = (
            EXPECTED_SIZE[2],
            EXPECTED_SIZE[1],
            EXPECTED_SIZE[0],
        )

        if data.shape != expected_numpy_shape:
            print(" | FAIL")
            errors.append(
                f"{subject_id}: unexpected NumPy "
                f"shape {data.shape}; "
                f"expected {expected_numpy_shape}"
            )
            continue

        if not np.isfinite(data).all():
            print(" | FAIL")
            errors.append(
                f"{subject_id}: nonfinite voxel values"
            )
            continue

        minimum = float(
            np.min(data)
        )

        maximum = float(
            np.max(data)
        )

        mean = float(
            np.mean(data)
        )

        std = float(
            np.std(data)
        )

        if (
            minimum < INTENSITY_MIN
            - INTENSITY_ATOL
            or maximum > INTENSITY_MAX
            + INTENSITY_ATOL
        ):
            print(" | FAIL")
            errors.append(
                f"{subject_id}: intensity range "
                f"{minimum:.6f}..{maximum:.6f}"
            )
            continue

        # ----------------------------------------------------
        # Add channel dimension
        # ----------------------------------------------------

        tensor = np.expand_dims(
            data,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        if tensor.shape != EXPECTED_TENSOR_SHAPE:
            print(" | FAIL")
            errors.append(
                f"{subject_id}: tensor shape "
                f"{tensor.shape}"
            )
            continue

        # ----------------------------------------------------
        # Save tensor
        # ----------------------------------------------------

        np.save(
            output_path,
            tensor,
        )

        # ----------------------------------------------------
        # Reload and validate saved tensor
        # ----------------------------------------------------

        try:
            saved_tensor = np.load(
                output_path,
                allow_pickle=False,
            )
        except Exception as exc:
            print(" | FAIL")
            errors.append(
                f"{subject_id}: saved tensor could "
                f"not be reloaded: "
                f"{type(exc).__name__}"
            )
            continue

        tensor_valid, tensor_message = (
            validate_tensor(saved_tensor)
        )

        if not tensor_valid:
            print(" | FAIL")
            errors.append(
                f"{subject_id}: saved tensor invalid "
                f"{tensor_message}"
            )
            continue

        metadata_rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "acquisition_source": source,
                "mni_volume": str(
                    nifti_path
                ),
                "tensor_path": str(
                    output_path
                ),
                "tensor_shape": str(
                    saved_tensor.shape
                ),
                "dtype": str(
                    saved_tensor.dtype
                ),
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "std": std,
            }
        )

        print(" | PASS")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        metadata_rows
    )

    metadata_columns = [
        "subject_id",
        "label",
        "acquisition_source",
        "mni_volume",
        "tensor_path",
        "tensor_shape",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
    ]

    metadata = metadata[
        metadata_columns
    ].sort_values(
        "subject_id"
    ).reset_index(
        drop=True
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False,
        encoding="utf-8",
    )

    labels_output = metadata[
        [
            "subject_id",
            "label",
        ]
    ].copy()

    labels_output.to_csv(
        LABELS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final file validation
    # --------------------------------------------------------

    tensor_count = len(metadata)

    actual_tensor_files = sorted(
        IMAGE_DIR.glob("*.npy")
    )

    expected_tensor_names = {
        f"{sid}.npy"
        for sid in manifest[
            "subject_id"
        ].astype(str)
    }

    actual_tensor_names = {
        path.name
        for path in actual_tensor_files
    }

    unexpected_files = (
        actual_tensor_names
        - expected_tensor_names
    )

    missing_files = (
        expected_tensor_names
        - actual_tensor_names
    )

    if unexpected_files:
        errors.append(
            "Unexpected tensor files: "
            + ", ".join(
                sorted(unexpected_files)
            )
        )

    if missing_files:
        errors.append(
            "Missing tensor files: "
            + ", ".join(
                sorted(missing_files)
            )
        )

    if tensor_count != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} "
            f"metadata rows; found {tensor_count}"
        )

    # --------------------------------------------------------
    # Validate output labels
    # --------------------------------------------------------

    if not metadata.empty:

        metadata_labels = (
            metadata["label"]
            .value_counts()
            .to_dict()
        )

        metadata_control = int(
            metadata_labels.get(0, 0)
        )

        metadata_schizophrenia = int(
            metadata_labels.get(1, 0)
        )

        if metadata_control != EXPECTED_CONTROL:
            errors.append(
                "Tensor metadata control count "
                f"expected {EXPECTED_CONTROL}, "
                f"found {metadata_control}"
            )

        if metadata_schizophrenia != EXPECTED_SCHIZOPHRENIA:
            errors.append(
                "Tensor metadata schizophrenia count "
                f"expected {EXPECTED_SCHIZOPHRENIA}, "
                f"found {metadata_schizophrenia}"
            )

    # --------------------------------------------------------
    # Report statistics
    # --------------------------------------------------------

    rms_count = int(
        (
            metadata["acquisition_source"] == "RMS"
        ).sum()
    ) if not metadata.empty else 0

    legacy_count = int(
        (
            metadata["acquisition_source"] == "LEGACY"
        ).sum()
    ) if not metadata.empty else 0

    if not metadata.empty:
        tensor_control_count = int(
            (metadata["label"] == 0).sum()
        )

        tensor_schizophrenia_count = int(
            (metadata["label"] == 1).sum()
        )
    else:
        tensor_control_count = 0
        tensor_schizophrenia_count = 0

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = [
        "NeuroFusion AI - sMRI 3D CNN Dataset",
        "Creation and Validation Report",
        "=" * 72,
        "",
        f"Input manifest           : {MANIFEST}",
        f"Expected subjects       : {EXPECTED_SUBJECTS}",
        f"Successfully exported   : {tensor_count}",
        f"Tensor files found      : {len(actual_tensor_files)}",
        "",
        "Class distribution:",
        f"  Control               : {tensor_control_count}",
        f"  Schizophrenia-spectrum: {tensor_schizophrenia_count}",
        "",
        "Approved tensor shape:",
        f"  {EXPECTED_TENSOR_SHAPE}",
        "",
        "MRI spatial geometry:",
        f"  Size                  : {EXPECTED_SIZE}",
        f"  Spacing               : {EXPECTED_SPACING}",
        "",
        "Intensity convention:",
        "  Range                 : [0, 1]",
        "  Data type             : float32",
        "",
        "Selected acquisition:",
        f"  RMS                   : {rms_count}",
        f"  LEGACY                : {legacy_count}",
        "",
    ]

    if not metadata.empty:
        report_lines.extend(
            [
                "Tensor intensity diagnostics:",
                f"  Minimum               : "
                f"{metadata['min'].min():.6f}",
                f"  Maximum               : "
                f"{metadata['max'].max():.6f}",
                f"  Mean range            : "
                f"{metadata['mean'].min():.6f} .. "
                f"{metadata['mean'].max():.6f}",
                f"  Std range             : "
                f"{metadata['std'].min():.6f} .. "
                f"{metadata['std'].max():.6f}",
                "",
            ]
        )

    report_lines.append(
        "Validation errors:"
    )

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
            f"Tensor directory         : {IMAGE_DIR}",
            f"Labels file              : {LABELS_FILE}",
            f"Metadata file            : {METADATA_FILE}",
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
    print("DATASET CREATION SUMMARY")
    print("=" * 72)

    print(
        f"Expected subjects : {EXPECTED_SUBJECTS}"
    )

    print(
        f"Exported tensors  : {tensor_count}"
    )

    print(
        f"Tensor files      : {len(actual_tensor_files)}"
    )

    print(
        f"Control tensors   : {tensor_control_count}"
    )

    print(
        f"Schizophrenia     : {tensor_schizophrenia_count}"
    )

    print(
        f"RMS tensors       : {rms_count}"
    )

    print(
        f"Legacy tensors    : {legacy_count}"
    )

    print()
    print(
        f"Tensor shape      : {EXPECTED_TENSOR_SHAPE}"
    )

    print(
        "Tensor dtype      : float32"
    )

    print()
    print(
        f"Labels file       : {LABELS_FILE}"
    )

    print(
        f"Metadata file     : {METADATA_FILE}"
    )

    print(
        f"Validation report : {REPORT_FILE}"
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