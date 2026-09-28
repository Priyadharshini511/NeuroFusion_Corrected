from pathlib import Path

import nibabel as nib
import numpy as np


# ==============================================================
# NeuroFusion AI
# External ds004302 sMRI normalization
#
# Uses the SAME normalization rule as the COBRE training pipeline:
#
#   lower = 1st percentile of brain voxels
#   upper = 99th percentile of brain voxels
#   clip to [lower, upper]
#   scale to [0, 1]
#   outside brain mask = 0
# ==============================================================


INPUT_DIR = Path(
    "processed_data/smri/external_test/ds004302/pilot_hdbet"
)

OUTPUT_DIR = Path(
    "processed_data/smri/external_test/ds004302/pilot_normalized"
)

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0

RANGE_TOLERANCE = 1e-6


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("NeuroFusion AI - EXTERNAL ds004302 NORMALIZATION")
print("=" * 80)

print()
print("Input :", INPUT_DIR.resolve())
print("Output:", OUTPUT_DIR.resolve())
print()
print("Normalization:")
print("  Lower percentile:", LOWER_PERCENTILE)
print("  Upper percentile:", UPPER_PERCENTILE)
print("  Output range    : [0, 1]")


# ==============================================================
# FIND SUBJECTS
# ==============================================================

subjects = sorted(
    p.parent.parent
    for p in INPUT_DIR.glob(
        "sub-*/anat/*_T1w_n4_bet.nii.gz"
    )
)

if not subjects:
    raise FileNotFoundError(
        "No external HD-BET brain-extracted images found."
    )


print()
print("Subjects found:", len(subjects))


# ==============================================================
# PROCESS
# ==============================================================

success = 0
failed = 0


for subject_dir in subjects:

    subject_id = subject_dir.name

    anat_dir = subject_dir / "anat"

    image_path = (
        anat_dir /
        f"{subject_id}_T1w_n4_bet.nii.gz"
    )

    mask_path = (
        anat_dir /
        f"{subject_id}_T1w_n4_bet_bet.nii.gz"
    )

    output_subject_dir = (
        OUTPUT_DIR /
        subject_id /
        "anat"
    )

    output_subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_subject_dir /
        f"{subject_id}_T1w_n4_normalized.nii.gz"
    )

    print()
    print("-" * 80)
    print(subject_id)
    print("-" * 80)

    try:

        # ------------------------------------------------------
        # Load images
        # ------------------------------------------------------

        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))

        data = image.get_fdata(
            dtype=np.float32
        )

        mask_data = mask.get_fdata(
            dtype=np.float32
        )

        # ------------------------------------------------------
        # Basic validation
        # ------------------------------------------------------

        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D image, got {data.ndim}D"
            )

        if mask_data.ndim != 3:
            raise ValueError(
                f"Expected 3D mask, got {mask_data.ndim}D"
            )

        if data.shape != mask_data.shape:
            raise ValueError(
                f"Image/mask shape mismatch: "
                f"{data.shape} != {mask_data.shape}"
            )

        if not np.all(
            np.isfinite(data)
        ):
            raise ValueError(
                "Image contains non-finite values."
            )

        if not np.all(
            np.isfinite(mask_data)
        ):
            raise ValueError(
                "Mask contains non-finite values."
            )

        brain_voxels = (
            mask_data > 0.5
        )

        brain_values = data[
            brain_voxels
        ]

        if brain_values.size == 0:
            raise ValueError(
                "Brain mask contains no voxels."
            )

        # ------------------------------------------------------
        # Percentile normalization
        # ------------------------------------------------------

        lower = np.percentile(
            brain_values,
            LOWER_PERCENTILE
        )

        upper = np.percentile(
            brain_values,
            UPPER_PERCENTILE
        )

        if not np.isfinite(lower):
            raise ValueError(
                "Lower percentile is non-finite."
            )

        if not np.isfinite(upper):
            raise ValueError(
                "Upper percentile is non-finite."
            )

        if upper <= lower:
            raise ValueError(
                "Upper percentile <= lower percentile."
            )

        # ------------------------------------------------------
        # Clip
        # ------------------------------------------------------

        brain_clipped = np.clip(
            brain_values,
            lower,
            upper
        )

        # ------------------------------------------------------
        # Scale to [0,1]
        # ------------------------------------------------------

        brain_normalized = (
            (brain_clipped - lower)
            / (upper - lower)
        )

        # ------------------------------------------------------
        # Create output
        # ------------------------------------------------------

        normalized = np.zeros_like(
            data,
            dtype=np.float32
        )

        normalized[
            brain_voxels
        ] = brain_normalized.astype(
            np.float32
        )

        # ------------------------------------------------------
        # Validate output
        # ------------------------------------------------------

        normalized_brain = normalized[
            brain_voxels
        ]

        if not np.all(
            np.isfinite(normalized_brain)
        ):
            raise ValueError(
                "Normalized brain contains "
                "non-finite values."
            )

        minimum = float(
            np.min(normalized_brain)
        )

        maximum = float(
            np.max(normalized_brain)
        )

        if (
            minimum < -RANGE_TOLERANCE
            or
            maximum > 1.0 + RANGE_TOLERANCE
        ):
            raise ValueError(
                "Normalized values outside [0,1]."
            )

        # ------------------------------------------------------
        # Preserve affine/header
        # ------------------------------------------------------

        output_image = nib.Nifti1Image(
            normalized,
            image.affine,
            image.header.copy()
        )

        output_image.set_data_dtype(
            np.float32
        )

        nib.save(
            output_image,
            str(output_path)
        )

        # ------------------------------------------------------
        # Report
        # ------------------------------------------------------

        print(
            f"  Brain voxels : {brain_values.size:,}"
        )

        print(
            f"  P1           : {float(lower):.6f}"
        )

        print(
            f"  P99          : {float(upper):.6f}"
        )

        print(
            f"  Output min   : {minimum:.6f}"
        )

        print(
            f"  Output max   : {maximum:.6f}"
        )

        print(
            f"  Output       : {output_path}"
        )

        print(
            "  STATUS       : PASS"
        )

        success += 1

    except Exception as exc:

        print(
            f"  STATUS       : FAIL"
        )

        print(
            f"  ERROR        : {exc}"
        )

        failed += 1


# ==============================================================
# SUMMARY
# ==============================================================

print()
print("=" * 80)
print("EXTERNAL NORMALIZATION SUMMARY")
print("=" * 80)

print()
print("Subjects found :", len(subjects))
print("Successful      :", success)
print("Failed          :", failed)

if failed != 0:

    raise RuntimeError(
        "External normalization did not complete successfully."
    )

print()
print("STATUS: EXTERNAL NORMALIZATION COMPLETED SUCCESSFULLY")
print()