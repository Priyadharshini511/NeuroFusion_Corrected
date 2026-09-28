"""
======================================================================
OPENNEURO EXTERNAL TEST — HD-BET BRAIN EXTRACTION
======================================================================

Purpose
-------
Apply the SAME HD-BET operation used by the NeuroFusion sMRI
training/preprocessing pipeline to the 3-subject OpenNeuro pilot.

This is NOT a new preprocessing method.

Only the input/output paths and external subject discovery are adapted.

Input:
    pilot_n4/
        sub-01/anat/sub-01_T1w_n4.nii.gz
        sub-26/anat/sub-26_T1w_n4.nii.gz
        sub-55/anat/sub-55_T1w_n4.nii.gz

Output:
    pilot_hdbet/
        sub-01/anat/sub-01_T1w_n4_bet.nii.gz
        sub-01/anat/sub-01_T1w_n4_bet_bet.nii.gz
        ...

HD-BET configuration:
    - CUDA
    - --save_bet_mask
    - --verbose

No changes are made to the original N4 files.
======================================================================
"""

from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np


# ======================================================================
# PATHS
# ======================================================================

INPUT_ROOT = Path(
    "processed_data/smri/external_test/ds004302/pilot_n4"
)

OUTPUT_ROOT = Path(
    "processed_data/smri/external_test/ds004302/pilot_hdbet"
)


# ======================================================================
# EXPECTED SUBJECTS
# ======================================================================

SUBJECTS = [
    "sub-01",
    "sub-26",
    "sub-55",
]


# ======================================================================
# SETTINGS
# ======================================================================

EXPECTED_VOXEL_SIZE = np.array(
    [1.0, 1.0, 1.0],
    dtype=float,
)

VOXEL_TOLERANCE = 0.01


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("OPENNEURO EXTERNAL TEST — HD-BET BRAIN EXTRACTION")
print("=" * 80)

print()
print("N4 input root:")
print(f"  {INPUT_ROOT.resolve()}")

print()
print("HD-BET output root:")
print(f"  {OUTPUT_ROOT.resolve()}")

print()
print("Subjects:")
for subject in SUBJECTS:
    print(f"  {subject}")


# ======================================================================
# DIRECTORY CHECK
# ======================================================================

if not INPUT_ROOT.is_dir():
    raise FileNotFoundError(
        "\nExternal N4 input directory does not exist:\n"
        f"{INPUT_ROOT.resolve()}\n\n"
        "Run external_n4.py first."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# HELPERS
# ======================================================================

def get_paths(subject):
    """
    Build external N4 input, HD-BET brain output,
    and HD-BET mask paths.
    """

    input_file = (
        INPUT_ROOT
        / subject
        / "anat"
        / f"{subject}_T1w_n4.nii.gz"
    )

    subject_output_dir = (
        OUTPUT_ROOT
        / subject
        / "anat"
    )

    subject_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        subject_output_dir
        / f"{subject}_T1w_n4_bet.nii.gz"
    )

    mask_file = (
        subject_output_dir
        / f"{subject}_T1w_n4_bet_bet.nii.gz"
    )

    return (
        input_file,
        output_file,
        mask_file,
    )


def validate_nifti(
    path,
    expected_reference=None,
):
    """
    Validate an image against the N4 reference.
    """

    try:

        image = nib.load(
            str(path)
        )

        data = image.get_fdata(
            dtype=np.float32
        )

        if len(image.shape) != 3:
            return (
                False,
                f"not 3D: {image.shape}",
            )

        if not np.all(
            np.isfinite(data)
        ):
            return (
                False,
                "contains NaN/Inf",
            )

        if not np.all(
            np.isfinite(
                image.affine
            )
        ):
            return (
                False,
                "affine contains NaN/Inf",
            )

        voxel_size = np.asarray(
            image.header.get_zooms()[:3],
            dtype=float,
        )

        if not np.allclose(
            voxel_size,
            EXPECTED_VOXEL_SIZE,
            atol=VOXEL_TOLERANCE,
        ):
            return (
                False,
                f"unexpected voxel size: {voxel_size}",
            )

        if expected_reference is not None:

            if image.shape != expected_reference.shape:
                return (
                    False,
                    "shape does not match N4 input",
                )

            if not np.allclose(
                image.affine,
                expected_reference.affine,
                atol=0.01,
            ):
                return (
                    False,
                    "affine does not match N4 input",
                )

        if data.size == 0:
            return (
                False,
                "empty image",
            )

        if not np.any(
            data != 0
        ):
            return (
                False,
                "image contains only zeros",
            )

        return (
            True,
            "PASS",
        )

    except Exception as exc:

        return (
            False,
            str(exc),
        )


def validate_mask(
    path,
    reference,
):
    """
    Validate HD-BET mask.
    """

    try:

        mask = nib.load(
            str(path)
        )

        data = mask.get_fdata(
            dtype=np.float32
        )

        if mask.shape != reference.shape:
            return (
                False,
                "mask shape does not match input",
            )

        if not np.all(
            np.isfinite(data)
        ):
            return (
                False,
                "mask contains NaN/Inf",
            )

        if not np.allclose(
            mask.affine,
            reference.affine,
            atol=0.01,
        ):
            return (
                False,
                "mask affine does not match input",
            )

        unique_values = np.unique(
            data
        )

        if unique_values.size == 0:
            return (
                False,
                "empty mask",
            )

        if not np.any(
            data > 0
        ):
            return (
                False,
                "mask contains no foreground",
            )

        return (
            True,
            "PASS",
        )

    except Exception as exc:

        return (
            False,
            str(exc),
        )


# ======================================================================
# TRACKERS
# ======================================================================

processed = []
skipped = []
failed = []


# ======================================================================
# PROCESS SUBJECTS
# ======================================================================

for subject in SUBJECTS:

    print("\n" + "-" * 80)
    print(f"SUBJECT: {subject}")
    print("-" * 80)

    (
        input_file,
        output_file,
        mask_file,
    ) = get_paths(
        subject
    )

    print()
    print("Input:")
    print(f"  {input_file}")

    print()
    print("Brain output:")
    print(f"  {output_file}")

    print()
    print("Mask output:")
    print(f"  {mask_file}")

    # ------------------------------------------------------------------
    # Input check
    # ------------------------------------------------------------------

    if not input_file.is_file():

        print()
        print(
            "ERROR: N4 input missing."
        )

        failed.append(
            (
                subject,
                "missing N4 input",
            )
        )

        continue

    try:

        input_img = nib.load(
            str(input_file)
        )

    except Exception as exc:

        print()
        print(
            "ERROR: Could not read N4 input:",
            exc,
        )

        failed.append(
            (
                subject,
                f"unreadable N4 input: {exc}",
            )
        )

        continue

    input_valid, input_reason = (
        validate_nifti(
            input_file
        )
    )

    if not input_valid:

        print()
        print(
            "ERROR: N4 input validation failed:"
        )

        print(
            f"  {input_reason}"
        )

        failed.append(
            (
                subject,
                f"invalid N4 input: {input_reason}",
            )
        )

        continue

    print()
    print(
        "N4 input validation: PASS"
    )

    print(
        "  Shape:",
        input_img.shape,
    )

    print(
        "  Orientation:",
        nib.aff2axcodes(
            input_img.affine
        ),
    )

    # ------------------------------------------------------------------
    # Existing output check
    # ------------------------------------------------------------------

    if (
        output_file.is_file()
        and mask_file.is_file()
    ):

        brain_valid, brain_reason = (
            validate_nifti(
                output_file,
                expected_reference=input_img,
            )
        )

        mask_valid, mask_reason = (
            validate_mask(
                mask_file,
                input_img,
            )
        )

        if brain_valid and mask_valid:

            print()
            print(
                "Existing HD-BET outputs verified: PASS"
            )

            skipped.append(
                subject
            )

            continue

        print()
        print(
            "Existing HD-BET output is invalid."
        )

        print(
            f"  Brain: {brain_reason}"
        )

        print(
            f"  Mask : {mask_reason}"
        )

        print(
            "Regenerating."
        )

        if output_file.is_file():
            output_file.unlink()

        if mask_file.is_file():
            mask_file.unlink()

    # ------------------------------------------------------------------
    # HD-BET command
    # ------------------------------------------------------------------

    command = [
        "hd-bet",
        "-i",
        str(input_file),
        "-o",
        str(output_file),
        "-device",
        "cuda",
        "--save_bet_mask",
        "--verbose",
    ]

    print()
    print(
        "Running HD-BET using CUDA..."
    )

    print()
    print(
        "Command:"
    )

    print(
        " ".join(command)
    )

    try:

        result = subprocess.run(
            command,
            check=False,
        )

    except FileNotFoundError:

        print()
        print(
            "ERROR: 'hd-bet' command was not found."
        )

        print(
            "Make sure the virtual environment is active "
            "and HD-BET is installed."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print(
            "ERROR while starting HD-BET:"
        )

        print(
            f"  {exc}"
        )

        failed.append(
            (
                subject,
                f"HD-BET launch error: {exc}",
            )
        )

        continue

    # ------------------------------------------------------------------
    # Return code
    # ------------------------------------------------------------------

    print()
    print(
        "HD-BET return code:",
        result.returncode,
    )

    if result.returncode != 0:

        print(
            "HD-BET FAILED."
        )

        failed.append(
            (
                subject,
                f"HD-BET return code {result.returncode}",
            )
        )

        continue

    # ------------------------------------------------------------------
    # Output existence
    # ------------------------------------------------------------------

    print()
    print(
        "Checking HD-BET outputs..."
    )

    print(
        "  Brain output:",
        "FOUND"
        if output_file.is_file()
        else "MISSING",
    )

    print(
        "  Brain mask:",
        "FOUND"
        if mask_file.is_file()
        else "MISSING",
    )

    if not output_file.is_file():

        failed.append(
            (
                subject,
                "brain output missing",
            )
        )

        continue

    if not mask_file.is_file():

        failed.append(
            (
                subject,
                "brain mask missing",
            )
        )

        continue

    # ------------------------------------------------------------------
    # Validate outputs
    # ------------------------------------------------------------------

    brain_valid, brain_reason = (
        validate_nifti(
            output_file,
            expected_reference=input_img,
        )
    )

    mask_valid, mask_reason = (
        validate_mask(
            mask_file,
            input_img,
        )
    )

    print()
    print(
        "Output validation:"
    )

    print(
        "  Brain image:",
        "PASS"
        if brain_valid
        else "FAIL",
    )

    if not brain_valid:
        print(
            "    Reason:",
            brain_reason,
        )

    print(
        "  Brain mask:",
        "PASS"
        if mask_valid
        else "FAIL",
    )

    if not mask_valid:
        print(
            "    Reason:",
            mask_reason,
        )

    if brain_valid and mask_valid:

        print()
        print(
            f"HD-BET completed successfully: {subject}"
        )

        processed.append(
            subject
        )

    else:

        print()
        print(
            f"HD-BET output validation FAILED: {subject}"
        )

        failed.append(
            (
                subject,
                f"brain={brain_reason}; "
                f"mask={mask_reason}",
            )
        )


# ======================================================================
# FINAL VALIDATION
# ======================================================================

print("\n" + "=" * 80)
print("FINAL EXTERNAL HD-BET VALIDATION")
print("=" * 80)

final_valid = []
final_invalid = []

for subject in SUBJECTS:

    (
        input_file,
        output_file,
        mask_file,
    ) = get_paths(
        subject
    )

    if not input_file.is_file():

        final_invalid.append(
            (
                subject,
                "missing N4 input",
            )
        )

        continue

    if not output_file.is_file():

        final_invalid.append(
            (
                subject,
                "missing brain output",
            )
        )

        continue

    if not mask_file.is_file():

        final_invalid.append(
            (
                subject,
                "missing brain mask",
            )
        )

        continue

    try:

        input_img = nib.load(
            str(input_file)
        )

    except Exception as exc:

        final_invalid.append(
            (
                subject,
                f"N4 input unreadable: {exc}",
            )
        )

        continue

    brain_valid, brain_reason = (
        validate_nifti(
            output_file,
            expected_reference=input_img,
        )
    )

    mask_valid, mask_reason = (
        validate_mask(
            mask_file,
            input_img,
        )
    )

    if brain_valid and mask_valid:

        final_valid.append(
            subject
        )

    else:

        final_invalid.append(
            (
                subject,
                f"brain={brain_reason}; "
                f"mask={mask_reason}",
            )
        )


# ======================================================================
# SUMMARY
# ======================================================================

print()
print(
    "Subjects:",
    len(SUBJECTS),
)

print(
    "Processed:",
    len(processed),
)

print(
    "Skipped:",
    len(skipped),
)

print(
    "Failed:",
    len(failed),
)

print(
    "Final valid:",
    len(final_valid),
)

print(
    "Final invalid:",
    len(final_invalid),
)


if final_invalid:

    print()
    print(
        "Invalid subjects:"
    )

    for subject, reason in final_invalid:

        print(
            f"  {subject}: {reason}"
        )


print()
print(
    "Output root:"
)

print(
    OUTPUT_ROOT.resolve()
)

print("\n" + "=" * 80)

if len(final_valid) == len(SUBJECTS):

    print(
        "STATUS: EXTERNAL HD-BET COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "STATUS: EXTERNAL HD-BET REQUIRES REVIEW"
    )

print("=" * 80)