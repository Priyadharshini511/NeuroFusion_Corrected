

# src/run_hdbet_echo1.py

"""
======================================================================
COBRE sMRI HD-BET BRAIN EXTRACTION
======================================================================

Purpose
-------
Perform HD-BET brain extraction on the currently selected COBRE
sMRI MPRAGE volumes after N4 bias-field correction.

Supports
--------
1. 5ECHO MPRAGE
   - HD-BET is applied to Echo 1 only.
   - The resulting Echo 1 mask is later applied to Echoes 1-5.

2. RMS MPRAGE
   - HD-BET is applied directly to the single RMS volume.
   - RMS is NOT duplicated into five echoes.

Pipeline position
-----------------
    check_mprage_sessions.py
              |
              v
    convert_mprage_5e.py
              |
              v
    batch_reorient_smri.py
              |
              v
    batch_n4.py
              |
              v
    THIS SCRIPT
              |
              v
    apply_echo1_mask.py

Important
---------
- Generic across COBRE ZIP datasets.
- Uses the MPRAGE selection manifest.
- Processes exactly the acquisition selected in the manifest.
- Exactly one acquisition per subject.
- Existing valid outputs are preserved.
- Invalid/partial outputs are regenerated.
- Legacy/unrelated files are ignored.
- Original N4 images are never modified.
- HD-BET uses CUDA.
- Output remains canonical RAS.
- Geometry remains 1 mm isotropic.
"""

from pathlib import Path
import csv
import subprocess
import sys

import nibabel as nib
import numpy as np


# ======================================================================
# PATHS
# ======================================================================

N4_FOLDER = Path(
    "processed_data/smri/n4"
)

BET_FOLDER = Path(
    "processed_data/smri/mprage_5e_bet"
)

MANIFEST_FILE = Path(
    "processed_data/smri/mprage_selection/mprage_selection_manifest.csv"
)

BET_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# SETTINGS
# ======================================================================

EXPECTED_VOXEL_SIZE = np.array(
    [1.0, 1.0, 1.0],
    dtype=float,
)

VOXEL_TOLERANCE = 0.01

VALID_ACQUISITION_TYPES = {
    "5ECHO",
    "RMS",
}


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("COBRE sMRI HD-BET BRAIN EXTRACTION")
print("=" * 80)

print()
print("N4 input folder:")
print(f"  {N4_FOLDER.resolve()}")

print()
print("HD-BET output folder:")
print(f"  {BET_FOLDER.resolve()}")

print()
print("Selection manifest:")
print(f"  {MANIFEST_FILE.resolve()}")


# ======================================================================
# VALIDATE REQUIRED PATHS
# ======================================================================

if not N4_FOLDER.is_dir():

    raise FileNotFoundError(
        f"\nN4 input folder does not exist:\n"
        f"{N4_FOLDER.resolve()}\n\n"
        "Run batch_n4.py first."
    )


if not MANIFEST_FILE.is_file():

    raise FileNotFoundError(
        f"\nMPRAGE selection manifest does not exist:\n"
        f"{MANIFEST_FILE.resolve()}\n\n"
        "Run check_mprage_sessions.py first."
    )


# ======================================================================
# LOAD SELECTED ACQUISITIONS
# ======================================================================

def load_selected_acquisitions(
    manifest_file,
):
    """
    Load exactly the SELECTED acquisitions from the manifest.

    Returns
    -------
    list of tuples
        [(subject, acquisition_type), ...]

    Requirements
    ------------
    - selection_status must be SELECTED.
    - acquisition_type must be 5ECHO or RMS.
    - exactly one selected acquisition per subject.
    """

    selected = []

    with manifest_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:

            raise ValueError(
                "Selection manifest has no header."
            )

        required_columns = {
            "subject",
            "selection_status",
            "acquisition_type",
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                "Selection manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            status = str(
                row.get(
                    "selection_status",
                    "",
                )
            ).strip().upper()

            if status != "SELECTED":
                continue

            subject = str(
                row.get(
                    "subject",
                    "",
                )
            ).strip()

            acquisition_type = str(
                row.get(
                    "acquisition_type",
                    "",
                )
            ).strip().upper()

            if not subject:

                raise ValueError(
                    "Manifest contains a SELECTED row "
                    "without a subject ID."
                )

            if acquisition_type not in VALID_ACQUISITION_TYPES:

                raise ValueError(
                    f"Invalid acquisition type for "
                    f"{subject}: {acquisition_type!r}. "
                    f"Expected one of "
                    f"{sorted(VALID_ACQUISITION_TYPES)}."
                )

            selected.append(
                (
                    subject,
                    acquisition_type,
                )
            )

    if not selected:

        raise ValueError(
            "No SELECTED acquisitions were found "
            "in the MPRAGE selection manifest."
        )

    # --------------------------------------------------------------
    # Exactly one selected acquisition per subject
    # --------------------------------------------------------------

    subjects_seen = {}

    for subject, acquisition_type in selected:

        if subject in subjects_seen:

            raise ValueError(
                "\nMultiple SELECTED acquisitions found "
                "for subject:\n"
                f"  {subject}\n"
                f"  Existing type: "
                f"{subjects_seen[subject]}\n"
                f"  New type: "
                f"{acquisition_type}\n\n"
                "The manifest must contain exactly one "
                "selected acquisition per subject."
            )

        subjects_seen[subject] = acquisition_type

    return sorted(
        selected,
        key=lambda x: x[0],
    )


selected_acquisitions = load_selected_acquisitions(
    MANIFEST_FILE
)


# ======================================================================
# DISPLAY SELECTION
# ======================================================================

print()
print("=" * 80)
print("SELECTED ACQUISITIONS FROM MANIFEST")
print("=" * 80)

type_counts = {}

for subject, acquisition_type in selected_acquisitions:

    print(
        f"  {subject} -> {acquisition_type}"
    )

    type_counts[acquisition_type] = (
        type_counts.get(
            acquisition_type,
            0,
        )
        + 1
    )

print()
print(
    "Total selected subjects:",
    len(selected_acquisitions),
)

print()
print("Acquisition types:")

for acquisition_type in sorted(type_counts):

    print(
        f"  {acquisition_type}: "
        f"{type_counts[acquisition_type]}"
    )


# ======================================================================
# NIFTI VALIDATION
# ======================================================================

def validate_nifti(
    file_path,
    expected_reference=None,
    require_nonempty=True,
):
    """
    Validate a NIfTI image.

    Checks
    ------
    - readable
    - 3-D
    - finite voxel values
    - 1 mm isotropic voxel size
    - canonical RAS orientation
    - non-empty image
    - matching shape/affine when reference is supplied

    Returns
    -------
    (True, "VALID")
    or
    (False, "reason")
    """

    try:

        img = nib.load(
            str(file_path)
        )

        if len(img.shape) != 3:

            return (
                False,
                f"image is not 3-D: {img.shape}",
            )

        data = np.asanyarray(
            img.dataobj
        )

        if data.size == 0:

            return (
                False,
                "empty image",
            )

        if not np.all(
            np.isfinite(data)
        ):

            return (
                False,
                "non-finite voxel values",
            )

        # ----------------------------------------------------------
        # Voxel size
        # ----------------------------------------------------------

        voxel_size = np.asarray(
            img.header.get_zooms()[:3],
            dtype=float,
        )

        if not np.allclose(
            voxel_size,
            EXPECTED_VOXEL_SIZE,
            atol=VOXEL_TOLERANCE,
        ):

            return (
                False,
                f"unexpected voxel size: "
                f"{voxel_size}",
            )

        # ----------------------------------------------------------
        # Orientation
        # ----------------------------------------------------------

        orientation = nib.aff2axcodes(
            img.affine
        )

        if orientation != (
            "R",
            "A",
            "S",
        ):

            return (
                False,
                f"orientation is {orientation}; "
                f"expected ('R','A','S')",
            )

        # ----------------------------------------------------------
        # Reference geometry
        # ----------------------------------------------------------

        if expected_reference is not None:

            if img.shape != expected_reference.shape:

                return (
                    False,
                    f"shape mismatch: "
                    f"{img.shape} != "
                    f"{expected_reference.shape}",
                )

            if not np.allclose(
                img.affine,
                expected_reference.affine,
                atol=1e-3,
            ):

                return (
                    False,
                    "affine mismatch",
                )

            reference_voxel_size = np.asarray(
                expected_reference.header
                .get_zooms()[:3],
                dtype=float,
            )

            if not np.allclose(
                voxel_size,
                reference_voxel_size,
                atol=VOXEL_TOLERANCE,
            ):

                return (
                    False,
                    f"voxel size mismatch: "
                    f"{voxel_size} != "
                    f"{reference_voxel_size}",
                )

        # ----------------------------------------------------------
        # Non-empty image
        # ----------------------------------------------------------

        if require_nonempty:

            if not np.any(
                data != 0
            ):

                return (
                    False,
                    "all voxels are zero",
                )

        return (
            True,
            "VALID",
        )

    except Exception as exc:

        return (
            False,
            f"read/validation error: {exc}",
        )


# ======================================================================
# MASK VALIDATION
# ======================================================================

def validate_mask(
    mask_file,
    reference_img,
):
    """
    Validate an HD-BET mask against its N4 input reference.
    """

    try:

        mask_img = nib.load(
            str(mask_file)
        )

        mask_data = np.asanyarray(
            mask_img.dataobj
        )

        if mask_data.size == 0:

            return (
                False,
                "empty mask",
            )

        if not np.all(
            np.isfinite(mask_data)
        ):

            return (
                False,
                "mask contains non-finite values",
            )

        # ----------------------------------------------------------
        # Shape
        # ----------------------------------------------------------

        if mask_img.shape != reference_img.shape:

            return (
                False,
                f"shape mismatch: "
                f"{mask_img.shape} != "
                f"{reference_img.shape}",
            )

        # ----------------------------------------------------------
        # Affine
        # ----------------------------------------------------------

        if not np.allclose(
            mask_img.affine,
            reference_img.affine,
            atol=1e-3,
        ):

            return (
                False,
                "mask affine mismatch",
            )

        # ----------------------------------------------------------
        # Voxel size
        # ----------------------------------------------------------

        mask_voxel_size = np.asarray(
            mask_img.header.get_zooms()[:3],
            dtype=float,
        )

        reference_voxel_size = np.asarray(
            reference_img.header.get_zooms()[:3],
            dtype=float,
        )

        if not np.allclose(
            mask_voxel_size,
            reference_voxel_size,
            atol=VOXEL_TOLERANCE,
        ):

            return (
                False,
                f"mask voxel size mismatch: "
                f"{mask_voxel_size} != "
                f"{reference_voxel_size}",
            )

        # ----------------------------------------------------------
        # Orientation
        # ----------------------------------------------------------

        orientation = nib.aff2axcodes(
            mask_img.affine
        )

        if orientation != (
            "R",
            "A",
            "S",
        ):

            return (
                False,
                f"mask orientation is {orientation}; "
                f"expected ('R','A','S')",
            )

        # ----------------------------------------------------------
        # Binary mask
        # ----------------------------------------------------------

        unique_values = np.unique(
            mask_data
        )

        if unique_values.size == 0:

            return (
                False,
                "mask has no values",
            )

        if not np.all(
            np.isin(
                unique_values,
                [0, 1],
            )
        ):

            return (
                False,
                "mask is not binary",
            )

        # ----------------------------------------------------------
        # Non-empty mask
        # ----------------------------------------------------------

        mask_voxels = np.count_nonzero(
            mask_data
        )

        if mask_voxels == 0:

            return (
                False,
                "mask is completely empty",
            )

        # ----------------------------------------------------------
        # Coverage
        # ----------------------------------------------------------

        total_voxels = mask_data.size

        coverage = (
            mask_voxels
            / total_voxels
        )

        if coverage <= 0 or coverage >= 1:

            return (
                False,
                f"invalid mask coverage: "
                f"{coverage:.6f}",
            )

        return (
            True,
            "VALID",
        )

    except Exception as exc:

        return (
            False,
            f"mask validation error: {exc}",
        )


# ======================================================================
# BUILD HD-BET INPUT/OUTPUT PATHS
# ======================================================================

def get_paths(
    subject,
    acquisition_type,
):
    """
    Return input, brain-output and mask paths.
    """

    if acquisition_type == "5ECHO":

        input_file = (
            N4_FOLDER
            / f"{subject}_echo1_n4.nii.gz"
        )

        output_file = (
            BET_FOLDER
            / f"{subject}_echo1_n4_bet.nii.gz"
        )

        mask_file = (
            BET_FOLDER
            / f"{subject}_echo1_n4_bet_bet.nii.gz"
        )

    elif acquisition_type == "RMS":

        input_file = (
            N4_FOLDER
            / f"{subject}_rms_n4.nii.gz"
        )

        output_file = (
            BET_FOLDER
            / f"{subject}_rms_n4_bet.nii.gz"
        )

        mask_file = (
            BET_FOLDER
            / f"{subject}_rms_n4_bet_bet.nii.gz"
        )

    else:

        raise ValueError(
            f"Unsupported acquisition type: "
            f"{acquisition_type}"
        )

    return (
        input_file,
        output_file,
        mask_file,
    )


# ======================================================================
# TRACKERS
# ======================================================================

processed = []
skipped = []
failed = []


# ======================================================================
# PROCESS SELECTED ACQUISITIONS
# ======================================================================

for index, (
    subject,
    acquisition_type,
) in enumerate(
    selected_acquisitions,
    start=1,
):

    (
        input_file,
        output_file,
        mask_file,
    ) = get_paths(
        subject,
        acquisition_type,
    )

    print()
    print("=" * 80)
    print(
        f"SUBJECT {index}/"
        f"{len(selected_acquisitions)}: "
        f"{subject}"
    )
    print(
        f"ACQUISITION TYPE: "
        f"{acquisition_type}"
    )
    print("=" * 80)

    print("N4 input:")
    print(f"  {input_file}")

    print("Brain-extracted output:")
    print(f"  {output_file}")

    print("Brain mask:")
    print(f"  {mask_file}")

    # --------------------------------------------------------------
    # INPUT EXISTENCE
    # --------------------------------------------------------------

    if not input_file.is_file():

        print()
        print(
            "ERROR: N4 input is missing."
        )

        failed.append(
            (
                subject,
                "missing N4 input",
            )
        )

        continue

    # --------------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------------

    input_valid, input_reason = validate_nifti(
        input_file
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
                f"invalid N4 input: "
                f"{input_reason}",
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
            "ERROR: Could not load N4 input:"
        )
        print(
            f"  {exc}"
        )

        failed.append(
            (
                subject,
                f"cannot load N4 input: {exc}",
            )
        )

        continue

    # --------------------------------------------------------------
    # EXISTING OUTPUTS
    # --------------------------------------------------------------

    if (
        output_file.is_file()
        and mask_file.is_file()
    ):

        print()
        print(
            "Existing HD-BET outputs found."
        )

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

            print(
                "Existing brain image: PASS"
            )

            print(
                "Existing brain mask : PASS"
            )

            print(
                "SKIPPING - existing outputs "
                "are valid."
            )

            skipped.append(
                subject
            )

            continue

        print()
        print(
            "Existing HD-BET outputs are invalid "
            "or incomplete."
        )

        print(
            f"  Brain image: "
            f"{brain_reason}"
        )

        print(
            f"  Brain mask : "
            f"{mask_reason}"
        )

        print(
            "Regenerating HD-BET outputs."
        )

    # --------------------------------------------------------------
    # REMOVE INVALID / PARTIAL OUTPUTS
    # --------------------------------------------------------------

    if output_file.exists():

        try:

            output_file.unlink()

            print(
                "Removed existing brain output."
            )

        except Exception as exc:

            print()
            print(
                "ERROR: Could not remove "
                "existing brain output:"
            )
            print(
                f"  {exc}"
            )

            failed.append(
                (
                    subject,
                    "cannot remove invalid brain output",
                )
            )

            continue

    if mask_file.exists():

        try:

            mask_file.unlink()

            print(
                "Removed existing brain mask."
            )

        except Exception as exc:

            print()
            print(
                "ERROR: Could not remove "
                "existing brain mask:"
            )
            print(
                f"  {exc}"
            )

            failed.append(
                (
                    subject,
                    "cannot remove invalid brain mask",
                )
            )

            continue

    # --------------------------------------------------------------
    # HD-BET COMMAND
    # --------------------------------------------------------------

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
            "Make sure the virtual environment is "
            "activated and HD-BET is installed."
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

    # --------------------------------------------------------------
    # RETURN CODE
    # --------------------------------------------------------------

    if result.returncode != 0:

        print()
        print(
            f"HD-BET FAILED for {subject}"
        )

        print(
            f"Return code: "
            f"{result.returncode}"
        )

        failed.append(
            (
                subject,
                f"HD-BET return code "
                f"{result.returncode}",
            )
        )

        continue

    # --------------------------------------------------------------
    # OUTPUT EXISTENCE
    # --------------------------------------------------------------

    print()
    print(
        "Checking HD-BET output files..."
    )

    print(
        f"  Brain output: "
        f"{'FOUND' if output_file.is_file() else 'MISSING'}"
    )

    print(
        f"  Brain mask  : "
        f"{'FOUND' if mask_file.is_file() else 'MISSING'}"
    )

    if not output_file.is_file():

        failed.append(
            (
                subject,
                "brain output missing after HD-BET",
            )
        )

        continue

    if not mask_file.is_file():

        failed.append(
            (
                subject,
                "brain mask missing after HD-BET",
            )
        )

        continue

    # --------------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------------

    brain_valid, brain_reason = validate_nifti(
        output_file,
        expected_reference=input_img,
    )

    mask_valid, mask_reason = validate_mask(
        mask_file,
        input_img,
    )

    print()
    print(
        "Output validation:"
    )

    print(
        f"  Brain image: "
        f"{'PASS' if brain_valid else 'FAIL'}"
    )

    if not brain_valid:

        print(
            f"    Reason: "
            f"{brain_reason}"
        )

    print(
        f"  Brain mask : "
        f"{'PASS' if mask_valid else 'FAIL'}"
    )

    if not mask_valid:

        print(
            f"    Reason: "
            f"{mask_reason}"
        )

    if brain_valid and mask_valid:

        print()
        print(
            f"HD-BET completed successfully: "
            f"{subject}"
        )

        processed.append(
            subject
        )

    else:

        print()
        print(
            f"HD-BET output validation FAILED: "
            f"{subject}"
        )

        failed.append(
            (
                subject,
                f"brain={brain_reason}; "
                f"mask={mask_reason}",
            )
        )


# ======================================================================
# FINAL BATCH VALIDATION
# ======================================================================

print()
print()
print("=" * 80)
print("FINAL HD-BET VALIDATION")
print("=" * 80)

final_valid = []
final_invalid = []


for subject, acquisition_type in selected_acquisitions:

    (
        input_file,
        output_file,
        mask_file,
    ) = get_paths(
        subject,
        acquisition_type,
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
                "missing brain-extracted output",
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
# FINAL INVENTORY
# ======================================================================

expected_count = len(
    selected_acquisitions
)

print(
    f"Expected selected subjects : "
    f"{expected_count}"
)

print(
    f"Valid HD-BET subjects       : "
    f"{len(final_valid)}"
)

print(
    f"Invalid HD-BET subjects     : "
    f"{len(final_invalid)}"
)


if final_invalid:

    print()
    print(
        "FINAL INVALID SUBJECTS:"
    )

    for subject, reason in final_invalid:

        print(
            f"  {subject}: {reason}"
        )


# ======================================================================
# FINAL STATUS
# ======================================================================

print()
print("=" * 80)

if (
    len(final_valid) == expected_count
    and not final_invalid
):

    print(
        "STATUS: HD-BET VALIDATION PASSED"
    )

else:

    print(
        "STATUS: HD-BET VALIDATION FAILED"
    )

print("=" * 80)


# ======================================================================
# PROCESSING SUMMARY
# ======================================================================

print()
print(
    "Processing summary:"
)

print(
    f"  Newly processed : "
    f"{len(processed)}"
)

print(
    f"  Skipped valid   : "
    f"{len(skipped)}"
)

print(
    f"  Failed          : "
    f"{len(failed)}"
)

print()
print(
    "Output directory:"
)

print(
    f"  {BET_FOLDER.resolve()}"
)


if failed:

    print()
    print(
        "PROCESSING FAILURES:"
    )

    for subject, reason in failed:

        print(
            f"  {subject}: {reason}"
        )


print()
print("=" * 80)