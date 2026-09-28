

"""
======================================================================
COBRE sMRI BATCH REORIENTATION
======================================================================

Purpose
-------
Reorient the currently selected COBRE sMRI MPRAGE NIfTI volumes into
canonical RAS orientation.

Supports
--------
1. 5ECHO MPRAGE
   subject_echo1.nii.gz ... subject_echo5.nii.gz

2. RMS MPRAGE
   subject_rms.nii.gz

Pipeline position
-----------------
    check_mprage_sessions.py
              |
              v
    convert_mprage_5e.py
              |
              v
    THIS SCRIPT
              |
              v
    N4 bias correction

Important
---------
- Generic across all COBRE ZIP datasets.
- Uses mprage_selection_manifest.csv.
- Does not hardcode subject IDs.
- Does not hardcode ZIP names.
- Does not hardcode sessions.
- Does not hardcode series numbers.
- Supports both 5ECHO and RMS acquisitions.
- Processes exactly the acquisition selected in the manifest.
- Existing valid outputs are preserved.
- Invalid existing outputs are regenerated.
- Legacy/unrelated files are ignored.
- Original converted NIfTI files are never modified.
- Reorientation only changes image orientation/array ordering.
- It does NOT perform resampling to a new voxel size.
- Do NOT run N4 until this step passes verification.

======================================================================
"""

import os
import csv
import argparse

import numpy as np
import nibabel as nib

from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
    inv_ornt_aff,
)


# ======================================================================
# COMMAND-LINE ARGUMENTS
# ======================================================================

parser = argparse.ArgumentParser(
    description=(
        "Reorient current COBRE sMRI MPRAGE NIfTI volumes "
        "to canonical RAS orientation."
    )
)

parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Input directory containing converted MPRAGE NIfTIs.",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for reoriented NIfTIs.",
)

parser.add_argument(
    "--manifest",
    type=str,
    default=None,
    help="Path to mprage_selection_manifest.csv.",
)

args = parser.parse_args()


# ======================================================================
# PATHS
# ======================================================================

if args.input is not None:
    INPUT_DIR = os.path.abspath(args.input)
else:
    INPUT_DIR = os.path.abspath(
        os.path.join(
            "processed_data",
            "smri",
            "mprage_5e",
        )
    )


if args.output is not None:
    OUTPUT_DIR = os.path.abspath(args.output)
else:
    OUTPUT_DIR = os.path.abspath(
        os.path.join(
            "processed_data",
            "smri",
            "reoriented",
        )
    )


if args.manifest is not None:
    MANIFEST_PATH = os.path.abspath(args.manifest)
else:
    MANIFEST_PATH = os.path.abspath(
        os.path.join(
            "processed_data",
            "smri",
            "mprage_selection",
            "mprage_selection_manifest.csv",
        )
    )


# ======================================================================
# EXPECTED INPUT GEOMETRY
# ======================================================================

EXPECTED_INPUT_SHAPE = (
    256,
    256,
    192,
)

EXPECTED_INPUT_VOXEL_SIZE = (
    1.0,
    1.0,
    1.0,
)

EXPECTED_ECHOES = (
    1,
    2,
    3,
    4,
    5,
)

VALID_ACQUISITION_TYPES = {
    "5ECHO",
    "RMS",
}


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("COBRE sMRI BATCH REORIENTATION")
print("=" * 80)

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSelection manifest:")
print(MANIFEST_PATH)

print("\nTarget orientation: canonical RAS")


# ======================================================================
# VERIFY DIRECTORIES / MANIFEST
# ======================================================================

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(
        "\nInput directory not found:\n"
        f"{INPUT_DIR}\n\n"
        "Run convert_mprage_5e.py first."
    )


if not os.path.isfile(MANIFEST_PATH):
    raise FileNotFoundError(
        "\nSelection manifest not found:\n"
        f"{MANIFEST_PATH}\n\n"
        "Run check_mprage_sessions.py first."
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ======================================================================
# LOAD SELECTED ACQUISITIONS FROM MANIFEST
# ======================================================================

selected_acquisitions = []

with open(
    MANIFEST_PATH,
    "r",
    newline="",
    encoding="utf-8",
) as csv_file:

    reader = csv.DictReader(csv_file)

    if not reader.fieldnames:
        raise RuntimeError(
            "Selection manifest contains no columns."
        )

    required_columns = {
        "subject",
        "selection_status",
        "acquisition_type",
    }

    missing_columns = (
        required_columns
        - set(reader.fieldnames)
    )

    if missing_columns:
        raise RuntimeError(
            "\nSelection manifest is missing required columns:\n"
            + ", ".join(sorted(missing_columns))
        )

    for row in reader:

        status = (
            row.get(
                "selection_status",
                "",
            )
            .strip()
            .upper()
        )

        if status != "SELECTED":
            continue

        subject = (
            row.get(
                "subject",
                "",
            )
            .strip()
        )

        acquisition_type = (
            row.get(
                "acquisition_type",
                "",
            )
            .strip()
            .upper()
        )

        if not subject:
            raise RuntimeError(
                "A SELECTED manifest row has an empty subject."
            )

        if acquisition_type not in VALID_ACQUISITION_TYPES:
            raise RuntimeError(
                f"Invalid acquisition_type for {subject}: "
                f"{acquisition_type!r}. "
                f"Expected one of {sorted(VALID_ACQUISITION_TYPES)}."
            )

        selected_acquisitions.append(
            (
                subject,
                acquisition_type,
            )
        )


# ======================================================================
# VALIDATE ONE SELECTION PER SUBJECT
# ======================================================================

subject_types = {}

for subject, acquisition_type in selected_acquisitions:

    if subject in subject_types:

        raise RuntimeError(
            "\nMultiple SELECTED acquisitions found for subject:\n"
            f"{subject}\n"
            f"Existing type: {subject_types[subject]}\n"
            f"New type: {acquisition_type}\n\n"
            "The manifest must contain exactly one selected "
            "acquisition per subject."
        )

    subject_types[subject] = acquisition_type


selected_acquisitions = sorted(
    selected_acquisitions,
    key=lambda x: x[0],
)


if not selected_acquisitions:
    raise RuntimeError(
        "\nNo SELECTED acquisitions found in the MPRAGE manifest."
    )


print(
    "\nCurrent selected acquisitions:",
    len(selected_acquisitions),
)


type_counts = {}

for _, acquisition_type in selected_acquisitions:
    type_counts[acquisition_type] = (
        type_counts.get(acquisition_type, 0) + 1
    )


print("\nAcquisition types:")

for acquisition_type in sorted(type_counts):
    print(
        f"  {acquisition_type}:",
        type_counts[acquisition_type],
    )


# ======================================================================
# HELPER: LOAD + VERIFY NIFTI
# ======================================================================

def load_and_verify_input(path):
    """
    Load an input NIfTI and verify:

    - readable
    - 3D
    - expected original dimensions
    - 1 mm voxel size
    - finite affine
    - finite voxel data
    - non-empty data
    """

    try:

        img = nib.load(path)

        if len(img.shape) != 3:
            return (
                None,
                False,
                f"Expected 3D image, got shape {img.shape}",
            )

        if img.shape != EXPECTED_INPUT_SHAPE:
            return (
                None,
                False,
                f"Expected input shape "
                f"{EXPECTED_INPUT_SHAPE}, got {img.shape}",
            )

        voxel_size = img.header.get_zooms()[:3]

        if not np.all(
            np.isfinite(voxel_size)
        ):
            return (
                None,
                False,
                "Invalid voxel size",
            )

        if not np.allclose(
            voxel_size,
            EXPECTED_INPUT_VOXEL_SIZE,
            atol=0.01,
        ):
            return (
                None,
                False,
                f"Expected 1 mm voxel size, got {voxel_size}",
            )

        if not np.all(
            np.isfinite(img.affine)
        ):
            return (
                None,
                False,
                "Input affine contains NaN/Inf",
            )

        data = img.get_fdata(
            dtype=np.float32
        )

        if not np.all(
            np.isfinite(data)
        ):
            return (
                None,
                False,
                "Input contains NaN/Inf",
            )

        if data.size == 0:
            return (
                None,
                False,
                "Input contains no voxels",
            )

        if not np.any(
            data != 0
        ):
            return (
                None,
                False,
                "Input contains only zeros",
            )

        return (
            img,
            True,
            "PASS",
        )

    except Exception as e:

        return (
            None,
            False,
            str(e),
        )


# ======================================================================
# HELPER: VERIFY RAS OUTPUT
# ======================================================================

def verify_ras_output(path):
    """
    Verify an existing/generated output.

    Important:
    Reorientation can change array shape because axes may be permuted.
    Therefore output shape is NOT required to equal the original shape.

    The output must:
    - be readable
    - be 3D
    - have canonical RAS orientation
    - retain 1 mm voxel spacing
    - contain finite data
    - contain non-zero voxels
    """

    try:

        img = nib.load(path)

        if len(img.shape) != 3:
            return (
                False,
                f"Output is not 3D: {img.shape}",
            )

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
                f"Output orientation is {orientation}, "
                "expected ('R', 'A', 'S')",
            )

        voxel_size = img.header.get_zooms()[:3]

        if not np.allclose(
            voxel_size,
            EXPECTED_INPUT_VOXEL_SIZE,
            atol=0.01,
        ):
            return (
                False,
                f"Output voxel size is {voxel_size}, "
                "expected 1 mm",
            )

        if not np.all(
            np.isfinite(img.affine)
        ):
            return (
                False,
                "Output affine contains NaN/Inf",
            )

        data = img.get_fdata(
            dtype=np.float32
        )

        if not np.all(
            np.isfinite(data)
        ):
            return (
                False,
                "Output contains NaN/Inf",
            )

        if data.size == 0:
            return (
                False,
                "Output contains no voxels",
            )

        if not np.any(
            data != 0
        ):
            return (
                False,
                "Output contains only zeros",
            )

        return (
            True,
            "PASS",
        )

    except Exception as e:

        return (
            False,
            str(e),
        )


# ======================================================================
# HELPER: REORIENT ONE NIFTI
# ======================================================================

def reorient_one(
    subject,
    input_path,
    output_path,
    label,
):
    """
    Reorient one NIfTI to canonical RAS.
    """

    print(
        f"\n{label}"
    )

    print(
        "  Input :",
        input_path,
    )

    print(
        "  Output:",
        output_path,
    )

    # ------------------------------------------------------------------
    # Existing output
    # ------------------------------------------------------------------

    if os.path.isfile(output_path):

        valid, reason = verify_ras_output(
            output_path
        )

        if valid:

            print(
                "  Existing output verified: PASS"
            )

            return (
                "SKIPPED",
                output_path,
            )

        print(
            "  Existing output invalid:",
            reason,
        )

        print(
            "  Regenerating output."
        )

        try:

            os.remove(
                output_path
            )

        except Exception as e:

            raise RuntimeError(
                "Could not remove invalid existing output: "
                f"{e}"
            )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    img, valid, reason = load_and_verify_input(
        input_path
    )

    if not valid:

        raise RuntimeError(
            f"Input verification failed: {reason}"
        )

    print(
        "  Input shape:",
        img.shape,
    )

    print(
        "  Input orientation:",
        nib.aff2axcodes(img.affine),
    )

    print(
        "  Input voxel size:",
        img.header.get_zooms()[:3],
    )

    # ------------------------------------------------------------------
    # Determine orientation transform
    # ------------------------------------------------------------------

    current_ornt = io_orientation(
        img.affine
    )

    target_ornt = axcodes2ornt(
        ("R", "A", "S")
    )

    transform = ornt_transform(
        current_ornt,
        target_ornt,
    )

    # ------------------------------------------------------------------
    # Apply orientation
    # ------------------------------------------------------------------

    data = img.get_fdata(
        dtype=np.float32
    )

    reoriented_data = apply_orientation(
        data,
        transform,
    )

    new_affine = (
        img.affine
        @ inv_ornt_aff(
            transform,
            img.shape,
        )
    )

    # ------------------------------------------------------------------
    # Create output
    # ------------------------------------------------------------------

    output_img = nib.Nifti1Image(
        reoriented_data.astype(
            np.float32
        ),
        new_affine,
    )

    output_img.header.set_xyzt_units(
        "mm"
    )

    output_img.header["descrip"] = (
        f"COBRE sMRI "
        f"Subject={subject} "
        f"{label} "
        f"Canonical=RAS"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    try:

        nib.save(
            output_img,
            output_path,
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to save output: {e}"
        )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    valid, reason = verify_ras_output(
        output_path
    )

    if not valid:

        try:

            os.remove(
                output_path
            )

        except Exception:
            pass

        raise RuntimeError(
            f"Output verification failed: {reason}"
        )

    check = nib.load(
        output_path
    )

    print(
        "  Output shape:",
        check.shape,
    )

    print(
        "  Output orientation:",
        nib.aff2axcodes(
            check.affine
        ),
    )

    print(
        "  Output voxel size:",
        check.header.get_zooms()[:3],
    )

    print(
        "  STATUS: PASS"
    )

    return (
        "PROCESSED",
        output_path,
    )


# ======================================================================
# PROCESSING COUNTERS
# ======================================================================

processed_files = 0
skipped_files = 0

successful_subjects = []
failed_subjects = []


# ======================================================================
# PROCESS SELECTED ACQUISITIONS
# ======================================================================

print("\n" + "=" * 80)
print("PROCESSING SELECTED ACQUISITIONS")
print("=" * 80)


for subject, acquisition_type in selected_acquisitions:

    print("\n" + "-" * 80)
    print(
        "SUBJECT:",
        subject,
    )
    print(
        "ACQUISITION TYPE:",
        acquisition_type,
    )
    print("-" * 80)

    subject_outputs = []
    subject_success = True

    try:

        # ==============================================================
        # 5ECHO
        # ==============================================================

        if acquisition_type == "5ECHO":

            for echo in EXPECTED_ECHOES:

                input_path = os.path.join(
                    INPUT_DIR,
                    f"{subject}_echo{echo}.nii.gz",
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"{subject}_echo{echo}_reoriented.nii.gz",
                )

                status, output = reorient_one(
                    subject=subject,
                    input_path=input_path,
                    output_path=output_path,
                    label=f"Echo {echo}",
                )

                subject_outputs.append(
                    output
                )

                if status == "PROCESSED":
                    processed_files += 1
                elif status == "SKIPPED":
                    skipped_files += 1

        # ==============================================================
        # RMS
        # ==============================================================

        elif acquisition_type == "RMS":

            input_path = os.path.join(
                INPUT_DIR,
                f"{subject}_rms.nii.gz",
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{subject}_rms_reoriented.nii.gz",
            )

            status, output = reorient_one(
                subject=subject,
                input_path=input_path,
                output_path=output_path,
                label="RMS",
            )

            subject_outputs.append(
                output
            )

            if status == "PROCESSED":
                processed_files += 1
            elif status == "SKIPPED":
                skipped_files += 1

        else:

            raise RuntimeError(
                f"Unsupported acquisition type: "
                f"{acquisition_type}"
            )

    except Exception as e:

        print(
            "\nERROR:",
            e,
        )

        subject_success = False

    # ==================================================================
    # HANDLE SUBJECT FAILURE
    # ==================================================================

    if not subject_success:

        print(
            "\nSUBJECT STATUS: FAILED"
        )

        # Remove outputs generated during this incomplete subject run.
        for path in subject_outputs:

            try:

                if os.path.isfile(path):

                    os.remove(
                        path
                    )

                    print(
                        "Removed incomplete output:",
                        os.path.basename(path),
                    )

            except Exception as cleanup_error:

                print(
                    "WARNING: Could not remove:",
                    path,
                    cleanup_error,
                )

        failed_subjects.append(
            subject
        )

        continue

    # ==================================================================
    # FINAL SUBJECT CHECK
    # ==================================================================

    expected_subject_files = (
        5
        if acquisition_type == "5ECHO"
        else 1
    )

    if len(subject_outputs) != expected_subject_files:

        print(
            "\nSUBJECT STATUS: FAILED"
        )

        print(
            "Expected files:",
            expected_subject_files,
        )

        print(
            "Obtained files:",
            len(subject_outputs),
        )

        failed_subjects.append(
            subject
        )

        continue

    all_valid = True

    for path in subject_outputs:

        valid, reason = verify_ras_output(
            path
        )

        if not valid:

            print(
                "Final verification failed:",
                os.path.basename(path),
                reason,
            )

            all_valid = False
            break

    if not all_valid:

        print(
            "\nSUBJECT STATUS: FAILED FINAL VERIFICATION"
        )

        failed_subjects.append(
            subject
        )

        continue

    # ==================================================================
    # SUCCESS
    # ==================================================================

    successful_subjects.append(
        subject
    )

    print(
        "\nSUBJECT STATUS: PASS"
    )

    if acquisition_type == "5ECHO":

        print(
            "All five echoes are present and canonical RAS."
        )

    else:

        print(
            "RMS structural volume is canonical RAS."
        )


# ======================================================================
# FINAL OUTPUT INVENTORY
# ======================================================================

print("\n" + "=" * 80)
print("REORIENTATION OUTPUT INVENTORY")
print("=" * 80)


expected_output_files = 0
current_output_files = 0

for subject, acquisition_type in selected_acquisitions:

    if acquisition_type == "5ECHO":

        expected_output_files += 5

        for echo in EXPECTED_ECHOES:

            path = os.path.join(
                OUTPUT_DIR,
                f"{subject}_echo{echo}_reoriented.nii.gz",
            )

            if os.path.isfile(path):

                valid, _ = verify_ras_output(
                    path
                )

                if valid:
                    current_output_files += 1

    elif acquisition_type == "RMS":

        expected_output_files += 1

        path = os.path.join(
            OUTPUT_DIR,
            f"{subject}_rms_reoriented.nii.gz",
        )

        if os.path.isfile(path):

            valid, _ = verify_ras_output(
                path
            )

            if valid:
                current_output_files += 1


print(
    "Selected subjects:",
    len(selected_acquisitions),
)

print(
    "Expected current outputs:",
    expected_output_files,
)

print(
    "Valid current outputs present:",
    current_output_files,
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("REORIENTATION SUMMARY")
print("=" * 80)

print(
    "Selected subjects          :",
    len(selected_acquisitions),
)

print(
    "Subjects passed            :",
    len(successful_subjects),
)

print(
    "Subjects failed            :",
    len(failed_subjects),
)

print(
    "New files processed        :",
    processed_files,
)

print(
    "Existing files skipped     :",
    skipped_files,
)

print(
    "Expected current outputs   :",
    expected_output_files,
)

print(
    "Valid current outputs      :",
    current_output_files,
)


if failed_subjects:

    print(
        "\nFailed subjects:"
    )

    for subject in failed_subjects:

        print(
            "  ",
            subject,
        )


# ======================================================================
# FINAL STATUS
# ======================================================================

print("\n" + "=" * 80)

if (
    not failed_subjects
    and current_output_files == expected_output_files
):

    print(
        "STATUS: REORIENTATION COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "STATUS: OUTPUT COUNT REQUIRES REVIEW"
    )

print("=" * 80)

print(
    "\nOriginal converted NIfTIs were not modified."
)

print(
    "Target orientation: canonical RAS."
)

print(
    "\nReoriented files are located under:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDo NOT run N4 until this reorientation step "
    "has been verified."
)

print("=" * 80)