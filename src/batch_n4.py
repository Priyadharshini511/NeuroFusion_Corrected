
# src\batch_n4.py

"""
======================================================================
COBRE sMRI BATCH N4 BIAS FIELD CORRECTION
======================================================================

Purpose
-------
Perform N4 bias-field correction on the currently selected COBRE
sMRI MPRAGE volumes.

Supports
--------
1. 5ECHO MPRAGE
   subject_echo1_reoriented.nii.gz
   subject_echo2_reoriented.nii.gz
   subject_echo3_reoriented.nii.gz
   subject_echo4_reoriented.nii.gz
   subject_echo5_reoriented.nii.gz

2. RMS MPRAGE
   subject_rms_reoriented.nii.gz

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
    THIS SCRIPT
              |
              v
    run_hdbet_echo1.py

Important
---------
- Generic across COBRE ZIP datasets.
- Uses the MPRAGE selection manifest.
- Supports both 5ECHO and RMS acquisitions.
- Processes exactly the acquisition selected in the manifest.
- RMS is processed as ONE 3-D structural volume.
- RMS is never duplicated into five echoes.
- N4 is applied independently to each actual input volume.
- Existing valid outputs are preserved.
- Invalid existing outputs are regenerated.
- Legacy/unrelated files are ignored.
- Original reoriented images are never modified.
- Geometry must remain 1 mm isotropic.
- Output must remain canonical RAS.

Dimension conventions
---------------------
For the current reoriented NIfTI files:

    SimpleITK GetSize():
        (192, 256, 256)

    SimpleITK GetArrayFromImage().shape:
        (256, 256, 192)

    nibabel NIfTI shape:
        (192, 256, 256)

SimpleITK image size uses X,Y,Z ordering.
SimpleITK's NumPy conversion returns Z,Y,X ordering.
Nibabel reports the NIfTI array shape in X,Y,Z ordering.

These are different representations of the same 3-D image.
"""

import os
import csv
import argparse

import numpy as np
import SimpleITK as sitk
import nibabel as nib


# ======================================================================
# COMMAND-LINE ARGUMENTS
# ======================================================================

parser = argparse.ArgumentParser(
    description=(
        "Perform N4 bias-field correction on the current "
        "COBRE sMRI MPRAGE dataset."
    )
)

parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Input directory containing reoriented NIfTI files.",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for N4-corrected NIfTI files.",
)

parser.add_argument(
    "--manifest",
    type=str,
    default=None,
    help="Path to the MPRAGE selection manifest.",
)

args = parser.parse_args()


# ======================================================================
# PATHS
# ======================================================================

INPUT_DIR = os.path.abspath(
    args.input
    if args.input is not None
    else os.path.join(
        "processed_data",
        "smri",
        "reoriented",
    )
)

OUTPUT_DIR = os.path.abspath(
    args.output
    if args.output is not None
    else os.path.join(
        "processed_data",
        "smri",
        "n4",
    )
)

MANIFEST_PATH = os.path.abspath(
    args.manifest
    if args.manifest is not None
    else os.path.join(
        "processed_data",
        "smri",
        "mprage_selection",
        "mprage_selection_manifest.csv",
    )
)


# ======================================================================
# CONSTANTS
# ======================================================================

# NIfTI / nibabel reports image dimensions in X,Y,Z.
EXPECTED_NIFTI_SHAPE = (
    192,
    256,
    256,
)

# SimpleITK GetSize() also reports X,Y,Z.
EXPECTED_SITK_SIZE = (
    192,
    256,
    256,
)

# SimpleITK GetArrayFromImage() returns Z,Y,X.
EXPECTED_ARRAY_SHAPE = (
    EXPECTED_SITK_SIZE[2],
    EXPECTED_SITK_SIZE[1],
    EXPECTED_SITK_SIZE[0],
)

EXPECTED_VOXEL_SIZE = (
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
print("COBRE sMRI BATCH N4 BIAS FIELD CORRECTION")
print("=" * 80)

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSelection manifest:")
print(MANIFEST_PATH)


# ======================================================================
# DIRECTORY / MANIFEST CHECKS
# ======================================================================

if not os.path.isdir(INPUT_DIR):

    raise FileNotFoundError(
        f"\nInput directory not found:\n{INPUT_DIR}\n\n"
        "Run batch_reorient_smri.py first."
    )


if not os.path.isfile(MANIFEST_PATH):

    raise FileNotFoundError(
        f"\nSelection manifest not found:\n{MANIFEST_PATH}\n\n"
        "Run check_mprage_sessions.py first."
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ======================================================================
# LOAD SELECTED ACQUISITIONS
# ======================================================================

selected_acquisitions = []

with open(
    MANIFEST_PATH,
    "r",
    newline="",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    if not reader.fieldnames:

        raise RuntimeError(
            "Selection manifest contains no columns."
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

        raise RuntimeError(
            "Selection manifest is missing columns: "
            + ", ".join(sorted(missing))
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
                "A SELECTED row contains an empty subject."
            )

        if acquisition_type not in VALID_ACQUISITION_TYPES:

            raise RuntimeError(
                f"Invalid acquisition_type for {subject}: "
                f"{acquisition_type!r}. "
                f"Expected one of "
                f"{sorted(VALID_ACQUISITION_TYPES)}."
            )

        selected_acquisitions.append(
            (
                subject,
                acquisition_type,
            )
        )


# ======================================================================
# VALIDATE ONE SELECTED ACQUISITION PER SUBJECT
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
        "\nNo SELECTED acquisitions found in the manifest."
    )


print(
    "\nSelected acquisitions:",
    len(selected_acquisitions),
)


# ======================================================================
# ACQUISITION TYPE SUMMARY
# ======================================================================

type_counts = {}

for _, acquisition_type in selected_acquisitions:

    type_counts[acquisition_type] = (
        type_counts.get(
            acquisition_type,
            0,
        )
        + 1
    )


print("\nAcquisition types:")

for acquisition_type in sorted(type_counts):

    print(
        f"  {acquisition_type}:",
        type_counts[acquisition_type],
    )


# ======================================================================
# VERIFY ONE NIFTI
# ======================================================================

def verify_nifti(
    path,
    require_ras=True,
):
    """
    Verify a reoriented or N4 NIfTI.

    Checks:

    SimpleITK image size:
        (192, 256, 256)

    SimpleITK NumPy array:
        (256, 256, 192)

    NIfTI / nibabel shape:
        (192, 256, 256)

    Additional checks:
    - readable image
    - 3-D geometry
    - 1 mm isotropic spacing
    - finite voxel data
    - finite affine
    - non-empty image
    - canonical RAS orientation
    """

    try:

        # --------------------------------------------------------------
        # SimpleITK checks
        # --------------------------------------------------------------

        image = sitk.ReadImage(
            path
        )

        size = tuple(
            int(x)
            for x in image.GetSize()
        )

        spacing = tuple(
            float(x)
            for x in image.GetSpacing()
        )

        # SimpleITK GetSize() uses X,Y,Z.
        if size != EXPECTED_SITK_SIZE:

            return (
                False,
                f"Unexpected SimpleITK size {size}; "
                f"expected {EXPECTED_SITK_SIZE}",
            )

        if not np.allclose(
            spacing,
            EXPECTED_VOXEL_SIZE,
            atol=0.01,
        ):

            return (
                False,
                f"Unexpected spacing {spacing}; "
                f"expected {EXPECTED_VOXEL_SIZE}",
            )

        # SimpleITK converts image data to NumPy as Z,Y,X.
        array = sitk.GetArrayFromImage(
            image
        )

        if array.shape != EXPECTED_ARRAY_SHAPE:

            return (
                False,
                f"Unexpected SimpleITK array shape "
                f"{array.shape}; "
                f"expected {EXPECTED_ARRAY_SHAPE}",
            )

        if not np.all(
            np.isfinite(array)
        ):

            return (
                False,
                "NaN/Inf values detected",
            )

        if array.size == 0:

            return (
                False,
                "Empty image",
            )

        if not np.any(
            array != 0
        ):

            return (
                False,
                "Image contains only zeros",
            )

        # --------------------------------------------------------------
        # NIfTI / nibabel checks
        # --------------------------------------------------------------

        nii = nib.load(
            path
        )

        if len(nii.shape) != 3:

            return (
                False,
                f"NIfTI is not 3-D: {nii.shape}",
            )

        nii_shape = tuple(
            int(x)
            for x in nii.shape
        )

        if nii_shape != EXPECTED_NIFTI_SHAPE:

            return (
                False,
                f"Unexpected NIfTI shape {nii_shape}; "
                f"expected {EXPECTED_NIFTI_SHAPE}",
            )

        nii_voxel_size = tuple(
            float(x)
            for x in nii.header.get_zooms()[:3]
        )

        if not np.allclose(
            nii_voxel_size,
            EXPECTED_VOXEL_SIZE,
            atol=0.01,
        ):

            return (
                False,
                f"Unexpected NIfTI voxel size "
                f"{nii_voxel_size}",
            )

        if not np.all(
            np.isfinite(nii.affine)
        ):

            return (
                False,
                "NIfTI affine contains NaN/Inf",
            )

        if require_ras:

            orientation = nib.aff2axcodes(
                nii.affine
            )

            if orientation != (
                "R",
                "A",
                "S",
            ):

                return (
                    False,
                    f"Orientation is {orientation}; "
                    f"expected ('R','A','S')",
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
# BUILD EXPECTED INPUT LIST
# ======================================================================

expected_input_files = []

for subject, acquisition_type in selected_acquisitions:

    if acquisition_type == "5ECHO":

        for echo in EXPECTED_ECHOES:

            expected_input_files.append(
                os.path.join(
                    INPUT_DIR,
                    f"{subject}_echo{echo}_reoriented.nii.gz",
                )
            )

    elif acquisition_type == "RMS":

        expected_input_files.append(
            os.path.join(
                INPUT_DIR,
                f"{subject}_rms_reoriented.nii.gz",
            )
        )


# ======================================================================
# VERIFY COMPLETE CURRENT INPUT SET
# ======================================================================

print("\n" + "=" * 80)
print("VERIFYING REORIENTED INPUTS")
print("=" * 80)


missing_inputs = []
invalid_inputs = []


for path in expected_input_files:

    if not os.path.isfile(path):

        missing_inputs.append(
            path
        )

        continue

    valid, reason = verify_nifti(
        path
    )

    if not valid:

        invalid_inputs.append(
            (
                path,
                reason,
            )
        )


# ----------------------------------------------------------------------
# Report missing inputs
# ----------------------------------------------------------------------

if missing_inputs:

    print(
        "\nMissing reoriented inputs:"
    )

    for path in missing_inputs:

        print(
            "  ",
            os.path.basename(path),
        )


# ----------------------------------------------------------------------
# Report invalid inputs
# ----------------------------------------------------------------------

if invalid_inputs:

    print(
        "\nInvalid reoriented inputs:"
    )

    for path, reason in invalid_inputs:

        print(
            "  ",
            os.path.basename(path),
            "->",
            reason,
        )


# ----------------------------------------------------------------------
# Stop if input set is incomplete
# ----------------------------------------------------------------------

if missing_inputs or invalid_inputs:

    raise RuntimeError(
        "\nN4 processing stopped because the current "
        "reoriented input set is incomplete or invalid."
    )


print(
    "\nComplete current input set verified:",
    len(expected_input_files),
    "volumes."
)


# ======================================================================
# N4 PROCESSING FUNCTION
# ======================================================================

def process_n4(
    subject,
    input_path,
    output_path,
    label,
):
    """
    Perform N4 bias-field correction on one actual
    structural volume.
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

    # ==============================================================
    # EXISTING OUTPUT CHECK
    # ==============================================================

    if os.path.isfile(output_path):

        valid, reason = verify_nifti(
            output_path
        )

        if valid:

            print(
                "  Existing N4 output verified: PASS"
            )

            return "SKIPPED"

        print(
            "  Existing N4 output is invalid:",
            reason,
        )

        print(
            "  Regenerating invalid output."
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

    # ==============================================================
    # INPUT VERIFICATION
    # ==============================================================

    valid, reason = verify_nifti(
        input_path
    )

    if not valid:

        raise RuntimeError(
            f"Input verification failed: {reason}"
        )

    # ==============================================================
    # READ IMAGE
    # ==============================================================

    image = sitk.ReadImage(
        input_path
    )

    print(
        "  Input size:",
        image.GetSize(),
    )

    print(
        "  Input spacing:",
        image.GetSpacing(),
    )

    # ==============================================================
    # CAST TO FLOAT32
    # ==============================================================

    image = sitk.Cast(
        image,
        sitk.sitkFloat32,
    )

    # ==============================================================
    # CREATE N4 MASK
    # ==============================================================

    mask = sitk.OtsuThreshold(
        image,
        0,
        1,
        200,
    )

    mask = sitk.BinaryMorphologicalClosing(
        mask,
        [2, 2, 2],
    )

    # ==============================================================
    # N4 BIAS FIELD CORRECTION
    # ==============================================================

    corrector = (
        sitk.N4BiasFieldCorrectionImageFilter()
    )

    corrector.SetMaximumNumberOfIterations(
        [30, 20, 10]
    )

    corrector.SetNumberOfControlPoints(
        [4, 4, 4]
    )

    print(
        "  Running N4 bias correction..."
    )

    corrected = corrector.Execute(
        image,
        mask,
    )

    # ==============================================================
    # PRESERVE SPATIAL INFORMATION
    # ==============================================================

    corrected = sitk.Cast(
        corrected,
        sitk.sitkFloat32,
    )

    corrected.CopyInformation(
        image
    )

    # ==============================================================
    # SAVE
    # ==============================================================

    try:

        sitk.WriteImage(
            corrected,
            output_path,
            True,
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to save N4 output: {e}"
        )

    # ==============================================================
    # VERIFY SAVED OUTPUT
    # ==============================================================

    valid, reason = verify_nifti(
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
            f"N4 output verification failed: {reason}"
        )

    check = sitk.ReadImage(
        output_path
    )

    print(
        "  Output size:",
        check.GetSize(),
    )

    print(
        "  Output spacing:",
        check.GetSpacing(),
    )

    print(
        "  N4 output verified: PASS"
    )

    return "PROCESSED"


# ======================================================================
# COUNTERS
# ======================================================================

processed = 0
skipped = 0

successful_subjects = []
failed_subjects = []


# ======================================================================
# START N4
# ======================================================================

print("\n" + "=" * 80)
print("STARTING N4 CORRECTION")
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

    subject_success = True

    try:

        # ==============================================================
        # 5ECHO
        # ==============================================================

        if acquisition_type == "5ECHO":

            for echo in EXPECTED_ECHOES:

                input_path = os.path.join(
                    INPUT_DIR,
                    f"{subject}_echo{echo}_reoriented.nii.gz",
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"{subject}_echo{echo}_n4.nii.gz",
                )

                status = process_n4(
                    subject=subject,
                    input_path=input_path,
                    output_path=output_path,
                    label=f"Echo {echo}",
                )

                if status == "PROCESSED":

                    processed += 1

                elif status == "SKIPPED":

                    skipped += 1

        # ==============================================================
        # RMS
        # ==============================================================

        elif acquisition_type == "RMS":

            input_path = os.path.join(
                INPUT_DIR,
                f"{subject}_rms_reoriented.nii.gz",
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{subject}_rms_n4.nii.gz",
            )

            status = process_n4(
                subject=subject,
                input_path=input_path,
                output_path=output_path,
                label="RMS",
            )

            if status == "PROCESSED":

                processed += 1

            elif status == "SKIPPED":

                skipped += 1

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

    # ==============================================================
    # SUBJECT STATUS
    # ==============================================================

    if subject_success:

        successful_subjects.append(
            subject
        )

        print(
            "\nSUBJECT STATUS: PASS"
        )

    else:

        failed_subjects.append(
            subject
        )

        print(
            "\nSUBJECT STATUS: FAILED"
        )


# ======================================================================
# BUILD EXPECTED OUTPUT LIST
# ======================================================================

expected_outputs = []

for subject, acquisition_type in selected_acquisitions:

    if acquisition_type == "5ECHO":

        for echo in EXPECTED_ECHOES:

            expected_outputs.append(
                os.path.join(
                    OUTPUT_DIR,
                    f"{subject}_echo{echo}_n4.nii.gz",
                )
            )

    elif acquisition_type == "RMS":

        expected_outputs.append(
            os.path.join(
                OUTPUT_DIR,
                f"{subject}_rms_n4.nii.gz",
            )
        )


# ======================================================================
# FINAL OUTPUT INVENTORY
# ======================================================================

print("\n" + "=" * 80)
print("N4 OUTPUT INVENTORY")
print("=" * 80)


valid_outputs_present = 0
invalid_outputs_present = 0
missing_outputs = []


for path in expected_outputs:

    if not os.path.isfile(path):

        missing_outputs.append(
            path
        )

        continue

    valid, reason = verify_nifti(
        path
    )

    if valid:

        valid_outputs_present += 1

    else:

        invalid_outputs_present += 1

        print(
            "Invalid output:",
            os.path.basename(path),
            "->",
            reason,
        )


print(
    "Selected subjects:",
    len(selected_acquisitions),
)

print(
    "Expected current N4 volumes:",
    len(expected_outputs),
)

print(
    "Valid current N4 volumes present:",
    valid_outputs_present,
)

print(
    "Invalid current N4 volumes:",
    invalid_outputs_present,
)

print(
    "Missing current N4 volumes:",
    len(missing_outputs),
)


if missing_outputs:

    print(
        "\nMissing N4 outputs:"
    )

    for path in missing_outputs:

        print(
            "  ",
            os.path.basename(path),
        )


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("N4 CORRECTION SUMMARY")
print("=" * 80)

print(
    "Selected subjects       :",
    len(selected_acquisitions),
)

print(
    "Newly processed         :",
    processed,
)

print(
    "Existing valid skipped  :",
    skipped,
)

print(
    "Valid outputs present   :",
    valid_outputs_present,
)

print(
    "Invalid outputs         :",
    invalid_outputs_present,
)

print(
    "Missing outputs         :",
    len(missing_outputs),
)

print(
    "Failed subjects         :",
    len(failed_subjects),
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
    and valid_outputs_present == len(expected_outputs)
    and invalid_outputs_present == 0
    and not missing_outputs
):

    print(
        "STATUS: N4 COMPLETED SUCCESSFULLY"
    )

elif failed_subjects:

    print(
        "STATUS: N4 COMPLETED WITH FAILURES"
    )

else:

    print(
        "STATUS: N4 OUTPUT COUNT REQUIRES REVIEW"
    )

print("=" * 80)

print(
    "\nOriginal reoriented NIfTIs were not modified."
)

print(
    "N4 was applied only to the acquisitions selected "
    "in the MPRAGE manifest."
)

print(
    "\nN4 outputs are located under:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDo NOT run HD-BET until this N4 step "
    "has been verified."
)

print("=" * 80)