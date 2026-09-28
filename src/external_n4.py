import os
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk


# ======================================================================
# OPENNEURO EXTERNAL TEST — N4 BIAS FIELD CORRECTION
# ======================================================================

INPUT_ROOT = Path(
    "processed_data/smri/external_test/ds004302/pilot"
)

OUTPUT_ROOT = Path(
    "processed_data/smri/external_test/ds004302/pilot_n4"
)

SUBJECTS = [
    "sub-01",
    "sub-26",
    "sub-55",
]


# ======================================================================
# SETTINGS
# ======================================================================

EXPECTED_ORIENTATION = (
    "R",
    "A",
    "S",
)

EXPECTED_VOXEL_SIZE = np.array(
    [1.0, 1.0, 1.0],
    dtype=float,
)

VOXEL_TOLERANCE = 0.01


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("OPENNEURO EXTERNAL TEST — N4 BIAS FIELD CORRECTION")
print("=" * 80)

print()
print("Input root:")
print(INPUT_ROOT.resolve())

print()
print("Output root:")
print(OUTPUT_ROOT.resolve())


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# VERIFY INPUT
# ======================================================================

def verify_input(path):

    try:

        image = nib.load(path)

        data = image.get_fdata(
            dtype=np.float32
        )

        if len(image.shape) != 3:
            return (
                False,
                f"Expected 3D image, got {image.shape}",
            )

        orientation = nib.aff2axcodes(
            image.affine
        )

        if orientation != EXPECTED_ORIENTATION:

            return (
                False,
                f"Expected RAS, got {orientation}",
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
                f"Expected ~1 mm voxels, got {voxel_size}",
            )

        if not np.all(
            np.isfinite(image.affine)
        ):

            return (
                False,
                "Affine contains NaN/Inf",
            )

        if not np.all(
            np.isfinite(data)
        ):

            return (
                False,
                "Image contains NaN/Inf",
            )

        if not np.any(
            data != 0
        ):

            return (
                False,
                "Image contains only zeros",
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
# VERIFY OUTPUT
# ======================================================================

def verify_output(path):

    try:

        image = nib.load(path)

        data = image.get_fdata(
            dtype=np.float32
        )

        orientation = nib.aff2axcodes(
            image.affine
        )

        voxel_size = np.asarray(
            image.header.get_zooms()[:3],
            dtype=float,
        )

        if len(image.shape) != 3:
            return False, f"Not 3D: {image.shape}"

        if orientation != EXPECTED_ORIENTATION:
            return False, f"Orientation: {orientation}"

        if not np.allclose(
            voxel_size,
            EXPECTED_VOXEL_SIZE,
            atol=VOXEL_TOLERANCE,
        ):
            return False, f"Voxel size: {voxel_size}"

        if not np.all(
            np.isfinite(data)
        ):
            return False, "NaN/Inf in output"

        if not np.any(
            data != 0
        ):
            return False, "Output contains only zeros"

        if not np.all(
            np.isfinite(image.affine)
        ):
            return False, "Invalid affine"

        return True, "PASS"

    except Exception as e:

        return False, str(e)


# ======================================================================
# PROCESS ONE SUBJECT
# ======================================================================

def process_subject(subject):

    input_path = (
        INPUT_ROOT
        / subject
        / "anat"
        / f"{subject}_T1w.nii.gz"
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

    output_path = (
        subject_output_dir
        / f"{subject}_T1w_n4.nii.gz"
    )

    print()
    print("-" * 80)
    print("SUBJECT:", subject)
    print("-" * 80)

    print("Input:")
    print(" ", input_path)

    print("Output:")
    print(" ", output_path)

    # --------------------------------------------------------------
    # Input verification
    # --------------------------------------------------------------

    valid, reason = verify_input(
        input_path
    )

    if not valid:

        raise RuntimeError(
            f"Input verification failed: {reason}"
        )

    original = nib.load(
        input_path
    )

    print()
    print("Input shape:")
    print(" ", original.shape)

    print("Input voxel size:")
    print(
        " ",
        original.header.get_zooms()[:3]
    )

    print("Input orientation:")
    print(
        " ",
        nib.aff2axcodes(
            original.affine
        )
    )

    # --------------------------------------------------------------
    # Existing output
    # --------------------------------------------------------------

    if output_path.exists():

        valid, reason = verify_output(
            output_path
        )

        if valid:

            print()
            print(
                "Existing N4 output verified: PASS"
            )

            return "SKIPPED"

        print()
        print(
            "Existing N4 output invalid:",
            reason,
        )

        output_path.unlink()

    # --------------------------------------------------------------
    # Read with SimpleITK
    # --------------------------------------------------------------

    print()
    print("Reading image with SimpleITK...")

    image = sitk.ReadImage(
        str(input_path)
    )

    print(
        "SimpleITK size:",
        image.GetSize()
    )

    print(
        "SimpleITK spacing:",
        image.GetSpacing()
    )

    # --------------------------------------------------------------
    # Float32
    # --------------------------------------------------------------

    image = sitk.Cast(
        image,
        sitk.sitkFloat32,
    )

    # --------------------------------------------------------------
    # N4 mask
    # --------------------------------------------------------------

    print(
        "Creating Otsu N4 mask..."
    )

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

    # --------------------------------------------------------------
    # N4
    # --------------------------------------------------------------

    print()
    print(
        "Running N4 bias correction..."
    )

    corrector = (
        sitk.N4BiasFieldCorrectionImageFilter()
    )

    corrector.SetMaximumNumberOfIterations(
        [30, 20, 10]
    )

    corrector.SetNumberOfControlPoints(
        [4, 4, 4]
    )

    corrected = corrector.Execute(
        image,
        mask,
    )

    corrected = sitk.Cast(
        corrected,
        sitk.sitkFloat32,
    )

    corrected.CopyInformation(
        image
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    print()
    print(
        "Saving N4 output..."
    )

    sitk.WriteImage(
        corrected,
        str(output_path),
    )

    # --------------------------------------------------------------
    # Verify
    # --------------------------------------------------------------

    valid, reason = verify_output(
        output_path
    )

    if not valid:

        try:
            output_path.unlink()
        except Exception:
            pass

        raise RuntimeError(
            f"N4 output verification failed: {reason}"
        )

    result = nib.load(
        output_path
    )

    print()
    print(
        "Output shape:"
    )
    print(
        " ",
        result.shape
    )

    print(
        "Output voxel size:"
    )
    print(
        " ",
        result.header.get_zooms()[:3]
    )

    print(
        "Output orientation:"
    )
    print(
        " ",
        nib.aff2axcodes(
            result.affine
        )
    )

    data = result.get_fdata(
        dtype=np.float32
    )

    print(
        "Output intensity min:",
        float(data.min())
    )

    print(
        "Output intensity max:",
        float(data.max())
    )

    print(
        "Output intensity mean:",
        float(data.mean())
    )

    print()
    print(
        "STATUS: PASS"
    )

    return "PROCESSED"


# ======================================================================
# MAIN
# ======================================================================

processed = 0
skipped = 0
failed = []

for subject in SUBJECTS:

    try:

        status = process_subject(
            subject
        )

        if status == "PROCESSED":
            processed += 1

        elif status == "SKIPPED":
            skipped += 1

    except Exception as e:

        print()
        print(
            "ERROR:",
            e
        )

        failed.append(
            subject
        )


# ======================================================================
# SUMMARY
# ======================================================================

print()
print("=" * 80)
print("EXTERNAL N4 SUMMARY")
print("=" * 80)

print(
    "Subjects:",
    len(SUBJECTS)
)

print(
    "Processed:",
    processed
)

print(
    "Skipped:",
    skipped
)

print(
    "Failed:",
    len(failed)
)

if failed:

    print()
    print("Failed subjects:")

    for subject in failed:
        print(
            " ",
            subject
        )

print()
print(
    "Output directory:"
)

print(
    OUTPUT_ROOT.resolve()
)

print()
print("=" * 80)

if not failed:

    print(
        "STATUS: EXTERNAL N4 COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "STATUS: EXTERNAL N4 COMPLETED WITH FAILURES"
    )

print("=" * 80)