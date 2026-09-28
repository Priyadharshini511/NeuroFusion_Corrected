# src/external_register_mni.py

from pathlib import Path

import SimpleITK as sitk
import numpy as np


# =============================================================================
# NeuroFusion AI
# External ds004302 sMRI -> MNI152 Registration
#
# Pipeline:
#
#   ds004302 normalized T1
#          ↓
#   affine registration to MNI152 2mm
#          ↓
#   MNI-space T1
#          ↓
#   exact MNI-grid validation
#
# Pilot subjects:
#   sub-01
#   sub-26
#   sub-55
#
# IMPORTANT:
#   - This script is ONLY for the external ds004302 dataset.
#   - It does NOT modify COBRE/training data.
#   - It does NOT perform AAL extraction.
#   - It does NOT perform model prediction.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

INPUT_DIR = Path(
    "processed_data/smri/external_test/ds004302/pilot_normalized"
)

MNI_PATH = Path(
    "processed_data/smri/mni/mni152_template_2mm.nii.gz"
)

OUTPUT_DIR = Path(
    "processed_data/smri/external_test/ds004302/pilot_mni"
)

TRANSFORM_DIR = Path(
    "processed_data/smri/external_test/ds004302/pilot_transforms"
)


# =============================================================================
# SETTINGS
# =============================================================================

VOXEL_TOLERANCE = 0.01
GEOMETRY_TOLERANCE = 1e-4

MAX_ITERATIONS = 200
NUMBER_OF_HISTOGRAM_BINS = 50
SAMPLING_PERCENTAGE = 0.20

RANDOM_SEED = 42


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRANSFORM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("NeuroFusion AI - EXTERNAL ds004302 MNI152 REGISTRATION")
print("=" * 80)

print()
print("Input directory:")
print(f"  {INPUT_DIR.resolve()}")

print()
print("MNI template:")
print(f"  {MNI_PATH.resolve()}")

print()
print("Output directory:")
print(f"  {OUTPUT_DIR.resolve()}")

print()
print("Transform directory:")
print(f"  {TRANSFORM_DIR.resolve()}")


# =============================================================================
# REQUIRED PATH CHECKS
# =============================================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(
        f"External normalized input directory not found:\n"
        f"{INPUT_DIR.resolve()}"
    )


if not MNI_PATH.exists():

    raise FileNotFoundError(
        f"MNI152 template not found:\n"
        f"{MNI_PATH.resolve()}"
    )


# =============================================================================
# LOAD MNI TEMPLATE
# =============================================================================

print()
print("-" * 80)
print("LOADING MNI152 TEMPLATE")
print("-" * 80)

fixed = sitk.ReadImage(
    str(MNI_PATH),
    sitk.sitkFloat32
)


MNI_SIZE = fixed.GetSize()
MNI_SPACING = fixed.GetSpacing()
MNI_ORIGIN = fixed.GetOrigin()
MNI_DIRECTION = fixed.GetDirection()


print()
print("MNI template information:")

print(
    f"  Size      : {MNI_SIZE}"
)

print(
    f"  Spacing   : {MNI_SPACING}"
)

print(
    f"  Origin    : {MNI_ORIGIN}"
)

print(
    f"  Direction : {MNI_DIRECTION}"
)


# =============================================================================
# VALIDATE MNI TEMPLATE
# =============================================================================

def validate_template(image):

    if image.GetDimension() != 3:

        raise RuntimeError(
            "MNI template is not 3D."
        )

    array = sitk.GetArrayViewFromImage(
        image
    )

    if array.size == 0:

        raise RuntimeError(
            "MNI template is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):

        raise RuntimeError(
            "MNI template contains non-finite values."
        )

    if np.count_nonzero(array) == 0:

        raise RuntimeError(
            "MNI template contains no non-zero voxels."
        )


validate_template(fixed)

print(
    "\nMNI template numerical validation: PASS"
)


# =============================================================================
# VALIDATE INPUT
# =============================================================================

def validate_input(image):

    if image.GetDimension() != 3:

        return False, (
            f"Expected 3D image, "
            f"got {image.GetDimension()}D"
        )

    array = sitk.GetArrayViewFromImage(
        image
    )

    if array.size == 0:

        return False, "Image is empty"

    if not np.all(
        np.isfinite(array)
    ):

        return False, (
            "Image contains non-finite values"
        )

    if np.count_nonzero(array) == 0:

        return False, (
            "Image contains no non-zero voxels"
        )

    spacing = np.asarray(
        image.GetSpacing(),
        dtype=float
    )

    # External normalized images should still be approximately
    # 1 mm isotropic before MNI registration.

    if not np.allclose(
        spacing,
        [1.0, 1.0, 1.0],
        atol=VOXEL_TOLERANCE
    ):

        return False, (
            f"Unexpected spacing: {tuple(spacing)}"
        )

    return True, "VALID"


# =============================================================================
# VALIDATE MNI OUTPUT
# =============================================================================

def validate_mni_output(
    image
):

    if image.GetDimension() != 3:

        return False, (
            "Output is not 3D"
        )

    array = sitk.GetArrayViewFromImage(
        image
    )

    if array.size == 0:

        return False, (
            "Output is empty"
        )

    if not np.all(
        np.isfinite(array)
    ):

        return False, (
            "Output contains non-finite values"
        )

    if np.count_nonzero(array) == 0:

        return False, (
            "Output contains no non-zero voxels"
        )

    # -------------------------------------------------------------------------
    # Exact MNI grid
    # -------------------------------------------------------------------------

    if image.GetSize() != MNI_SIZE:

        return False, (
            f"Size mismatch: "
            f"{image.GetSize()} != {MNI_SIZE}"
        )

    if not np.allclose(
        image.GetSpacing(),
        MNI_SPACING,
        atol=VOXEL_TOLERANCE
    ):

        return False, (
            "Spacing mismatch with MNI template"
        )

    if not np.allclose(
        image.GetOrigin(),
        MNI_ORIGIN,
        atol=GEOMETRY_TOLERANCE
    ):

        return False, (
            "Origin mismatch with MNI template"
        )

    if not np.allclose(
        image.GetDirection(),
        MNI_DIRECTION,
        atol=GEOMETRY_TOLERANCE
    ):

        return False, (
            "Direction mismatch with MNI template"
        )

    return True, "VALID"


# =============================================================================
# GET INPUT SUBJECTS
# =============================================================================

input_files = sorted(
    INPUT_DIR.rglob(
        "*_normalized.nii.gz"
    )
)


if not input_files:

    raise RuntimeError(
        "No normalized external NIfTI files found."
    )


print()
print("-" * 80)
print("EXTERNAL INPUT INVENTORY")
print("-" * 80)

print(
    f"\nNormalized files found: {len(input_files)}"
)

for path in input_files:

    print(
        f"  {path}"
    )


# =============================================================================
# REGISTRATION FUNCTION
# =============================================================================

def register_to_mni(
    moving,
    fixed
):

    # -------------------------------------------------------------------------
    # Initial affine transform
    # -------------------------------------------------------------------------

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # -------------------------------------------------------------------------
    # Registration method
    # -------------------------------------------------------------------------

    registration = sitk.ImageRegistrationMethod()

    # Mutual information is appropriate for intensity-based
    # registration between subject MRI and template MRI.

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=NUMBER_OF_HISTOGRAM_BINS
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(
        SAMPLING_PERCENTAGE
    )

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.01,
        numberOfIterations=MAX_ITERATIONS,
        gradientMagnitudeTolerance=1e-6
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False
    )

    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=[4, 2, 1]
    )

    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[2, 1, 0]
    )

    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # -------------------------------------------------------------------------
    # Deterministic sampling seed
    # -------------------------------------------------------------------------

    registration.SetMetricSamplingPercentage(
        SAMPLING_PERCENTAGE
    )

    # -------------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------------

    final_transform = registration.Execute(
        fixed,
        moving
    )

    return (
        final_transform,
        registration
    )


# =============================================================================
# PROCESS EACH SUBJECT
# =============================================================================

successful = []
failed = []


print()
print("=" * 80)
print("STARTING EXTERNAL MNI REGISTRATION")
print("=" * 80)


for input_path in input_files:

    # -------------------------------------------------------------------------
    # Subject ID
    # -------------------------------------------------------------------------

    subject_id = input_path.parent.parent.name

    print()
    print("-" * 80)
    print(subject_id)
    print("-" * 80)

    print(
        f"Input: {input_path}"
    )

    try:

        # ---------------------------------------------------------------------
        # Read input
        # ---------------------------------------------------------------------

        moving = sitk.ReadImage(
            str(input_path),
            sitk.sitkFloat32
        )

        valid, message = validate_input(
            moving
        )

        if not valid:

            print(
                f"  INPUT VALIDATION: FAIL"
            )

            print(
                f"  Reason: {message}"
            )

            failed.append(
                (subject_id, message)
            )

            continue

        print(
            "  Input validation: PASS"
        )

        print(
            f"  Input size    : {moving.GetSize()}"
        )

        print(
            f"  Input spacing : {moving.GetSpacing()}"
        )

        # ---------------------------------------------------------------------
        # Output paths
        # ---------------------------------------------------------------------

        subject_output_dir = (
            OUTPUT_DIR /
            subject_id /
            "anat"
        )

        subject_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            subject_output_dir /
            f"{subject_id}_T1w_mni.nii.gz"
        )

        transform_path = (
            TRANSFORM_DIR /
            f"{subject_id}_to_MNI152_affine.tfm"
        )

        # ---------------------------------------------------------------------
        # Skip valid existing output
        # ---------------------------------------------------------------------

        if output_path.exists():

            existing = sitk.ReadImage(
                str(output_path),
                sitk.sitkFloat32
            )

            valid_existing, existing_message = (
                validate_mni_output(existing)
            )

            if valid_existing:

                print(
                    "  Existing MNI output: VALID"
                )

                successful.append(
                    subject_id
                )

                continue

            else:

                print(
                    "  Existing MNI output invalid."
                )

                print(
                    f"  Reason: {existing_message}"
                )

        # ---------------------------------------------------------------------
        # Registration
        # ---------------------------------------------------------------------

        print(
            "\n  Running affine registration..."
        )

        final_transform, registration = (
            register_to_mni(
                moving,
                fixed
            )
        )

        # ---------------------------------------------------------------------
        # Save transform
        # ---------------------------------------------------------------------

        sitk.WriteTransform(
            final_transform,
            str(transform_path)
        )

        print(
            f"  Transform saved:"
        )

        print(
            f"    {transform_path}"
        )

        # ---------------------------------------------------------------------
        # Resample into exact MNI grid
        # ---------------------------------------------------------------------

        print(
            "\n  Resampling into exact MNI grid..."
        )

        registered = sitk.Resample(
            moving,
            fixed,
            final_transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32
        )

        # ---------------------------------------------------------------------
        # Validate registered output
        # ---------------------------------------------------------------------

        valid_output, output_message = (
            validate_mni_output(
                registered
            )
        )

        if not valid_output:

            print(
                "  MNI OUTPUT VALIDATION: FAIL"
            )

            print(
                f"  Reason: {output_message}"
            )

            failed.append(
                (subject_id, output_message)
            )

            continue

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        sitk.WriteImage(
            registered,
            str(output_path),
            useCompression=True
        )

        # ---------------------------------------------------------------------
        # Reload and validate saved file
        # ---------------------------------------------------------------------

        saved = sitk.ReadImage(
            str(output_path),
            sitk.sitkFloat32
        )

        saved_valid, saved_message = (
            validate_mni_output(
                saved
            )
        )

        if not saved_valid:

            print(
                "  SAVED OUTPUT VALIDATION: FAIL"
            )

            print(
                f"  Reason: {saved_message}"
            )

            failed.append(
                (subject_id, saved_message)
            )

            continue

        # ---------------------------------------------------------------------
        # Final information
        # ---------------------------------------------------------------------

        print(
            "\n  MNI OUTPUT VALIDATION: PASS"
        )

        print(
            f"  Output size      : {saved.GetSize()}"
        )

        print(
            f"  Output spacing   : {saved.GetSpacing()}"
        )

        print(
            f"  Output origin    : {saved.GetOrigin()}"
        )

        print(
            f"  Output direction : {saved.GetDirection()}"
        )

        print(
            f"  Output:"
        )

        print(
            f"    {output_path}"
        )

        print(
            "\n  STATUS: PASS"
        )

        successful.append(
            subject_id
        )

    except Exception as exc:

        print(
            "\n  STATUS: FAIL"
        )

        print(
            f"  Error: {exc}"
        )

        failed.append(
            (subject_id, str(exc))
        )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("EXTERNAL MNI REGISTRATION SUMMARY")
print("=" * 80)

print()
print(
    f"Input subjects : {len(input_files)}"
)

print(
    f"Successful     : {len(successful)}"
)

print(
    f"Failed         : {len(failed)}"
)


if successful:

    print()
    print("Successful subjects:")

    for subject in successful:

        print(
            f"  {subject}"
        )


if failed:

    print()
    print("Failed subjects:")

    for subject, reason in failed:

        print(
            f"  {subject} | {reason}"
        )


print()

if failed:

    print(
        "STATUS: EXTERNAL MNI REGISTRATION "
        "COMPLETED WITH FAILURES"
    )

else:

    print(
        "STATUS: EXTERNAL MNI REGISTRATION "
        "COMPLETED SUCCESSFULLY"
    )

print()

print("=" * 80)