"""
NeuroFusion AI
OpenNeuro ds000115 - MNI152 2 mm Affine Registration

This script is intentionally compatible with the SimpleITK 2.5.x API
installed in the NeuroFusion environment.

Pipeline position:
    Reorientation -> N4 -> HD-BET -> Normalization
    -> MNI152 registration
    -> VBM

This script ONLY performs MNI152 affine registration.
It does NOT perform VBM.

Important fixes compared with the previous versions:
1. Does NOT call SetMetricSamplingReproducibilitySeed()
   because that method is not available in the installed SimpleITK.
2. Uses SetShrinkFactorsPerLevel(), not SetShrinkFactors().
3. Does NOT call sitk.Version_CompiledDate(), which is not available.
4. Uses deterministic REGULAR metric sampling instead of RANDOM sampling.
5. Explicitly casts images to Float32 before registration.
6. Validates every output against the exact project MNI geometry.
7. Re-runs incomplete/corrupt previous outputs automatically.
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(".")

DATASET_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "openneuro_ds000115"
)

INPUT_ROOT = (
    DATASET_ROOT
    / "preprocessed"
    / "skull_stripped"
)

MNI_TEMPLATE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "mni"
    / "mni152_template_2mm.nii.gz"
)

OUTPUT_ROOT = DATASET_ROOT / "registered"

TRANSFORM_ROOT = DATASET_ROOT / "transforms"

MANIFEST = DATASET_ROOT / "openneuro_smri_manifest.csv"

REPORT_FILE = DATASET_ROOT / "mni_registration_report.csv"


# ======================================================================
# EXPECTED PROJECT MNI152 2 mm GEOMETRY
# ======================================================================

EXPECTED_SIZE = (99, 117, 95)

EXPECTED_SPACING = (2.0, 2.0, 2.0)

EXPECTED_ORIGIN = (98.0, 134.0, -72.0)

EXPECTED_DIRECTION = (
    -1.0, 0.0, 0.0,
     0.0, -1.0, 0.0,
     0.0, 0.0, 1.0,
)


# ======================================================================
# REGISTRATION PARAMETERS
# ======================================================================

# Mattes mutual information
MI_HISTOGRAM_BINS = 50

# Deterministic regular sampling.
# We deliberately do NOT use RANDOM sampling because the installed
# SimpleITK version does not expose SetMetricSamplingReproducibilitySeed.
METRIC_SAMPLING_PERCENTAGE = 0.20

# Multi-resolution registration
SHRINK_FACTORS = [4, 2, 1]
SMOOTHING_SIGMAS = [2, 1, 0]

# Gradient descent
LEARNING_RATE = 1.0
NUMBER_OF_ITERATIONS = 100
CONVERGENCE_MINIMUM_VALUE = 1e-6
CONVERGENCE_WINDOW_SIZE = 10


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def approximately_equal_tuple(a, b, tolerance=1e-5):
    if len(a) != len(b):
        return False

    return all(
        abs(float(x) - float(y)) <= tolerance
        for x, y in zip(a, b)
    )


def print_sitk_api_check():
    """Print the exact registration API capabilities being used."""

    registration = sitk.ImageRegistrationMethod()

    required_methods = [
        "SetMetricAsMattesMutualInformation",
        "SetMetricSamplingStrategy",
        "SetMetricSamplingPercentage",
        "SetInterpolator",
        "SetShrinkFactorsPerLevel",
        "SetSmoothingSigmasPerLevel",
        "SetSmoothingSigmasAreSpecifiedInPhysicalUnits",
        "SetOptimizerAsGradientDescent",
        "SetOptimizerScalesFromPhysicalShift",
        "SetInitialTransform",
    ]

    missing = [
        method
        for method in required_methods
        if not hasattr(registration, method)
    ]

    print("\nSimpleITK registration API check:")

    if missing:
        print("MISSING METHODS:")
        for method in missing:
            print(f"  - {method}")

        raise RuntimeError(
            "Installed SimpleITK does not provide the required "
            "registration API."
        )

    for method in required_methods:
        print(f"  {method}: OK")

    print("Registration API check: PASS")


def validate_image_finite(image, name):
    """Validate image contents and return basic statistics."""

    array = sitk.GetArrayFromImage(image)

    if array.size == 0:
        raise ValueError(f"{name} contains zero voxels.")

    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())

    if nan_count:
        raise ValueError(
            f"{name} contains {nan_count} NaN voxels."
        )

    if inf_count:
        raise ValueError(
            f"{name} contains {inf_count} Inf voxels."
        )

    if not np.any(array != 0):
        raise ValueError(
            f"{name} appears to be completely empty."
        )

    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "nan": nan_count,
        "inf": inf_count,
    }


# ======================================================================
# MNI TEMPLATE VALIDATION
# ======================================================================

def validate_mni_template(template_path):
    print("\n" + "=" * 70)
    print("VALIDATING MNI152 TEMPLATE")
    print("=" * 70)

    if not template_path.exists():
        raise FileNotFoundError(
            f"MNI template not found:\n{template_path.resolve()}"
        )

    template = sitk.ReadImage(str(template_path))

    print(f"\nTemplate: {template_path}")
    print(f"Size:      {template.GetSize()}")
    print(f"Spacing:   {template.GetSpacing()}")
    print(f"Origin:    {template.GetOrigin()}")
    print(f"Direction: {template.GetDirection()}")

    if template.GetSize() != EXPECTED_SIZE:
        raise ValueError(
            f"MNI template size mismatch.\n"
            f"Expected: {EXPECTED_SIZE}\n"
            f"Found:    {template.GetSize()}"
        )

    if not approximately_equal_tuple(
        template.GetSpacing(),
        EXPECTED_SPACING,
    ):
        raise ValueError(
            f"MNI template spacing mismatch.\n"
            f"Expected: {EXPECTED_SPACING}\n"
            f"Found:    {template.GetSpacing()}"
        )

    if not approximately_equal_tuple(
        template.GetOrigin(),
        EXPECTED_ORIGIN,
    ):
        raise ValueError(
            f"MNI template origin mismatch.\n"
            f"Expected: {EXPECTED_ORIGIN}\n"
            f"Found:    {template.GetOrigin()}"
        )

    if not approximately_equal_tuple(
        template.GetDirection(),
        EXPECTED_DIRECTION,
    ):
        raise ValueError(
            f"MNI template direction mismatch.\n"
            f"Expected: {EXPECTED_DIRECTION}\n"
            f"Found:    {template.GetDirection()}"
        )

    validate_image_finite(template, "MNI template")

    # Registration metrics work more reliably with floating point images.
    template = sitk.Cast(template, sitk.sitkFloat32)

    print("\nMNI template validation: PASS")

    return template


# ======================================================================
# REGISTERED IMAGE VALIDATION
# ======================================================================

def validate_registered_image(image_path):
    image = sitk.ReadImage(str(image_path))

    if image.GetSize() != EXPECTED_SIZE:
        raise ValueError(
            f"Registered image size mismatch.\n"
            f"Expected: {EXPECTED_SIZE}\n"
            f"Found:    {image.GetSize()}"
        )

    if not approximately_equal_tuple(
        image.GetSpacing(),
        EXPECTED_SPACING,
    ):
        raise ValueError(
            f"Registered image spacing mismatch.\n"
            f"Expected: {EXPECTED_SPACING}\n"
            f"Found:    {image.GetSpacing()}"
        )

    if not approximately_equal_tuple(
        image.GetOrigin(),
        EXPECTED_ORIGIN,
    ):
        raise ValueError(
            f"Registered image origin mismatch.\n"
            f"Expected: {EXPECTED_ORIGIN}\n"
            f"Found:    {image.GetOrigin()}"
        )

    if not approximately_equal_tuple(
        image.GetDirection(),
        EXPECTED_DIRECTION,
    ):
        raise ValueError(
            "Registered image direction mismatch."
        )

    stats = validate_image_finite(
        image,
        "Registered image",
    )

    return image, stats


# ======================================================================
# AFFINE MNI REGISTRATION
# ======================================================================

def register_to_mni(moving_image, fixed_image):
    """
    Perform 3D affine registration:

        moving = subject skull-stripped T1
        fixed  = MNI152 2 mm template

    The returned transform maps moving image coordinates into
    fixed/template space.
    """

    print("Creating initial affine transform...")

    affine_transform = sitk.AffineTransform(3)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        affine_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()

    # --------------------------------------------------------------
    # Metric
    # --------------------------------------------------------------

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=MI_HISTOGRAM_BINS
    )

    # IMPORTANT:
    # REGULAR is deterministic and avoids the unavailable
    # SetMetricSamplingReproducibilitySeed() API.
    registration.SetMetricSamplingStrategy(
        registration.REGULAR
    )

    registration.SetMetricSamplingPercentage(
        METRIC_SAMPLING_PERCENTAGE
    )

    # --------------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------------

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    # --------------------------------------------------------------
    # Multi-resolution
    # --------------------------------------------------------------

    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=SHRINK_FACTORS
    )

    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=SMOOTHING_SIGMAS
    )

    registration.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(
        True
    )

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    registration.SetOptimizerAsGradientDescent(
        learningRate=LEARNING_RATE,
        numberOfIterations=NUMBER_OF_ITERATIONS,
        convergenceMinimumValue=CONVERGENCE_MINIMUM_VALUE,
        convergenceWindowSize=CONVERGENCE_WINDOW_SIZE,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    # --------------------------------------------------------------
    # Transform
    # --------------------------------------------------------------

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False,
    )

    print("Executing affine registration...")

    final_transform = registration.Execute(
        fixed_image,
        moving_image,
    )

    return final_transform, registration


# ======================================================================
# SUBJECT PROCESSING
# ======================================================================

def process_subject(subject, template):
    input_path = (
        INPUT_ROOT
        / f"{subject}_T1w_ras_n4_bet.nii.gz"
    )

    registered_path = (
        OUTPUT_ROOT
        / f"{subject}_T1w_mni.nii.gz"
    )

    transform_path = (
        TRANSFORM_ROOT
        / f"{subject}_to_MNI152_2mm.tfm"
    )

    base_result = {
        "subject_id": subject,
        "input_path": str(input_path),
        "registered_path": str(registered_path),
        "transform_path": str(transform_path),
    }

    print("\n" + "-" * 70)
    print(f"Subject: {subject}")

    # --------------------------------------------------------------
    # Input check
    # --------------------------------------------------------------

    if not input_path.exists():
        print("ERROR: input image not found.")
        print(input_path)

        return {
            **base_result,
            "status": "MISSING_INPUT",
            "error": "Input image not found.",
        }

    # --------------------------------------------------------------
    # Existing output check
    # --------------------------------------------------------------

    if registered_path.exists() and transform_path.exists():
        print("Existing MNI output found. Validating...")

        try:
            _, stats = validate_registered_image(
                registered_path
            )

            print("Existing output validation: PASS")
            print("MNI registration: already exists -> SKIP")

            return {
                **base_result,
                "shape": str(EXPECTED_SIZE),
                "spacing": str(EXPECTED_SPACING),
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "nan": stats["nan"],
                "inf": stats["inf"],
                "status": "SUCCESS_EXISTING",
                "error": "",
            }

        except Exception as exc:
            print(
                "Existing output is invalid/incomplete."
            )
            print(f"Reason: {exc}")
            print("Removing invalid output and re-running...")

            try:
                registered_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

            try:
                transform_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

    elif registered_path.exists() or transform_path.exists():
        print(
            "Incomplete previous output detected. "
            "Removing it before re-running..."
        )

        try:
            registered_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        try:
            transform_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

    # --------------------------------------------------------------
    # Load moving image
    # --------------------------------------------------------------

    print("Loading skull-stripped image...")

    try:
        moving_original = sitk.ReadImage(
            str(input_path)
        )

        stats = validate_image_finite(
            moving_original,
            "Moving image",
        )

        print(
            f"Moving size: {moving_original.GetSize()}"
        )

        print(
            f"Moving spacing: "
            f"{moving_original.GetSpacing()}"
        )

        print(
            f"Moving origin: "
            f"{moving_original.GetOrigin()}"
        )

        print(
            f"Moving direction: "
            f"{moving_original.GetDirection()}"
        )

        print(
            f"Moving intensity range: "
            f"{stats['min']:.4g} to {stats['max']:.4g}"
        )

        print("Moving image validation: PASS")

    except Exception as exc:
        print(f"Input validation ERROR: {exc}")

        return {
            **base_result,
            "status": "INPUT_VALIDATION_FAILED",
            "error": str(exc),
        }

    # --------------------------------------------------------------
    # Cast to float32 for registration
    # --------------------------------------------------------------

    moving = sitk.Cast(
        moving_original,
        sitk.sitkFloat32,
    )

    # --------------------------------------------------------------
    # Affine registration
    # --------------------------------------------------------------

    print("\nAffine registration: running...")

    try:
        final_transform, registration = register_to_mni(
            moving_image=moving,
            fixed_image=template,
        )

        print("Affine registration: DONE")

        try:
            metric_value = float(
                registration.GetMetricValue()
            )
        except Exception:
            metric_value = np.nan

        print(
            f"Final metric value: {metric_value}"
        )

    except Exception as exc:
        print(
            f"\nMNI registration ERROR: {exc}"
        )
        traceback.print_exc()

        return {
            **base_result,
            "status": "REGISTRATION_FAILED",
            "error": str(exc),
        }

    # --------------------------------------------------------------
    # Resample into exact MNI template geometry
    # --------------------------------------------------------------

    print("Resampling subject into MNI152 2 mm space...")

    try:
        registered = sitk.Resample(
            moving,
            template,
            final_transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

        # Force exact metadata from template.
        registered.CopyInformation(template)

        print(
            f"Registered size: {registered.GetSize()}"
        )

        print(
            f"Registered spacing: "
            f"{registered.GetSpacing()}"
        )

    except Exception as exc:
        print(
            f"Resampling ERROR: {exc}"
        )

        return {
            **base_result,
            "status": "RESAMPLING_FAILED",
            "error": str(exc),
        }

    # --------------------------------------------------------------
    # Validate before writing
    # --------------------------------------------------------------

    try:
        temp_registered_path = (
            OUTPUT_ROOT
            / f"{subject}_T1w_mni.tmp.nii.gz"
        )

        sitk.WriteImage(
            registered,
            str(temp_registered_path),
            useCompression=True,
        )

        # Read back from disk so validation checks the actual file.
        saved_image = sitk.ReadImage(
            str(temp_registered_path)
        )

        if saved_image.GetSize() != EXPECTED_SIZE:
            raise ValueError(
                "Temporary registered image has wrong size."
            )

        if not approximately_equal_tuple(
            saved_image.GetSpacing(),
            EXPECTED_SPACING,
        ):
            raise ValueError(
                "Temporary registered image has wrong spacing."
            )

        if not approximately_equal_tuple(
            saved_image.GetOrigin(),
            EXPECTED_ORIGIN,
        ):
            raise ValueError(
                "Temporary registered image has wrong origin."
            )

        if not approximately_equal_tuple(
            saved_image.GetDirection(),
            EXPECTED_DIRECTION,
        ):
            raise ValueError(
                "Temporary registered image has wrong direction."
            )

        saved_stats = validate_image_finite(
            saved_image,
            "Temporary registered image",
        )

        print("MNI geometry validation: PASS")

    except Exception as exc:
        print(
            f"Output pre-validation ERROR: {exc}"
        )

        try:
            temp_registered_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return {
            **base_result,
            "status": "OUTPUT_VALIDATION_FAILED",
            "error": str(exc),
        }

    # --------------------------------------------------------------
    # Save transform and final output
    # --------------------------------------------------------------

    try:
        sitk.WriteTransform(
            final_transform,
            str(transform_path),
        )

        # Atomic-ish replacement: the validated temporary image becomes
        # the final image only after validation has passed.
        if registered_path.exists():
            registered_path.unlink()

        temp_registered_path.replace(
            registered_path
        )

        print("Transformation saved.")
        print("Registered image saved.")

    except Exception as exc:
        print(
            f"Saving output ERROR: {exc}"
        )

        try:
            temp_registered_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return {
            **base_result,
            "status": "OUTPUT_SAVE_FAILED",
            "error": str(exc),
        }

    # --------------------------------------------------------------
    # Final validation of actual final files
    # --------------------------------------------------------------

    print("Final validation: running...")

    try:
        _, final_stats = validate_registered_image(
            registered_path
        )

        if not transform_path.exists():
            raise FileNotFoundError(
                "Transform file was not created."
            )

        print(
            f"Validation: shape={EXPECTED_SIZE} "
            f"spacing={list(EXPECTED_SPACING)} "
            f"NaN={final_stats['nan']} "
            f"Inf={final_stats['inf']}"
        )

        print("STATUS: SUCCESS")

        return {
            **base_result,
            "shape": str(EXPECTED_SIZE),
            "spacing": str(EXPECTED_SPACING),
            "min": final_stats["min"],
            "max": final_stats["max"],
            "mean": final_stats["mean"],
            "std": final_stats["std"],
            "nan": final_stats["nan"],
            "inf": final_stats["inf"],
            "status": "SUCCESS",
            "error": "",
        }

    except Exception as exc:
        print(
            f"Final validation ERROR: {exc}"
        )

        return {
            **base_result,
            "status": "OUTPUT_VALIDATION_FAILED",
            "error": str(exc),
        }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("NeuroFusion AI - OpenNeuro MNI152 Registration")
    print("=" * 70)

    print("\nSimpleITK:")
    print(sitk.Version())

    print("\nPython-compatible registration API will be checked.")

    try:
        print_sitk_api_check()
    except Exception as exc:
        print("\nERROR: SimpleITK API check failed.")
        print(exc)
        sys.exit(1)

    print("\nInput root:")
    print(INPUT_ROOT.resolve())

    print("\nMNI template:")
    print(MNI_TEMPLATE.resolve())

    print("\nOutput root:")
    print(OUTPUT_ROOT.resolve())

    print("\nTransform root:")
    print(TRANSFORM_ROOT.resolve())

    # --------------------------------------------------------------
    # Required paths
    # --------------------------------------------------------------

    if not INPUT_ROOT.exists():
        print(
            "\nERROR: skull-stripped input directory not found."
        )
        print(INPUT_ROOT.resolve())
        sys.exit(1)

    if not MANIFEST.exists():
        print(
            "\nERROR: OpenNeuro manifest not found."
        )
        print(MANIFEST.resolve())
        sys.exit(1)

    if not MNI_TEMPLATE.exists():
        print(
            "\nERROR: MNI152 template not found."
        )
        print(MNI_TEMPLATE.resolve())
        sys.exit(1)

    # --------------------------------------------------------------
    # Validate template
    # --------------------------------------------------------------

    try:
        template = validate_mni_template(
            MNI_TEMPLATE
        )
    except Exception as exc:
        print(
            "\nMNI TEMPLATE VALIDATION FAILED:"
        )
        print(exc)
        sys.exit(1)

    # --------------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFORM_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Read manifest
    # --------------------------------------------------------------

    try:
        df = pd.read_csv(
            MANIFEST
        )
    except Exception as exc:
        print(
            f"\nERROR: Could not read manifest: {exc}"
        )
        sys.exit(1)

    if "subject_id" not in df.columns:
        print(
            "\nERROR: manifest must contain a "
            "'subject_id' column."
        )
        print(
            f"Available columns: {list(df.columns)}"
        )
        sys.exit(1)

    subjects = (
        df["subject_id"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    subjects = [
        subject
        for subject in subjects
        if subject
    ]

    if not subjects:
        print(
            "\nERROR: no subjects found in manifest."
        )
        sys.exit(1)

    print(
        f"\nSubjects in manifest: {len(subjects)}"
    )

    # --------------------------------------------------------------
    # Process subjects
    # --------------------------------------------------------------

    results = []

    for index, subject in enumerate(
        subjects,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"[{index}/{len(subjects)}] {subject}"
        )
        print("=" * 70)

        result = process_subject(
            subject,
            template,
        )

        results.append(result)

    # --------------------------------------------------------------
    # Save report
    # --------------------------------------------------------------

    report = pd.DataFrame(
        results
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MNI REGISTRATION COMPLETE")
    print("=" * 70)

    print(
        f"\nReport:\n"
        f"{REPORT_FILE.resolve()}"
    )

    if not report.empty and "status" in report.columns:

        print("\nStatus:")
        print(
            report["status"]
            .value_counts()
        )

        successful = int(
            report["status"]
            .isin(
                [
                    "SUCCESS",
                    "SUCCESS_EXISTING",
                ]
            )
            .sum()
        )

        total = len(subjects)

        print(
            f"\nSuccessful subjects: "
            f"{successful}/{total}"
        )

        if successful == total:
            print(
                "\nOVERALL STATUS: PASS"
            )
            print(
                "All subjects are registered in "
                "MNI152 2 mm space."
            )
        else:
            print(
                "\nOVERALL STATUS: CHECK FAILED SUBJECTS"
            )

            failed = report[
                ~report["status"].isin(
                    [
                        "SUCCESS",
                        "SUCCESS_EXISTING",
                    ]
                )
            ]

            if not failed.empty:
                print("\nFailed subjects:")
                print(
                    failed[
                        [
                            "subject_id",
                            "status",
                            "error",
                        ]
                    ].to_string(index=False)
                )

    print(
        f"\nRegistered images:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        f"\nTransforms:\n"
        f"{TRANSFORM_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()
