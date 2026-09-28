

# src/batch_register_mni.py

from pathlib import Path
import csv

import SimpleITK as sitk
import numpy as np


# ==============================================================
# GENERIC sMRI BATCH MNI152 AFFINE REGISTRATION
#
# Supports:
#
#   RMS MPRAGE
#       normalized RMS
#           ↓
#       affine registration to MNI152
#           ↓
#       one MNI RMS volume
#
#   5-ECHO MPRAGE
#       Echo 1 normalized
#           ↓
#       affine registration to MNI152
#           ↓
#       Echo 1 MNI
#           +
#       SAME affine transform
#           ↓
#       Echoes 2-5 MNI
#
# SUBJECT SELECTION:
#   Uses mprage_selection_manifest.csv
#
# IMPORTANT:
#   - One selected acquisition per subject.
#   - RMS produces one MNI volume.
#   - 5ECHO produces five MNI volumes.
#   - Echoes 2-5 never receive independent registration.
#   - Original normalized inputs are never modified.
#   - Existing valid outputs are skipped.
# ==============================================================


# ==============================================================
# PATHS
# ==============================================================

INPUT_DIR = Path(
    "processed_data/smri/mprage_5e_normalized"
)

MNI_PATH = Path(
    "processed_data/smri/mni/mni152_template_2mm.nii.gz"
)

OUTPUT_DIR = Path(
    "processed_data/smri/registered"
)

TRANSFORM_DIR = Path(
    "processed_data/smri/transforms"
)

MANIFEST_FILE = Path(
    "processed_data/smri/mprage_selection/"
    "mprage_selection_manifest.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRANSFORM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# SETTINGS
# ==============================================================

ECHOES = [1, 2, 3, 4, 5]

VOXEL_TOLERANCE = 0.01

GEOMETRY_TOLERANCE = 1e-4

NONEMPTY_MIN_VOXELS = 1


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("GENERIC sMRI BATCH MNI152 AFFINE REGISTRATION")
print("=" * 70)

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

print()
print("Selection manifest:")
print(f"  {MANIFEST_FILE.resolve()}")


# ==============================================================
# REQUIRED PATH CHECKS
# ==============================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(
        f"Normalized input directory does not exist:\n"
        f"{INPUT_DIR.resolve()}"
    )


if not MNI_PATH.exists():

    raise FileNotFoundError(
        f"MNI152 template was not found:\n"
        f"{MNI_PATH.resolve()}"
    )


if not MANIFEST_FILE.exists():

    raise FileNotFoundError(
        f"Selection manifest was not found:\n"
        f"{MANIFEST_FILE.resolve()}"
    )


# ==============================================================
# LOAD SELECTED ACQUISITIONS
# ==============================================================

def load_selected_acquisitions(manifest_file):
    """
    Load exactly one SELECTED acquisition per subject.

    Required manifest columns:
        subject
        selection_status

    acquisition_type must be:
        RMS
        5ECHO
    """

    selected = []

    with manifest_file.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:

            raise ValueError(
                "Selection manifest has no header."
            )

        required_columns = {
            "subject",
            "selection_status"
        }

        missing = (
            required_columns -
            set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                "Selection manifest is missing required "
                "columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            status = str(
                row.get(
                    "selection_status",
                    ""
                )
            ).strip().upper()

            if status != "SELECTED":
                continue

            subject = str(
                row.get(
                    "subject",
                    ""
                )
            ).strip()

            if not subject:

                raise ValueError(
                    "A SELECTED manifest row has no "
                    "subject ID."
                )

            acquisition_type = str(
                row.get(
                    "acquisition_type",
                    ""
                )
            ).strip().upper()

            if acquisition_type not in {
                "RMS",
                "5ECHO"
            }:

                raise ValueError(
                    f"Unsupported acquisition type "
                    f"for {subject}: "
                    f"{acquisition_type}"
                )

            selected.append(
                {
                    "subject": subject,
                    "acquisition_type": acquisition_type,
                    "session": str(
                        row.get(
                            "session",
                            ""
                        )
                    ).strip(),
                    "series": str(
                        row.get(
                            "series",
                            ""
                        )
                    ).strip()
                }
            )

    if not selected:

        raise ValueError(
            "No SELECTED acquisitions were found "
            "in the selection manifest."
        )

    subjects = [
        item["subject"]
        for item in selected
    ]

    if len(subjects) != len(set(subjects)):

        duplicates = sorted(
            {
                subject
                for subject in subjects
                if subjects.count(subject) > 1
            }
        )

        raise ValueError(
            "Duplicate selected subjects found:\n"
            + "\n".join(duplicates)
        )

    return sorted(
        selected,
        key=lambda x: x["subject"]
    )


selected_acquisitions = (
    load_selected_acquisitions(
        MANIFEST_FILE
    )
)


# ==============================================================
# DISPLAY CURRENT COHORT
# ==============================================================

print()
print("-" * 70)
print("SELECTED ACQUISITIONS FROM MANIFEST")
print("-" * 70)

for item in selected_acquisitions:

    print(
        f"  {item['subject']} | "
        f"{item['acquisition_type']} | "
        f"{item['session']} | "
        f"{item['series']}"
    )

print()
print(
    f"Total selected subjects: "
    f"{len(selected_acquisitions)}"
)


# ==============================================================
# ACQUISITION SUMMARY
# ==============================================================

type_counts = {
    "RMS": 0,
    "5ECHO": 0
}

expected_outputs = 0

for item in selected_acquisitions:

    acquisition_type = item[
        "acquisition_type"
    ]

    type_counts[
        acquisition_type
    ] += 1

    if acquisition_type == "RMS":

        expected_outputs += 1

    else:

        expected_outputs += len(ECHOES)


print()
print("Acquisition types:")

print(
    f"  RMS   : "
    f"{type_counts['RMS']}"
)

print(
    f"  5ECHO : "
    f"{type_counts['5ECHO']}"
)

print()
print(
    f"Expected MNI output volumes: "
    f"{expected_outputs}"
)

print(
    f"Expected transforms: "
    f"{len(selected_acquisitions)}"
)


# ==============================================================
# LOAD MNI TEMPLATE
# ==============================================================

print()
print("-" * 70)
print("LOADING MNI152 TEMPLATE")
print("-" * 70)

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
    "  Size      :",
    MNI_SIZE
)

print(
    "  Spacing   :",
    MNI_SPACING
)

print(
    "  Origin    :",
    MNI_ORIGIN
)

print(
    "  Direction :",
    MNI_DIRECTION
)


# ==============================================================
# MNI TEMPLATE VALIDATION
# ==============================================================

def validate_mni_template(image):

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

    if np.count_nonzero(array) < (
        NONEMPTY_MIN_VOXELS
    ):

        raise RuntimeError(
            "MNI template contains no non-zero voxels."
        )


validate_mni_template(
    fixed
)

print(
    "MNI template numerical validation: PASS"
)


# ==============================================================
# INPUT IMAGE VALIDATION
# ==============================================================

def validate_input_image(
    image,
    reference=None
):
    """
    Validate normalized sMRI input.

    Checks:
      - 3D
      - finite
      - non-empty
      - 1mm isotropic
      - optional geometry against reference
    """

    if image.GetDimension() != 3:

        return (
            False,
            f"expected 3D image, "
            f"got dimension "
            f"{image.GetDimension()}"
        )

    array = sitk.GetArrayViewFromImage(
        image
    )

    if array.size == 0:

        return (
            False,
            "image is empty"
        )

    if not np.all(
        np.isfinite(array)
    ):

        return (
            False,
            "image contains non-finite values"
        )

    if np.count_nonzero(array) == 0:

        return (
            False,
            "image contains no non-zero voxels"
        )

    spacing = np.asarray(
        image.GetSpacing(),
        dtype=float
    )

    if not np.allclose(
        spacing,
        [1.0, 1.0, 1.0],
        atol=VOXEL_TOLERANCE
    ):

        return (
            False,
            f"unexpected input spacing: "
            f"{tuple(spacing)}"
        )

    if reference is not None:

        if image.GetSize() != (
            reference.GetSize()
        ):

            return (
                False,
                f"size mismatch: "
                f"{image.GetSize()} != "
                f"{reference.GetSize()}"
            )

        if not np.allclose(
            image.GetSpacing(),
            reference.GetSpacing(),
            atol=VOXEL_TOLERANCE
        ):

            return (
                False,
                "spacing mismatch with reference"
            )

        if not np.allclose(
            image.GetOrigin(),
            reference.GetOrigin(),
            atol=GEOMETRY_TOLERANCE
        ):

            return (
                False,
                "origin mismatch with reference"
            )

        if not np.allclose(
            image.GetDirection(),
            reference.GetDirection(),
            atol=GEOMETRY_TOLERANCE
        ):

            return (
                False,
                "direction mismatch with reference"
            )

    return (
        True,
        "VALID"
    )


# ==============================================================
# MNI OUTPUT VALIDATION
# ==============================================================

def validate_mni_output(
    output_path,
    reference
):
    """
    Validate registered output against exact MNI grid.
    """

    try:

        image = sitk.ReadImage(
            str(output_path),
            sitk.sitkFloat32
        )

        if image.GetDimension() != 3:

            return (
                False,
                "MNI output is not 3D",
                None
            )

        array = sitk.GetArrayViewFromImage(
            image
        )

        if array.size == 0:

            return (
                False,
                "MNI output is empty",
                None
            )

        if not np.all(
            np.isfinite(array)
        ):

            return (
                False,
                "MNI output contains non-finite values",
                None
            )

        if np.count_nonzero(array) == 0:

            return (
                False,
                "MNI output contains no non-zero voxels",
                None
            )

        # ------------------------------------------------------
        # Exact MNI grid
        # ------------------------------------------------------

        if image.GetSize() != reference.GetSize():

            return (
                False,
                f"size mismatch: "
                f"{image.GetSize()} != "
                f"{reference.GetSize()}",
                None
            )

        if not np.allclose(
            image.GetSpacing(),
            reference.GetSpacing(),
            atol=VOXEL_TOLERANCE
        ):

            return (
                False,
                "spacing mismatch with MNI template",
                None
            )

        if not np.allclose(
            image.GetOrigin(),
            reference.GetOrigin(),
            atol=GEOMETRY_TOLERANCE
        ):

            return (
                False,
                "origin mismatch with MNI template",
                None
            )

        if not np.allclose(
            image.GetDirection(),
            reference.GetDirection(),
            atol=GEOMETRY_TOLERANCE
        ):

            return (
                False,
                "direction mismatch with MNI template",
                None
            )

        return (
            True,
            "VALID",
            image
        )

    except Exception as exc:

        return (
            False,
            f"MNI output validation error: {exc}",
            None
        )


# ==============================================================
# TRANSFORM VALIDATION
# ==============================================================

def validate_transform(
    transform_path
):
    """
    Validate a 3D affine transform.

    Handles:
      - direct AffineTransform
      - one-element CompositeTransform
        containing an AffineTransform
    """

    try:

        transform = sitk.ReadTransform(
            str(transform_path)
        )

        # ------------------------------------------------------
        # Direct affine
        # ------------------------------------------------------

        if (
            transform.GetDimension() == 3
            and
            "AffineTransform"
            in transform.GetName()
        ):

            parameters = np.asarray(
                transform.GetParameters(),
                dtype=float
            )

            if parameters.size != 12:

                return (
                    False,
                    f"unexpected affine parameter count: "
                    f"{parameters.size}, expected 12",
                    None
                )

            if not np.all(
                np.isfinite(parameters)
            ):

                return (
                    False,
                    "transform contains non-finite parameters",
                    None
                )

            fixed_parameters = np.asarray(
                transform.GetFixedParameters(),
                dtype=float
            )

            if not np.all(
                np.isfinite(fixed_parameters)
            ):

                return (
                    False,
                    "transform contains non-finite "
                    "fixed parameters",
                    None
                )

            return (
                True,
                "VALID",
                transform
            )

        # ------------------------------------------------------
        # One-element composite containing affine
        # ------------------------------------------------------

        if isinstance(
            transform,
            sitk.CompositeTransform
        ):

            number_of_transforms = (
                transform.GetNumberOfTransforms()
            )

            if number_of_transforms != 1:

                return (
                    False,
                    "CompositeTransform contains "
                    f"{number_of_transforms} transforms; "
                    "expected exactly 1",
                    None
                )

            component = (
                transform.GetNthTransform(0)
            )

            if (
                component.GetDimension() != 3
                or
                "AffineTransform"
                not in component.GetName()
            ):

                return (
                    False,
                    "CompositeTransform does not contain "
                    "a 3D affine transform",
                    None
                )

            parameters = np.asarray(
                component.GetParameters(),
                dtype=float
            )

            if parameters.size != 12:

                return (
                    False,
                    f"unexpected affine parameter count: "
                    f"{parameters.size}, expected 12",
                    None
                )

            if not np.all(
                np.isfinite(parameters)
            ):

                return (
                    False,
                    "affine component contains "
                    "non-finite parameters",
                    None
                )

            fixed_parameters = np.asarray(
                component.GetFixedParameters(),
                dtype=float
            )

            if not np.all(
                np.isfinite(fixed_parameters)
            ):

                return (
                    False,
                    "affine component contains "
                    "non-finite fixed parameters",
                    None
                )

            return (
                True,
                "VALID",
                transform
            )

        return (
            False,
            f"unexpected transform type: "
            f"{transform.GetName()}",
            None
        )

    except Exception as exc:

        return (
            False,
            f"transform validation error: {exc}",
            None
        )


# ==============================================================
# CONVERT REGISTRATION RESULT TO AFFINE
# ==============================================================

def extract_affine_transform(
    transform
):
    """
    Convert the registration result into a standalone
    3D AffineTransform.

    SimpleITK may return a CompositeTransform when an
    initial transform is supplied with inPlace=False.

    If a one-component composite is returned, its affine
    component is copied into a standalone AffineTransform.
    """

    # ----------------------------------------------------------
    # Direct affine
    # ----------------------------------------------------------

    if (
        transform.GetDimension() == 3
        and
        "AffineTransform"
        in transform.GetName()
        and
        not isinstance(
            transform,
            sitk.CompositeTransform
        )
    ):

        affine = sitk.AffineTransform(
            transform
        )

        return affine


    # ----------------------------------------------------------
    # Composite
    # ----------------------------------------------------------

    if isinstance(
        transform,
        sitk.CompositeTransform
    ):

        number_of_transforms = (
            transform.GetNumberOfTransforms()
        )

        if number_of_transforms != 1:

            raise RuntimeError(
                "Registration returned a CompositeTransform "
                f"with {number_of_transforms} components. "
                "Expected exactly one affine component."
            )

        component = (
            transform.GetNthTransform(0)
        )

        if component.GetDimension() != 3:

            raise RuntimeError(
                "Composite transform component is not 3D."
            )

        if "AffineTransform" not in (
            component.GetName()
        ):

            raise RuntimeError(
                "Composite transform component is not "
                f"an affine transform: "
                f"{component.GetName()}"
            )

        affine = sitk.AffineTransform(
            component
        )

        return affine


    raise RuntimeError(
        "Registration returned unsupported "
        f"transform type: {transform.GetName()}"
    )


# ==============================================================
# RESAMPLE TO MNI
# ==============================================================

def resample_to_mni(
    moving,
    transform
):

    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )


# ==============================================================
# PROCESSING COUNTERS
# ==============================================================

registered_volumes = 0

skipped_volumes = 0

failed_subjects = 0

failure_details = []


# ==============================================================
# PROCESS SELECTED ACQUISITIONS
# ==============================================================

for index, item in enumerate(
    selected_acquisitions,
    start=1
):

    subject = item["subject"]

    acquisition_type = item[
        "acquisition_type"
    ]

    print()
    print("=" * 70)

    print(
        f"SUBJECT {index}/"
        f"{len(selected_acquisitions)}: "
        f"{subject}"
    )

    print(
        f"Acquisition type: "
        f"{acquisition_type}"
    )

    print("=" * 70)


    subject_failed = False


    # ==========================================================
    # DETERMINE INPUT/OUTPUT
    # ==========================================================

    if acquisition_type == "RMS":

        registration_input = (
            INPUT_DIR /
            f"{subject}_rms_n4_normalized.nii.gz"
        )

        registration_output = (
            OUTPUT_DIR /
            f"{subject}_rms_n4_mni.nii.gz"
        )

    else:

        registration_input = (
            INPUT_DIR /
            f"{subject}_echo1_n4_normalized.nii.gz"
        )

        registration_output = (
            OUTPUT_DIR /
            f"{subject}_echo1_mni.nii.gz"
        )


    transform_path = (
        TRANSFORM_DIR /
        f"{subject}_to_mni.tfm"
    )


    print()
    print("Registration input:")
    print(
        f"  {registration_input}"
    )

    print()
    print("Registration output:")
    print(
        f"  {registration_output}"
    )

    print()
    print("Transform:")
    print(
        f"  {transform_path}"
    )


    # ==========================================================
    # CHECK INPUT
    # ==========================================================

    if not registration_input.exists():

        print()
        print(
            "ERROR: Registration input is missing."
        )

        failed_subjects += 1

        failure_details.append(
            (
                subject,
                f"missing normalized "
                f"{acquisition_type} input"
            )
        )

        continue


    # ==========================================================
    # READ INPUT
    # ==========================================================

    try:

        moving = sitk.ReadImage(
            str(registration_input),
            sitk.sitkFloat32
        )

    except Exception as exc:

        print()
        print(
            "ERROR: Could not read "
            "registration input:"
        )

        print(
            f"  {exc}"
        )

        failed_subjects += 1

        failure_details.append(
            (
                subject,
                f"registration input read error: "
                f"{exc}"
            )
        )

        continue


    # ==========================================================
    # VALIDATE INPUT
    # ==========================================================

    (
        input_valid,
        input_reason
    ) = validate_input_image(
        moving
    )

    if not input_valid:

        print()
        print(
            "ERROR: Registration input "
            "validation failed:"
        )

        print(
            f"  {input_reason}"
        )

        failed_subjects += 1

        failure_details.append(
            (
                subject,
                f"invalid registration input: "
                f"{input_reason}"
            )
        )

        continue


    print()
    print("Registration input geometry:")

    print(
        "  Size     :",
        moving.GetSize()
    )

    print(
        "  Spacing  :",
        moving.GetSpacing()
    )

    print(
        "  Origin   :",
        moving.GetOrigin()
    )

    print(
        "  Direction:",
        moving.GetDirection()
    )

    print(
        "Registration input validation: PASS"
    )


    # ==========================================================
    # CHECK EXISTING VALID REGISTRATION
    # ==========================================================

    registration_ready = False

    transform = None


    if (
        transform_path.exists()
        and
        registration_output.exists()
    ):

        print()
        print(
            "Existing registration found."
        )


        (
            transform_valid,
            transform_reason,
            existing_transform
        ) = validate_transform(
            transform_path
        )


        (
            output_valid,
            output_reason,
            _
        ) = validate_mni_output(
            registration_output,
            fixed
        )


        if (
            transform_valid
            and
            output_valid
        ):

            print(
                "Existing transform validation: PASS"
            )

            print(
                "Existing MNI output validation: PASS"
            )

            print(
                "Skipping new registration."
            )

            transform = existing_transform

            registration_ready = True

            skipped_volumes += 1

        else:

            print(
                "Existing registration is invalid."
            )

            if not transform_valid:

                print(
                    f"  Transform: "
                    f"{transform_reason}"
                )

            if not output_valid:

                print(
                    f"  MNI output: "
                    f"{output_reason}"
                )

            print(
                "Registration will be recreated."
            )


    # ==========================================================
    # NEW REGISTRATION
    # ==========================================================

    if not registration_ready:

        # ------------------------------------------------------
        # Remove invalid output
        # ------------------------------------------------------

        if registration_output.exists():

            try:

                registration_output.unlink()

                print(
                    "Removed invalid existing "
                    "MNI output."
                )

            except Exception as exc:

                print(
                    "ERROR: Could not remove invalid "
                    "MNI output:"
                )

                print(
                    f"  {exc}"
                )

                failed_subjects += 1

                failure_details.append(
                    (
                        subject,
                        "could not remove invalid "
                        "MNI output"
                    )
                )

                continue


        # ------------------------------------------------------
        # Remove invalid transform
        # ------------------------------------------------------

        if transform_path.exists():

            try:

                transform_path.unlink()

                print(
                    "Removed invalid existing "
                    "transform."
                )

            except Exception as exc:

                print(
                    "ERROR: Could not remove invalid "
                    "transform:"
                )

                print(
                    f"  {exc}"
                )

                failed_subjects += 1

                failure_details.append(
                    (
                        subject,
                        "could not remove invalid "
                        "transform"
                    )
                )

                continue


        # ======================================================
        # INITIAL AFFINE TRANSFORM
        # ======================================================

        print()
        print(
            "Creating initial affine transform..."
        )

        initial_transform = (
            sitk.CenteredTransformInitializer(
                fixed,
                moving,
                sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        )


        # ======================================================
        # REGISTRATION METHOD
        # ======================================================

        registration = (
            sitk.ImageRegistrationMethod()
        )


        # ------------------------------------------------------
        # Mattes Mutual Information
        # ------------------------------------------------------

        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=50
        )


        # ------------------------------------------------------
        # Random metric sampling
        # ------------------------------------------------------

        registration.SetMetricSamplingStrategy(
            registration.RANDOM
        )

        registration.SetMetricSamplingPercentage(
            0.20
        )


        # ------------------------------------------------------
        # Interpolation
        # ------------------------------------------------------

        registration.SetInterpolator(
            sitk.sitkLinear
        )


        # ------------------------------------------------------
        # Multi-resolution
        # ------------------------------------------------------

        registration.SetShrinkFactorsPerLevel(
            shrinkFactors=[4, 2, 1]
        )

        registration.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[2, 1, 0]
        )

        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


        # ------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------

        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )

        registration.SetOptimizerScalesFromPhysicalShift()


        # ------------------------------------------------------
        # Initial transform
        #
        # IMPORTANT:
        # inPlace=True ensures the supplied affine object
        # itself is optimized rather than producing a
        # composite wrapper around it.
        # ------------------------------------------------------

        registration.SetInitialTransform(
            initial_transform,
            inPlace=True
        )


        # ======================================================
        # RUN REGISTRATION
        # ======================================================

        print()
        print(
            f"Starting {acquisition_type} -> "
            "MNI152 affine registration..."
        )

        print(
            "CPU processing..."
        )


        try:

            registration_result = (
                registration.Execute(
                    fixed,
                    moving
                )
            )

        except Exception as exc:

            print()
            print(
                "ERROR: Registration execution failed:"
            )

            print(
                f"  {exc}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"registration execution error: "
                    f"{exc}"
                )
            )

            continue


        print()
        print(
            "Registration completed."
        )

        print(
            "Returned transform type:",
            registration_result.GetName()
        )

        print(
            "Final metric value:",
            registration.GetMetricValue()
        )

        print(
            "Optimizer iterations:",
            registration.GetOptimizerIteration()
        )


        # ======================================================
        # EXTRACT STANDALONE AFFINE
        # ======================================================

        try:

            transform = (
                extract_affine_transform(
                    registration_result
                )
            )

        except Exception as exc:

            print()
            print(
                "ERROR: Could not extract final "
                "3D affine transform:"
            )

            print(
                f"  {exc}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"affine extraction error: "
                    f"{exc}"
                )
            )

            continue


        # ======================================================
        # VALIDATE FINAL AFFINE
        # ======================================================

        if transform.GetDimension() != 3:

            print(
                "ERROR: Final transform is not 3D."
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    "final transform is not 3D"
                )
            )

            continue


        if "AffineTransform" not in (
            transform.GetName()
        ):

            print(
                "ERROR: Final transform is not affine:"
            )

            print(
                f"  {transform.GetName()}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"unexpected final transform: "
                    f"{transform.GetName()}"
                )
            )

            continue


        parameters = np.asarray(
            transform.GetParameters(),
            dtype=float
        )

        if parameters.size != 12:

            print(
                "ERROR: Unexpected affine "
                "parameter count:"
            )

            print(
                f"  {parameters.size}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"unexpected affine parameter count: "
                    f"{parameters.size}"
                )
            )

            continue


        if not np.all(
            np.isfinite(parameters)
        ):

            print(
                "ERROR: Final affine transform "
                "contains non-finite parameters."
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    "non-finite affine parameters"
                )
            )

            continue


        print(
            "Final affine transform validation: PASS"
        )


        # ======================================================
        # SAVE TRANSFORM
        # ======================================================

        try:

            sitk.WriteTransform(
                transform,
                str(transform_path)
            )

        except Exception as exc:

            print(
                "ERROR: Could not save transform:"
            )

            print(
                f"  {exc}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"transform save error: "
                    f"{exc}"
                )
            )

            continue


        print()
        print(
            "Transform saved:"
        )

        print(
            f"  {transform_path}"
        )


        # ======================================================
        # VERIFY SAVED TRANSFORM
        # ======================================================

        (
            saved_transform_valid,
            saved_transform_reason,
            saved_transform
        ) = validate_transform(
            transform_path
        )


        if not saved_transform_valid:

            print(
                "ERROR: Saved transform "
                "failed validation:"
            )

            print(
                f"  {saved_transform_reason}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"saved transform invalid: "
                    f"{saved_transform_reason}"
                )
            )

            continue


        print(
            "Saved transform validation: PASS"
        )

        transform = saved_transform


        # ======================================================
        # RESAMPLE REGISTRATION SOURCE
        # ======================================================

        try:

            registered = (
                resample_to_mni(
                    moving,
                    transform
                )
            )

        except Exception as exc:

            print(
                "ERROR: MNI resampling failed:"
            )

            print(
                f"  {exc}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"MNI resampling error: "
                    f"{exc}"
                )
            )

            continue


        # ======================================================
        # SAVE MNI OUTPUT
        # ======================================================

        try:

            sitk.WriteImage(
                registered,
                str(registration_output),
                useCompression=True
            )

        except Exception as exc:

            print(
                "ERROR: Could not save MNI output:"
            )

            print(
                f"  {exc}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"MNI output save error: "
                    f"{exc}"
                )
            )

            continue


        # ======================================================
        # VALIDATE MNI OUTPUT
        # ======================================================

        (
            output_valid,
            output_reason,
            _
        ) = validate_mni_output(
            registration_output,
            fixed
        )


        if not output_valid:

            print(
                "ERROR: MNI output validation failed:"
            )

            print(
                f"  {output_reason}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject,
                    f"MNI output invalid: "
                    f"{output_reason}"
                )
            )

            continue


        print()
        print(
            "MNI output validation: PASS"
        )

        print(
            "  Size     :",
            MNI_SIZE
        )

        print(
            "  Spacing  :",
            MNI_SPACING
        )

        print(
            "  Origin   :",
            MNI_ORIGIN
        )

        print(
            "  Direction:",
            MNI_DIRECTION
        )

        registered_volumes += 1


    # ==========================================================
    # FINAL TRANSFORM CHECK
    # ==========================================================

    if not transform_path.exists():

        print()
        print(
            "ERROR: Subject transform does not exist."
        )

        subject_failed = True

        failure_details.append(
            (
                subject,
                "missing subject transform"
            )
        )

    else:

        (
            transform_valid,
            transform_reason,
            transform
        ) = validate_transform(
            transform_path
        )

        if not transform_valid:

            print()
            print(
                "ERROR: Subject transform failed "
                "final validation:"
            )

            print(
                f"  {transform_reason}"
            )

            subject_failed = True

            failure_details.append(
                (
                    subject,
                    f"invalid transform: "
                    f"{transform_reason}"
                )
            )


    # ==========================================================
    # RMS FINAL CHECK
    # ==========================================================

    if (
        not subject_failed
        and
        acquisition_type == "RMS"
    ):

        output_path = (
            OUTPUT_DIR /
            f"{subject}_rms_n4_mni.nii.gz"
        )

        if not output_path.exists():

            subject_failed = True

            failure_details.append(
                (
                    subject,
                    "missing RMS MNI output"
                )
            )

        else:

            (
                output_valid,
                output_reason,
                _
            ) = validate_mni_output(
                output_path,
                fixed
            )

            if not output_valid:

                subject_failed = True

                failure_details.append(
                    (
                        subject,
                        f"invalid RMS MNI output: "
                        f"{output_reason}"
                    )
                )

            else:

                print()
                print(
                    "RMS MNI output final validation: PASS"
                )


    # ==========================================================
    # 5ECHO FINAL CHECK
    # ==========================================================

    if (
        not subject_failed
        and
        acquisition_type == "5ECHO"
    ):

        for echo in [2, 3, 4, 5]:

            input_path = (
                INPUT_DIR /
                f"{subject}_echo{echo}_n4_normalized.nii.gz"
            )

            output_path = (
                OUTPUT_DIR /
                f"{subject}_echo{echo}_mni.nii.gz"
            )


            print()
            print(
                "-" * 70
            )

            print(
                f"Applying Echo-1 transform "
                f"to Echo {echo}..."
            )

            print(
                "-" * 70
            )


            # --------------------------------------------------
            # Input
            # --------------------------------------------------

            if not input_path.exists():

                print(
                    f"ERROR: Echo {echo} "
                    "normalized input is missing."
                )

                subject_failed = True

                failure_details.append(
                    (
                        f"{subject}_echo{echo}",
                        "missing normalized input"
                    )
                )

                continue


            try:

                echo_image = sitk.ReadImage(
                    str(input_path),
                    sitk.sitkFloat32
                )

            except Exception as exc:

                print(
                    f"ERROR: Could not read "
                    f"Echo {echo}:"
                )

                print(
                    f"  {exc}"
                )

                subject_failed = True

                failure_details.append(
                    (
                        f"{subject}_echo{echo}",
                        f"read error: {exc}"
                    )
                )

                continue


            # --------------------------------------------------
            # Validate same geometry as Echo 1
            # --------------------------------------------------

            (
                echo_valid,
                echo_reason
            ) = validate_input_image(
                echo_image,
                reference=moving
            )

            if not echo_valid:

                print(
                    f"ERROR: Echo {echo} "
                    "validation failed:"
                )

                print(
                    f"  {echo_reason}"
                )

                subject_failed = True

                failure_details.append(
                    (
                        f"{subject}_echo{echo}",
                        f"invalid input: "
                        f"{echo_reason}"
                    )
                )

                continue


            # --------------------------------------------------
            # Existing output
            # --------------------------------------------------

            if output_path.exists():

                (
                    existing_valid,
                    existing_reason,
                    _
                ) = validate_mni_output(
                    output_path,
                    fixed
                )

                if existing_valid:

                    print(
                        f"Echo {echo} MNI output "
                        "is already valid."
                    )

                    print(
                        "Skipping."
                    )

                    skipped_volumes += 1

                    continue


                print(
                    f"Existing Echo {echo} "
                    "output is invalid:"
                )

                print(
                    f"  {existing_reason}"
                )

                print(
                    "Recreating."
                )

                try:

                    output_path.unlink()

                except Exception as exc:

                    print(
                        "ERROR: Could not remove "
                        "invalid output:"
                    )

                    print(
                        f"  {exc}"
                    )

                    subject_failed = True

                    failure_details.append(
                        (
                            f"{subject}_echo{echo}",
                            "could not remove invalid output"
                        )
                    )

                    continue


            # --------------------------------------------------
            # SAME Echo-1 transform
            # --------------------------------------------------

            try:

                echo_registered = (
                    resample_to_mni(
                        echo_image,
                        transform
                    )
                )

            except Exception as exc:

                print(
                    f"ERROR: Echo {echo} "
                    "MNI resampling failed:"
                )

                print(
                    f"  {exc}"
                )

                subject_failed = True

                failure_details.append(
                    (
                        f"{subject}_echo{echo}",
                        f"resampling error: {exc}"
                    )
                )

                continue


            # --------------------------------------------------
            # Save
            # --------------------------------------------------

            try:

                sitk.WriteImage(
                    echo_registered,
                    str(output_path),
                    useCompression=True
                )

            except Exception as exc:

                print(
                    f"ERROR: Could not save "
                    f"Echo {echo} MNI output:"
                )

                print(
                    f"  {exc}"
                )

                subject_failed = True

                failure_details.append(
                    (
                        f"{subject}_echo{echo}",
                        f"save error: {exc}"
                    )
                )

                continue


            # --------------------------------------------------
            # Validate
            # --------------------------------------------------

            (
                output_valid,
                output_reason,
                _
            ) = validate_mni_output(
                output_path,
                fixed
            )


            if not output_valid:

                print(
                    f"ERROR: Echo {echo} "
                    "MNI validation failed:"
                )

                print(
                    f"  {output_reason}"
                )

                subject_failed = True

                failure_details.append(
                    (
                        f"{subject}_echo{echo}",
                        f"MNI validation failed: "
                        f"{output_reason}"
                    )
                )

                continue


            print(
                f"Echo {echo} MNI output "
                "validation: PASS"
            )

            registered_volumes += 1


    # ==========================================================
    # SUBJECT RESULT
    # ==========================================================

    if subject_failed:

        failed_subjects += 1

        print()
        print(
            f"SUBJECT {subject}: FAILED/PARTIAL"
        )

    else:

        print()
        print(
            f"SUBJECT {subject}: COMPLETED"
        )


# ==============================================================
# FINAL BATCH VALIDATION
# ==============================================================

print()
print("=" * 70)
print("FINAL MNI152 BATCH VALIDATION")
print("=" * 70)


final_valid_subjects = []

final_invalid_entries = []


for item in selected_acquisitions:

    subject = item["subject"]

    acquisition_type = item[
        "acquisition_type"
    ]


    # ==========================================================
    # TRANSFORM
    # ==========================================================

    transform_path = (
        TRANSFORM_DIR /
        f"{subject}_to_mni.tfm"
    )


    if not transform_path.exists():

        final_invalid_entries.append(
            (
                subject,
                "missing transform"
            )
        )

        continue


    (
        transform_valid,
        transform_reason,
        _
    ) = validate_transform(
        transform_path
    )


    if not transform_valid:

        final_invalid_entries.append(
            (
                subject,
                f"invalid transform: "
                f"{transform_reason}"
            )
        )

        continue


    subject_valid = True


    # ==========================================================
    # RMS
    # ==========================================================

    if acquisition_type == "RMS":

        output_path = (
            OUTPUT_DIR /
            f"{subject}_rms_n4_mni.nii.gz"
        )


        if not output_path.exists():

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject,
                    "missing RMS MNI output"
                )
            )

        else:

            (
                output_valid,
                output_reason,
                _
            ) = validate_mni_output(
                output_path,
                fixed
            )


            if not output_valid:

                subject_valid = False

                final_invalid_entries.append(
                    (
                        subject,
                        f"invalid RMS MNI output: "
                        f"{output_reason}"
                    )
                )


    # ==========================================================
    # 5ECHO
    # ==========================================================

    else:

        for echo in ECHOES:

            output_path = (
                OUTPUT_DIR /
                f"{subject}_echo{echo}_mni.nii.gz"
            )


            if not output_path.exists():

                subject_valid = False

                final_invalid_entries.append(
                    (
                        f"{subject}_echo{echo}",
                        "missing MNI output"
                    )
                )

                continue


            (
                output_valid,
                output_reason,
                _
            ) = validate_mni_output(
                output_path,
                fixed
            )


            if not output_valid:

                subject_valid = False

                final_invalid_entries.append(
                    (
                        f"{subject}_echo{echo}",
                        f"invalid MNI output: "
                        f"{output_reason}"
                    )
                )


    if subject_valid:

        final_valid_subjects.append(
            subject
        )


# ==============================================================
# FINAL INVENTORY
# ==============================================================

actual_current_outputs = 0

actual_current_transforms = 0


for item in selected_acquisitions:

    subject = item["subject"]

    acquisition_type = item[
        "acquisition_type"
    ]


    # ----------------------------------------------------------
    # Transform
    # ----------------------------------------------------------

    transform_path = (
        TRANSFORM_DIR /
        f"{subject}_to_mni.tfm"
    )


    if transform_path.exists():

        (
            transform_valid,
            _,
            _
        ) = validate_transform(
            transform_path
        )

        if transform_valid:

            actual_current_transforms += 1


    # ----------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------

    if acquisition_type == "RMS":

        output_path = (
            OUTPUT_DIR /
            f"{subject}_rms_n4_mni.nii.gz"
        )


        if output_path.exists():

            (
                output_valid,
                _,
                _
            ) = validate_mni_output(
                output_path,
                fixed
            )

            if output_valid:

                actual_current_outputs += 1


    else:

        for echo in ECHOES:

            output_path = (
                OUTPUT_DIR /
                f"{subject}_echo{echo}_mni.nii.gz"
            )


            if output_path.exists():

                (
                    output_valid,
                    _,
                    _
                ) = validate_mni_output(
                    output_path,
                    fixed
                )

                if output_valid:

                    actual_current_outputs += 1


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()

print(
    f"Selected subjects          : "
    f"{len(selected_acquisitions)}"
)

print(
    f"Expected MNI outputs       : "
    f"{expected_outputs}"
)

print(
    f"Actual valid MNI outputs   : "
    f"{actual_current_outputs}"
)

print(
    f"Expected transforms        : "
    f"{len(selected_acquisitions)}"
)

print(
    f"Actual valid transforms    : "
    f"{actual_current_transforms}"
)

print(
    f"Valid complete subjects    : "
    f"{len(final_valid_subjects)}"
)

print(
    f"Invalid/missing entries    : "
    f"{len(final_invalid_entries)}"
)


# ==============================================================
# INVALID DETAILS
# ==============================================================

if final_invalid_entries:

    print()
    print(
        "FINAL INVALID/MISSING ENTRIES:"
    )

    for item, reason in final_invalid_entries:

        print(
            f"  {item}: {reason}"
        )


# ==============================================================
# PROCESSING SUMMARY
# ==============================================================

print()
print("Processing summary:")

print(
    f"  Newly registered volumes : "
    f"{registered_volumes}"
)

print(
    f"  Skipped valid volumes    : "
    f"{skipped_volumes}"
)

print(
    f"  Failed subjects          : "
    f"{failed_subjects}"
)


# ==============================================================
# FAILURE DETAILS
# ==============================================================

if failure_details:

    print()
    print(
        "PROCESSING FAILURE DETAILS:"
    )

    for item, reason in failure_details:

        print(
            f"  {item}: {reason}"
        )


# ==============================================================
# OUTPUT PATHS
# ==============================================================

print()
print("MNI output directory:")

print(
    f"  {OUTPUT_DIR.resolve()}"
)

print()
print("Transform directory:")

print(
    f"  {TRANSFORM_DIR.resolve()}"
)


# ==============================================================
# FINAL STATUS
# ==============================================================

print()
print("=" * 70)


if (
    len(final_valid_subjects)
    == len(selected_acquisitions)

    and
    actual_current_outputs
    == expected_outputs

    and
    actual_current_transforms
    == len(selected_acquisitions)

    and
    not final_invalid_entries
):

    print(
        "STATUS: MNI152 REGISTRATION "
        "VALIDATION PASSED"
    )

else:

    print(
        "STATUS: MNI152 REGISTRATION "
        "VALIDATION FAILED"
    )


print("=" * 70)