# src/qc_mni_current.py

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk


# ================================================================
# GENERIC MNI REGISTERED sMRI QUALITY CONTROL
#
# Supports:
#   - RMS single-volume acquisitions
#   - 5ECHO acquisitions
#
# Uses the actual manifest structure:
#   subject
#   session
#   series
#   selection_status
#   acquisition_type
#
# QC ONLY:
#   No MRI images or transforms are modified.
# ================================================================


print("=" * 70)
print("GENERIC MNI REGISTERED sMRI QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

REGISTERED_DIR = os.path.join(
    "processed_data", "smri", "registered"
)

TRANSFORMS_DIR = os.path.join(
    "processed_data", "smri", "transforms"
)

MNI_TEMPLATE = os.path.join(
    "processed_data", "smri", "mni",
    "mni152_template_2mm.nii.gz"
)

MANIFEST_PATH = os.path.join(
    "processed_data", "smri",
    "mprage_selection",
    "mprage_selection_manifest.csv"
)

QC_DIR = os.path.join(
    "processed_data", "smri", "qc_mni"
)

SUMMARY_CSV = os.path.join(
    QC_DIR, "mni_qc_summary.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ECHOES = [1, 2, 3, 4, 5]

GEOMETRY_ATOL = 1e-5


# ================================================================
# MANIFEST
# ================================================================

def get_selected_acquisitions():

    if not os.path.isfile(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Selection manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "subject",
        "session",
        "series",
        "selection_status",
        "acquisition_type"
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    selected = manifest[
        manifest["selection_status"]
        .astype(str)
        .str.upper()
        .eq("SELECTED")
    ].copy()

    if selected.empty:
        raise ValueError(
            "No SELECTED acquisitions found in manifest."
        )

    selected["subject"] = (
        selected["subject"]
        .astype(str)
        .str.strip()
    )

    selected["session"] = (
        selected["session"]
        .astype(str)
        .str.strip()
    )

    selected["series"] = (
        selected["series"]
        .astype(str)
        .str.strip()
    )

    selected["acquisition_type"] = (
        selected["acquisition_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if selected["subject"].eq("").any():
        raise ValueError(
            "Manifest contains empty subject IDs."
        )

    if selected["subject"].duplicated().any():

        duplicates = sorted(
            selected.loc[
                selected["subject"].duplicated(),
                "subject"
            ].unique()
        )

        raise ValueError(
            "Duplicate SELECTED subjects found: "
            f"{duplicates}"
        )

    invalid_types = set(
        selected["acquisition_type"]
    ) - {"RMS", "5ECHO"}

    if invalid_types:
        raise ValueError(
            "Unsupported acquisition types: "
            f"{sorted(invalid_types)}"
        )

    return selected.sort_values(
        "subject"
    ).reset_index(drop=True)


# ================================================================
# EXPECTED FILENAMES
# ================================================================

def get_expected_registered_files(
    subject,
    acquisition_type
):

    if acquisition_type == "RMS":

        return [
            f"{subject}_rms_n4_mni.nii.gz"
        ]

    if acquisition_type == "5ECHO":

        return [
            f"{subject}_echo{echo}_n4_mni.nii.gz"
            for echo in EXPECTED_ECHOES
        ]

    raise ValueError(
        f"Unsupported acquisition type: "
        f"{acquisition_type}"
    )


def get_expected_transform_file(subject):

    return f"{subject}_to_mni.tfm"


# ================================================================
# IMAGE VALIDATION
# ================================================================

def read_valid_image(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    image = sitk.ReadImage(path)

    if image.GetDimension() != 3:

        raise ValueError(
            f"Expected 3D image, got "
            f"{image.GetDimension()}D."
        )

    data = sitk.GetArrayFromImage(image)

    if data.size == 0:

        raise ValueError(
            "Image contains no voxels."
        )

    if not np.all(
        np.isfinite(data)
    ):

        raise ValueError(
            "Image contains NaN or infinite values."
        )

    if np.count_nonzero(data) == 0:

        raise ValueError(
            "Image contains no non-zero voxels."
        )

    return image, data


# ================================================================
# GEOMETRY
# ================================================================

def geometry_matches(
    image_a,
    image_b
):

    return (
        image_a.GetSize()
        == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=GEOMETRY_ATOL
        )
        and np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=GEOMETRY_ATOL
        )
        and np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=GEOMETRY_ATOL
        )
    )


# ================================================================
# AFFINE
# ================================================================

def affine_from_sitk(image):

    spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64
    )

    origin = np.asarray(
        image.GetOrigin(),
        dtype=np.float64
    )

    direction = np.asarray(
        image.GetDirection(),
        dtype=np.float64
    ).reshape(3, 3)

    affine = np.eye(4)

    affine[:3, :3] = (
        direction
        @ np.diag(spacing)
    )

    affine[:3, 3] = origin

    return affine


def validate_affine(image):

    affine = affine_from_sitk(image)

    if not np.all(
        np.isfinite(affine)
    ):
        return False

    determinant = np.linalg.det(
        affine[:3, :3]
    )

    if not np.isfinite(determinant):
        return False

    if abs(determinant) < 1e-8:
        return False

    return True


# ================================================================
# TRANSFORM
# ================================================================

def validate_transform(
    transform_path
):

    if not os.path.isfile(
        transform_path
    ):

        return False, "Transform file missing."

    try:

        transform = sitk.ReadTransform(
            transform_path
        )

        transform_name = (
            transform.GetName()
        )

        parameters = np.asarray(
            transform.GetParameters(),
            dtype=np.float64
        )

        fixed_parameters = np.asarray(
            transform.GetFixedParameters(),
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(parameters)
        ):

            return (
                False,
                "Transform contains "
                "non-finite parameters."
            )

        if not np.all(
            np.isfinite(fixed_parameters)
        ):

            return (
                False,
                "Transform contains "
                "non-finite fixed parameters."
            )

        if transform_name.lower() != "affinetransform":

            return (
                False,
                f"Unexpected transform type: "
                f"{transform_name}"
            )

        if len(parameters) != 12:

            return (
                False,
                "3D affine transform should "
                "contain 12 parameters."
            )

        return True, transform_name

    except Exception as e:

        return (
            False,
            f"Unable to read transform: {e}"
        )


# ================================================================
# PATH VALIDATION
# ================================================================

for path, description in [

    (
        REGISTERED_DIR,
        "registered MRI directory"
    ),

    (
        TRANSFORMS_DIR,
        "transform directory"
    )
]:

    if not os.path.isdir(path):

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


if not os.path.isfile(
    MNI_TEMPLATE
):

    raise FileNotFoundError(
        f"MNI template not found:\n"
        f"{MNI_TEMPLATE}"
    )


os.makedirs(
    QC_DIR,
    exist_ok=True
)


# ================================================================
# LOAD MNI TEMPLATE
# ================================================================

print("\n" + "=" * 70)
print("LOADING MNI REFERENCE")
print("=" * 70)

mni_template, mni_data = (
    read_valid_image(
        MNI_TEMPLATE
    )
)

print(
    "MNI size:",
    mni_template.GetSize()
)

print(
    "MNI spacing:",
    mni_template.GetSpacing()
)

print(
    "MNI origin:",
    mni_template.GetOrigin()
)

print(
    "MNI direction:",
    mni_template.GetDirection()
)

if not validate_affine(
    mni_template
):

    raise ValueError(
        "MNI template affine is invalid."
    )

print(
    "MNI template validation: PASS"
)


# ================================================================
# CURRENT COHORT
# ================================================================

selected = get_selected_acquisitions()

print("\n" + "-" * 70)
print("CURRENT COHORT")
print("-" * 70)

for _, row in selected.iterrows():

    print(
        f"  {row['subject']} | "
        f"{row['acquisition_type']} | "
        f"{row['session']} | "
        f"{row['series']}"
    )

print(
    "\nSelected subjects:",
    len(selected)
)

rms_count = int(
    (
        selected["acquisition_type"]
        == "RMS"
    ).sum()
)

echo_count = int(
    (
        selected["acquisition_type"]
        == "5ECHO"
    ).sum()
)

print(
    "Acquisition types:"
)

print(
    f"  RMS   : {rms_count}"
)

print(
    f"  5ECHO : {echo_count}"
)


# ================================================================
# EXPECTED CURRENT FILES
# ================================================================

expected_files = set()

for _, row in selected.iterrows():

    expected_files.update(
        get_expected_registered_files(
            row["subject"],
            row["acquisition_type"]
        )
    )


expected_transform_files = {
    get_expected_transform_file(
        subject
    )
    for subject in selected["subject"]
}


print(
    "\nExpected MNI registered volumes:",
    len(expected_files)
)

print(
    "Expected transforms:",
    len(expected_transform_files)
)


# ================================================================
# SUBJECT QC
# ================================================================

results = []

passed = 0
failed = 0


for index, row in selected.iterrows():

    subject = row["subject"]
    acquisition_type = row["acquisition_type"]

    print("\n" + "=" * 70)
    print(
        f"QC {index + 1}/{len(selected)} — "
        f"{subject} | {acquisition_type}"
    )
    print("=" * 70)

    errors = []

    checks = {
        "all_registered_images_valid": False,
        "all_mni_geometry_valid": False,
        "all_affines_valid": False,
        "transform_valid": False
    }

    expected_subject_files = (
        get_expected_registered_files(
            subject,
            acquisition_type
        )
    )

    # ------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------

    transform_filename = (
        get_expected_transform_file(
            subject
        )
    )

    transform_path = os.path.join(
        TRANSFORMS_DIR,
        transform_filename
    )

    print("\nTransform:")
    print(
        f"  {transform_filename}"
    )

    transform_ok, transform_message = (
        validate_transform(
            transform_path
        )
    )

    checks[
        "transform_valid"
    ] = transform_ok

    if transform_ok:

        print(
            "  Transform: PASS"
        )

    else:

        print(
            "  Transform: FAIL"
        )

        errors.append(
            transform_message
        )

    # ------------------------------------------------------------
    # Registered image(s)
    # ------------------------------------------------------------

    all_images_valid = True
    all_geometry_valid = True
    all_affines_valid = True

    for filename in expected_subject_files:

        registered_path = os.path.join(
            REGISTERED_DIR,
            filename
        )

        print(
            "\nRegistered volume:"
        )

        print(
            f"  {filename}"
        )

        try:

            image, data = (
                read_valid_image(
                    registered_path
                )
            )

            print(
                "  Readability: PASS"
            )

            # ----------------------------------------------------
            # MNI grid
            # ----------------------------------------------------

            if geometry_matches(
                image,
                mni_template
            ):

                print(
                    "  MNI grid: PASS"
                )

            else:

                all_geometry_valid = False
                all_images_valid = False

                errors.append(
                    f"{filename}: MNI geometry "
                    "does not match template."
                )

                print(
                    "  MNI grid: FAIL"
                )

            # ----------------------------------------------------
            # Affine
            # ----------------------------------------------------

            if validate_affine(
                image
            ):

                print(
                    "  Affine validity: PASS"
                )

            else:

                all_affines_valid = False
                all_images_valid = False

                errors.append(
                    f"{filename}: invalid affine."
                )

                print(
                    "  Affine validity: FAIL"
                )

        except Exception as e:

            all_images_valid = False
            all_geometry_valid = False
            all_affines_valid = False

            errors.append(
                f"{filename}: {e}"
            )

            print(
                "  Status: FAIL"
            )

    checks[
        "all_registered_images_valid"
    ] = (
        all_images_valid
    )

    checks[
        "all_mni_geometry_valid"
    ] = (
        all_geometry_valid
    )

    checks[
        "all_affines_valid"
    ] = (
        all_affines_valid
    )

    # ------------------------------------------------------------
    # Subject result
    # ------------------------------------------------------------

    if len(errors) == 0:

        status = "PASS"
        passed += 1

    else:

        status = "FAIL"
        failed += 1

    results.append(
        {
            "subject": subject,
            "acquisition_type": acquisition_type,
            "registered_volume_count": len(
                expected_subject_files
            ),
            "status": status,
            "error_count": len(errors),
            "errors": " | ".join(errors),
            **checks
        }
    )

    print(
        "\nSUBJECT STATUS:",
        status
    )

    if errors:

        for error in errors:

            print(
                " -",
                error
            )


# ================================================================
# CURRENT FILE COMPLETENESS
# ================================================================

print("\n" + "=" * 70)
print("CURRENT COHORT FILE COMPLETENESS")
print("=" * 70)


actual_registered_files = {
    filename
    for filename in os.listdir(
        REGISTERED_DIR
    )
    if filename.lower().endswith(
        (".nii", ".nii.gz")
    )
}

actual_current_files = (
    actual_registered_files
    & expected_files
)

missing_files = sorted(
    expected_files
    - actual_current_files
)

legacy_files = sorted(
    actual_registered_files
    - expected_files
)


actual_transform_files = {
    filename
    for filename in os.listdir(
        TRANSFORMS_DIR
    )
    if filename.lower().endswith(
        ".tfm"
    )
}

actual_current_transforms = (
    actual_transform_files
    & expected_transform_files
)

missing_transforms = sorted(
    expected_transform_files
    - actual_current_transforms
)

legacy_transforms = sorted(
    actual_transform_files
    - expected_transform_files
)


print(
    "Expected current registered files:",
    len(expected_files)
)

print(
    "Found current registered files:",
    len(actual_current_files)
)

print(
    "Missing current registered files:",
    len(missing_files)
)

print(
    "Legacy/unrelated registered files ignored:",
    len(legacy_files)
)

print(
    "\nExpected current transforms:",
    len(expected_transform_files)
)

print(
    "Found current transforms:",
    len(actual_current_transforms)
)

print(
    "Missing current transforms:",
    len(missing_transforms)
)

print(
    "Legacy/unrelated transforms ignored:",
    len(legacy_transforms)
)


if missing_files:

    print(
        "\nMissing registered files:"
    )

    for filename in missing_files:

        print(
            " ",
            filename
        )


if missing_transforms:

    print(
        "\nMissing transforms:"
    )

    for filename in missing_transforms:

        print(
            " ",
            filename
        )


if legacy_files:

    print(
        "\nLegacy/unrelated registered files "
        "are ignored."
    )

    for filename in legacy_files[:20]:

        print(
            " ",
            filename
        )

    if len(legacy_files) > 20:

        print(
            f"  ... and "
            f"{len(legacy_files) - 20} more."
        )


# ================================================================
# SAVE SUMMARY
# ================================================================

summary_df = pd.DataFrame(
    results
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("MNI QC SUMMARY")
print("=" * 70)

print(
    "Selected subjects:",
    len(selected)
)

print(
    "Expected registered files:",
    len(expected_files)
)

print(
    "Found current registered files:",
    len(actual_current_files)
)

print(
    "Expected transforms:",
    len(expected_transform_files)
)

print(
    "Found current transforms:",
    len(actual_current_transforms)
)

print(
    "Subjects PASS:",
    passed
)

print(
    "Subjects FAIL:",
    failed
)

print(
    "Missing registered files:",
    len(missing_files)
)

print(
    "Missing transforms:",
    len(missing_transforms)
)

print(
    "Legacy registered files ignored:",
    len(legacy_files)
)

print(
    "Legacy transforms ignored:",
    len(legacy_transforms)
)

print(
    "Summary CSV:",
    SUMMARY_CSV
)


cohort_pass = (
    failed == 0
    and len(missing_files) == 0
    and len(missing_transforms) == 0
    and len(actual_current_files)
    == len(expected_files)
    and len(actual_current_transforms)
    == len(expected_transform_files)
)


if cohort_pass:

    print(
        "\nSTATUS: AUTOMATED MNI QC PASS"
    )

    print(
        "All current selected subjects, "
        "registered volumes, and transforms passed."
    )

else:

    print(
        "\nSTATUS: MNI QC FAILED"
    )


print("=" * 70)