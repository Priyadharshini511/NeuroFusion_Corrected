# # src\resample_aal_to_mni.py

# import os
# import numpy as np
# import SimpleITK as sitk


# # ================================================================
# # AAL ATLAS -> MNI152 2mm RESAMPLING
# #
# # Generic atlas-level preprocessing step.
# #
# # Input:
# #   Original AAL SPM12 atlas
# #
# # Output:
# #   AAL atlas resampled onto the project's MNI152 2mm grid
# #
# # Method:
# #   Nearest-neighbor interpolation
# #
# # Important:
# #   This script is independent of subject count / dataset ZIP.
# # ================================================================


# print("=" * 70)
# print("AAL ATLAS -> MNI152 2mm RESAMPLING")
# print("=" * 70)


# # ================================================================
# # PATHS
# # ================================================================

# SOURCE_ATLAS = (
#     r"processed_data\smri\atlas\aal_SPM12"
#     r"\aal\ROI_MNI_V4.nii"
# )

# MNI_TEMPLATE = (
#     r"processed_data\smri\mni"
#     r"\mni152_template_2mm.nii.gz"
# )

# OUTPUT_DIR = (
#     r"processed_data\smri\atlas"
#     r"\aal_mni"
# )

# OUTPUT_ATLAS = os.path.join(
#     OUTPUT_DIR,
#     "AAL_MNI152_2mm.nii.gz"
# )


# # ================================================================
# # EXPECTED AAL CONFIGURATION
# # ================================================================

# EXPECTED_ROIS = 116

# EXPECTED_LABELS = np.arange(
#     1,
#     EXPECTED_ROIS + 1,
#     dtype=np.int32
# )

# ATOL = 1e-5


# # ================================================================
# # HELPERS
# # ================================================================

# def geometry_matches(image_a, image_b, atol=1e-5):

#     return (
#         image_a.GetSize() == image_b.GetSize()
#         and np.allclose(
#             image_a.GetSpacing(),
#             image_b.GetSpacing(),
#             atol=atol
#         )
#         and np.allclose(
#             image_a.GetOrigin(),
#             image_b.GetOrigin(),
#             atol=atol
#         )
#         and np.allclose(
#             image_a.GetDirection(),
#             image_b.GetDirection(),
#             atol=atol
#         )
#     )


# def validate_atlas(
#     atlas,
#     expected_labels,
#     description
# ):

#     print("\n" + "-" * 70)
#     print(f"VALIDATING {description}")
#     print("-" * 70)

#     data = sitk.GetArrayFromImage(atlas)

#     if data.size == 0:
#         raise ValueError(
#             f"{description} contains no voxels."
#         )

#     if not np.all(np.isfinite(data)):
#         raise ValueError(
#             f"{description} contains NaN or infinite values."
#         )

#     rounded = np.rint(data)

#     max_error = np.max(
#         np.abs(data - rounded)
#     )

#     print(
#         "Maximum integer deviation:",
#         max_error
#     )

#     if max_error > ATOL:
#         raise ValueError(
#             f"{description} contains non-integer labels."
#         )

#     positive_labels = np.unique(
#         rounded[rounded > 0]
#     ).astype(np.int32)

#     print(
#         "Positive labels:",
#         len(positive_labels)
#     )

#     print(
#         "Labels:",
#         positive_labels.tolist()
#     )

#     if not np.array_equal(
#         positive_labels,
#         expected_labels
#     ):
#         missing = sorted(
#             set(expected_labels.tolist())
#             - set(positive_labels.tolist())
#         )

#         unexpected = sorted(
#             set(positive_labels.tolist())
#             - set(expected_labels.tolist())
#         )

#         raise ValueError(
#             f"{description} label verification failed.\n"
#             f"Missing labels    : {missing}\n"
#             f"Unexpected labels : {unexpected}"
#         )

#     if np.any(data < 0):
#         raise ValueError(
#             f"{description} contains negative labels."
#         )

#     print(
#         f"{description} validation: PASS"
#     )

#     return True


# def validate_existing_output():

#     if not os.path.exists(OUTPUT_ATLAS):
#         return False

#     print("\nExisting AAL/MNI output detected.")
#     print("Validating existing output before reuse...")

#     try:

#         output_atlas = sitk.ReadImage(
#             OUTPUT_ATLAS,
#             sitk.sitkUInt16
#         )

#         if not geometry_matches(
#             output_atlas,
#             mni_template
#         ):
#             print(
#                 "Existing output geometry is invalid."
#             )
#             return False

#         validate_atlas(
#             output_atlas,
#             EXPECTED_LABELS,
#             "EXISTING RESAMPLED AAL ATLAS"
#         )

#         print(
#             "\nExisting valid AAL/MNI atlas will be reused."
#         )

#         return True

#     except Exception as e:

#         print(
#             "\nExisting output failed validation:"
#         )
#         print(str(e))

#         return False


# # ================================================================
# # PATH CHECKS
# # ================================================================

# print("\n" + "-" * 70)
# print("PATH CONFIGURATION")
# print("-" * 70)

# print("Source AAL atlas:")
# print(SOURCE_ATLAS)

# print("\nMNI template:")
# print(MNI_TEMPLATE)

# print("\nOutput atlas:")
# print(OUTPUT_ATLAS)


# if not os.path.isfile(SOURCE_ATLAS):

#     raise FileNotFoundError(
#         f"\nSource AAL atlas not found:\n{SOURCE_ATLAS}"
#     )


# if not os.path.isfile(MNI_TEMPLATE):

#     raise FileNotFoundError(
#         f"\nMNI template not found:\n{MNI_TEMPLATE}"
#     )


# os.makedirs(
#     OUTPUT_DIR,
#     exist_ok=True
# )


# # ================================================================
# # LOAD MNI TEMPLATE
# # ================================================================

# print("\n" + "=" * 70)
# print("LOADING MNI TEMPLATE")
# print("=" * 70)

# mni_template = sitk.ReadImage(
#     MNI_TEMPLATE,
#     sitk.sitkFloat32
# )

# mni_data = sitk.GetArrayFromImage(
#     mni_template
# )

# if not np.all(np.isfinite(mni_data)):

#     raise ValueError(
#         "MNI template contains NaN or infinite values."
#     )


# print("MNI size:")
# print(mni_template.GetSize())

# print(
#     "MNI spacing:",
#     mni_template.GetSpacing()
# )

# print(
#     "MNI origin:",
#     mni_template.GetOrigin()
# )

# print(
#     "MNI direction:",
#     mni_template.GetDirection()
# )


# # ================================================================
# # REUSE VALID EXISTING OUTPUT
# # ================================================================

# if validate_existing_output():

#     print("\n" + "=" * 70)
#     print("AAL RESAMPLING STATUS")
#     print("=" * 70)
#     print("STATUS: SUCCESS")
#     print("Existing valid atlas reused.")
#     print("=" * 70)

#     raise SystemExit(0)


# # ================================================================
# # LOAD SOURCE AAL ATLAS
# # ================================================================

# print("\n" + "=" * 70)
# print("LOADING SOURCE AAL ATLAS")
# print("=" * 70)

# source_atlas = sitk.ReadImage(
#     SOURCE_ATLAS,
#     sitk.sitkFloat32
# )

# print("Source size:")
# print(source_atlas.GetSize())

# print(
#     "Source spacing:",
#     source_atlas.GetSpacing()
# )

# print(
#     "Source origin:",
#     source_atlas.GetOrigin()
# )

# print(
#     "Source direction:",
#     source_atlas.GetDirection()
# )


# # ================================================================
# # VALIDATE SOURCE ATLAS
# # ================================================================

# validate_atlas(
#     source_atlas,
#     EXPECTED_LABELS,
#     "SOURCE AAL ATLAS"
# )


# # ================================================================
# # RESAMPLE
# # ================================================================

# print("\n" + "=" * 70)
# print("RESAMPLING AAL ATLAS")
# print("=" * 70)

# print("Interpolation: NEAREST NEIGHBOR")
# print("Reference grid: MNI152 2mm")
# print("Expected ROIs:", EXPECTED_ROIS)


# resampled_atlas = sitk.Resample(
#     source_atlas,
#     mni_template,
#     sitk.Transform(),
#     sitk.sitkNearestNeighbor,
#     0,
#     sitk.sitkUInt16
# )


# # ================================================================
# # VALIDATE OUTPUT GEOMETRY
# # ================================================================

# if not geometry_matches(
#     resampled_atlas,
#     mni_template
# ):

#     raise RuntimeError(
#         "Resampled AAL atlas does not match "
#         "the MNI template grid."
#     )


# print("\nResampled geometry:")
# print("Size     :", resampled_atlas.GetSize())
# print("Spacing  :", resampled_atlas.GetSpacing())
# print("Origin   :", resampled_atlas.GetOrigin())
# print("Direction:", resampled_atlas.GetDirection())


# # ================================================================
# # VALIDATE RESAMPLED LABELS
# # ================================================================

# validate_atlas(
#     resampled_atlas,
#     EXPECTED_LABELS,
#     "RESAMPLED AAL ATLAS"
# )


# # ================================================================
# # SAVE
# # ================================================================

# print("\n" + "=" * 70)
# print("SAVING RESAMPLED AAL ATLAS")
# print("=" * 70)

# sitk.WriteImage(
#     resampled_atlas,
#     OUTPUT_ATLAS,
#     useCompression=True
# )

# print(
#     "Saved:",
#     OUTPUT_ATLAS
# )


# # ================================================================
# # RELOAD AND FINAL VALIDATION
# # ================================================================

# print("\n" + "=" * 70)
# print("FINAL OUTPUT VALIDATION")
# print("=" * 70)

# final_atlas = sitk.ReadImage(
#     OUTPUT_ATLAS,
#     sitk.sitkUInt16
# )

# if not geometry_matches(
#     final_atlas,
#     mni_template
# ):

#     raise RuntimeError(
#         "Saved AAL atlas geometry does not match MNI template."
#     )


# validate_atlas(
#     final_atlas,
#     EXPECTED_LABELS,
#     "SAVED AAL ATLAS"
# )


# # ================================================================
# # FINAL STATUS
# # ================================================================

# print("\n" + "=" * 70)
# print("AAL RESAMPLING STATUS")
# print("=" * 70)

# print("STATUS: SUCCESS")
# print("AAL labels verified:", EXPECTED_ROIS)
# print("MNI grid verified: PASS")
# print("Nearest-neighbor interpolation: PASS")
# print("Output verified after saving: PASS")

# print("=" * 70)



# src/resample_aal_to_mni.py

from pathlib import Path
import numpy as np
import SimpleITK as sitk


# ================================================================
# AAL ATLAS -> MNI152 2mm RESAMPLING
#
# Generic atlas-level preprocessing.
#
# The source AAL atlas uses the standard AAL integer ROI IDs
# (e.g. 2001, 2002, ..., 9170), NOT labels 1-116.
#
# The original AAL ROI IDs are preserved.
#
# Resampling:
#   nearest-neighbor interpolation
#
# Output:
#   AAL_MNI152_2mm.nii.gz
# ================================================================


print("=" * 70)
print("AAL ATLAS -> MNI152 2mm RESAMPLING")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

SOURCE_ATLAS = Path(
    r"processed_data\smri\atlas\aal_SPM12\aal\ROI_MNI_V4.nii"
)

MNI_TEMPLATE = Path(
    r"processed_data\smri\mni\mni152_template_2mm.nii.gz"
)

OUTPUT_DIR = Path(
    r"processed_data\smri\atlas\aal_mni"
)

OUTPUT_ATLAS = (
    OUTPUT_DIR / "AAL_MNI152_2mm.nii.gz"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ROIS = 116
LABEL_TOLERANCE = 1e-5
GEOMETRY_TOLERANCE = 1e-5


# ================================================================
# HELPERS
# ================================================================

def geometry_matches(
    image_a,
    image_b,
    tolerance=GEOMETRY_TOLERANCE
):
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=tolerance
        )
        and np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=tolerance
        )
        and np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=tolerance
        )
    )


def validate_mni_template(image):

    if image.GetDimension() != 3:
        raise ValueError(
            "MNI template must be 3D."
        )

    if image.GetSize() != (99, 117, 95):
        raise ValueError(
            f"Unexpected MNI template size: "
            f"{image.GetSize()}"
        )

    if not np.allclose(
        image.GetSpacing(),
        (2.0, 2.0, 2.0),
        atol=GEOMETRY_TOLERANCE
    ):
        raise ValueError(
            f"Unexpected MNI spacing: "
            f"{image.GetSpacing()}"
        )

    expected_origin = (
        98.0,
        134.0,
        -72.0
    )

    if not np.allclose(
        image.GetOrigin(),
        expected_origin,
        atol=GEOMETRY_TOLERANCE
    ):
        raise ValueError(
            f"Unexpected MNI origin: "
            f"{image.GetOrigin()}"
        )

    expected_direction = (
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, 1.0
    )

    if not np.allclose(
        image.GetDirection(),
        expected_direction,
        atol=GEOMETRY_TOLERANCE
    ):
        raise ValueError(
            f"Unexpected MNI direction: "
            f"{image.GetDirection()}"
        )

    data = sitk.GetArrayFromImage(
        image
    ).astype(np.float64)

    if data.size == 0:
        raise ValueError(
            "MNI template is empty."
        )

    if not np.all(
        np.isfinite(data)
    ):
        raise ValueError(
            "MNI template contains NaN or infinite values."
        )

    print(
        "MNI template validation: PASS"
    )


def get_aal_labels(image):

    data = sitk.GetArrayFromImage(
        image
    ).astype(np.float64)

    if data.size == 0:
        raise ValueError(
            "AAL atlas is empty."
        )

    if not np.all(
        np.isfinite(data)
    ):
        raise ValueError(
            "AAL atlas contains NaN or infinite values."
        )

    rounded = np.rint(data)

    max_error = np.max(
        np.abs(
            data - rounded
        )
    )

    print(
        f"Maximum integer deviation: {max_error}"
    )

    if max_error > LABEL_TOLERANCE:
        raise ValueError(
            "AAL atlas contains non-integer labels."
        )

    if np.any(data < 0):
        raise ValueError(
            "AAL atlas contains negative labels."
        )

    positive_labels = np.unique(
        rounded[rounded > 0]
    ).astype(np.int32)

    if len(positive_labels) != EXPECTED_ROIS:

        raise ValueError(
            "Unexpected number of AAL ROI labels.\n"
            f"Expected: {EXPECTED_ROIS}\n"
            f"Found   : {len(positive_labels)}"
        )

    print(
        f"Positive ROI labels: {len(positive_labels)}"
    )

    print(
        "Labels:",
        positive_labels.tolist()
    )

    return positive_labels


def validate_atlas(
    image,
    expected_labels,
    mni_template,
    name
):

    print("\n" + "-" * 70)
    print(f"VALIDATING {name.upper()}")
    print("-" * 70)

    if not geometry_matches(
        image,
        mni_template
    ):
        raise ValueError(
            f"{name} geometry does not match "
            "the MNI152 2mm template."
        )

    print(
        "MNI geometry: PASS"
    )

    labels = get_aal_labels(
        image
    )

    if not np.array_equal(
        labels,
        expected_labels
    ):

        missing = sorted(
            set(expected_labels.tolist())
            - set(labels.tolist())
        )

        unexpected = sorted(
            set(labels.tolist())
            - set(expected_labels.tolist())
        )

        raise ValueError(
            f"{name} label verification failed.\n"
            f"Missing labels    : {missing}\n"
            f"Unexpected labels : {unexpected}"
        )

    print(
        "AAL label verification: PASS"
    )

    return labels


# ================================================================
# PATH CHECKS
# ================================================================

if not SOURCE_ATLAS.is_file():
    raise FileNotFoundError(
        f"Source AAL atlas not found:\n{SOURCE_ATLAS}"
    )

if not MNI_TEMPLATE.is_file():
    raise FileNotFoundError(
        f"MNI template not found:\n{MNI_TEMPLATE}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# LOAD MNI TEMPLATE
# ================================================================

print("\n" + "=" * 70)
print("LOADING MNI152 TEMPLATE")
print("=" * 70)

mni_template = sitk.ReadImage(
    str(MNI_TEMPLATE)
)

print(
    "MNI template geometry:"
)

print(
    "  Size      :",
    mni_template.GetSize()
)

print(
    "  Spacing   :",
    mni_template.GetSpacing()
)

print(
    "  Origin    :",
    mni_template.GetOrigin()
)

print(
    "  Direction :",
    mni_template.GetDirection()
)

validate_mni_template(
    mni_template
)


# ================================================================
# LOAD SOURCE AAL
# ================================================================

print("\n" + "=" * 70)
print("LOADING SOURCE AAL ATLAS")
print("=" * 70)

source_atlas = sitk.ReadImage(
    str(SOURCE_ATLAS)
)

print(
    "Source AAL geometry:"
)

print(
    "  Size      :",
    source_atlas.GetSize()
)

print(
    "  Spacing   :",
    source_atlas.GetSpacing()
)

print(
    "  Origin    :",
    source_atlas.GetOrigin()
)

print(
    "  Direction :",
    source_atlas.GetDirection()
)


# ================================================================
# VALIDATE SOURCE LABELS
# ================================================================

source_labels = get_aal_labels(
    source_atlas
)

print(
    "Source AAL ROI count:",
    len(source_labels)
)


# ================================================================
# REUSE EXISTING OUTPUT IF VALID
# ================================================================

if OUTPUT_ATLAS.is_file():

    print("\n" + "=" * 70)
    print("EXISTING AAL/MNI ATLAS DETECTED")
    print("=" * 70)

    try:

        existing = sitk.ReadImage(
            str(OUTPUT_ATLAS)
        )

        validate_atlas(
            existing,
            source_labels,
            mni_template,
            "Existing resampled AAL atlas"
        )

        print("\nExisting resampled atlas is VALID.")
        print("No resampling required.")
        print("\nSTATUS: SUCCESS")
        print("=" * 70)

        raise SystemExit(0)

    except Exception as e:

        print(
            "\nExisting AAL/MNI atlas failed validation:"
        )

        print(
            f"  {e}"
        )

        print(
            "\nExisting output will be regenerated."
        )


# ================================================================
# RESAMPLE
# ================================================================

print("\n" + "=" * 70)
print("RESAMPLING AAL ATLAS TO MNI152 2mm")
print("=" * 70)

print(
    "Interpolation: NEAREST NEIGHBOR"
)

print(
    "ROI labels preserved: YES"
)

resampled = sitk.Resample(
    source_atlas,
    mni_template,
    sitk.Transform(),
    sitk.sitkNearestNeighbor,
    0,
    sitk.sitkUInt16
)


# ================================================================
# VALIDATE RESAMPLED ATLAS
# ================================================================

validate_atlas(
    resampled,
    source_labels,
    mni_template,
    "Resampled AAL atlas"
)


# ================================================================
# SAVE
# ================================================================

print("\n" + "=" * 70)
print("SAVING RESAMPLED AAL ATLAS")
print("=" * 70)

sitk.WriteImage(
    resampled,
    str(OUTPUT_ATLAS),
    useCompression=True
)

print(
    "Saved:",
    OUTPUT_ATLAS
)


# ================================================================
# RELOAD AND FINAL VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("RELOADING SAVED AAL ATLAS")
print("=" * 70)

final_atlas = sitk.ReadImage(
    str(OUTPUT_ATLAS)
)

final_labels = validate_atlas(
    final_atlas,
    source_labels,
    mni_template,
    "Saved AAL/MNI atlas"
)


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("AAL ATLAS RESAMPLING STATUS")
print("=" * 70)

print("STATUS: SUCCESS")
print(
    f"ROI count: {len(final_labels)}"
)
print(
    "MNI grid: 99 x 117 x 95"
)
print(
    "Voxel size: 2 x 2 x 2 mm"
)
print(
    "Interpolation: nearest neighbor"
)
print(
    "Original AAL labels preserved: YES"
)

print("=" * 70)