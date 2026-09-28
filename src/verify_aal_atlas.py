# src\verify_aal_atlas.py

import os
import numpy as np
import SimpleITK as sitk


# ================================================================
# AAL ATLAS QUALITY CONTROL
#
# Generic atlas-level verification.
#
# Checks:
#   1. MNI template exists/readable
#   2. AAL atlas exists/readable
#   3. Exact MNI/AAL grid compatibility
#   4. Finite atlas values
#   5. Integer ROI labels
#   6. Background label = 0
#   7. Exactly 116 positive AAL labels
#   8. No unexpected labels
#   9. Non-empty atlas
#
# This script is independent of subjects.
# It does NOT use a hardcoded MRI subject.
# It does NOT modify any files.
# ================================================================


print("=" * 70)
print("AAL ATLAS QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

MNI_TEMPLATE = (
    r"processed_data\smri"
    r"\mni\mni152_template_2mm.nii.gz"
)

AAL_ATLAS = (
    r"processed_data\smri"
    r"\atlas\aal_mni"
    r"\AAL_MNI152_2mm.nii.gz"
)

QC_DIR = (
    r"processed_data\smri"
    r"\qc_atlas"
)

SUMMARY_FILE = os.path.join(
    QC_DIR,
    "aal_atlas_qc_summary.txt"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ROIS = 116

EXPECTED_LABELS = np.arange(
    1,
    EXPECTED_ROIS + 1,
    dtype=np.int32
)

GEOMETRY_ATOL = 1e-5
INTEGER_ATOL = 1e-5


# ================================================================
# HELPERS
# ================================================================

def geometry_matches(
    image_a,
    image_b
):

    size_ok = (
        image_a.GetSize()
        == image_b.GetSize()
    )

    spacing_ok = np.allclose(
        image_a.GetSpacing(),
        image_b.GetSpacing(),
        atol=GEOMETRY_ATOL
    )

    origin_ok = np.allclose(
        image_a.GetOrigin(),
        image_b.GetOrigin(),
        atol=GEOMETRY_ATOL
    )

    direction_ok = np.allclose(
        image_a.GetDirection(),
        image_b.GetDirection(),
        atol=GEOMETRY_ATOL
    )

    return (
        size_ok
        and spacing_ok
        and origin_ok
        and direction_ok
    )


def read_image(
    path,
    pixel_type
):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    image = sitk.ReadImage(
        path,
        pixel_type
    )

    data = sitk.GetArrayFromImage(
        image
    )

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

    return image, data


# ================================================================
# PREPARE QC DIRECTORY
# ================================================================

os.makedirs(
    QC_DIR,
    exist_ok=True
)


# ================================================================
# RESULT TRACKING
# ================================================================

checks = {}

errors = []


# ================================================================
# CHECK MNI TEMPLATE
# ================================================================

print("\n" + "=" * 70)
print("1. MNI TEMPLATE")
print("=" * 70)

try:

    mni_image, mni_data = read_image(
        MNI_TEMPLATE,
        sitk.sitkFloat32
    )

    checks[
        "mni_readable"
    ] = True

    print(
        "Readable: PASS"
    )

    print(
        "Size:",
        mni_image.GetSize()
    )

    print(
        "Spacing:",
        mni_image.GetSpacing()
    )

    print(
        "Origin:",
        mni_image.GetOrigin()
    )

    print(
        "Direction:",
        mni_image.GetDirection()
    )

except Exception as e:

    checks[
        "mni_readable"
    ] = False

    errors.append(
        f"MNI template validation failed: {e}"
    )

    mni_image = None
    mni_data = None

    print(
        "Readable: FAIL"
    )


# ================================================================
# CHECK AAL ATLAS
# ================================================================

print("\n" + "=" * 70)
print("2. AAL ATLAS")
print("=" * 70)

try:

    aal_image, aal_data = read_image(
        AAL_ATLAS,
        sitk.sitkFloat32
    )

    checks[
        "aal_readable"
    ] = True

    print(
        "Readable: PASS"
    )

    print(
        "Size:",
        aal_image.GetSize()
    )

    print(
        "Spacing:",
        aal_image.GetSpacing()
    )

    print(
        "Origin:",
        aal_image.GetOrigin()
    )

    print(
        "Direction:",
        aal_image.GetDirection()
    )

except Exception as e:

    checks[
        "aal_readable"
    ] = False

    errors.append(
        f"AAL atlas validation failed: {e}"
    )

    aal_image = None
    aal_data = None

    print(
        "Readable: FAIL"
    )


# ================================================================
# GRID COMPATIBILITY
# ================================================================

print("\n" + "=" * 70)
print("3. MNI / AAL GRID COMPATIBILITY")
print("=" * 70)

if (
    mni_image is not None
    and aal_image is not None
):

    checks[
        "mni_aal_geometry_match"
    ] = geometry_matches(
        mni_image,
        aal_image
    )

    if checks[
        "mni_aal_geometry_match"
    ]:

        print(
            "Grid compatibility: PASS"
        )

    else:

        print(
            "Grid compatibility: FAIL"
        )

        errors.append(
            "AAL atlas does not exactly match "
            "the MNI template grid."
        )

else:

    checks[
        "mni_aal_geometry_match"
    ] = False

    errors.append(
        "Cannot compare MNI/AAL geometry."
    )


# ================================================================
# ATLAS FINITE / NON-EMPTY
# ================================================================

print("\n" + "=" * 70)
print("4. ATLAS DATA VALIDATION")
print("=" * 70)

if aal_data is not None:

    finite_ok = np.all(
        np.isfinite(aal_data)
    )

    checks[
        "aal_finite"
    ] = finite_ok

    if finite_ok:

        print(
            "Finite values: PASS"
        )

    else:

        print(
            "Finite values: FAIL"
        )

        errors.append(
            "AAL atlas contains non-finite values."
        )


    nonzero_count = np.count_nonzero(
        aal_data
    )

    nonempty_ok = (
        nonzero_count > 0
    )

    checks[
        "aal_nonempty"
    ] = nonempty_ok

    print(
        "Non-zero voxels:",
        f"{nonzero_count:,}"
    )

    if nonempty_ok:

        print(
            "Non-empty: PASS"
        )

    else:

        print(
            "Non-empty: FAIL"
        )

        errors.append(
            "AAL atlas contains no positive voxels."
        )

else:

    checks[
        "aal_finite"
    ] = False

    checks[
        "aal_nonempty"
    ] = False


# ================================================================
# INTEGER LABEL VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("5. ROI LABEL VALIDATION")
print("=" * 70)

if aal_data is not None:

    rounded = np.rint(
        aal_data
    )

    maximum_integer_error = np.max(
        np.abs(
            aal_data
            - rounded
        )
    )

    print(
        "Maximum integer deviation:",
        maximum_integer_error
    )

    integer_ok = (
        maximum_integer_error
        <= INTEGER_ATOL
    )

    checks[
        "aal_integer_labels"
    ] = integer_ok

    if integer_ok:

        print(
            "Integer labels: PASS"
        )

    else:

        print(
            "Integer labels: FAIL"
        )

        errors.append(
            "AAL atlas contains non-integer ROI values."
        )

else:

    checks[
        "aal_integer_labels"
    ] = False


# ================================================================
# BACKGROUND VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("6. BACKGROUND LABEL")
print("=" * 70)

if aal_data is not None:

    negative_values = np.sum(
        aal_data < -INTEGER_ATOL
    )

    background_ok = (
        negative_values == 0
    )

    checks[
        "no_negative_labels"
    ] = background_ok

    if background_ok:

        print(
            "No negative labels: PASS"
        )

    else:

        print(
            "No negative labels: FAIL"
        )

        errors.append(
            "AAL atlas contains negative label values."
        )

else:

    checks[
        "no_negative_labels"
    ] = False


# ================================================================
# EXACT AAL LABEL SET
# ================================================================

print("\n" + "=" * 70)
print("7. EXACT AAL LABEL SET")
print("=" * 70)

if aal_data is not None:

    positive_labels = np.unique(
        rounded[
            rounded > 0
        ]
    ).astype(
        np.int32
    )

    print(
        "Positive ROI count:",
        len(positive_labels)
    )

    print(
        "Expected ROI count:",
        EXPECTED_ROIS
    )

    print(
        "Observed labels:",
        positive_labels.tolist()
    )

    label_set_ok = np.array_equal(
        positive_labels,
        EXPECTED_LABELS
    )

    checks[
        "exact_116_labels"
    ] = label_set_ok

    if label_set_ok:

        print(
            "Exact labels 1-116: PASS"
        )

    else:

        missing = sorted(
            set(
                EXPECTED_LABELS.tolist()
            )
            -
            set(
                positive_labels.tolist()
            )
        )

        unexpected = sorted(
            set(
                positive_labels.tolist()
            )
            -
            set(
                EXPECTED_LABELS.tolist()
            )
        )

        print(
            "Missing labels:",
            missing
        )

        print(
            "Unexpected labels:",
            unexpected
        )

        errors.append(
            "AAL atlas does not contain exactly "
            "labels 1-116."
        )

else:

    checks[
        "exact_116_labels"
    ] = False


# ================================================================
# LABEL COUNTS
# ================================================================

print("\n" + "=" * 70)
print("8. ROI VOXEL COUNTS")
print("=" * 70)

roi_counts = {}

if aal_data is not None:

    for label in EXPECTED_LABELS:

        count = int(
            np.sum(
                rounded == label
            )
        )

        roi_counts[
            int(label)
        ] = count

    zero_voxel_rois = [
        label
        for label, count in roi_counts.items()
        if count == 0
    ]

    if zero_voxel_rois:

        checks[
            "all_rois_nonempty"
        ] = False

        errors.append(
            "One or more AAL ROIs contain zero voxels: "
            f"{zero_voxel_rois}"
        )

        print(
            "All ROIs non-empty: FAIL"
        )

    else:

        checks[
            "all_rois_nonempty"
        ] = True

        print(
            "All ROIs non-empty: PASS"
        )

    print(
        "Minimum ROI voxel count:",
        min(
            roi_counts.values()
        )
    )

    print(
        "Maximum ROI voxel count:",
        max(
            roi_counts.values()
        )
    )

else:

    checks[
        "all_rois_nonempty"
    ] = False


# ================================================================
# WRITE SUMMARY
# ================================================================

summary_lines = []

summary_lines.append(
    "AAL ATLAS QUALITY CONTROL"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append(
    f"MNI template: {MNI_TEMPLATE}"
)

summary_lines.append(
    f"AAL atlas: {AAL_ATLAS}"
)

summary_lines.append(
    f"Expected ROIs: {EXPECTED_ROIS}"
)

summary_lines.append("")

for name, value in checks.items():

    summary_lines.append(
        f"{name}: {value}"
    )

summary_lines.append("")

if errors:

    summary_lines.append(
        "ERRORS:"
    )

    for error in errors:

        summary_lines.append(
            f"- {error}"
        )

else:

    summary_lines.append(
        "ERRORS: None"
    )


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("AAL ATLAS QC SUMMARY")
print("=" * 70)

print(
    "Expected ROIs:",
    EXPECTED_ROIS
)

print(
    "Checks passed:",
    sum(
        bool(value)
        for value in checks.values()
    ),
    "/",
    len(checks)
)

print(
    "Errors:",
    len(errors)
)

print(
    "Summary:",
    SUMMARY_FILE
)


if len(errors) == 0:

    print(
        "\nSTATUS: AUTOMATED QC PASS"
    )

else:

    print(
        "\nSTATUS: QC FAILED"
    )

    print("\nProblems:")

    for error in errors:

        print(
            " -",
            error
        )


print("=" * 70)