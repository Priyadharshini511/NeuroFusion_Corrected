# src\qc_masked_batch.py

import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk


# ================================================================
# MASKED sMRI QUALITY CONTROL
#
# Checks all 5 echoes for the currently selected subjects.
#
# Input:
#   N4-corrected echoes
#   HD-BET Echo-1 mask
#   Masked 5-echo MRI
#
# Checks:
#   - Exact subject/echo completeness
#   - Readability
#   - Finite values
#   - Non-empty data
#   - RAS/direction consistency
#   - Geometry consistency with N4
#   - Echo-to-echo affine consistency
#   - Background remains zero
#   - Values outside HD-BET mask are zero
#
# This script is QC only. It does not modify preprocessing data.
# ================================================================


print("=" * 70)
print("MASKED sMRI QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

N4_DIR = (
    r"processed_data\smri"
    r"\n4"
)

MASK_DIR = (
    r"processed_data\smri"
    r"\mprage_5e_bet"
)

MASKED_DIR = (
    r"processed_data\smri"
    r"\mprage_5e_masked"
)

MANIFEST_PATH = (
    r"processed_data\smri"
    r"\mprage_selection"
    r"\mprage_selection_manifest.csv"
)

QC_DIR = (
    r"processed_data\smri"
    r"\qc_masked"
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "masked_qc_summary.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ECHOES = [1, 2, 3, 4, 5]

GEOMETRY_ATOL = 1e-5


# ================================================================
# HELPERS
# ================================================================

def get_selected_subjects():

    if not os.path.isfile(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Selection manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "subject_id",
        "selection_status"
    }

    missing = (
        required
        - set(manifest.columns)
    )

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
    ]

    if selected.empty:
        raise ValueError(
            "No SELECTED subjects found in manifest."
        )

    subjects = (
        selected["subject_id"]
        .astype(str)
        .str.strip()
    )

    if subjects.eq("").any():
        raise ValueError(
            "Manifest contains empty subject IDs."
        )

    if subjects.duplicated().any():

        duplicates = sorted(
            subjects[
                subjects.duplicated()
            ].unique()
        )

        raise ValueError(
            "Duplicate SELECTED subjects found: "
            f"{duplicates}"
        )

    return sorted(
        subjects.unique()
    )


def read_valid_image(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    image = sitk.ReadImage(path)

    data = sitk.GetArrayFromImage(
        image
    )

    if data.size == 0:
        raise ValueError(
            "Image is empty."
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


# ================================================================
# PATH CHECKS
# ================================================================

for path, description in [
    (N4_DIR, "N4 directory"),
    (MASK_DIR, "HD-BET mask directory"),
    (MASKED_DIR, "masked MRI directory")
]:

    if not os.path.isdir(path):

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


os.makedirs(
    QC_DIR,
    exist_ok=True
)


# ================================================================
# CURRENT SUBJECT COHORT
# ================================================================

subjects = get_selected_subjects()

expected_pairs = {
    (subject, echo)
    for subject in subjects
    for echo in EXPECTED_ECHOES
}

expected_count = len(
    expected_pairs
)


print("\nCurrent cohort:")
print(
    "Subjects:",
    len(subjects)
)

print(
    "Echoes:",
    EXPECTED_ECHOES
)

print(
    "Expected masked files:",
    expected_count
)


# ================================================================
# QC LOOP
# ================================================================

results = []

passed = 0
failed = 0


for index, subject_id in enumerate(
    subjects,
    start=1
):

    print("\n" + "=" * 70)
    print(
        f"QC {index}/{len(subjects)} — {subject_id}"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Load Echo-1 N4 as reference geometry.
    # ------------------------------------------------------------

    reference_path = os.path.join(
        N4_DIR,
        f"{subject_id}_echo1_n4.nii.gz"
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{subject_id}_echo1_n4_bet_mask.nii.gz"
    )

    checks = {
        "reference_valid": False,
        "mask_valid": False,
        "mask_binary": False,
        "mask_geometry_matches_n4": False,
        "all_echoes_valid": False,
        "all_echoes_geometry_match_n4": False,
        "all_echoes_affine_consistent": False,
        "all_background_zero": False,
        "all_values_inside_mask": False
    }

    errors = []

    reference_image = None
    mask_image = None
    mask_data = None

    # ------------------------------------------------------------
    # Reference N4
    # ------------------------------------------------------------

    try:

        reference_image, reference_data = (
            read_valid_image(
                reference_path
            )
        )

        checks[
            "reference_valid"
        ] = True

        print(
            "Reference N4 Echo-1: PASS"
        )

    except Exception as e:

        errors.append(
            f"Reference N4 validation failed: {e}"
        )

        print(
            "Reference N4: FAIL"
        )


    # ------------------------------------------------------------
    # HD-BET mask
    # ------------------------------------------------------------

    if reference_image is not None:

        try:

            mask_image, mask_data = (
                read_valid_image(
                    mask_path
                )
            )

            checks[
                "mask_valid"
            ] = True

            unique_mask = np.unique(
                mask_data
            )

            if np.all(
                np.isin(
                    unique_mask,
                    [0, 1]
                )
            ):

                checks[
                    "mask_binary"
                ] = True

            else:

                errors.append(
                    "HD-BET mask is not binary."
                )

            if geometry_matches(
                mask_image,
                reference_image
            ):

                checks[
                    "mask_geometry_matches_n4"
                ] = True

            else:

                errors.append(
                    "HD-BET mask geometry does not "
                    "match N4 Echo-1."
                )

            print(
                "HD-BET mask: PASS"
            )

        except Exception as e:

            errors.append(
                f"HD-BET mask validation failed: {e}"
            )

            print(
                "HD-BET mask: FAIL"
            )


    # ------------------------------------------------------------
    # Process five masked echoes
    # ------------------------------------------------------------

    echo_images = {}
    echo_data = {}

    all_echoes_ok = True
    all_geometry_ok = True
    all_affines_ok = True
    all_background_zero = True
    all_inside_mask = True

    reference_affine = None

    for echo in EXPECTED_ECHOES:

        masked_path = os.path.join(
            MASKED_DIR,
            f"{subject_id}_echo{echo}_masked.nii.gz"
        )

        print(
            f"\nEcho {echo}:"
        )

        try:

            image, data = read_valid_image(
                masked_path
            )

            echo_images[echo] = image
            echo_data[echo] = data

            # ----------------------------------------------------
            # Geometry
            # ----------------------------------------------------

            if reference_image is not None:

                if geometry_matches(
                    image,
                    reference_image
                ):

                    print(
                        "  Geometry: PASS"
                    )

                else:

                    all_geometry_ok = False
                    all_echoes_ok = False

                    errors.append(
                        f"Echo {echo}: geometry does not "
                        "match N4 reference."
                    )

                    print(
                        "  Geometry: FAIL"
                    )

            # ----------------------------------------------------
            # Affine
            # ----------------------------------------------------

            affine = affine_from_sitk(
                image
            )

            if reference_affine is None:

                reference_affine = affine

            else:

                if np.allclose(
                    affine,
                    reference_affine,
                    atol=GEOMETRY_ATOL
                ):

                    print(
                        "  Affine: PASS"
                    )

                else:

                    all_affines_ok = False
                    all_echoes_ok = False

                    errors.append(
                        f"Echo {echo}: affine differs "
                        "from another echo."
                    )

                    print(
                        "  Affine: FAIL"
                    )

            # ----------------------------------------------------
            # Background zero
            # ----------------------------------------------------

            background = (
                data[
                    mask_data == 0
                ]
                if mask_data is not None
                and mask_data.shape == data.shape
                else np.array([])
            )

            if background.size == 0:

                background_zero = True

            else:

                background_zero = np.allclose(
                    background,
                    0,
                    atol=1e-6
                )

            if background_zero:

                print(
                    "  Background zero: PASS"
                )

            else:

                all_background_zero = False
                all_echoes_ok = False

                errors.append(
                    f"Echo {echo}: non-zero values "
                    "outside mask."
                )

                print(
                    "  Background zero: FAIL"
                )

            # ----------------------------------------------------
            # Masked region contains data
            # ----------------------------------------------------

            if mask_data is not None:

                if mask_data.shape == data.shape:

                    inside = data[
                        mask_data > 0
                    ]

                    if inside.size == 0:

                        all_inside_mask = False
                        all_echoes_ok = False

                        errors.append(
                            f"Echo {echo}: no voxels inside mask."
                        )

                    elif not np.any(
                        np.abs(inside) > 0
                    ):

                        all_inside_mask = False
                        all_echoes_ok = False

                        errors.append(
                            f"Echo {echo}: masked brain region "
                            "contains no signal."
                        )

                else:

                    all_inside_mask = False
                    all_echoes_ok = False

            print(
                "  Finite/non-empty: PASS"
            )

        except Exception as e:

            all_echoes_ok = False

            errors.append(
                f"Echo {echo} validation failed: {e}"
            )

            print(
                "  Status: FAIL"
            )


    # ------------------------------------------------------------
    # Final subject checks
    # ------------------------------------------------------------

    checks[
        "all_echoes_valid"
    ] = (
        all_echoes_ok
        and len(echo_images)
        == len(EXPECTED_ECHOES)
    )

    checks[
        "all_echoes_geometry_match_n4"
    ] = (
        all_geometry_ok
        and checks[
            "reference_valid"
        ]
    )

    checks[
        "all_echoes_affine_consistent"
    ] = all_affines_ok

    checks[
        "all_background_zero"
    ] = all_background_zero

    checks[
        "all_values_inside_mask"
    ] = all_inside_mask


    # ------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------

    if len(errors) == 0:

        status = "PASS"
        passed += 1

    else:

        status = "FAIL"
        failed += 1


    results.append(
        {
            "subject_id": subject_id,
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
# FINAL COHORT-LEVEL FILE CHECK
# ================================================================

print("\n" + "=" * 70)
print("COHORT-LEVEL FILE COMPLETENESS CHECK")
print("=" * 70)

actual_pairs = set()

for filename in os.listdir(
    MASKED_DIR
):

    if not filename.endswith(
        "_masked.nii.gz"
    ):
        continue

    match = re.match(
        r"^(A\d+)_echo([1-5])_masked\.nii\.gz$",
        filename
    )

    if match is None:
        continue

    subject_id = match.group(1)
    echo = int(
        match.group(2)
    )

    if subject_id in subjects:

        actual_pairs.add(
            (
                subject_id,
                echo
            )
        )


missing_pairs = sorted(
    expected_pairs
    - actual_pairs
)

unexpected_pairs = sorted(
    actual_pairs
    - expected_pairs
)


print(
    "Expected current files:",
    expected_count
)

print(
    "Found current files:",
    len(actual_pairs)
)

print(
    "Missing pairs:",
    len(missing_pairs)
)

print(
    "Unexpected pairs:",
    len(unexpected_pairs)
)


if missing_pairs:

    print("\nMissing:")

    for subject, echo in missing_pairs:
        print(
            f"  {subject} Echo {echo}"
        )


if unexpected_pairs:

    print("\nUnexpected:")

    for subject, echo in unexpected_pairs:
        print(
            f"  {subject} Echo {echo}"
        )


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("MASKED sMRI QC SUMMARY")
print("=" * 70)

print(
    "Selected subjects:",
    len(subjects)
)

print(
    "Expected files:",
    expected_count
)

print(
    "Found current files:",
    len(actual_pairs)
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
    "Summary CSV:",
    SUMMARY_CSV
)


cohort_pass = (
    len(missing_pairs) == 0
    and len(unexpected_pairs) == 0
    and failed == 0
)


if cohort_pass:

    print(
        "\nSTATUS: AUTOMATED QC PASS"
    )

else:

    print(
        "\nSTATUS: QC FAILED"
    )


print("=" * 70)