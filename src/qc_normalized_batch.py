# # src\qc_normalized_batch.py

# import os
# import re
# import numpy as np
# import pandas as pd
# import SimpleITK as sitk


# # ================================================================
# # NORMALIZED sMRI QUALITY CONTROL
# #
# # Generic, manifest-driven QC for all 5 normalized echoes.
# #
# # Expected normalization methodology:
# #   1. Brain voxels = masked input > 0
# #   2. 1st percentile / 99th percentile
# #   3. Clip brain intensities to those limits
# #   4. Scale brain intensities to [0, 1]
# #   5. Background remains exactly 0
# #
# # This script verifies the output.
# # It does NOT perform normalization and does NOT modify data.
# # ================================================================


# print("=" * 70)
# print("NORMALIZED sMRI QUALITY CONTROL")
# print("=" * 70)


# # ================================================================
# # PATHS
# # ================================================================

# MASKED_DIR = (
#     r"processed_data\smri"
#     r"\mprage_5e_masked"
# )

# NORMALIZED_DIR = (
#     r"processed_data\smri"
#     r"\mprage_5e_normalized"
# )

# MANIFEST_PATH = (
#     r"processed_data\smri"
#     r"\mprage_selection"
#     r"\mprage_selection_manifest.csv"
# )

# QC_DIR = (
#     r"processed_data\smri"
#     r"\qc_normalized"
# )

# SUMMARY_CSV = os.path.join(
#     QC_DIR,
#     "normalized_qc_summary.csv"
# )


# # ================================================================
# # CONFIGURATION
# # ================================================================

# EXPECTED_ECHOES = [1, 2, 3, 4, 5]

# GEOMETRY_ATOL = 1e-5

# # Floating-point tolerance for [0, 1].
# VALUE_ATOL = 1e-5

# # Background should be zero within this tolerance.
# BACKGROUND_ATOL = 1e-6

# # Minimum number of non-zero voxels required.
# MIN_NONZERO_VOXELS = 10


# # ================================================================
# # HELPERS
# # ================================================================

# def get_selected_subjects():

#     if not os.path.isfile(MANIFEST_PATH):

#         raise FileNotFoundError(
#             f"Selection manifest not found:\n{MANIFEST_PATH}"
#         )

#     manifest = pd.read_csv(
#         MANIFEST_PATH
#     )

#     required_columns = {
#         "subject_id",
#         "selection_status"
#     }

#     missing = (
#         required_columns
#         - set(manifest.columns)
#     )

#     if missing:

#         raise ValueError(
#             "Manifest is missing required columns: "
#             f"{sorted(missing)}"
#         )

#     selected = manifest[
#         manifest["selection_status"]
#         .astype(str)
#         .str.upper()
#         .eq("SELECTED")
#     ].copy()

#     if selected.empty:

#         raise ValueError(
#             "No SELECTED subjects found in manifest."
#         )

#     subjects = (
#         selected["subject_id"]
#         .astype(str)
#         .str.strip()
#     )

#     if subjects.eq("").any():

#         raise ValueError(
#             "Manifest contains empty subject IDs."
#         )

#     if subjects.duplicated().any():

#         duplicates = sorted(
#             subjects[
#                 subjects.duplicated()
#             ].unique()
#         )

#         raise ValueError(
#             "Duplicate SELECTED subjects found: "
#             f"{duplicates}"
#         )

#     return sorted(
#         subjects.unique()
#     )


# def read_image(path):

#     if not os.path.isfile(path):

#         raise FileNotFoundError(
#             f"File not found: {path}"
#         )

#     image = sitk.ReadImage(
#         path
#     )

#     data = sitk.GetArrayFromImage(
#         image
#     ).astype(
#         np.float32
#     )

#     if data.size == 0:

#         raise ValueError(
#             "Image contains no voxels."
#         )

#     if not np.all(
#         np.isfinite(data)
#     ):

#         raise ValueError(
#             "Image contains NaN or infinite values."
#         )

#     return image, data


# def geometry_matches(
#     image_a,
#     image_b
# ):

#     return (
#         image_a.GetSize()
#         == image_b.GetSize()
#         and np.allclose(
#             image_a.GetSpacing(),
#             image_b.GetSpacing(),
#             atol=GEOMETRY_ATOL
#         )
#         and np.allclose(
#             image_a.GetOrigin(),
#             image_b.GetOrigin(),
#             atol=GEOMETRY_ATOL
#         )
#         and np.allclose(
#             image_a.GetDirection(),
#             image_b.GetDirection(),
#             atol=GEOMETRY_ATOL
#         )
#     )


# def affine_from_sitk(image):

#     spacing = np.asarray(
#         image.GetSpacing(),
#         dtype=np.float64
#     )

#     origin = np.asarray(
#         image.GetOrigin(),
#         dtype=np.float64
#     )

#     direction = np.asarray(
#         image.GetDirection(),
#         dtype=np.float64
#     ).reshape(3, 3)

#     affine = np.eye(4)

#     affine[:3, :3] = (
#         direction
#         @ np.diag(spacing)
#     )

#     affine[:3, 3] = origin

#     return affine


# def check_normalized_values(
#     data
# ):

#     nonzero = data[
#         np.abs(data) > BACKGROUND_ATOL
#     ]

#     if nonzero.size == 0:

#         return False, None, None

#     minimum = float(
#         np.min(nonzero)
#     )

#     maximum = float(
#         np.max(nonzero)
#     )

#     valid = (
#         minimum >= -VALUE_ATOL
#         and maximum <= 1.0 + VALUE_ATOL
#     )

#     return (
#         valid,
#         minimum,
#         maximum
#     )


# # ================================================================
# # PATH CHECKS
# # ================================================================

# for path, description in [
#     (
#         MASKED_DIR,
#         "masked MRI directory"
#     ),
#     (
#         NORMALIZED_DIR,
#         "normalized MRI directory"
#     )
# ]:

#     if not os.path.isdir(path):

#         raise FileNotFoundError(
#             f"{description} not found:\n{path}"
#         )


# os.makedirs(
#     QC_DIR,
#     exist_ok=True
# )


# # ================================================================
# # CURRENT SUBJECT COHORT
# # ================================================================

# subjects = get_selected_subjects()

# expected_pairs = {
#     (
#         subject,
#         echo
#     )
#     for subject in subjects
#     for echo in EXPECTED_ECHOES
# }

# expected_count = len(
#     expected_pairs
# )


# print("\n" + "-" * 70)
# print("CURRENT COHORT")
# print("-" * 70)

# print(
#     "Selected subjects:",
#     len(subjects)
# )

# print(
#     "Expected echoes:",
#     EXPECTED_ECHOES
# )

# print(
#     "Expected normalized files:",
#     expected_count
# )


# # ================================================================
# # QC LOOP
# # ================================================================

# results = []

# passed = 0
# failed = 0


# for index, subject_id in enumerate(
#     subjects,
#     start=1
# ):

#     print("\n" + "=" * 70)
#     print(
#         f"QC {index}/{len(subjects)} — {subject_id}"
#     )
#     print("=" * 70)

#     subject_errors = []

#     subject_checks = {
#         "all_masked_inputs_valid": False,
#         "all_normalized_outputs_valid": False,
#         "all_geometry_match": False,
#         "all_affines_match": False,
#         "all_background_zero": False,
#         "all_normalized_range_valid": False,
#         "all_nonempty": False
#     }

#     masked_images = {}
#     masked_data = {}

#     normalized_images = {}
#     normalized_data = {}

#     all_inputs_valid = True
#     all_outputs_valid = True
#     all_geometry_valid = True
#     all_affines_valid = True
#     all_background_valid = True
#     all_range_valid = True
#     all_nonempty = True

#     reference_masked = None
#     reference_normalized = None

#     reference_affine = None


#     # ============================================================
#     # PROCESS EACH ECHO
#     # ============================================================

#     for echo in EXPECTED_ECHOES:

#         print(
#             f"\nEcho {echo}"
#         )

#         masked_path = os.path.join(
#             MASKED_DIR,
#             f"{subject_id}_echo{echo}_masked.nii.gz"
#         )

#         normalized_path = os.path.join(
#             NORMALIZED_DIR,
#             f"{subject_id}_echo{echo}_normalized.nii.gz"
#         )

#         # --------------------------------------------------------
#         # Load masked input
#         # --------------------------------------------------------

#         try:

#             masked_image, masked_array = (
#                 read_image(
#                     masked_path
#                 )
#             )

#             masked_images[echo] = masked_image
#             masked_data[echo] = masked_array

#             print(
#                 "  Masked input: PASS"
#             )

#         except Exception as e:

#             all_inputs_valid = False
#             all_geometry_valid = False
#             all_affines_valid = False

#             subject_errors.append(
#                 f"Echo {echo}: masked input failed: {e}"
#             )

#             print(
#                 "  Masked input: FAIL"
#             )

#             continue


#         # --------------------------------------------------------
#         # Load normalized output
#         # --------------------------------------------------------

#         try:

#             normalized_image, normalized_array = (
#                 read_image(
#                     normalized_path
#                 )
#             )

#             normalized_images[echo] = (
#                 normalized_image
#             )

#             normalized_data[echo] = (
#                 normalized_array
#             )

#             print(
#                 "  Normalized output: PASS"
#             )

#         except Exception as e:

#             all_outputs_valid = False
#             all_geometry_valid = False
#             all_affines_valid = False
#             all_background_valid = False
#             all_range_valid = False

#             subject_errors.append(
#                 f"Echo {echo}: normalized output failed: {e}"
#             )

#             print(
#                 "  Normalized output: FAIL"
#             )

#             continue


#         # --------------------------------------------------------
#         # Non-empty check
#         # --------------------------------------------------------

#         nonzero_count = np.count_nonzero(
#             np.abs(normalized_array)
#             > BACKGROUND_ATOL
#         )

#         if nonzero_count < MIN_NONZERO_VOXELS:

#             all_nonempty = False
#             all_outputs_valid = False

#             subject_errors.append(
#                 f"Echo {echo}: normalized image is empty "
#                 f"or nearly empty ({nonzero_count} voxels)."
#             )

#             print(
#                 "  Non-empty: FAIL"
#             )

#         else:

#             print(
#                 "  Non-empty: PASS"
#             )


#         # --------------------------------------------------------
#         # Geometry: normalized vs masked
#         # --------------------------------------------------------

#         if geometry_matches(
#             normalized_image,
#             masked_image
#         ):

#             print(
#                 "  Geometry vs masked input: PASS"
#             )

#         else:

#             all_geometry_valid = False
#             all_outputs_valid = False

#             subject_errors.append(
#                 f"Echo {echo}: normalized geometry "
#                 "does not match masked input."
#             )

#             print(
#                 "  Geometry vs masked input: FAIL"
#             )


#         # --------------------------------------------------------
#         # Affine consistency
#         # --------------------------------------------------------

#         normalized_affine = affine_from_sitk(
#             normalized_image
#         )

#         if reference_affine is None:

#             reference_affine = normalized_affine

#         else:

#             if np.allclose(
#                 normalized_affine,
#                 reference_affine,
#                 atol=GEOMETRY_ATOL
#             ):

#                 print(
#                     "  Echo affine consistency: PASS"
#                 )

#             else:

#                 all_affines_valid = False
#                 all_outputs_valid = False

#                 subject_errors.append(
#                     f"Echo {echo}: normalized affine "
#                     "differs from other echoes."
#                 )

#                 print(
#                     "  Echo affine consistency: FAIL"
#                 )


#         # --------------------------------------------------------
#         # Background preservation
#         #
#         # The masked input has background == 0.
#         # The normalized output must also remain 0 there.
#         # --------------------------------------------------------

#         if masked_array.shape != normalized_array.shape:

#             all_background_valid = False
#             all_outputs_valid = False

#             subject_errors.append(
#                 f"Echo {echo}: masked and normalized "
#                 "array shapes differ."
#             )

#         else:

#             background_values = normalized_array[
#                 masked_array == 0
#             ]

#             if background_values.size == 0:

#                 background_ok = True

#             else:

#                 background_ok = np.allclose(
#                     background_values,
#                     0,
#                     atol=BACKGROUND_ATOL
#                 )

#             if background_ok:

#                 print(
#                     "  Background zero: PASS"
#                 )

#             else:

#                 all_background_valid = False
#                 all_outputs_valid = False

#                 subject_errors.append(
#                     f"Echo {echo}: normalized output "
#                     "contains non-zero background."
#                 )

#                 print(
#                     "  Background zero: FAIL"
#                 )


#         # --------------------------------------------------------
#         # Normalized [0,1] range
#         # --------------------------------------------------------

#         range_valid, minimum, maximum = (
#             check_normalized_values(
#                 normalized_array
#             )
#         )

#         if range_valid:

#             print(
#                 "  Normalized range: PASS "
#                 f"[{minimum:.6f}, {maximum:.6f}]"
#             )

#         else:

#             all_range_valid = False
#             all_outputs_valid = False

#             if minimum is None:

#                 subject_errors.append(
#                     f"Echo {echo}: no non-zero "
#                     "normalized brain voxels."
#                 )

#             else:

#                 subject_errors.append(
#                     f"Echo {echo}: normalized values "
#                     f"outside [0,1]: "
#                     f"[{minimum:.6f}, {maximum:.6f}]"
#                 )

#             print(
#                 "  Normalized range: FAIL"
#             )


#         # --------------------------------------------------------
#         # Keep reference geometry
#         # --------------------------------------------------------

#         if reference_masked is None:
#             reference_masked = masked_image

#         if reference_normalized is None:
#             reference_normalized = normalized_image


#     # ============================================================
#     # FINAL SUBJECT CHECKS
#     # ============================================================

#     subject_checks[
#         "all_masked_inputs_valid"
#     ] = (
#         all_inputs_valid
#         and len(masked_images)
#         == len(EXPECTED_ECHOES)
#     )

#     subject_checks[
#         "all_normalized_outputs_valid"
#     ] = (
#         all_outputs_valid
#         and len(normalized_images)
#         == len(EXPECTED_ECHOES)
#     )

#     subject_checks[
#         "all_geometry_match"
#     ] = all_geometry_valid

#     subject_checks[
#         "all_affines_match"
#     ] = all_affines_valid

#     subject_checks[
#         "all_background_zero"
#     ] = all_background_valid

#     subject_checks[
#         "all_normalized_range_valid"
#     ] = all_range_valid

#     subject_checks[
#         "all_nonempty"
#     ] = all_nonempty


#     # ============================================================
#     # STATUS
#     # ============================================================

#     if len(subject_errors) == 0:

#         status = "PASS"
#         passed += 1

#     else:

#         status = "FAIL"
#         failed += 1


#     results.append(
#         {
#             "subject_id": subject_id,
#             "status": status,
#             "error_count": len(subject_errors),
#             "errors": " | ".join(
#                 subject_errors
#             ),
#             **subject_checks
#         }
#     )


#     print(
#         "\nSUBJECT STATUS:",
#         status
#     )

#     if subject_errors:

#         for error in subject_errors:

#             print(
#                 " -",
#                 error
#             )


# # ================================================================
# # COHORT-LEVEL FILE COMPLETENESS
# # ================================================================

# print("\n" + "=" * 70)
# print("COHORT-LEVEL FILE COMPLETENESS")
# print("=" * 70)


# actual_pairs = set()

# pattern = re.compile(
#     r"^(A\d+)_echo([1-5])_normalized\.nii\.gz$"
# )


# for filename in os.listdir(
#     NORMALIZED_DIR
# ):

#     match = pattern.match(
#         filename
#     )

#     if match is None:
#         continue

#     subject_id = match.group(1)
#     echo = int(
#         match.group(2)
#     )

#     if subject_id in subjects:

#         actual_pairs.add(
#             (
#                 subject_id,
#                 echo
#             )
#         )


# missing_pairs = sorted(
#     expected_pairs
#     - actual_pairs
# )

# unexpected_pairs = sorted(
#     actual_pairs
#     - expected_pairs
# )


# print(
#     "Expected current files:",
#     expected_count
# )

# print(
#     "Found current files:",
#     len(actual_pairs)
# )

# print(
#     "Missing pairs:",
#     len(missing_pairs)
# )

# print(
#     "Unexpected pairs:",
#     len(unexpected_pairs)
# )


# if missing_pairs:

#     print("\nMissing:")

#     for subject, echo in missing_pairs:

#         print(
#             f"  {subject} Echo {echo}"
#         )


# if unexpected_pairs:

#     print("\nUnexpected:")

#     for subject, echo in unexpected_pairs:

#         print(
#             f"  {subject} Echo {echo}"
#         )


# # ================================================================
# # SAVE SUMMARY
# # ================================================================

# summary_df = pd.DataFrame(
#     results
# )

# summary_df.to_csv(
#     SUMMARY_CSV,
#     index=False
# )


# # ================================================================
# # FINAL STATUS
# # ================================================================

# print("\n" + "=" * 70)
# print("NORMALIZED sMRI QC SUMMARY")
# print("=" * 70)

# print(
#     "Selected subjects:",
#     len(subjects)
# )

# print(
#     "Expected files:",
#     expected_count
# )

# print(
#     "Found current files:",
#     len(actual_pairs)
# )

# print(
#     "Subjects PASS:",
#     passed
# )

# print(
#     "Subjects FAIL:",
#     failed
# )

# print(
#     "Summary CSV:",
#     SUMMARY_CSV
# )


# cohort_pass = (
#     len(missing_pairs) == 0
#     and len(unexpected_pairs) == 0
#     and failed == 0
# )


# if cohort_pass:

#     print(
#         "\nSTATUS: AUTOMATED QC PASS"
#     )

# else:

#     print(
#         "\nSTATUS: QC FAILED"
#     )


# print("=" * 70)



# src\qc_normalized_batch.py

from pathlib import Path
import csv

import nibabel as nib
import numpy as np


# ================================================================
# GENERIC NORMALIZED sMRI QUALITY CONTROL
#
# Supports:
#   1. 5-ECHO MPRAGE
#      - validates all five normalized echoes
#
#   2. RMS MPRAGE
#      - validates the single normalized RMS volume
#
# INPUT:
#   processed_data/smri/mprage_5e_masked
#
# OUTPUT:
#   processed_data/smri/mprage_5e_normalized
#
# SUBJECT SELECTION:
#   processed_data/smri/mprage_selection/
#       mprage_selection_manifest.csv
#
# IMPORTANT:
#   - Manifest-driven
#   - Exactly one selected acquisition per subject
#   - Supports RMS and 5ECHO
#   - Legacy files in shared folders are ignored
#   - Exact expected filenames are derived from acquisition type
#   - Does NOT modify MRI data
#   - QC only
#
# NORMALIZATION METHODOLOGY VERIFIED:
#   1. Brain voxels = masked input > 0
#   2. 1st percentile
#   3. 99th percentile
#   4. Clip to percentile range
#   5. Scale brain values to [0,1]
#   6. Background remains exactly 0
# ================================================================


# ================================================================
# PATHS
# ================================================================

MASKED_DIR = Path(
    "processed_data/smri/mprage_5e_masked"
)

NORMALIZED_DIR = Path(
    "processed_data/smri/mprage_5e_normalized"
)

MANIFEST_PATH = Path(
    "processed_data/smri/mprage_selection/"
    "mprage_selection_manifest.csv"
)

QC_DIR = Path(
    "processed_data/smri/qc_normalized"
)

SUMMARY_CSV = (
    QC_DIR /
    "normalized_qc_summary.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

ECHOES = [1, 2, 3, 4, 5]

VOXEL_TOLERANCE = 0.01

AFFINE_TOLERANCE = 1e-3

VALUE_ATOL = 1e-5

BACKGROUND_ATOL = 1e-6

MIN_NONZERO_VOXELS = 10


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("GENERIC NORMALIZED sMRI QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATH CHECKS
# ================================================================

print("\nMasked input folder:")
print(
    f"  {MASKED_DIR.resolve()}"
)

print("\nNormalized output folder:")
print(
    f"  {NORMALIZED_DIR.resolve()}"
)

print("\nSelection manifest:")
print(
    f"  {MANIFEST_PATH.resolve()}"
)


if not MASKED_DIR.exists():

    raise FileNotFoundError(
        "Masked input directory does not exist:\n"
        f"{MASKED_DIR.resolve()}"
    )


if not NORMALIZED_DIR.exists():

    raise FileNotFoundError(
        "Normalized output directory does not exist:\n"
        f"{NORMALIZED_DIR.resolve()}"
    )


if not MANIFEST_PATH.exists():

    raise FileNotFoundError(
        "Selection manifest does not exist:\n"
        f"{MANIFEST_PATH.resolve()}"
    )


QC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# LOAD SELECTED ACQUISITIONS
# ================================================================

def load_selected_acquisitions(
    manifest_path
):
    """
    Load exactly one SELECTED acquisition per subject.

    Required manifest columns:
        subject
        selection_status
        acquisition_type

    Supports:
        RMS
        5ECHO
    """

    selected = []

    with manifest_path.open(
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
            "selection_status",
            "acquisition_type"
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
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
                    "A SELECTED manifest row has "
                    "no subject ID."
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
                    "acquisition_type":
                        acquisition_type,
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
            "No SELECTED acquisitions found "
            "in manifest."
        )

    subjects = [
        item["subject"]
        for item in selected
    ]

    if len(subjects) != len(
        set(subjects)
    ):

        duplicates = sorted(
            {
                subject
                for subject in subjects
                if subjects.count(subject) > 1
            }
        )

        raise ValueError(
            "More than one SELECTED acquisition "
            "exists for subject(s):\n"
            + "\n".join(duplicates)
        )

    return sorted(
        selected,
        key=lambda x: x["subject"]
    )


selected_acquisitions = (
    load_selected_acquisitions(
        MANIFEST_PATH
    )
)


# ================================================================
# CURRENT COHORT
# ================================================================

print("\n" + "-" * 70)
print("CURRENT COHORT")
print("-" * 70)

for item in selected_acquisitions:

    print(
        f"  {item['subject']} | "
        f"{item['acquisition_type']} | "
        f"{item['session']} | "
        f"{item['series']}"
    )


print(
    f"\nSelected subjects: "
    f"{len(selected_acquisitions)}"
)


# ================================================================
# ACQUISITION SUMMARY
# ================================================================

type_counts = {
    "RMS": 0,
    "5ECHO": 0
}


for item in selected_acquisitions:

    type_counts[
        item["acquisition_type"]
    ] += 1


print("\nAcquisition types:")

print(
    f"  RMS   : "
    f"{type_counts['RMS']}"
)

print(
    f"  5ECHO : "
    f"{type_counts['5ECHO']}"
)


# ================================================================
# BUILD EXACT EXPECTED FILENAMES
# ================================================================

expected_files = set()


for item in selected_acquisitions:

    subject = item["subject"]

    acquisition_type = (
        item["acquisition_type"]
    )

    if acquisition_type == "RMS":

        expected_files.add(
            f"{subject}_rms_n4_normalized.nii.gz"
        )

    elif acquisition_type == "5ECHO":

        for echo in ECHOES:

            expected_files.add(
                f"{subject}_echo{echo}_n4_normalized.nii.gz"
            )


expected_count = len(
    expected_files
)


print(
    f"\nExpected normalized volumes: "
    f"{expected_count}"
)


# ================================================================
# NIFTI READER
# ================================================================

def read_nifti(
    file_path
):
    """
    Load and validate basic NIfTI contents.

    Returns:
        image
        data
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    try:

        image = nib.load(
            str(file_path)
        )

    except Exception as exc:

        raise ValueError(
            f"Could not read NIfTI: {exc}"
        )

    try:

        data = image.get_fdata(
            dtype=np.float32
        )

    except Exception as exc:

        raise ValueError(
            f"Could not load voxel data: {exc}"
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
# ORIENTATION CHECK
# ================================================================

def is_ras(
    image
):

    orientation = nib.aff2axcodes(
        image.affine
    )

    return orientation == (
        "R",
        "A",
        "S"
    )


# ================================================================
# GEOMETRY CHECK
# ================================================================

def geometry_matches(
    image_a,
    image_b
):

    if image_a.shape != image_b.shape:

        return False

    voxel_a = np.asarray(
        image_a.header.get_zooms()[:3],
        dtype=float
    )

    voxel_b = np.asarray(
        image_b.header.get_zooms()[:3],
        dtype=float
    )

    if not np.allclose(
        voxel_a,
        voxel_b,
        atol=VOXEL_TOLERANCE
    ):

        return False

    if not np.allclose(
        image_a.affine,
        image_b.affine,
        atol=AFFINE_TOLERANCE
    ):

        return False

    return True


# ================================================================
# BASIC IMAGE VALIDATION
# ================================================================

def validate_basic_image(
    image,
    data
):
    """
    Validate:
      - readable
      - finite
      - RAS
      - 1mm isotropic
      - non-empty
    """

    if data.size == 0:

        return (
            False,
            "empty image"
        )

    if not np.all(
        np.isfinite(data)
    ):

        return (
            False,
            "non-finite values"
        )

    if not is_ras(
        image
    ):

        return (
            False,
            f"orientation is "
            f"{nib.aff2axcodes(image.affine)}, "
            f"expected RAS"
        )

    voxel_size = np.asarray(
        image.header.get_zooms()[:3],
        dtype=float
    )

    if not np.allclose(
        voxel_size,
        [1.0, 1.0, 1.0],
        atol=VOXEL_TOLERANCE
    ):

        return (
            False,
            f"voxel size is {voxel_size}, "
            f"expected 1mm isotropic"
        )

    if np.count_nonzero(
        data
    ) == 0:

        return (
            False,
            "image contains no non-zero voxels"
        )

    return (
        True,
        "PASS"
    )


# ================================================================
# NORMALIZED OUTPUT VALIDATION
# ================================================================

def validate_normalized_output(
    normalized_image,
    normalized_data,
    masked_image,
    masked_data
):
    """
    Validate normalized output.

    Checks:
      - basic NIfTI validity
      - geometry matches masked input
      - non-empty
      - background remains zero
      - positive brain values exist
      - brain values are within [0,1]
    """

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------

    valid, reason = validate_basic_image(
        normalized_image,
        normalized_data
    )

    if not valid:

        return (
            False,
            reason,
            None,
            None
        )

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------

    if not geometry_matches(
        normalized_image,
        masked_image
    ):

        return (
            False,
            "normalized geometry does not "
            "match masked input",
            None,
            None
        )

    # ------------------------------------------------------------
    # Positive brain voxels
    # ------------------------------------------------------------

    brain_mask = (
        masked_data > 0
    )

    brain_count = np.count_nonzero(
        brain_mask
    )

    if brain_count < MIN_NONZERO_VOXELS:

        return (
            False,
            f"masked brain contains only "
            f"{brain_count} voxels",
            None,
            None
        )

    # ------------------------------------------------------------
    # Background must remain zero
    # ------------------------------------------------------------

    background = normalized_data[
        ~brain_mask
    ]

    if background.size > 0:

        if not np.allclose(
            background,
            0.0,
            atol=BACKGROUND_ATOL
        ):

            return (
                False,
                "background contains non-zero values",
                None,
                None
            )

    # ------------------------------------------------------------
    # Brain values
    # ------------------------------------------------------------

    brain_values = normalized_data[
        brain_mask
    ]

    if not np.all(
        np.isfinite(brain_values)
    ):

        return (
            False,
            "brain contains non-finite values",
            None,
            None
        )

    minimum = float(
        np.min(brain_values)
    )

    maximum = float(
        np.max(brain_values)
    )

    # ------------------------------------------------------------
    # [0,1] range
    # ------------------------------------------------------------

    if minimum < -VALUE_ATOL:

        return (
            False,
            f"minimum brain value "
            f"{minimum:.8f} is below 0",
            None,
            None
        )

    if maximum > (
        1.0 + VALUE_ATOL
    ):

        return (
            False,
            f"maximum brain value "
            f"{maximum:.8f} exceeds 1",
            None,
            None
        )

    # ------------------------------------------------------------
    # Non-empty normalized brain
    # ------------------------------------------------------------

    positive_count = np.count_nonzero(
        brain_values > BACKGROUND_ATOL
    )

    if positive_count < MIN_NONZERO_VOXELS:

        return (
            False,
            f"normalized brain contains only "
            f"{positive_count} positive voxels",
            None,
            None
        )

    return (
        True,
        "PASS",
        minimum,
        maximum
    )


# ================================================================
# QC RESULTS
# ================================================================

results = []

passed_subjects = 0

failed_subjects = 0


# ================================================================
# SUBJECT QC
# ================================================================

for index, item in enumerate(
    selected_acquisitions,
    start=1
):

    subject = item["subject"]

    acquisition_type = (
        item["acquisition_type"]
    )

    print("\n" + "=" * 70)

    print(
        f"SUBJECT {index}/"
        f"{len(selected_acquisitions)}: "
        f"{subject} | "
        f"{acquisition_type}"
    )

    print("=" * 70)


    subject_errors = []

    subject_pass = True

    volumes_checked = 0


    # ============================================================
    # RMS
    # ============================================================

    if acquisition_type == "RMS":

        masked_file = (
            MASKED_DIR /
            f"{subject}_rms_n4_masked.nii.gz"
        )

        normalized_file = (
            NORMALIZED_DIR /
            f"{subject}_rms_n4_normalized.nii.gz"
        )

        print("\nMasked input:")
        print(
            f"  {masked_file}"
        )

        print("\nNormalized output:")
        print(
            f"  {normalized_file}"
        )


        # --------------------------------------------------------
        # Masked input
        # --------------------------------------------------------

        try:

            masked_image, masked_data = (
                read_nifti(
                    masked_file
                )
            )

            valid, reason = (
                validate_basic_image(
                    masked_image,
                    masked_data
                )
            )

            if not valid:

                raise ValueError(
                    reason
                )

            print(
                "Masked input: PASS"
            )

        except Exception as exc:

            subject_pass = False

            subject_errors.append(
                f"Masked input failed: {exc}"
            )

            print(
                "Masked input: FAIL"
            )

            masked_image = None
            masked_data = None


        # --------------------------------------------------------
        # Normalized output
        # --------------------------------------------------------

        if masked_image is not None:

            try:

                normalized_image, normalized_data = (
                    read_nifti(
                        normalized_file
                    )
                )

                print(
                    "Normalized output: PASS"
                )

                (
                    output_valid,
                    output_reason,
                    minimum,
                    maximum
                ) = validate_normalized_output(
                    normalized_image,
                    normalized_data,
                    masked_image,
                    masked_data
                )

                volumes_checked += 1


                # ------------------------------------------------
                # Geometry
                # ------------------------------------------------

                if geometry_matches(
                    normalized_image,
                    masked_image
                ):

                    print(
                        "Geometry match: PASS"
                    )

                else:

                    print(
                        "Geometry match: FAIL"
                    )


                # ------------------------------------------------
                # Affine
                # ------------------------------------------------

                if np.allclose(
                    normalized_image.affine,
                    masked_image.affine,
                    atol=AFFINE_TOLERANCE
                ):

                    print(
                        "Affine match: PASS"
                    )

                else:

                    print(
                        "Affine match: FAIL"
                    )


                # ------------------------------------------------
                # Background
                # ------------------------------------------------

                background = (
                    normalized_data[
                        masked_data == 0
                    ]
                )

                background_ok = (
                    background.size == 0
                    or np.allclose(
                        background,
                        0.0,
                        atol=BACKGROUND_ATOL
                    )
                )

                if background_ok:

                    print(
                        "Background zero: PASS"
                    )

                else:

                    print(
                        "Background zero: FAIL"
                    )


                # ------------------------------------------------
                # Range
                # ------------------------------------------------

                if output_valid:

                    print(
                        "Normalized range: PASS "
                        f"[{minimum:.6f}, "
                        f"{maximum:.6f}]"
                    )

                else:

                    # Determine range independently so
                    # the diagnostic is still informative.

                    brain_values = (
                        normalized_data[
                            masked_data > 0
                        ]
                    )

                    if brain_values.size > 0:

                        diagnostic_min = float(
                            np.min(
                                brain_values
                            )
                        )

                        diagnostic_max = float(
                            np.max(
                                brain_values
                            )
                        )

                        print(
                            "Normalized range: FAIL "
                            f"[{diagnostic_min:.6f}, "
                            f"{diagnostic_max:.6f}]"
                        )

                    else:

                        print(
                            "Normalized range: FAIL"
                        )


                # ------------------------------------------------
                # Final volume status
                # ------------------------------------------------

                if not output_valid:

                    subject_pass = False

                    subject_errors.append(
                        f"Normalized output failed: "
                        f"{output_reason}"
                    )

            except Exception as exc:

                subject_pass = False

                subject_errors.append(
                    f"Normalized output failed: "
                    f"{exc}"
                )

                print(
                    "Normalized output: FAIL"
                )


    # ============================================================
    # 5ECHO
    # ============================================================

    elif acquisition_type == "5ECHO":

        reference_affine = None

        for echo in ECHOES:

            masked_file = (
                MASKED_DIR /
                f"{subject}_echo{echo}_n4_masked.nii.gz"
            )

            normalized_file = (
                NORMALIZED_DIR /
                f"{subject}_echo{echo}_n4_normalized.nii.gz"
            )

            print(
                f"\nEcho {echo}"
            )

            # ----------------------------------------------------
            # Masked input
            # ----------------------------------------------------

            try:

                masked_image, masked_data = (
                    read_nifti(
                        masked_file
                    )
                )

                valid, reason = (
                    validate_basic_image(
                        masked_image,
                        masked_data
                    )
                )

                if not valid:

                    raise ValueError(
                        reason
                    )

                print(
                    "  Masked input: PASS"
                )

            except Exception as exc:

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: "
                    f"masked input failed: {exc}"
                )

                print(
                    "  Masked input: FAIL"
                )

                continue


            # ----------------------------------------------------
            # Normalized output
            # ----------------------------------------------------

            try:

                normalized_image, normalized_data = (
                    read_nifti(
                        normalized_file
                    )
                )

                print(
                    "  Normalized output: PASS"
                )

            except Exception as exc:

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: "
                    f"normalized output failed: {exc}"
                )

                print(
                    "  Normalized output: FAIL"
                )

                continue


            volumes_checked += 1


            # ----------------------------------------------------
            # Geometry
            # ----------------------------------------------------

            geometry_ok = geometry_matches(
                normalized_image,
                masked_image
            )

            if geometry_ok:

                print(
                    "  Geometry match: PASS"
                )

            else:

                print(
                    "  Geometry match: FAIL"
                )

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: geometry mismatch"
                )


            # ----------------------------------------------------
            # Affine
            # ----------------------------------------------------

            affine_ok = np.allclose(
                normalized_image.affine,
                masked_image.affine,
                atol=AFFINE_TOLERANCE
            )

            if affine_ok:

                print(
                    "  Affine match: PASS"
                )

            else:

                print(
                    "  Affine match: FAIL"
                )

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: affine mismatch"
                )


            # ----------------------------------------------------
            # Echo-to-echo affine consistency
            # ----------------------------------------------------

            if reference_affine is None:

                reference_affine = (
                    normalized_image.affine.copy()
                )

            else:

                echo_affine_ok = np.allclose(
                    normalized_image.affine,
                    reference_affine,
                    atol=AFFINE_TOLERANCE
                )

                if echo_affine_ok:

                    print(
                        "  Echo affine consistency: PASS"
                    )

                else:

                    print(
                        "  Echo affine consistency: FAIL"
                    )

                    subject_pass = False

                    subject_errors.append(
                        f"Echo {echo}: "
                        "affine differs from other echoes"
                    )


            # ----------------------------------------------------
            # Background
            # ----------------------------------------------------

            background = (
                normalized_data[
                    masked_data == 0
                ]
            )

            background_ok = (
                background.size == 0
                or np.allclose(
                    background,
                    0.0,
                    atol=BACKGROUND_ATOL
                )
            )

            if background_ok:

                print(
                    "  Background zero: PASS"
                )

            else:

                print(
                    "  Background zero: FAIL"
                )

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: "
                    "non-zero background"
                )


            # ----------------------------------------------------
            # Range
            # ----------------------------------------------------

            brain_values = (
                normalized_data[
                    masked_data > 0
                ]
            )

            if brain_values.size == 0:

                print(
                    "  Normalized range: FAIL"
                )

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: "
                    "no brain voxels"
                )

            else:

                minimum = float(
                    np.min(
                        brain_values
                    )
                )

                maximum = float(
                    np.max(
                        brain_values
                    )
                )

                range_ok = (
                    minimum >= -VALUE_ATOL
                    and
                    maximum <= (
                        1.0 + VALUE_ATOL
                    )
                )

                if range_ok:

                    print(
                        "  Normalized range: PASS "
                        f"[{minimum:.6f}, "
                        f"{maximum:.6f}]"
                    )

                else:

                    print(
                        "  Normalized range: FAIL "
                        f"[{minimum:.6f}, "
                        f"{maximum:.6f}]"
                    )

                    subject_pass = False

                    subject_errors.append(
                        f"Echo {echo}: "
                        f"values outside [0,1]"
                    )


            # ----------------------------------------------------
            # Non-empty
            # ----------------------------------------------------

            positive_count = np.count_nonzero(
                brain_values
                > BACKGROUND_ATOL
            )

            if positive_count >= (
                MIN_NONZERO_VOXELS
            ):

                print(
                    "  Non-empty: PASS"
                )

            else:

                print(
                    "  Non-empty: FAIL"
                )

                subject_pass = False

                subject_errors.append(
                    f"Echo {echo}: "
                    f"only {positive_count} "
                    "positive voxels"
                )


    # ============================================================
    # UNKNOWN TYPE
    # ============================================================

    else:

        subject_pass = False

        subject_errors.append(
            f"Unsupported acquisition type: "
            f"{acquisition_type}"
        )


    # ============================================================
    # SUBJECT RESULT
    # ============================================================

    if subject_pass:

        status = "PASS"

        passed_subjects += 1

    else:

        status = "FAIL"

        failed_subjects += 1


    results.append(
        {
            "subject": subject,
            "acquisition_type":
                acquisition_type,
            "status": status,
            "volumes_checked":
                volumes_checked,
            "error_count":
                len(subject_errors),
            "errors":
                " | ".join(
                    subject_errors
                )
        }
    )


    print(
        "\nSTATUS:",
        status
    )


    if subject_errors:

        print(
            "Errors:"
        )

        for error in subject_errors:

            print(
                f"  - {error}"
            )


# ================================================================
# EXACT CURRENT FILE COMPLETENESS
# ================================================================

print("\n" + "=" * 70)
print("CURRENT COHORT FILE COMPLETENESS")
print("=" * 70)


# ---------------------------------------------------------------
# Actual files that exactly match expected current filenames
# ---------------------------------------------------------------

actual_current_files = set()

for filename in NORMALIZED_DIR.iterdir():

    if not filename.is_file():
        continue

    if filename.name in expected_files:

        actual_current_files.add(
            filename.name
        )


# ---------------------------------------------------------------
# Missing files
# ---------------------------------------------------------------

missing_files = sorted(
    expected_files
    - actual_current_files
)


# ---------------------------------------------------------------
# Legacy/unrelated files
#
# These are deliberately ignored.
# They are NOT considered QC failures.
# ---------------------------------------------------------------

all_files = {
    p.name
    for p in NORMALIZED_DIR.iterdir()
    if p.is_file()
}

legacy_files = sorted(
    all_files
    - expected_files
)


print(
    "Expected current normalized files:",
    len(expected_files)
)

print(
    "Found current normalized files:",
    len(actual_current_files)
)

print(
    "Missing current files:",
    len(missing_files)
)

print(
    "Legacy/unrelated files ignored:",
    len(legacy_files)
)


# ---------------------------------------------------------------
# Missing current files
# ---------------------------------------------------------------

if missing_files:

    print(
        "\nMissing current files:"
    )

    for filename in missing_files:

        print(
            f"  {filename}"
        )


# ---------------------------------------------------------------
# Legacy files
# ---------------------------------------------------------------

if legacy_files:

    print(
        "\nLegacy/unrelated files are ignored "
        "and do not affect current QC."
    )

    # Display only a limited list to keep output manageable.

    display_limit = 20

    for filename in legacy_files[
        :display_limit
    ]:

        print(
            f"  {filename}"
        )

    if len(legacy_files) > display_limit:

        print(
            f"  ... and "
            f"{len(legacy_files) - display_limit} "
            f"more."
        )


# ================================================================
# SAVE SUMMARY
# ================================================================

import pandas as pd


summary_df = pd.DataFrame(
    results
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("NORMALIZED sMRI QC SUMMARY")
print("=" * 70)

print(
    "Selected subjects:",
    len(selected_acquisitions)
)

print(
    "Expected normalized volumes:",
    expected_count
)

print(
    "Found current normalized files:",
    len(actual_current_files)
)

print(
    "Missing current files:",
    len(missing_files)
)

print(
    "Legacy/unrelated files ignored:",
    len(legacy_files)
)

print(
    "Subjects PASS:",
    passed_subjects
)

print(
    "Subjects FAIL:",
    failed_subjects
)

print(
    "Summary CSV:",
    SUMMARY_CSV.resolve()
)


# ================================================================
# FINAL STATUS
# ================================================================

cohort_pass = (
    len(missing_files) == 0
    and passed_subjects
    == len(selected_acquisitions)
    and failed_subjects == 0
)


print("\n" + "=" * 70)

if cohort_pass:

    print(
        "STATUS: AUTOMATED NORMALIZATION QC PASS"
    )

    print(
        "All current selected subjects and "
        "expected normalized volumes passed."
    )

else:

    print(
        "STATUS: NORMALIZATION QC FAILED"
    )

print("=" * 70)