# src\qc_n4.py

"""
======================================================================
COBRE sMRI — N4 BIAS-FIELD CORRECTION QUALITY CONTROL
======================================================================

Purpose
-------
Independently verify N4 bias-field corrected sMRI volumes.

This script DOES NOT perform N4 correction.
This script DOES NOT modify MRI data.

It compares:

    reoriented/
          |
          v
        N4 QC
          |
          v
    n4/

Supports
--------
1. 5ECHO MPRAGE
   subject_echo1_reoriented.nii.gz
   ...
   subject_echo5_reoriented.nii.gz

2. RMS MPRAGE
   subject_rms_reoriented.nii.gz

QC checks
---------
- Current selected cohort from MPRAGE manifest
- Exactly one selected acquisition per subject
- Input/output existence
- NIfTI readability
- 3-D image
- Expected dimensions
- 1 mm isotropic voxel size
- Canonical RAS orientation
- Finite voxel values
- Non-empty images
- Input/output geometry consistency
- Input/output array-shape consistency
- Input/output intensity statistics
- Mean absolute difference
- Maximum absolute difference
- Percentage of changed voxels
- Detection of completely identical input/output
- Final cohort completeness
- CSV summary

Important
---------
- Generic across COBRE ZIP datasets.
- Manifest-driven.
- RMS remains one 3-D volume.
- 5ECHO checks all five actual echoes.
- Legacy/unrelated files are ignored.
- Original files are never modified.
"""

import os
import csv
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib


# ======================================================================
# COMMAND-LINE ARGUMENTS
# ======================================================================

parser = argparse.ArgumentParser(
    description=(
        "Independent QC for COBRE sMRI N4 bias-field corrected volumes."
    )
)

parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Directory containing reoriented NIfTI files.",
)

parser.add_argument(
    "--n4",
    type=str,
    default=None,
    help="Directory containing N4-corrected NIfTI files.",
)

parser.add_argument(
    "--manifest",
    type=str,
    default=None,
    help="Path to the MPRAGE selection manifest.",
)

parser.add_argument(
    "--qc-dir",
    type=str,
    default=None,
    help="Directory for N4 QC results.",
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

N4_DIR = os.path.abspath(
    args.n4
    if args.n4 is not None
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

QC_DIR = os.path.abspath(
    args.qc_dir
    if args.qc_dir is not None
    else os.path.join(
        "processed_data",
        "smri",
        "qc_n4",
    )
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "n4_qc_summary.csv",
)


# ======================================================================
# CONSTANTS
# ======================================================================

EXPECTED_NIFTI_SHAPE = (
    192,
    256,
    256,
)

EXPECTED_SITK_SIZE = (
    192,
    256,
    256,
)

EXPECTED_ARRAY_SHAPE = (
    256,
    256,
    192,
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

GEOMETRY_ATOL = 1e-5

# A voxel is considered changed if its absolute difference
# exceeds this numerical tolerance.
CHANGE_ATOL = 1e-6

# Minimum number of non-zero voxels for a valid image.
MIN_NONZERO_VOXELS = 10


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("COBRE sMRI — N4 BIAS-FIELD CORRECTION QUALITY CONTROL")
print("=" * 80)

print("\nReoriented input directory:")
print(INPUT_DIR)

print("\nN4 output directory:")
print(N4_DIR)

print("\nSelection manifest:")
print(MANIFEST_PATH)

print("\nQC output directory:")
print(QC_DIR)


# ======================================================================
# DIRECTORY CHECKS
# ======================================================================

for path, description in [
    (
        INPUT_DIR,
        "reoriented input directory",
    ),
    (
        N4_DIR,
        "N4 output directory",
    ),
]:

    if not os.path.isdir(path):

        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


if not os.path.isfile(MANIFEST_PATH):

    raise FileNotFoundError(
        f"\nSelection manifest not found:\n{MANIFEST_PATH}"
    )


os.makedirs(
    QC_DIR,
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
                f"Invalid acquisition type for {subject}: "
                f"{acquisition_type!r}"
            )

        selected_acquisitions.append(
            (
                subject,
                acquisition_type,
            )
        )


# ======================================================================
# EXACTLY ONE ACQUISITION PER SUBJECT
# ======================================================================

subject_types = {}

for subject, acquisition_type in selected_acquisitions:

    if subject in subject_types:

        raise RuntimeError(
            "\nMultiple SELECTED acquisitions found for subject: "
            f"{subject}"
        )

    subject_types[subject] = acquisition_type


selected_acquisitions = sorted(
    selected_acquisitions,
    key=lambda x: x[0],
)


if not selected_acquisitions:

    raise RuntimeError(
        "\nNo SELECTED acquisitions found in manifest."
    )


print(
    "\nSelected acquisitions:",
    len(selected_acquisitions),
)


# ======================================================================
# ACQUISITION SUMMARY
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
# IMAGE READER
# ======================================================================

def read_and_validate(
    path,
):
    """
    Read one NIfTI and perform basic structural validation.
    """

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    # --------------------------------------------------------------
    # SimpleITK
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

    if size != EXPECTED_SITK_SIZE:

        raise ValueError(
            f"Unexpected SimpleITK size {size}; "
            f"expected {EXPECTED_SITK_SIZE}"
        )

    if not np.allclose(
        spacing,
        EXPECTED_VOXEL_SIZE,
        atol=0.01,
    ):

        raise ValueError(
            f"Unexpected spacing {spacing}; "
            f"expected {EXPECTED_VOXEL_SIZE}"
        )

    array = sitk.GetArrayFromImage(
        image
    ).astype(
        np.float64
    )

    if array.shape != EXPECTED_ARRAY_SHAPE:

        raise ValueError(
            f"Unexpected array shape {array.shape}; "
            f"expected {EXPECTED_ARRAY_SHAPE}"
        )

    if array.size == 0:

        raise ValueError(
            "Image contains no voxels."
        )

    if not np.all(
        np.isfinite(array)
    ):

        raise ValueError(
            "Image contains NaN or Inf values."
        )

    nonzero_count = int(
        np.count_nonzero(
            array != 0
        )
    )

    if nonzero_count < MIN_NONZERO_VOXELS:

        raise ValueError(
            "Image is empty or nearly empty: "
            f"{nonzero_count} non-zero voxels."
        )

    # --------------------------------------------------------------
    # NIfTI / nibabel
    # --------------------------------------------------------------

    nii = nib.load(
        path
    )

    nii_shape = tuple(
        int(x)
        for x in nii.shape
    )

    if nii_shape != EXPECTED_NIFTI_SHAPE:

        raise ValueError(
            f"Unexpected NIfTI shape {nii_shape}; "
            f"expected {EXPECTED_NIFTI_SHAPE}"
        )

    nii_spacing = tuple(
        float(x)
        for x in nii.header.get_zooms()[:3]
    )

    if not np.allclose(
        nii_spacing,
        EXPECTED_VOXEL_SIZE,
        atol=0.01,
    ):

        raise ValueError(
            f"Unexpected NIfTI voxel size {nii_spacing}"
        )

    if not np.all(
        np.isfinite(nii.affine)
    ):

        raise ValueError(
            "NIfTI affine contains NaN/Inf."
        )

    orientation = nib.aff2axcodes(
        nii.affine
    )

    if orientation != (
        "R",
        "A",
        "S",
    ):

        raise ValueError(
            f"Orientation is {orientation}; "
            "expected ('R','A','S')."
        )

    return image, array, nii


# ======================================================================
# GEOMETRY COMPARISON
# ======================================================================

def geometry_matches(
    image_a,
    image_b,
):
    """
    Compare SimpleITK spatial geometry.
    """

    return (
        image_a.GetSize()
        == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=GEOMETRY_ATOL,
        )
        and np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=GEOMETRY_ATOL,
        )
        and np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=GEOMETRY_ATOL,
        )
    )


# ======================================================================
# ARRAY STATISTICS
# ======================================================================

def calculate_difference_statistics(
    input_array,
    output_array,
):
    """
    Calculate quantitative input-vs-N4 differences.
    """

    if input_array.shape != output_array.shape:

        raise ValueError(
            "Input and output array shapes differ."
        )

    difference = (
        output_array
        - input_array
    )

    absolute_difference = np.abs(
        difference
    )

    changed_mask = (
        absolute_difference
        > CHANGE_ATOL
    )

    total_voxels = (
        input_array.size
    )

    changed_voxels = int(
        np.count_nonzero(
            changed_mask
        )
    )

    changed_percentage = (
        100.0
        * changed_voxels
        / total_voxels
    )

    mean_absolute_difference = float(
        np.mean(
            absolute_difference
        )
    )

    maximum_absolute_difference = float(
        np.max(
            absolute_difference
        )
    )

    mean_input = float(
        np.mean(input_array)
    )

    mean_output = float(
        np.mean(output_array)
    )

    minimum_input = float(
        np.min(input_array)
    )

    maximum_input = float(
        np.max(input_array)
    )

    minimum_output = float(
        np.min(output_array)
    )

    maximum_output = float(
        np.max(output_array)
    )

    identical = bool(
        np.allclose(
            input_array,
            output_array,
            atol=CHANGE_ATOL,
            rtol=0.0,
        )
    )

    return {
        "input_min": minimum_input,
        "input_max": maximum_input,
        "input_mean": mean_input,
        "n4_min": minimum_output,
        "n4_max": maximum_output,
        "n4_mean": mean_output,
        "mean_absolute_difference":
            mean_absolute_difference,
        "maximum_absolute_difference":
            maximum_absolute_difference,
        "changed_voxels":
            changed_voxels,
        "total_voxels":
            total_voxels,
        "changed_percentage":
            changed_percentage,
        "identical_input_output":
            identical,
    }


# ======================================================================
# EXPECTED FILE BUILDER
# ======================================================================

def build_expected_files(
    subject,
    acquisition_type,
):

    pairs = []

    if acquisition_type == "5ECHO":

        for echo in EXPECTED_ECHOES:

            pairs.append(
                {
                    "subject": subject,
                    "acquisition_type": acquisition_type,
                    "echo": echo,
                    "input": os.path.join(
                        INPUT_DIR,
                        f"{subject}_echo{echo}_reoriented.nii.gz",
                    ),
                    "output": os.path.join(
                        N4_DIR,
                        f"{subject}_echo{echo}_n4.nii.gz",
                    ),
                }
            )

    elif acquisition_type == "RMS":

        pairs.append(
            {
                "subject": subject,
                "acquisition_type": acquisition_type,
                "echo": "",
                "input": os.path.join(
                    INPUT_DIR,
                    f"{subject}_rms_reoriented.nii.gz",
                ),
                "output": os.path.join(
                    N4_DIR,
                    f"{subject}_rms_n4.nii.gz",
                ),
            }
        )

    return pairs


# ======================================================================
# BUILD CURRENT FILE SET
# ======================================================================

expected_files = []

for subject, acquisition_type in selected_acquisitions:

    expected_files.extend(
        build_expected_files(
            subject,
            acquisition_type,
        )
    )


print("\n" + "=" * 80)
print("EXPECTED CURRENT QC SET")
print("=" * 80)

print(
    "Expected current volumes:",
    len(expected_files),
)


# ======================================================================
# QC LOOP
# ======================================================================

results = []

passed = 0
failed = 0


for index, item in enumerate(
    expected_files,
    start=1,
):

    subject = item["subject"]
    acquisition_type = item["acquisition_type"]
    echo = item["echo"]
    input_path = item["input"]
    output_path = item["output"]

    label = (
        f"{subject} Echo {echo}"
        if echo != ""
        else f"{subject} RMS"
    )

    print("\n" + "-" * 80)
    print(
        f"QC {index}/{len(expected_files)} — {label}"
    )
    print("-" * 80)

    row = {
        "subject": subject,
        "acquisition_type": acquisition_type,
        "echo": echo,
        "status": "FAIL",
        "input_valid": False,
        "n4_valid": False,
        "geometry_match": False,
        "input_shape": "",
        "n4_shape": "",
        "input_voxel_size": "",
        "n4_voxel_size": "",
        "input_min": np.nan,
        "input_max": np.nan,
        "input_mean": np.nan,
        "n4_min": np.nan,
        "n4_max": np.nan,
        "n4_mean": np.nan,
        "mean_absolute_difference": np.nan,
        "maximum_absolute_difference": np.nan,
        "changed_voxels": 0,
        "total_voxels": 0,
        "changed_percentage": np.nan,
        "identical_input_output": "",
        "error": "",
    }

    try:

        # ----------------------------------------------------------
        # INPUT
        # ----------------------------------------------------------

        input_image, input_array, input_nii = (
            read_and_validate(
                input_path
            )
        )

        row["input_valid"] = True
        row["input_shape"] = str(
            tuple(input_nii.shape)
        )
        row["input_voxel_size"] = str(
            tuple(
                round(
                    float(x),
                    6,
                )
                for x in input_nii.header.get_zooms()[:3]
            )
        )

        print(
            "Input validation: PASS"
        )

        # ----------------------------------------------------------
        # N4 OUTPUT
        # ----------------------------------------------------------

        n4_image, n4_array, n4_nii = (
            read_and_validate(
                output_path
            )
        )

        row["n4_valid"] = True
        row["n4_shape"] = str(
            tuple(n4_nii.shape)
        )
        row["n4_voxel_size"] = str(
            tuple(
                round(
                    float(x),
                    6,
                )
                for x in n4_nii.header.get_zooms()[:3]
            )
        )

        print(
            "N4 output validation: PASS"
        )

        # ----------------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------------

        geometry_ok = geometry_matches(
            input_image,
            n4_image,
        )

        row["geometry_match"] = geometry_ok

        if geometry_ok:

            print(
                "Input/N4 geometry: PASS"
            )

        else:

            raise ValueError(
                "Input and N4 spatial geometry do not match."
            )

        # ----------------------------------------------------------
        # DIFFERENCE STATISTICS
        # ----------------------------------------------------------

        stats = calculate_difference_statistics(
            input_array,
            n4_array,
        )

        for key, value in stats.items():

            row[key] = value

        print(
            f"Input intensity range: "
            f"[{stats['input_min']:.6f}, "
            f"{stats['input_max']:.6f}]"
        )

        print(
            f"N4 intensity range: "
            f"[{stats['n4_min']:.6f}, "
            f"{stats['n4_max']:.6f}]"
        )

        print(
            f"Input mean: "
            f"{stats['input_mean']:.6f}"
        )

        print(
            f"N4 mean: "
            f"{stats['n4_mean']:.6f}"
        )

        print(
            f"Mean absolute difference: "
            f"{stats['mean_absolute_difference']:.6f}"
        )

        print(
            f"Maximum absolute difference: "
            f"{stats['maximum_absolute_difference']:.6f}"
        )

        print(
            f"Changed voxels: "
            f"{stats['changed_voxels']:,} / "
            f"{stats['total_voxels']:,}"
        )

        print(
            f"Changed percentage: "
            f"{stats['changed_percentage']:.6f}%"
        )

        print(
            "Identical input/output:",
            stats["identical_input_output"],
        )

        # ----------------------------------------------------------
        # N4 EFFECT CHECK
        # ----------------------------------------------------------

        if stats["identical_input_output"]:

            raise ValueError(
                "N4 output is numerically identical "
                "to the reoriented input."
            )

        if (
            stats["changed_voxels"] == 0
            or stats["mean_absolute_difference"] <= CHANGE_ATOL
        ):

            raise ValueError(
                "No meaningful numerical change detected "
                "between input and N4 output."
            )

        # ----------------------------------------------------------
        # PASS
        # ----------------------------------------------------------

        row["status"] = "PASS"

        passed += 1

        print(
            "\nSTATUS: PASS"
        )

    except Exception as e:

        row["error"] = str(e)

        failed += 1

        print(
            "\nSTATUS: FAIL"
        )

        print(
            "Reason:",
            e,
        )

    results.append(
        row
    )


# ======================================================================
# CURRENT COHORT COMPLETENESS
# ======================================================================

print("\n" + "=" * 80)
print("CURRENT COHORT COMPLETENESS")
print("=" * 80)


expected_paths = {
    item["output"]
    for item in expected_files
}


actual_current_paths = set()

for filename in os.listdir(
    N4_DIR
):

    if not filename.endswith(
        ".nii.gz"
    ):
        continue

    full_path = os.path.join(
        N4_DIR,
        filename,
    )

    if full_path in expected_paths:

        actual_current_paths.add(
            full_path
        )


missing_outputs = (
    expected_paths
    - actual_current_paths
)

unexpected_current_outputs = (
    actual_current_paths
    - expected_paths
)


print(
    "Expected current N4 files:",
    len(expected_paths),
)

print(
    "Found current N4 files:",
    len(actual_current_paths),
)

print(
    "Missing current files:",
    len(missing_outputs),
)

print(
    "Unexpected current files:",
    len(unexpected_current_outputs),
)


if missing_outputs:

    print("\nMissing:")

    for path in sorted(
        missing_outputs
    ):

        print(
            "  ",
            os.path.basename(path),
        )


if unexpected_current_outputs:

    print("\nUnexpected:")

    for path in sorted(
        unexpected_current_outputs
    ):

        print(
            "  ",
            os.path.basename(path),
        )


# ======================================================================
# SAVE CSV
# ======================================================================

summary_df = pd.DataFrame(
    results
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ======================================================================
# FINAL STATUS
# ======================================================================

print("\n" + "=" * 80)
print("N4 INDEPENDENT QC SUMMARY")
print("=" * 80)

print(
    "Selected subjects:",
    len(selected_acquisitions),
)

print(
    "Expected volumes:",
    len(expected_files),
)

print(
    "Volumes passed:",
    passed,
)

print(
    "Volumes failed:",
    failed,
)

print(
    "Missing current files:",
    len(missing_outputs),
)

print(
    "Unexpected current files:",
    len(unexpected_current_outputs),
)

print(
    "Summary CSV:",
    SUMMARY_CSV,
)


cohort_pass = (
    failed == 0
    and len(missing_outputs) == 0
    and len(unexpected_current_outputs) == 0
    and passed == len(expected_files)
)


if cohort_pass:

    print(
        "\nSTATUS: INDEPENDENT N4 QC PASS"
    )

else:

    print(
        "\nSTATUS: INDEPENDENT N4 QC FAILED"
    )


print("=" * 80)

print(
    "\nThis QC was read-only."
)

print(
    "No MRI files were modified."
)

print(
    "Only acquisitions selected in the MPRAGE manifest "
    "were evaluated."
)

print("=" * 80)