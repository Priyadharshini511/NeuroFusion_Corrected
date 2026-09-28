from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd


# ======================================================================
# NeuroFusion AI
# External OpenNeuro VBM -> AAL 116 ROI Feature Extraction
#
# CURRENT COHORT:
#   OpenNeuro ds000115
#   43 subjects
#
# PURPOSE:
#   Extract the same type of 116 regional GM-volume features used
#   in the NeuroFusion Exp20 sMRI pipeline.
#
# IMPORTANT:
#   - Uses UNSMOOTHED mwc1 files.
#   - Does NOT use smoothed s8mwc1 files for feature extraction.
#   - Uses AAL_MNI152_2mm atlas.
#   - Uses nearest-neighbour atlas resampling.
#   - Does NOT import scipy.
#   - Does NOT modify the original 119-subject training matrix.
#   - Produces a separate external 43-subject feature matrix.
#
# The previous 3-subject script was used as the methodological reference.
# ======================================================================


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


VBM_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "openneuro_ds000115"
    / "vbm"
)


ATLAS_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "atlas"
    / "aal_mni"
    / "AAL_MNI152_2mm.nii.gz"
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "openneuro_ds000115"
    / "features"
)


OUTPUT_FILE = (
    OUTPUT_ROOT
    / "openneuro_43_subjects_aal_116_features.csv"
)


QC_FILE = (
    OUTPUT_ROOT
    / "openneuro_43_subjects_aal_116_features_QC.txt"
)


# ======================================================================
# HEADER
# ======================================================================

print()
print("=" * 80)
print("NeuroFusion AI - OpenNeuro 43-Subject AAL Feature Extraction")
print("=" * 80)

print()
print("Project root:")
print(f"  {PROJECT_ROOT}")

print()
print("VBM root:")
print(f"  {VBM_ROOT}")

print()
print("AAL atlas:")
print(f"  {ATLAS_FILE}")

print()
print("Output:")
print(f"  {OUTPUT_FILE}")

print()
print("Method:")
print("  Unsmoothened modulated GM (mwc1)")
print("  AAL 116 ROI atlas")
print("  Nearest-neighbour atlas resampling")
print("  Regional GM volume = sum(GM values) x voxel volume")
print("  SciPy dependency: NONE")


# ======================================================================
# VALIDATE PATHS
# ======================================================================

print()
print("=" * 80)
print("PRE-FLIGHT CHECKS")
print("=" * 80)


if not VBM_ROOT.is_dir():
    raise FileNotFoundError(
        f"\nVBM directory not found:\n{VBM_ROOT}"
    )


if not ATLAS_FILE.is_file():
    raise FileNotFoundError(
        f"\nAAL atlas not found:\n{ATLAS_FILE}"
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("VBM directory: PASS")

print("AAL atlas: PASS")

print("Output directory: READY")


# ======================================================================
# LOAD AAL ATLAS
# ======================================================================

print()
print("=" * 80)
print("LOADING AAL ATLAS")
print("=" * 80)


atlas_img = nib.load(
    str(ATLAS_FILE)
)


atlas_data = atlas_img.get_fdata(
    dtype=np.float64
)


if atlas_data.ndim != 3:
    raise ValueError(
        f"AAL atlas must be 3-D. Found shape: {atlas_data.shape}"
    )


atlas_labels = np.unique(
    atlas_data
)


atlas_labels = atlas_labels[
    atlas_labels > 0
]


atlas_labels = np.sort(
    atlas_labels.astype(np.int32)
)


print()
print("Atlas shape:")
print(f"  {atlas_img.shape}")


print()
print("Atlas voxel size:")
print(
    f"  {atlas_img.header.get_zooms()[:3]}"
)


print()
print("Non-zero AAL ROI labels:")
print(f"  {len(atlas_labels)}")


if len(atlas_labels) != 116:
    raise ValueError(
        "\nAAL atlas validation FAILED.\n"
        f"Expected 116 ROI labels but found {len(atlas_labels)}."
    )


print()
print("AAL atlas validation: PASS")


# ======================================================================
# DISCOVER MWC1 FILES
# ======================================================================

print()
print("=" * 80)
print("DISCOVERING EXTERNAL VBM SUBJECTS")
print("=" * 80)


mwc1_records = []


for subject_dir in sorted(
    VBM_ROOT.iterdir(),
    key=lambda p: p.name
):

    if not subject_dir.is_dir():
        continue


    # --------------------------------------------------------------
    # Search only for UNSMOOTHED mwc1 files.
    #
    # Expected:
    #   subject_dir/mwc1*.nii
    #
    # Explicitly exclude:
    #   smooth/
    #   s8mwc1*
    # --------------------------------------------------------------

    candidates = []

    for p in subject_dir.rglob("mwc1*.nii"):

        name_lower = p.name.lower()
        path_lower = str(p).lower()

        if "smooth" in path_lower:
            continue

        if name_lower.startswith("s8mwc1"):
            continue

        candidates.append(p)


    candidates = sorted(
        set(candidates)
    )


    if len(candidates) == 0:
        continue


    if len(candidates) > 1:
        raise RuntimeError(
            "\nMultiple unsmoothed mwc1 files found for "
            f"{subject_dir.name}:\n"
            + "\n".join(
                f"  {p}"
                for p in candidates
            )
        )


    mwc1_records.append(
        (
            subject_dir.name,
            candidates[0]
        )
    )


print()
print("External VBM subjects discovered:")
print(f"  {len(mwc1_records)}")


if len(mwc1_records) != 43:
    raise RuntimeError(
        "\nExpected exactly 43 external VBM subjects.\n"
        f"Found: {len(mwc1_records)}\n\n"
        "Check the VBM output directory before continuing."
    )


print()
print("Complete external subject list:")


for index, (
    subject,
    mwc1_file
) in enumerate(
    mwc1_records,
    start=1
):

    print(
        f"  {index:02d} -> "
        f"{subject} -> "
        f"{mwc1_file.name}"
    )


# ======================================================================
# NEAREST-NEIGHBOUR ATLAS RESAMPLING
#
# This replaces scipy.ndimage.affine_transform.
#
# We map each target VBM voxel into atlas coordinates using:
#
#     inv(atlas_affine) @ target_affine
#
# and nearest-neighbour rounding.
#
# Processing is done slice-by-slice to avoid creating an enormous
# full coordinate array in memory.
# ======================================================================

def resample_atlas_nearest_neighbor(
    atlas_data,
    atlas_affine,
    target_shape,
    target_affine,
):
    """
    Resample the AAL atlas onto the target mwc1 voxel grid.

    Uses nearest-neighbour interpolation.

    This is intentionally implemented with NumPy only so that
    scipy is not required.
    """

    target_shape = tuple(
        int(x)
        for x in target_shape
    )


    if len(target_shape) != 3:
        raise ValueError(
            f"Target image must be 3-D. "
            f"Found shape: {target_shape}"
        )


    # --------------------------------------------------------------
    # Transformation:
    #
    # target voxel coordinates
    #          ->
    # world coordinates
    #          ->
    # atlas voxel coordinates
    #
    # atlas_voxel =
    #     inv(atlas_affine)
    #     @ target_affine
    #     @ target_voxel
    # --------------------------------------------------------------

    transform = (
        np.linalg.inv(atlas_affine)
        @ target_affine
    )


    output = np.zeros(
        target_shape,
        dtype=np.int32
    )


    # --------------------------------------------------------------
    # Generate X/Y coordinate grid once.
    # Z is processed in slices.
    # --------------------------------------------------------------

    nx = target_shape[0]
    ny = target_shape[1]
    nz = target_shape[2]


    x = np.arange(
        nx,
        dtype=np.float64
    )


    y = np.arange(
        ny,
        dtype=np.float64
    )


    xx, yy = np.meshgrid(
        x,
        y,
        indexing="ij"
    )


    ones = np.ones_like(
        xx,
        dtype=np.float64
    )


    # --------------------------------------------------------------
    # Process one Z slice at a time.
    # --------------------------------------------------------------

    for z in range(nz):

        zz = np.full_like(
            xx,
            float(z)
        )


        coords = np.stack(
            [
                xx,
                yy,
                zz,
                ones,
            ],
            axis=0
        )


        atlas_coords = (
            transform
            @ coords.reshape(
                4,
                -1
            )
        )


        # ----------------------------------------------------------
        # Nearest-neighbour interpolation
        # ----------------------------------------------------------

        ax = np.rint(
            atlas_coords[0]
        ).astype(
            np.int64
        )


        ay = np.rint(
            atlas_coords[1]
        ).astype(
            np.int64
        )


        az = np.rint(
            atlas_coords[2]
        ).astype(
            np.int64
        )


        valid = (
            (ax >= 0)
            & (ax < atlas_data.shape[0])
            & (ay >= 0)
            & (ay < atlas_data.shape[1])
            & (az >= 0)
            & (az < atlas_data.shape[2])
        )


        slice_result = np.zeros(
            nx * ny,
            dtype=np.int32
        )


        if np.any(valid):

            valid_ax = ax[valid]
            valid_ay = ay[valid]
            valid_az = az[valid]


            values = atlas_data[
                valid_ax,
                valid_ay,
                valid_az
            ]


            values = np.rint(
                values
            ).astype(
                np.int32
            )


            slice_result[
                valid
            ] = values


        output[:, :, z] = (
            slice_result.reshape(
                nx,
                ny
            )
        )


    return output


# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

feature_rows = []

qc_lines = []


qc_lines.append(
    "NeuroFusion AI - OpenNeuro 43-Subject AAL Feature Extraction"
)

qc_lines.append(
    "=" * 80
)

qc_lines.append(
    f"VBM root: {VBM_ROOT}"
)

qc_lines.append(
    f"AAL atlas: {ATLAS_FILE}"
)

qc_lines.append(
    f"Number of subjects: {len(mwc1_records)}"
)

qc_lines.append(
    "Feature count: 116"
)

qc_lines.append(
    "Feature source: unsmoothed modulated GM (mwc1)"
)

qc_lines.append(
    "Atlas resampling: nearest neighbour"
)

qc_lines.append(
    "SciPy used: NO"
)


for subject_index, (
    subject,
    mwc1_file
) in enumerate(
    mwc1_records,
    start=1
):

    print()
    print()
    print("#" * 80)
    print(
        f"[{subject_index}/43] "
        f"PROCESSING {subject}"
    )
    print("#" * 80)


    print()
    print("MWC1:")
    print(f"  {mwc1_file}")


    # --------------------------------------------------------------
    # Load MWC1
    # --------------------------------------------------------------

    img = nib.load(
        str(mwc1_file)
    )


    data = img.get_fdata(
        dtype=np.float64
    )


    print()
    print("Image shape:")
    print(f"  {img.shape}")


    print()
    print("Voxel size:")
    print(
        f"  {img.header.get_zooms()[:3]}"
    )


    # --------------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------------

    if data.ndim != 3:

        raise ValueError(
            f"{subject}: MWC1 is not 3-D: "
            f"{data.shape}"
        )


    if not np.all(
        np.isfinite(data)
    ):

        raise ValueError(
            f"{subject}: MWC1 contains "
            "non-finite values."
        )


    nonzero_count = np.count_nonzero(
        data
    )


    if nonzero_count == 0:

        raise ValueError(
            f"{subject}: MWC1 is completely zero."
        )


    print()
    print("Finite voxels:")
    print(
        f"  {np.count_nonzero(np.isfinite(data))}"
        f" / {data.size}"
    )


    print()
    print("Non-zero voxels:")
    print(
        f"  {nonzero_count}"
    )


    print()
    print("Minimum:")
    print(
        f"  {np.min(data):.8f}"
    )


    print()
    print("Maximum:")
    print(
        f"  {np.max(data):.8f}"
    )


    print()
    print("Mean:")
    print(
        f"  {np.mean(data):.8f}"
    )


    # --------------------------------------------------------------
    # Resample AAL atlas to MWC1 grid
    # --------------------------------------------------------------

    print()
    print(
        "Resampling AAL atlas "
        "to MWC1 grid..."
    )


    resampled_atlas = (
        resample_atlas_nearest_neighbor(
            atlas_data=atlas_data,
            atlas_affine=atlas_img.affine,
            target_shape=img.shape,
            target_affine=img.affine,
        )
    )


    # --------------------------------------------------------------
    # Validate atlas labels after resampling
    # --------------------------------------------------------------

    resampled_labels = np.unique(
        resampled_atlas
    )


    resampled_labels = (
        resampled_labels[
            resampled_labels > 0
        ]
    )


    missing_labels = sorted(
        set(
            atlas_labels.tolist()
        )
        -
        set(
            resampled_labels.tolist()
        )
    )


    if missing_labels:

        raise ValueError(
            f"{subject}: missing AAL labels "
            f"after resampling:\n"
            f"{missing_labels}"
        )


    print()
    print(
        "AAL labels after resampling:"
    )

    print(
        f"  {len(resampled_labels)} / 116"
    )


    print(
        "Atlas resampling: PASS"
    )


    # --------------------------------------------------------------
    # Voxel volume
    #
    # Same definition as the reference Exp20 script:
    #
    # abs(det(img.affine[:3,:3]))
    # --------------------------------------------------------------

    voxel_volume = abs(
        np.linalg.det(
            img.affine[:3, :3]
        )
    )


    if not np.isfinite(
        voxel_volume
    ):

        raise ValueError(
            f"{subject}: invalid voxel volume."
        )


    if voxel_volume <= 0:

        raise ValueError(
            f"{subject}: voxel volume must "
            "be positive."
        )


    print()
    print("Voxel volume:")
    print(
        f"  {voxel_volume:.6f} mm^3"
    )


    # --------------------------------------------------------------
    # Create feature row
    #
    # No diagnosis label is invented here.
    #
    # This is an external cohort feature matrix.
    # --------------------------------------------------------------

    row = {
        "subject_id": subject,
    }


    # --------------------------------------------------------------
    # Extract 116 AAL regional GM volumes
    #
    # GMVOL =
    #     sum(GM voxel values in ROI)
    #     *
    #     voxel volume
    # --------------------------------------------------------------

    feature_values = []


    for roi_label in atlas_labels:

        roi_mask = (
            resampled_atlas
            == roi_label
        )


        roi_voxel_count = (
            np.count_nonzero(
                roi_mask
            )
        )


        if roi_voxel_count == 0:

            raise ValueError(
                f"{subject}: "
                f"AAL ROI {roi_label} "
                "has zero voxels."
            )


        roi_sum = np.sum(
            data[
                roi_mask
            ]
        )


        roi_volume = (
            roi_sum
            * voxel_volume
        )


        if not np.isfinite(
            roi_volume
        ):

            raise ValueError(
                f"{subject}: "
                f"ROI {roi_label} "
                "produced non-finite "
                "feature value."
            )


        row[
            f"GMVOL_AAL_{roi_label}"
        ] = float(
            roi_volume
        )


        feature_values.append(
            float(
                roi_volume
            )
        )


    # --------------------------------------------------------------
    # Validate 116 features
    # --------------------------------------------------------------

    if len(feature_values) != 116:

        raise RuntimeError(
            f"{subject}: expected 116 "
            f"features but got "
            f"{len(feature_values)}."
        )


    feature_array = np.asarray(
        feature_values,
        dtype=np.float64
    )


    if not np.all(
        np.isfinite(
            feature_array
        )
    ):

        raise RuntimeError(
            f"{subject}: feature vector "
            "contains non-finite values."
        )


    feature_rows.append(
        row
    )


    print()
    print(
        "ROI features extracted:"
    )

    print(
        "  116 / 116"
    )


    print()
    print(
        "Feature value range:"
    )

    print(
        f"  {np.min(feature_array):.6f}"
        f" to "
        f"{np.max(feature_array):.6f}"
    )


    # --------------------------------------------------------------
    # QC record
    # --------------------------------------------------------------

    qc_lines.append(
        ""
    )

    qc_lines.append(
        f"Subject: {subject}"
    )

    qc_lines.append(
        f"MWC1: {mwc1_file}"
    )

    qc_lines.append(
        f"Image shape: {img.shape}"
    )

    qc_lines.append(
        "AAL labels: 116 / 116"
    )

    qc_lines.append(
        f"Voxel volume: {voxel_volume:.6f} mm^3"
    )

    qc_lines.append(
        f"Non-zero MWC1 voxels: {nonzero_count}"
    )

    qc_lines.append(
        f"Feature minimum: {np.min(feature_array):.6f}"
    )

    qc_lines.append(
        f"Feature maximum: {np.max(feature_array):.6f}"
    )

    qc_lines.append(
        "Status: PASS"
    )


    print()
    print(
        f"FEATURE EXTRACTION PASSED: "
        f"{subject}"
    )


# ======================================================================
# BUILD DATAFRAME
# ======================================================================

print()
print()
print("=" * 80)
print("BUILDING 43 x 116 FEATURE MATRIX")
print("=" * 80)


feature_columns = [
    f"GMVOL_AAL_{label}"
    for label in atlas_labels
]


columns = [
    "subject_id"
] + feature_columns


features_df = pd.DataFrame(
    feature_rows,
    columns=columns
)


# ======================================================================
# FINAL VALIDATION
# ======================================================================

print()
print("=" * 80)
print("FINAL EXTERNAL FEATURE VALIDATION")
print("=" * 80)


print()
print("Rows:")
print(
    f"  {len(features_df)}"
)


print()
print("Columns:")
print(
    f"  {len(features_df.columns)}"
)


print()
print("Expected:")
print(
    "  43 subjects x 116 features"
)


print()
print("Actual feature matrix:")


feature_matrix = (
    features_df[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )
)


print(
    f"  {feature_matrix.shape}"
)


# --------------------------------------------------------------
# Exact shape validation
# --------------------------------------------------------------

if feature_matrix.shape != (
    43,
    116
):

    raise RuntimeError(
        "\nFINAL VALIDATION FAILED.\n"
        f"Expected feature matrix shape "
        f"(43, 116).\n"
        f"Found: {feature_matrix.shape}"
    )


# --------------------------------------------------------------
# Subject ID validation
# --------------------------------------------------------------

if (
    features_df["subject_id"]
    .duplicated()
    .any()
):

    duplicates = (
        features_df.loc[
            features_df[
                "subject_id"
            ].duplicated(),
            "subject_id"
        ]
        .tolist()
    )


    raise RuntimeError(
        "\nDuplicate subject IDs detected:\n"
        + "\n".join(
            str(x)
            for x in duplicates
        )
    )


# --------------------------------------------------------------
# Finite feature validation
# --------------------------------------------------------------

if not np.all(
    np.isfinite(
        feature_matrix
    )
):

    raise RuntimeError(
        "Feature matrix contains "
        "non-finite values."
    )


# --------------------------------------------------------------
# Empty feature validation
# --------------------------------------------------------------

if np.all(
    feature_matrix == 0
):

    raise RuntimeError(
        "Entire feature matrix is zero."
    )


print()
print(
    "Subject ID validation: PASS"
)


print(
    "116-feature validation: PASS"
)


print(
    "Finite-value validation: PASS"
)


print(
    "Feature matrix validation: PASS"
)


# ======================================================================
# SAVE FEATURE MATRIX
# ======================================================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 80)
print("FEATURE MATRIX SAVED")
print("=" * 80)


print()
print(
    f"CSV:\n  {OUTPUT_FILE}"
)


# ======================================================================
# SAVE QC REPORT
# ======================================================================

qc_lines.append("")
qc_lines.append("=" * 80)
qc_lines.append("FINAL VALIDATION")
qc_lines.append("=" * 80)
qc_lines.append("")
qc_lines.append(
    f"Subjects processed: {len(features_df)}"
)
qc_lines.append(
    "Expected subjects: 43"
)
qc_lines.append(
    "Feature columns: 116"
)
qc_lines.append(
    "Expected matrix shape: (43, 116)"
)
qc_lines.append(
    f"Actual matrix shape: {feature_matrix.shape}"
)
qc_lines.append(
    "Duplicate subject IDs: NO"
)
qc_lines.append(
    "Non-finite features: NO"
)
qc_lines.append(
    "Status: PASS"
)


QC_FILE.write_text(
    "\n".join(qc_lines),
    encoding="utf-8"
)


print()
print(
    f"QC report:\n  {QC_FILE}"
)


# ======================================================================
# SHOW SUBJECT LIST
# ======================================================================

print()
print("=" * 80)
print("FINAL SUBJECT LIST")
print("=" * 80)


print(
    features_df[
        ["subject_id"]
    ].to_string(
        index=False
    )
)


# ======================================================================
# FINAL STATUS
# ======================================================================

print()
print()
print("=" * 80)
print("STATUS: OPENNEURO AAL FEATURE EXTRACTION PASSED")
print("=" * 80)


print()
print(
    "43 subjects processed successfully."
)

print(
    "116 AAL GM-volume features extracted per subject."
)

print(
    f"Feature matrix: {feature_matrix.shape}"
)

print(
    "No SciPy dependency was used."
)

print()
print(
    "The original 119-subject Exp20 training matrix "
    "was not modified."
)

print()
print("NEXT STEP:")
print(
    "Use this external feature matrix for "
    "fixed-model / external-subject inference."
)

print()
print(
    "DONE."
)