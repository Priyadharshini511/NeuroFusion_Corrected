
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform


# ======================================================================
# NeuroFusion AI - External Exp20 AAL Feature Extraction
# ======================================================================
#
# External subjects:
#   sub-01
#   sub-26
#   sub-55
#
# These are UNSEEN external subjects.
# Their clinical labels are NOT used during feature extraction.
#
# Output:
#   subject_id
#   116 GMVOL_AAL_* features
#
# Age and gender are NOT added here because they are demographic
# variables required separately by the Exp20 nuisance-controlled model.
# ======================================================================


PROJECT_ROOT = Path(".")


VBM_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "external_test"
    / "ds004302"
    / "pilot_vbm"
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "external_test"
    / "features"
)


ATLAS_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "atlas"
    / "aal_mni"
    / "AAL_MNI152_2mm.nii.gz"
)


OUTPUT_FILE = (
    OUTPUT_ROOT
    / "external_regional_gm_volume_features.csv"
)


# ======================================================================
# EXTERNAL SUBJECTS
# ======================================================================

SUBJECTS = [
    "sub-01",
    "sub-26",
    "sub-55",
]


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("NeuroFusion AI - External Exp20 AAL Feature Extraction")
print("=" * 80)

print()
print("External VBM root:")
print(f"  {VBM_ROOT.resolve()}")

print()
print("AAL atlas:")
print(f"  {ATLAS_FILE.resolve()}")

print()
print("Output:")
print(f"  {OUTPUT_FILE.resolve()}")


# ======================================================================
# VALIDATE PATHS
# ======================================================================

if not VBM_ROOT.is_dir():
    raise FileNotFoundError(
        f"External VBM directory not found:\n"
        f"{VBM_ROOT.resolve()}"
    )


if not ATLAS_FILE.is_file():
    raise FileNotFoundError(
        f"AAL atlas not found:\n"
        f"{ATLAS_FILE.resolve()}"
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


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


atlas_data = atlas_img.get_fdata()


atlas_labels = np.unique(
    atlas_data
)


atlas_labels = atlas_labels[
    atlas_labels > 0
]


atlas_labels = np.sort(
    atlas_labels.astype(int)
)


print()
print("Atlas shape:")
print(f"  {atlas_img.shape}")


print()
print("Nonzero ROI labels:")
print(f"  {len(atlas_labels)}")


if len(atlas_labels) != 116:

    raise ValueError(
        f"Expected exactly 116 AAL ROI labels, "
        f"found {len(atlas_labels)}."
    )


print()
print("AAL atlas validation: PASS")


# ======================================================================
# FIND MWC1
# ======================================================================

def find_mwc1(subject):

    """
    Find the unsmoothed modulated GM map.

    Exp20 uses:
        mwc1*.nii

    and explicitly excludes:
        smooth/
    """

    subject_dir = (
        VBM_ROOT
        / subject
    )


    if not subject_dir.is_dir():

        raise FileNotFoundError(
            f"Subject VBM directory not found:\n"
            f"{subject_dir}"
        )


    candidates = sorted(
        p
        for p in subject_dir.glob(
            "mwc1*.nii"
        )
        if "smooth" not in str(p).lower()
    )


    if len(candidates) != 1:

        raise ValueError(
            f"Expected exactly one unsmoothed "
            f"mwc1 file for {subject}.\n"
            f"Found {len(candidates)}:\n"
            + "\n".join(
                str(p)
                for p in candidates
            )
        )


    return candidates[0]


# ======================================================================
# EXTRACT FEATURES
# ======================================================================

feature_rows = []


for subject in SUBJECTS:

    print()
    print("=" * 80)
    print(f"SUBJECT: {subject}")
    print("=" * 80)


    # ------------------------------------------------------------------
    # FIND MWC1
    # ------------------------------------------------------------------

    mwc1_file = find_mwc1(
        subject
    )


    print()
    print("mwc1:")
    print(f"  {mwc1_file}")


    # ------------------------------------------------------------------
    # LOAD IMAGE
    # ------------------------------------------------------------------

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


    # ------------------------------------------------------------------
    # NUMERICAL VALIDATION
    # ------------------------------------------------------------------

    if data.ndim != 3:

        raise ValueError(
            f"{subject}: mwc1 is not 3-D: "
            f"{data.shape}"
        )


    finite_count = np.count_nonzero(
        np.isfinite(data)
    )


    total_voxels = data.size


    print()
    print("Finite voxels:")
    print(
        f"  {finite_count} / "
        f"{total_voxels}"
    )


    if not np.all(
        np.isfinite(data)
    ):

        raise ValueError(
            f"{subject}: mwc1 contains "
            f"non-finite values."
        )


    nonzero_count = np.count_nonzero(
        data
    )


    print()
    print("Non-zero voxels:")
    print(
        f"  {nonzero_count}"
    )


    if nonzero_count == 0:

        raise ValueError(
            f"{subject}: mwc1 is "
            f"completely zero."
        )


    print()
    print("Minimum:")
    print(
        f"  {np.min(data):.6f}"
    )


    print()
    print("Maximum:")
    print(
        f"  {np.max(data):.6f}"
    )


    print()
    print("Mean:")
    print(
        f"  {np.mean(data):.6f}"
    )


    # ------------------------------------------------------------------
    # RESAMPLE AAL ATLAS TO MWC1 GRID
    # ------------------------------------------------------------------

    transform = (
        np.linalg.inv(
            atlas_img.affine
        )
        @ img.affine
    )


    resampled_atlas = affine_transform(

        atlas_data,

        matrix=transform[:3, :3],

        offset=transform[:3, 3],

        output_shape=img.shape,

        order=0,

        mode="constant",

        cval=0,

        prefilter=False,
    )


    resampled_atlas = np.rint(
        resampled_atlas
    ).astype(np.int32)


    # ------------------------------------------------------------------
    # VERIFY AAL LABELS
    # ------------------------------------------------------------------

    resampled_labels = np.unique(
        resampled_atlas
    )


    resampled_labels = (
        resampled_labels[
            resampled_labels > 0
        ]
    )


    missing_labels = sorted(

        set(atlas_labels)
        -
        set(resampled_labels)

    )


    print()
    print("Resampled AAL labels:")
    print(
        f"  {len(resampled_labels)} / 116"
    )


    if missing_labels:

        raise ValueError(
            f"{subject}: missing AAL labels "
            f"after resampling:\n"
            f"{missing_labels}"
        )


    print(
        "  AAL resampling validation: PASS"
    )


    # ------------------------------------------------------------------
    # VOXEL VOLUME
    # ------------------------------------------------------------------

    voxel_volume = abs(
        np.linalg.det(
            img.affine[:3, :3]
        )
    )


    print()
    print("Voxel volume:")
    print(
        f"  {voxel_volume:.6f} mm^3"
    )


    # ------------------------------------------------------------------
    # CREATE SUBJECT ROW
    # ------------------------------------------------------------------

    row = {
        "subject_id": subject
    }


    # ------------------------------------------------------------------
    # EXTRACT 116 AAL GM-VOLUME FEATURES
    # ------------------------------------------------------------------

    for roi_label in atlas_labels:

        roi_mask = (
            resampled_atlas
            == roi_label
        )


        roi_voxel_count = np.count_nonzero(
            roi_mask
        )


        if roi_voxel_count == 0:

            raise ValueError(
                f"{subject}: ROI "
                f"{roi_label} has zero voxels."
            )


        roi_sum = np.sum(
            data[roi_mask]
        )


        roi_volume = (
            roi_sum
            * voxel_volume
        )


        row[
            f"GMVOL_AAL_{roi_label}"
        ] = roi_volume


    feature_rows.append(
        row
    )


    feature_values = [

        row[key]

        for key in row

        if key.startswith(
            "GMVOL_AAL_"
        )

    ]


    print()
    print("ROI features extracted:")
    print("  116 / 116")


    print()
    print(
        f"Feature value range: "
        f"{min(feature_values):.6f}"
        f" to "
        f"{max(feature_values):.6f}"
    )


    print()
    print(
        f"Feature extraction completed: "
        f"{subject}"
    )


# ======================================================================
# BUILD DATAFRAME
# ======================================================================

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
print("Expected rows:")
print(
    f"  {len(SUBJECTS)}"
)


print()
print("Columns:")
print(
    f"  {len(features_df.columns)}"
)


print()
print("Expected columns:")
print("  117")


print()
print("Expected feature columns:")
print("  116")


print()
print("Actual feature columns:")
print(
    f"  {len(features_df.columns) - 1}"
)


# ----------------------------------------------------------------------
# ROW VALIDATION
# ----------------------------------------------------------------------

if len(features_df) != len(SUBJECTS):

    raise ValueError(
        f"Expected {len(SUBJECTS)} subjects, "
        f"found {len(features_df)}."
    )


# ----------------------------------------------------------------------
# SUBJECT ID VALIDATION
# ----------------------------------------------------------------------

if features_df[
    "subject_id"
].duplicated().any():

    raise ValueError(
        "Duplicate subject IDs detected."
    )


if set(
    features_df["subject_id"]
) != set(SUBJECTS):

    raise ValueError(
        "External subject IDs do not "
        "match expected subjects."
    )


# ----------------------------------------------------------------------
# FEATURE COUNT VALIDATION
# ----------------------------------------------------------------------

if len(feature_columns) != 116:

    raise ValueError(
        f"Expected 116 features, "
        f"found {len(feature_columns)}."
    )


# ----------------------------------------------------------------------
# NUMERICAL VALIDATION
# ----------------------------------------------------------------------

feature_matrix = (
    features_df[
        feature_columns
    ].to_numpy(
        dtype=float
    )
)


if not np.all(
    np.isfinite(
        feature_matrix
    )
):

    raise ValueError(
        "Feature matrix contains "
        "NaN or infinite values."
    )


if feature_matrix.shape != (
    len(SUBJECTS),
    116
):

    raise ValueError(
        "Feature matrix has unexpected shape: "
        f"{feature_matrix.shape}"
    )


# ----------------------------------------------------------------------
# DISPLAY SUBJECTS
# ----------------------------------------------------------------------

print()
print("Subject IDs:")

print(
    features_df[
        ["subject_id"]
    ].to_string(
        index=False
    )
)


print()
print("Feature matrix shape:")

print(
    feature_matrix.shape
)


# ======================================================================
# SAVE
# ======================================================================

features_df.to_csv(

    OUTPUT_FILE,

    index=False

)


print()
print("Saved:")

print(
    f"  {OUTPUT_FILE.resolve()}"
)


print()
print("=" * 80)
print(
    "STATUS: EXTERNAL AAL FEATURE EXTRACTION PASSED"
)
print("=" * 80)