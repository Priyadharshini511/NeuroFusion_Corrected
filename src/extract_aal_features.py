

# src/extract_aal_features.py

from pathlib import Path
import re

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker


# ================================================================
# GENERIC AAL sMRI ROI FEATURE EXTRACTION
#
# Supports:
#
#   RMS:
#       A000XXXXX_rms_n4_mni.nii.gz
#
#   5ECHO:
#       A000XXXXX_echo1_mni.nii.gz
#       A000XXXXX_echo2_mni.nii.gz
#       A000XXXXX_echo3_mni.nii.gz
#       A000XXXXX_echo4_mni.nii.gz
#       A000XXXXX_echo5_mni.nii.gz
#
# Cohort source:
#   mprage_selection_manifest.csv
#
# Only:
#   selection_status = SELECTED
#
# Manifest subject column:
#   subject
#
# Features:
#   Mean intensity of each of the 116 AAL ROIs.
#
# Important:
#   The actual integer ROI IDs stored in the atlas are preserved.
#
# No echo averaging.
# No feature standardization.
# No atlas relabeling.
# ================================================================


print("=" * 70)
print("GENERIC AAL sMRI ROI FEATURE EXTRACTION")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

MRI_DIR = Path(
    r"processed_data\smri\registered"
)

ATLAS_PATH = Path(
    r"processed_data\smri\atlas\aal_mni"
    r"\AAL_MNI152_2mm.nii.gz"
)

MANIFEST_PATH = Path(
    r"processed_data\smri\mprage_selection"
    r"\mprage_selection_manifest.csv"
)

OUTPUT_DIR = Path(
    r"processed_data\smri\features\aal"
)

COMBINED_OUTPUT = (
    OUTPUT_DIR
    / "COBRE_AAL_sMRI_features.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ROIS = 116

ECHOES = [1, 2, 3, 4, 5]

GEOMETRY_ATOL = 1e-5
AFFINE_ATOL = 1e-3


RMS_PATTERN = re.compile(
    r"^(A\d+)_rms_n4_mni\.nii\.gz$"
)

ECHO_PATTERN = re.compile(
    r"^(A\d+)_echo([1-5])_mni\.nii\.gz$"
)


# ================================================================
# ACQUISITION TYPE NORMALIZATION
# ================================================================

def normalize_acquisition_type(value):

    value = str(value).strip().upper()

    if value == "RMS":
        return "RMS"

    if value in {
        "5ECHO",
        "5-ECHO",
        "5 ECHO",
        "5_ECHO"
    }:
        return "5ECHO"

    raise ValueError(
        f"Unknown acquisition type in manifest: {value}"
    )


# ================================================================
# LOAD SELECTION MANIFEST
# ================================================================

def load_manifest():

    if not MANIFEST_PATH.is_file():

        raise FileNotFoundError(
            f"Selection manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "subject",
        "selection_status",
        "acquisition_type"
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:

        raise ValueError(
            "Selection manifest is missing required columns: "
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
            "No SELECTED rows found in selection manifest."
        )

    # ------------------------------------------------------------
    # Manifest uses "subject".
    # Internally we use "subject_id".
    # ------------------------------------------------------------

    selected["subject_id"] = (
        selected["subject"]
        .astype(str)
        .str.strip()
    )

    if selected["subject_id"].eq("").any():

        raise ValueError(
            "Selection manifest contains empty subject IDs."
        )

    if selected["subject_id"].duplicated().any():

        duplicates = sorted(
            selected.loc[
                selected["subject_id"].duplicated(),
                "subject_id"
            ].unique()
        )

        raise ValueError(
            "Multiple SELECTED rows found for subject(s): "
            f"{duplicates}"
        )

    selected["acquisition_type"] = (
        selected["acquisition_type"]
        .apply(normalize_acquisition_type)
    )

    selected = selected[
        [
            "subject_id",
            "acquisition_type"
        ]
    ].copy()

    selected = selected.sort_values(
        "subject_id"
    ).reset_index(
        drop=True
    )

    return selected


# ================================================================
# AAL ATLAS VALIDATION
# ================================================================

def validate_atlas(atlas):

    # ------------------------------------------------------------
    # Dimension
    # ------------------------------------------------------------

    if len(atlas.shape) != 3:

        raise ValueError(
            f"AAL atlas must be 3D. "
            f"Found shape: {atlas.shape}"
        )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    data = atlas.get_fdata(
        dtype=np.float32
    )

    if data.size == 0:

        raise ValueError(
            "AAL atlas is empty."
        )

    # ------------------------------------------------------------
    # Finite check
    # ------------------------------------------------------------

    if not np.all(
        np.isfinite(data)
    ):

        raise ValueError(
            "AAL atlas contains NaN or infinite values."
        )

    # ------------------------------------------------------------
    # Negative labels
    # ------------------------------------------------------------

    if np.any(data < 0):

        raise ValueError(
            "AAL atlas contains negative labels."
        )

    # ------------------------------------------------------------
    # Integer-label validation
    # ------------------------------------------------------------

    rounded = np.rint(
        data
    )

    max_error = np.max(
        np.abs(
            data - rounded
        )
    )

    print(
        "Maximum integer deviation:",
        float(max_error)
    )

    if max_error > 1e-5:

        raise ValueError(
            "AAL atlas contains non-integer labels."
        )

    # ------------------------------------------------------------
    # Get actual positive labels
    #
    # IMPORTANT:
    # We do NOT assume labels are 1..116.
    # ------------------------------------------------------------

    positive_labels = np.unique(
        rounded[rounded > 0]
    ).astype(np.int32)

    print(
        "Positive ROI labels:",
        len(positive_labels)
    )

    print(
        "AAL labels:",
        positive_labels.tolist()
    )

    # ------------------------------------------------------------
    # Exactly 116 ROIs
    # ------------------------------------------------------------

    if len(positive_labels) != EXPECTED_ROIS:

        raise ValueError(
            "Unexpected AAL ROI count.\n"
            f"Expected: {EXPECTED_ROIS}\n"
            f"Found   : {len(positive_labels)}"
        )

    print(
        "AAL label verification: PASS"
    )

    return positive_labels


# ================================================================
# MRI / AAL GEOMETRY VALIDATION
# ================================================================

def validate_geometry(
    mri,
    atlas,
    subject_id,
    volume_label
):

    # ------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------

    if mri.shape != atlas.shape:

        raise ValueError(
            f"{subject_id} {volume_label}: "
            "MRI/AAL shape mismatch.\n"
            f"MRI={mri.shape}\n"
            f"AAL={atlas.shape}"
        )

    # ------------------------------------------------------------
    # Voxel spacing
    # ------------------------------------------------------------

    mri_spacing = (
        mri.header.get_zooms()[:3]
    )

    atlas_spacing = (
        atlas.header.get_zooms()[:3]
    )

    if not np.allclose(
        mri_spacing,
        atlas_spacing,
        atol=GEOMETRY_ATOL
    ):

        raise ValueError(
            f"{subject_id} {volume_label}: "
            "MRI/AAL spacing mismatch.\n"
            f"MRI={mri_spacing}\n"
            f"AAL={atlas_spacing}"
        )

    # ------------------------------------------------------------
    # Affine
    # ------------------------------------------------------------

    if not np.allclose(
        mri.affine,
        atlas.affine,
        atol=AFFINE_ATOL
    ):

        raise ValueError(
            f"{subject_id} {volume_label}: "
            "MRI/AAL affine mismatch."
        )


# ================================================================
# EXISTING FEATURE FILE VALIDATION
# ================================================================

def validate_existing_feature_csv(
    csv_path,
    subject_id,
    acquisition_type,
    volume_label,
    feature_names
):

    if not csv_path.is_file():
        return None

    try:

        df = pd.read_csv(
            csv_path
        )

        expected_columns = [
            "subject_id",
            "acquisition_type",
            "volume"
        ] + feature_names

        if list(df.columns) != expected_columns:
            return None

        if df.shape != (
            1,
            len(expected_columns)
        ):
            return None

        if str(
            df.iloc[0]["subject_id"]
        ) != subject_id:
            return None

        if str(
            df.iloc[0]["acquisition_type"]
        ) != acquisition_type:
            return None

        if str(
            df.iloc[0]["volume"]
        ) != volume_label:
            return None

        values = df[
            feature_names
        ].to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(values)
        ):
            return None

        return df

    except Exception:

        return None


# ================================================================
# BUILD EXPECTED MRI INPUT INVENTORY
# ================================================================

def build_expected_inputs(selected):

    expected = []

    for _, row in selected.iterrows():

        subject_id = row[
            "subject_id"
        ]

        acquisition_type = row[
            "acquisition_type"
        ]

        # --------------------------------------------------------
        # RMS
        # --------------------------------------------------------

        if acquisition_type == "RMS":

            expected.append(
                {
                    "subject_id": subject_id,
                    "acquisition_type": "RMS",
                    "volume": "RMS",
                    "path": (
                        MRI_DIR
                        / f"{subject_id}_rms_n4_mni.nii.gz"
                    )
                }
            )

        # --------------------------------------------------------
        # 5ECHO
        # --------------------------------------------------------

        elif acquisition_type == "5ECHO":

            for echo in ECHOES:

                expected.append(
                    {
                        "subject_id": subject_id,
                        "acquisition_type": "5ECHO",
                        "volume": f"Echo {echo}",
                        "echo": echo,
                        "path": (
                            MRI_DIR
                            / f"{subject_id}_echo{echo}_mni.nii.gz"
                        )
                    }
                )

        else:

            raise ValueError(
                f"Unsupported acquisition type: "
                f"{acquisition_type}"
            )

    return expected


# ================================================================
# OUTPUT PATH
# ================================================================

def output_path_for(item):

    subject_id = item[
        "subject_id"
    ]

    acquisition_type = item[
        "acquisition_type"
    ]

    if acquisition_type == "RMS":

        return (
            OUTPUT_DIR
            / f"{subject_id}_rms_aal_features.csv"
        )

    echo = item[
        "echo"
    ]

    return (
        OUTPUT_DIR
        / f"{subject_id}_echo{echo}_aal_features.csv"
    )


# ================================================================
# NORMALIZE NILEARN OUTPUT
# ================================================================

def normalize_masker_output(
    features,
    subject_id,
    volume_label
):

    features = np.asarray(
        features,
        dtype=np.float64
    )

    print(
        "Masker output shape:",
        features.shape
    )

    # ------------------------------------------------------------
    # Single 3D image can return:
    #
    #   (116,)
    #
    # Some versions/configurations can return:
    #
    #   (1,116)
    # ------------------------------------------------------------

    if features.ndim == 1:

        if features.shape[0] != EXPECTED_ROIS:

            raise ValueError(
                f"{subject_id} {volume_label}: "
                "Unexpected 1D masker output shape: "
                f"{features.shape}"
            )

        feature_vector = features

    elif features.ndim == 2:

        if features.shape != (
            1,
            EXPECTED_ROIS
        ):

            raise ValueError(
                f"{subject_id} {volume_label}: "
                "Unexpected 2D masker output shape: "
                f"{features.shape}"
            )

        feature_vector = features[0]

    else:

        raise ValueError(
            f"{subject_id} {volume_label}: "
            "Unexpected masker output dimensions: "
            f"{features.shape}"
        )

    if not np.all(
        np.isfinite(feature_vector)
    ):

        raise ValueError(
            f"{subject_id} {volume_label}: "
            "Feature vector contains NaN or infinite values."
        )

    return feature_vector


# ================================================================
# PIPELINE CONFIGURATION
# ================================================================

print("\n" + "-" * 70)
print("PIPELINE CONFIGURATION")
print("-" * 70)

print(
    "MRI directory:",
    MRI_DIR
)

print(
    "AAL atlas:",
    ATLAS_PATH
)

print(
    "Selection manifest:",
    MANIFEST_PATH
)

print(
    "Output directory:",
    OUTPUT_DIR
)


# ================================================================
# PATH CHECKS
# ================================================================

if not MRI_DIR.is_dir():

    raise FileNotFoundError(
        f"MNI MRI directory not found:\n"
        f"{MRI_DIR}"
    )

if not ATLAS_PATH.is_file():

    raise FileNotFoundError(
        f"AAL atlas not found:\n"
        f"{ATLAS_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# LOAD CURRENT SUBJECT COHORT
# ================================================================

print("\n" + "=" * 70)
print("LOADING CURRENT SUBJECT COHORT")
print("=" * 70)

selected = load_manifest()

print(
    "Selected subjects:",
    len(selected)
)

print(
    "\nSelected acquisition types:"
)

print(
    selected[
        "acquisition_type"
    ].value_counts().to_string()
)


# ================================================================
# BUILD EXPECTED INPUTS
# ================================================================

expected_inputs = build_expected_inputs(
    selected
)

print(
    "\nExpected current MNI MRI volumes:",
    len(expected_inputs)
)

rms_count = sum(
    item["acquisition_type"] == "RMS"
    for item in expected_inputs
)

five_echo_count = sum(
    item["acquisition_type"] == "5ECHO"
    for item in expected_inputs
)

print(
    "  RMS volumes   :",
    rms_count
)

print(
    "  5ECHO volumes :",
    five_echo_count
)


# ================================================================
# VERIFY EXPECTED INPUT FILES
# ================================================================

print("\n" + "=" * 70)
print("VERIFYING CURRENT MNI MRI INPUTS")
print("=" * 70)

missing_inputs = []

for item in expected_inputs:

    if not item["path"].is_file():

        missing_inputs.append(
            str(item["path"])
        )

if missing_inputs:

    print(
        "\nMissing expected MRI files:"
    )

    for path in missing_inputs:

        print(
            " ",
            path
        )

    raise RuntimeError(
        f"{len(missing_inputs)} expected MRI file(s) missing."
    )

print(
    "All expected current MRI files: PASS"
)


# ================================================================
# DISCOVER REGISTERED FILES
# ================================================================

print("\n" + "=" * 70)
print("DISCOVERING REGISTERED MRI FILES")
print("=" * 70)

all_registered_files = sorted(
    [
        path
        for path in MRI_DIR.iterdir()
        if path.is_file()
        and path.name.endswith(
            ".nii.gz"
        )
    ]
)

current_subjects = set(
    selected[
        "subject_id"
    ].astype(str)
)

current_registered = []
legacy_registered = []

for path in all_registered_files:

    filename = path.name

    match_rms = RMS_PATTERN.match(
        filename
    )

    match_echo = ECHO_PATTERN.match(
        filename
    )

    subject_id = None

    if match_rms:

        subject_id = match_rms.group(1)

    elif match_echo:

        subject_id = match_echo.group(1)

    if subject_id in current_subjects:

        current_registered.append(
            path
        )

    else:

        legacy_registered.append(
            path
        )

print(
    "All registered NIfTI files:",
    len(all_registered_files)
)

print(
    "Current-cohort registered files:",
    len(current_registered)
)

print(
    "Legacy/unrelated registered files ignored:",
    len(legacy_registered)
)


# ================================================================
# LOAD AND VALIDATE AAL ATLAS
# ================================================================

print("\n" + "=" * 70)
print("LOADING VERIFIED AAL ATLAS")
print("=" * 70)

atlas_img = nib.load(
    str(ATLAS_PATH)
)

print(
    "Atlas shape:",
    atlas_img.shape
)

print(
    "Atlas voxel size:",
    atlas_img.header.get_zooms()[:3]
)

roi_labels = validate_atlas(
    atlas_img
)


# ================================================================
# FEATURE NAMES
# ================================================================

feature_names = [
    f"sMRI_AAL_{int(label)}"
    for label in roi_labels
]

if len(feature_names) != EXPECTED_ROIS:

    raise RuntimeError(
        "Incorrect number of AAL feature names."
    )

if len(set(feature_names)) != EXPECTED_ROIS:

    raise RuntimeError(
        "AAL feature names are not unique."
    )

print(
    "Feature names:",
    len(feature_names)
)


# ================================================================
# CREATE NILEARN MASKER
# ================================================================

print("\n" + "-" * 70)
print("CREATING AAL ROI MASKER")
print("-" * 70)

masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=False,
    resampling_target=None,
    background_label=0
)

print(
    "Standardization : DISABLED"
)

print(
    "Resampling      : DISABLED"
)

print(
    "Background      : 0"
)


# ================================================================
# EXTRACTION
# ================================================================

print("\n" + "=" * 70)
print("STARTING AAL ROI EXTRACTION")
print("=" * 70)

successful = 0
skipped = 0
failed = 0

all_feature_rows = []
failed_items = []


for index, item in enumerate(
    expected_inputs,
    start=1
):

    subject_id = item[
        "subject_id"
    ]

    acquisition_type = item[
        "acquisition_type"
    ]

    volume_label = item[
        "volume"
    ]

    mri_path = item[
        "path"
    ]

    print("\n" + "-" * 70)

    print(
        f"FILE {index}/{len(expected_inputs)}"
    )

    print("-" * 70)

    print(
        "Subject:",
        subject_id
    )

    print(
        "Acquisition:",
        acquisition_type
    )

    print(
        "Volume:",
        volume_label
    )

    print(
        "Input:",
        mri_path
    )

    try:

        # --------------------------------------------------------
        # OUTPUT PATH
        # --------------------------------------------------------

        output_csv = output_path_for(
            item
        )


        # --------------------------------------------------------
        # REUSE VALID EXISTING FEATURE FILE
        # --------------------------------------------------------

        existing_df = (
            validate_existing_feature_csv(
                output_csv,
                subject_id,
                acquisition_type,
                volume_label,
                feature_names
            )
        )

        if existing_df is not None:

            print(
                "Existing valid feature file: REUSED"
            )

            all_feature_rows.append(
                existing_df
            )

            successful += 1
            skipped += 1

            continue


        # --------------------------------------------------------
        # LOAD MRI
        # --------------------------------------------------------

        mri = nib.load(
            str(mri_path)
        )

        print(
            "MRI shape:",
            mri.shape
        )

        print(
            "MRI voxel size:",
            mri.header.get_zooms()[:3]
        )


        # --------------------------------------------------------
        # VERIFY RAS
        # --------------------------------------------------------

        orientation = nib.aff2axcodes(
            mri.affine
        )

        print(
            "MRI orientation:",
            orientation
        )

        if orientation != (
            "R",
            "A",
            "S"
        ):

            raise ValueError(
                f"MRI is not RAS oriented: "
                f"{orientation}"
            )


        # --------------------------------------------------------
        # VERIFY MRI/AAL GEOMETRY
        # --------------------------------------------------------

        validate_geometry(
            mri,
            atlas_img,
            subject_id,
            volume_label
        )

        print(
            "MRI/AAL geometry: PASS"
        )


        # --------------------------------------------------------
        # MRI DATA VALIDATION
        # --------------------------------------------------------

        mri_data = mri.get_fdata(
            dtype=np.float32
        )

        if mri_data.size == 0:

            raise ValueError(
                "MRI is empty."
            )

        if not np.all(
            np.isfinite(mri_data)
        ):

            raise ValueError(
                "MRI contains NaN or infinite values."
            )

        nonzero_voxels = np.count_nonzero(
            mri_data
        )

        print(
            "Non-zero MRI voxels:",
            f"{nonzero_voxels:,}"
        )

        if nonzero_voxels == 0:

            raise ValueError(
                "MRI contains no non-zero data."
            )


        # --------------------------------------------------------
        # AAL EXTRACTION
        # --------------------------------------------------------

        features = masker.fit_transform(
            mri
        )

        feature_vector = normalize_masker_output(
            features,
            subject_id,
            volume_label
        )

        print(
            "ROI feature count:",
            len(feature_vector)
        )


        # --------------------------------------------------------
        # CREATE FEATURE ROW
        # --------------------------------------------------------

        row = {
            "subject_id": subject_id,
            "acquisition_type": acquisition_type,
            "volume": volume_label
        }

        for feature_name, value in zip(
            feature_names,
            feature_vector
        ):

            row[feature_name] = float(
                value
            )

        df = pd.DataFrame(
            [row],
            columns=[
                "subject_id",
                "acquisition_type",
                "volume"
            ] + feature_names
        )


        # --------------------------------------------------------
        # VERIFY FEATURE ROW
        # --------------------------------------------------------

        expected_columns = (
            3 + EXPECTED_ROIS
        )

        if df.shape != (
            1,
            expected_columns
        ):

            raise ValueError(
                "Unexpected feature row shape.\n"
                f"Expected: (1, {expected_columns})\n"
                f"Found   : {df.shape}"
            )

        values = df[
            feature_names
        ].to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(values)
        ):

            raise ValueError(
                "Feature row contains "
                "non-finite values."
            )


        # --------------------------------------------------------
        # SAVE INDIVIDUAL FEATURE FILE
        # --------------------------------------------------------

        df.to_csv(
            output_csv,
            index=False
        )

        print(
            "Feature file saved:",
            output_csv
        )

        all_feature_rows.append(
            df
        )

        successful += 1

        print(
            "Status: SUCCESS"
        )


    except Exception as e:

        failed += 1

        failed_items.append(
            (
                subject_id,
                acquisition_type,
                volume_label,
                str(e)
            )
        )

        print(
            "Status: FAILED"
        )

        print(
            "Error:",
            str(e)
        )


# ================================================================
# EXTRACTION RESULT
# ================================================================

print("\n" + "=" * 70)
print("EXTRACTION RESULT")
print("=" * 70)

print(
    "Successful:",
    successful
)

print(
    "Reused valid:",
    skipped
)

print(
    "Failed:",
    failed
)

print(
    "Expected:",
    len(expected_inputs)
)


if failed:

    print(
        "\nFailed files:"
    )

    for (
        subject,
        acquisition,
        volume,
        error
    ) in failed_items:

        print(
            f"{subject} | "
            f"{acquisition} | "
            f"{volume} | "
            f"{error}"
        )

    raise RuntimeError(
        f"AAL extraction failed for "
        f"{failed} volume(s)."
    )


if successful != len(expected_inputs):

    raise RuntimeError(
        "Successful extraction count does not "
        "match expected input count."
    )


# ================================================================
# COMBINE FEATURE MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CREATING COMBINED FEATURE MATRIX")
print("=" * 70)

if not all_feature_rows:

    raise RuntimeError(
        "No feature rows were produced."
    )

combined_df = pd.concat(
    all_feature_rows,
    ignore_index=True
)

combined_df = combined_df.sort_values(
    by=[
        "subject_id",
        "acquisition_type",
        "volume"
    ]
).reset_index(
    drop=True
)


# ================================================================
# FINAL MATRIX STRUCTURE
# ================================================================

expected_columns = [
    "subject_id",
    "acquisition_type",
    "volume"
] + feature_names

if list(
    combined_df.columns
) != expected_columns:

    raise RuntimeError(
        "Final feature column structure is incorrect."
    )


expected_shape = (
    len(expected_inputs),
    3 + EXPECTED_ROIS
)

if combined_df.shape != expected_shape:

    raise RuntimeError(
        "Unexpected final matrix shape.\n"
        f"Expected: {expected_shape}\n"
        f"Found   : {combined_df.shape}"
    )


# ================================================================
# FINAL SUBJECT CHECK
# ================================================================

expected_subjects = set(
    selected[
        "subject_id"
    ].astype(str)
)

actual_subjects = set(
    combined_df[
        "subject_id"
    ].astype(str)
)

if actual_subjects != expected_subjects:

    missing = sorted(
        expected_subjects
        - actual_subjects
    )

    unexpected = sorted(
        actual_subjects
        - expected_subjects
    )

    raise RuntimeError(
        "Final matrix subject mismatch.\n"
        f"Missing subjects    : {missing}\n"
        f"Unexpected subjects : {unexpected}"
    )


# ================================================================
# FINAL ACQUISITION CHECK
# ================================================================

expected_rms_subjects = set(
    selected.loc[
        selected["acquisition_type"] == "RMS",
        "subject_id"
    ]
)

expected_5echo_subjects = set(
    selected.loc[
        selected["acquisition_type"] == "5ECHO",
        "subject_id"
    ]
)

actual_rms_subjects = set(
    combined_df.loc[
        combined_df["acquisition_type"] == "RMS",
        "subject_id"
    ].astype(str)
)

actual_5echo_subjects = set(
    combined_df.loc[
        combined_df["acquisition_type"] == "5ECHO",
        "subject_id"
    ].astype(str)
)


if actual_rms_subjects != expected_rms_subjects:

    raise RuntimeError(
        "RMS subject set mismatch."
    )


if actual_5echo_subjects != expected_5echo_subjects:

    raise RuntimeError(
        "5ECHO subject set mismatch."
    )


# ================================================================
# RMS ROW VALIDATION
# ================================================================

rms_rows = combined_df[
    combined_df["acquisition_type"]
    == "RMS"
]

if len(rms_rows) != len(
    expected_rms_subjects
):

    raise RuntimeError(
        "RMS row count mismatch.\n"
        f"Expected: {len(expected_rms_subjects)}\n"
        f"Found   : {len(rms_rows)}"
    )


if not rms_rows[
    "volume"
].eq("RMS").all():

    raise RuntimeError(
        "RMS rows contain an invalid volume label."
    )


# ================================================================
# 5ECHO ROW VALIDATION
# ================================================================

five_echo_rows = combined_df[
    combined_df["acquisition_type"]
    == "5ECHO"
]

expected_5echo_rows = (
    len(expected_5echo_subjects)
    * 5
)

if len(five_echo_rows) != (
    expected_5echo_rows
):

    raise RuntimeError(
        "5ECHO row count mismatch.\n"
        f"Expected: {expected_5echo_rows}\n"
        f"Found   : {len(five_echo_rows)}"
    )


if expected_5echo_rows > 0:

    expected_echo_volume_labels = {
        f"Echo {echo}"
        for echo in ECHOES
    }

    actual_echo_volume_labels = set(
        five_echo_rows[
            "volume"
        ].astype(str)
    )

    if actual_echo_volume_labels != (
        expected_echo_volume_labels
    ):

        raise RuntimeError(
            "5ECHO volume labels are incorrect.\n"
            f"Expected: "
            f"{sorted(expected_echo_volume_labels)}\n"
            f"Found: "
            f"{sorted(actual_echo_volume_labels)}"
        )


# ================================================================
# DUPLICATE CHECK
# ================================================================

duplicate_count = combined_df.duplicated(
    subset=[
        "subject_id",
        "acquisition_type",
        "volume"
    ]
).sum()

if duplicate_count != 0:

    raise RuntimeError(
        "Duplicate subject/acquisition/volume rows detected."
    )


# ================================================================
# FINAL FEATURE VALIDATION
# ================================================================

feature_matrix = combined_df[
    feature_names
].to_numpy(
    dtype=np.float64
)

if not np.all(
    np.isfinite(feature_matrix)
):

    raise RuntimeError(
        "Final feature matrix contains "
        "NaN or infinite values."
    )


missing_values = (
    combined_df.isna()
    .sum()
    .sum()
)

if missing_values != 0:

    raise RuntimeError(
        "Final feature matrix contains "
        f"{missing_values} missing values."
    )


# ================================================================
# SAVE COMBINED MATRIX
# ================================================================

combined_df.to_csv(
    COMBINED_OUTPUT,
    index=False
)

print(
    "\nCombined feature matrix saved:"
)

print(
    COMBINED_OUTPUT
)


# ================================================================
# RELOAD FINAL MATRIX
# ================================================================

print("\n" + "=" * 70)
print("RELOADING FINAL FEATURE MATRIX")
print("=" * 70)

final_df = pd.read_csv(
    COMBINED_OUTPUT
)

if final_df.shape != expected_shape:

    raise RuntimeError(
        "Reloaded feature matrix has unexpected shape.\n"
        f"Expected: {expected_shape}\n"
        f"Found   : {final_df.shape}"
    )


if list(
    final_df.columns
) != expected_columns:

    raise RuntimeError(
        "Reloaded feature matrix has incorrect columns."
    )


final_feature_matrix = final_df[
    feature_names
].to_numpy(
    dtype=np.float64
)

if not np.all(
    np.isfinite(final_feature_matrix)
):

    raise RuntimeError(
        "Reloaded feature matrix contains "
        "non-finite values."
    )


if final_df.isna().sum().sum() != 0:

    raise RuntimeError(
        "Reloaded feature matrix contains "
        "missing values."
    )


# ================================================================
# RELOAD DUPLICATE CHECK
# ================================================================

reload_duplicates = final_df.duplicated(
    subset=[
        "subject_id",
        "acquisition_type",
        "volume"
    ]
).sum()

if reload_duplicates != 0:

    raise RuntimeError(
        "Reloaded matrix contains duplicate rows."
    )


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("AAL FEATURE EXTRACTION STATUS")
print("=" * 70)

print(
    "STATUS: SUCCESS"
)

print(
    f"Subjects              : {len(selected)}"
)

print(
    f"Total feature rows    : {len(final_df)}"
)

print(
    f"RMS rows              : {len(rms_rows)}"
)

print(
    f"5ECHO rows            : {len(five_echo_rows)}"
)

print(
    f"AAL ROIs              : {EXPECTED_ROIS}"
)

print(
    f"Columns               : {EXPECTED_ROIS + 3}"
)

print(
    "Original AAL IDs      : PRESERVED"
)

print(
    "Duplicate rows        : 0"
)

print(
    "Missing values        : 0"
)

print(
    "Finite feature values : PASS"
)

print(
    "\nAAL ROI extraction completed successfully."
)

print("=" * 70)