from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform

PROJECT_ROOT = Path(".")

VBM_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "06_vbm"
    / "mwc1A00024372_sMRI_n4.nii"
)

ATLAS_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "atlas"
    / "aal_mni"
    / "AAL_MNI152_2mm.nii.gz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "07_aal_features"
)

OUTPUT_FILE = OUTPUT_DIR / "A00024372_aal_features.csv"


print("=" * 80)
print("NeuroFusion AI - DEMO AAL FEATURE EXTRACTION")
print("=" * 80)

print()
print("VBM:")
print(VBM_FILE.resolve())

print()
print("AAL atlas:")
print(ATLAS_FILE.resolve())

if not VBM_FILE.is_file():
    raise FileNotFoundError(
        f"VBM file not found:\n{VBM_FILE.resolve()}"
    )

if not ATLAS_FILE.is_file():
    raise FileNotFoundError(
        f"AAL atlas not found:\n{ATLAS_FILE.resolve()}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# LOAD VBM
# ----------------------------------------------------------------------

print()
print("Loading modulated GM map...")

vbm_img = nib.load(str(VBM_FILE))
vbm_data = vbm_img.get_fdata(dtype=np.float64)

print("VBM shape:", vbm_img.shape)
print("VBM voxel size:", vbm_img.header.get_zooms()[:3])


if vbm_data.ndim != 3:
    raise ValueError("VBM image is not 3-D.")

if not np.all(np.isfinite(vbm_data)):
    raise ValueError("VBM contains non-finite values.")

if np.count_nonzero(vbm_data) == 0:
    raise ValueError("VBM is completely zero.")


# ----------------------------------------------------------------------
# LOAD AAL
# ----------------------------------------------------------------------

print()
print("Loading AAL atlas...")

atlas_img = nib.load(str(ATLAS_FILE))
atlas_data = atlas_img.get_fdata()

atlas_labels = np.unique(atlas_data)
atlas_labels = atlas_labels[atlas_labels > 0]
atlas_labels = np.sort(atlas_labels.astype(int))

print("AAL shape:", atlas_img.shape)
print("AAL voxel size:", atlas_img.header.get_zooms()[:3])
print("AAL labels:", len(atlas_labels))

if len(atlas_labels) != 116:
    raise ValueError(
        f"Expected 116 AAL labels, found {len(atlas_labels)}"
    )


# ----------------------------------------------------------------------
# RESAMPLE AAL TO VBM GRID
# ----------------------------------------------------------------------

print()
print("Resampling AAL atlas to VBM grid...")

transform = (
    np.linalg.inv(atlas_img.affine)
    @ vbm_img.affine
)

resampled_atlas = affine_transform(
    atlas_data,
    matrix=transform[:3, :3],
    offset=transform[:3, 3],
    output_shape=vbm_img.shape,
    order=0,
    mode="constant",
    cval=0,
    prefilter=False,
)

resampled_atlas = np.rint(
    resampled_atlas
).astype(np.int32)


resampled_labels = np.unique(resampled_atlas)
resampled_labels = resampled_labels[
    resampled_labels > 0
]

print(
    "Resampled labels:",
    len(resampled_labels),
    "/ 116"
)

missing = sorted(
    set(atlas_labels) - set(resampled_labels)
)

if missing:
    raise ValueError(
        f"Missing AAL labels after resampling: {missing}"
    )

print("AAL resampling validation: PASS")


# ----------------------------------------------------------------------
# VOXEL VOLUME
# ----------------------------------------------------------------------

voxel_volume = abs(
    np.linalg.det(
        vbm_img.affine[:3, :3]
    )
)

print()
print("VBM voxel volume:")
print(f"{voxel_volume:.6f} mm^3")


# ----------------------------------------------------------------------
# EXTRACT 116 GM-VOLUME FEATURES
# ----------------------------------------------------------------------

print()
print("Extracting 116 regional GM-volume features...")

row = {
    "subject_id": "A00024372"
}


for roi_label in atlas_labels:

    mask = (
        resampled_atlas == roi_label
    )

    values = vbm_data[mask]

    if values.size == 0:
        raise ValueError(
            f"ROI {roi_label} has zero voxels."
        )

    gm_volume = (
        np.sum(values)
        * voxel_volume
    )

    row[
        f"GMVOL_AAL_{roi_label}"
    ] = gm_volume


df = pd.DataFrame([row])


# ----------------------------------------------------------------------
# VALIDATE FEATURE COUNT
# ----------------------------------------------------------------------

feature_columns = [
    c
    for c in df.columns
    if c.startswith("GMVOL_AAL_")
]

print()
print("Feature count:")
print(len(feature_columns))

if len(feature_columns) != 116:
    raise ValueError(
        f"Expected 116 features, found {len(feature_columns)}"
    )

if not np.all(
    np.isfinite(
        df[feature_columns].to_numpy()
    )
):
    raise ValueError(
        "AAL features contain NaN or infinite values."
    )

print("Feature validation: PASS")


# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 80)
print("AAL FEATURE EXTRACTION PASS")
print("=" * 80)

print()
print("Subject:")
print("  A00024372")

print()
print("Features:")
print("  116")

print()
print("Output:")
print(OUTPUT_FILE.resolve())

print()
print("Columns:")
print(df.columns.tolist())

print()
print("=" * 80)
