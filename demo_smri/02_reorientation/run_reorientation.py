import nibabel as nib
import numpy as np
from pathlib import Path
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
    inv_ornt_aff,
)

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = ROOT / "processed_data" / "smri" / "demo_single_subject" / "02_nifti" / "A00024372_sMRI.nii.gz"
OUTPUT_DIR = ROOT / "processed_data" / "smri" / "demo_single_subject" / "03_reoriented"
OUTPUT = OUTPUT_DIR / "A00024372_sMRI_reoriented.nii.gz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NeuroFusion sMRI DEMO - RAS REORIENTATION")
print("=" * 70)

print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")

if not INPUT.is_file():
    raise FileNotFoundError(f"Input not found: {INPUT}")

img = nib.load(INPUT)

print("\nInput shape:", img.shape)
print("Input orientation:", nib.aff2axcodes(img.affine))
print("Input voxel size:", img.header.get_zooms()[:3])

data = img.get_fdata(dtype=np.float32)

if not np.all(np.isfinite(data)):
    raise RuntimeError("Input contains NaN/Inf values.")

if not np.any(data != 0):
    raise RuntimeError("Input contains only zeros.")

current_ornt = io_orientation(img.affine)
target_ornt = axcodes2ornt(("R", "A", "S"))
transform = ornt_transform(current_ornt, target_ornt)

reoriented_data = apply_orientation(data, transform)

new_affine = img.affine @ inv_ornt_aff(
    transform,
    img.shape,
)

output_img = nib.Nifti1Image(
    reoriented_data.astype(np.float32),
    new_affine,
)

output_img.header.set_xyzt_units("mm")
output_img.header["descrip"] = (
    "NeuroFusion demo - A00024372 - canonical RAS"
)

nib.save(output_img, OUTPUT)

check = nib.load(OUTPUT)

print("\nOutput shape:", check.shape)
print("Output orientation:", nib.aff2axcodes(check.affine))
print("Output voxel size:", check.header.get_zooms()[:3])

if nib.aff2axcodes(check.affine) != ("R", "A", "S"):
    raise RuntimeError("RAS verification failed.")

if not np.allclose(
    check.header.get_zooms()[:3],
    (1.0, 1.0, 1.0),
    atol=0.01,
):
    raise RuntimeError("Voxel-size verification failed.")

print("\nSTATUS: RAS REORIENTATION PASS")
