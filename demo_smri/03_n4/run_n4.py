from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent.parent

INPUT = (
    ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "03_reoriented"
    / "A00024372_sMRI_reoriented.nii.gz"
)

OUTPUT_DIR = (
    ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "04_n4"
)

OUTPUT = OUTPUT_DIR / "A00024372_sMRI_n4.nii.gz"

print("=" * 70)
print("NeuroFusion sMRI DEMO - N4 BIAS CORRECTION")
print("=" * 70)

print("Input :", INPUT)
print("Output:", OUTPUT)

# ------------------------------------------------------------
# Verify ONLY the demo input
# ------------------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nDemo input not found:\n{INPUT}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

print("\nLoading demo image...")

img = sitk.ReadImage(str(INPUT))

print("Dimension:", img.GetDimension())
print("Size     :", img.GetSize())
print("Spacing  :", img.GetSpacing())

if img.GetDimension() != 3:
    raise RuntimeError(
        f"Expected 3D image, got {img.GetDimension()}D"
    )

# ------------------------------------------------------------
# Convert image to float
# ------------------------------------------------------------

img_float = sitk.Cast(
    img,
    sitk.sitkFloat32
)

# ------------------------------------------------------------
# Create N4 mask
# ------------------------------------------------------------

print("\nCreating foreground mask...")

mask = sitk.OtsuThreshold(
    img_float,
    0,
    1,
    200
)

# ------------------------------------------------------------
# N4 bias correction
# ------------------------------------------------------------

print("Running N4 bias-field correction...")

corrector = sitk.N4BiasFieldCorrectionImageFilter()

corrector.SetMaximumNumberOfIterations(
    [50, 50, 30, 20]
)

corrected = corrector.Execute(
    img_float,
    mask
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

print("\nSaving corrected image...")

sitk.WriteImage(
    corrected,
    str(OUTPUT),
    True
)

# ------------------------------------------------------------
# Verify output
# ------------------------------------------------------------

if not OUTPUT.exists():
    raise RuntimeError(
        f"N4 output was not created:\n{OUTPUT}"
    )

print("\nVerifying output...")

check = nib.load(str(OUTPUT))

data = check.get_fdata(dtype=np.float32)

if len(check.shape) != 3:
    raise RuntimeError(
        f"Output is not 3D: {check.shape}"
    )

if not np.all(np.isfinite(data)):
    raise RuntimeError(
        "Output contains NaN/Inf values."
    )

if not np.any(data != 0):
    raise RuntimeError(
        "Output contains only zeros."
    )

orientation = nib.aff2axcodes(check.affine)

voxel_size = check.header.get_zooms()[:3]

print("Output shape      :", check.shape)
print("Output orientation:", orientation)
print("Output voxel size :", voxel_size)

print("\n" + "=" * 70)
print("STATUS: DEMO N4 BIAS CORRECTION PASS")
print("=" * 70)

print("\nOnly this demo subject was processed:")
print("  A00024372")

print("\nOutput:")
print(OUTPUT)

print("\nNo OpenNeuro subjects were processed.")
print("No batch N4 script was executed.")
