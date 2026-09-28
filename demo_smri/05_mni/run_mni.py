from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nilearn import datasets

# ================================================================
# NeuroFusion sMRI DEMO - MNI AFFINE REGISTRATION
# Demo subject: A00024372 ONLY
# ================================================================

ROOT = Path(__file__).resolve().parent.parent.parent

INPUT = (
    ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "05_hd_bet"
    / "A00024372_sMRI_n4_bet.nii.gz"
)

OUTPUT_DIR = (
    ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "06_mni"
)

OUTPUT = OUTPUT_DIR / "A00024372_sMRI_mni.nii.gz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NeuroFusion sMRI DEMO - MNI AFFINE REGISTRATION")
print("=" * 70)

print("Input :", INPUT)
print("Output:", OUTPUT)

if not INPUT.exists():
    raise FileNotFoundError(f"Demo input not found: {INPUT}")

# ----------------------------------------------------------------
# Load demo image
# ----------------------------------------------------------------

print("\nLoading demo sMRI...")

moving = sitk.ReadImage(str(INPUT), sitk.sitkFloat32)

print("Moving image size    :", moving.GetSize())
print("Moving spacing       :", moving.GetSpacing())
print("Moving direction     :", moving.GetDirection())

# ----------------------------------------------------------------
# Load MNI152 template
# ----------------------------------------------------------------

print("\nLoading Nilearn MNI152 1 mm template...")

mni = datasets.load_mni152_template()

mni_data = np.asarray(mni.get_fdata(), dtype=np.float32)

# Convert MNI NIfTI to SimpleITK while preserving the MNI affine.
mni_img = nib.Nifti1Image(
    mni_data,
    mni.affine
)

mni_path = OUTPUT_DIR / "MNI152_template_1mm.nii.gz"

nib.save(
    mni_img,
    str(mni_path)
)

fixed = sitk.ReadImage(
    str(mni_path),
    sitk.sitkFloat32
)

print("MNI template size    :", fixed.GetSize())
print("MNI template spacing :", fixed.GetSpacing())

# ----------------------------------------------------------------
# Initial affine transform
# ----------------------------------------------------------------

print("\nCreating initial affine transform...")

initial_transform = sitk.CenteredTransformInitializer(
    fixed,
    moving,
    sitk.AffineTransform(3),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

# ----------------------------------------------------------------
# Registration
# ----------------------------------------------------------------

print("\nRunning affine registration...")
print("Only A00024372 is being processed.")

registration = sitk.ImageRegistrationMethod()

registration.SetMetricAsMattesMutualInformation(
    numberOfHistogramBins=50
)

registration.SetMetricSamplingStrategy(
    registration.RANDOM
)

registration.SetMetricSamplingPercentage(
    0.10
)

registration.SetInterpolator(
    sitk.sitkLinear
)

registration.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)

registration.SetOptimizerScalesFromPhysicalShift()

registration.SetShrinkFactorsPerLevel(
    shrinkFactors=[4, 2, 1]
)

registration.SetSmoothingSigmasPerLevel(
    smoothingSigmas=[2, 1, 0]
)

registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration.SetInitialTransform(
    initial_transform,
    inPlace=False
)

final_transform = registration.Execute(
    fixed,
    moving
)

print("\nRegistration completed.")

print(
    "Final metric value:",
    registration.GetMetricValue()
)

print(
    "Optimizer iterations:",
    registration.GetOptimizerIteration()
)

# ----------------------------------------------------------------
# Resample into MNI space
# ----------------------------------------------------------------

print("\nResampling subject into MNI space...")

registered = sitk.Resample(
    moving,
    fixed,
    final_transform,
    sitk.sitkLinear,
    0.0,
    sitk.sitkFloat32
)

sitk.WriteImage(
    registered,
    str(OUTPUT)
)

# ----------------------------------------------------------------
# Save transform
# ----------------------------------------------------------------

TRANSFORM = OUTPUT_DIR / "A00024372_to_MNI_affine.tfm"

sitk.WriteTransform(
    final_transform,
    str(TRANSFORM)
)

# ----------------------------------------------------------------
# Verification
# ----------------------------------------------------------------

print("\nVerifying MNI output...")

check = sitk.ReadImage(
    str(OUTPUT),
    sitk.sitkFloat32
)

array = sitk.GetArrayFromImage(check)

if array.size == 0:
    raise RuntimeError("MNI output contains no voxels.")

if not np.all(np.isfinite(array)):
    raise RuntimeError("MNI output contains NaN/Inf values.")

if not np.any(array != 0):
    raise RuntimeError("MNI output contains only zeros.")

if check.GetSize() != fixed.GetSize():
    raise RuntimeError(
        f"Output size {check.GetSize()} does not match "
        f"MNI template size {fixed.GetSize()}."
    )

if not np.allclose(
    check.GetSpacing(),
    fixed.GetSpacing(),
    atol=1e-4
):
    raise RuntimeError(
        "Output spacing does not match MNI template."
    )

print("Output size       :", check.GetSize())
print("Output spacing    :", check.GetSpacing())
print("Output direction  :", check.GetDirection())
print("Output voxel count:", array.size)

print("\n" + "=" * 70)
print("STATUS: DEMO MNI AFFINE REGISTRATION PASS")
print("=" * 70)

print("\nOnly this demo subject was processed:")
print("  A00024372")

print("\nRegistered image:")
print(OUTPUT)

print("\nAffine transform:")
print(TRANSFORM)

print("\nMNI template used:")
print(mni_path)

print("\nNo batch subjects were processed.")
print("No files under src/ were modified.")
print("=" * 70)
