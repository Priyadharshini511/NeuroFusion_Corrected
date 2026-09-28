from pathlib import Path
import subprocess
import sys
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

INPUT = (
    ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "04_n4"
    / "A00024372_sMRI_n4.nii.gz"
)

OUTPUT_DIR = (
    ROOT
    / "processed_data"
    / "smri"
    / "demo_single_subject"
    / "05_hd_bet"
)

OUTPUT = OUTPUT_DIR / "A00024372_sMRI_n4_bet.nii.gz"
MASK = OUTPUT_DIR / "A00024372_sMRI_n4_bet_mask.nii.gz"

print("=" * 70)
print("NeuroFusion sMRI DEMO - HD-BET SKULL STRIPPING")
print("=" * 70)

print("Input :", INPUT)
print("Output:", OUTPUT)
print("Mask  :", MASK)

# ------------------------------------------------------------
# Verify ONLY the demo input
# ------------------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nDemo N4 input not found:\n{INPUT}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Locate HD-BET executable from the active environment
# ------------------------------------------------------------

hd_bet = "hd-bet"

if sys.platform == "win32":
    possible = Path(sys.executable).parent / "Scripts" / "hd-bet.exe"

    if possible.exists():
        hd_bet = str(possible)

print("\nHD-BET executable:")
print(hd_bet)

# ------------------------------------------------------------
# Run HD-BET ONLY on this demo subject
# ------------------------------------------------------------

print("\nRunning HD-BET on A00024372 only...\n")

cmd = [
    hd_bet,
    "-i",
    str(INPUT),
    "-o",
    str(OUTPUT),
    "-device",
    "cpu",
    "--save_bet_mask",
]

print("Command:")
print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
print()

result = subprocess.run(cmd)

if result.returncode != 0:
    raise RuntimeError(
        f"HD-BET failed with exit code {result.returncode}"
    )

# ------------------------------------------------------------
# Verify output
# ------------------------------------------------------------

if not OUTPUT.exists():
    raise RuntimeError(
        f"HD-BET output was not created:\n{OUTPUT}"
    )

print("\nVerifying skull-stripped output...")

img = nib.load(str(OUTPUT))
data = img.get_fdata(dtype=np.float32)

if len(img.shape) != 3:
    raise RuntimeError(
        f"Output is not 3D: {img.shape}"
    )

if not np.all(np.isfinite(data)):
    raise RuntimeError(
        "HD-BET output contains NaN/Inf values."
    )

if not np.any(data != 0):
    raise RuntimeError(
        "HD-BET output contains only zeros."
    )

orientation = nib.aff2axcodes(img.affine)
voxel_size = img.header.get_zooms()[:3]

print("Output shape      :", img.shape)
print("Output orientation:", orientation)
print("Output voxel size :", voxel_size)

# ------------------------------------------------------------
# Check mask if generated
# ------------------------------------------------------------

if MASK.exists():

    mask_img = nib.load(str(MASK))
    mask_data = mask_img.get_fdata(dtype=np.float32)

    print("\nMask found:")
    print("Mask shape        :", mask_img.shape)
    print("Mask orientation  :", nib.aff2axcodes(mask_img.affine))

    if not np.any(mask_data != 0):
        raise RuntimeError(
            "HD-BET mask contains only zeros."
        )

    print("Mask verification : PASS")

else:
    print("\nWARNING: HD-BET mask was not found.")
    print("Continuing because the brain-extracted image exists.")

print("\n" + "=" * 70)
print("STATUS: DEMO HD-BET PASS")
print("=" * 70)

print("\nOnly this demo subject was processed:")
print("  A00024372")

print("\nOutput:")
print(OUTPUT)

if MASK.exists():
    print("\nBrain mask:")
    print(MASK)

print("\nNo batch subjects were processed.")
print("No files under src/ were modified.")
print("=" * 70)
