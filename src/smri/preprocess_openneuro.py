from pathlib import Path
import pandas as pd
import numpy as np
import SimpleITK as sitk
import subprocess
import shutil
import sys
import torch


# ============================================================
# NeuroFusion - OpenNeuro sMRI preprocessing
#
# Pipeline:
#   T1 MRI
#      ↓
#   Reorientation to RAS
#      ↓
#   N4 bias-field correction
#      ↓
#   HD-BET skull stripping
#      ↓
#   Robust intensity normalization
#      ↓
#   Validation
#
# Note:
#   DICOM → NIfTI is NOT performed here because the manifest
#   already points to NIfTI T1 images.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST = Path(
    "processed_data/smri/openneuro_ds000115/openneuro_smri_manifest.csv"
)

OUTPUT_ROOT = Path(
    "processed_data/smri/openneuro_ds000115/preprocessed"
)

RAS_DIR = OUTPUT_ROOT / "ras"
N4_DIR = OUTPUT_ROOT / "n4"
BET_DIR = OUTPUT_ROOT / "skull_stripped"
NORM_DIR = OUTPUT_ROOT / "normalized"

REPORT_PATH = OUTPUT_ROOT / "openneuro_preprocessing_report.csv"

# HD-BET executable
HD_BET = shutil.which("hd-bet")

# Use CUDA GPU if available
USE_GPU = torch.cuda.is_available()

# HD-BET accepts "cuda" for GPU and "cpu" for CPU
HD_BET_DEVICE = "cuda" if USE_GPU else "cpu"


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

def print_environment():

    print("\nEnvironment:")

    print(
        "SimpleITK:",
        sitk.Version()
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "GPU: None"
        )

    print(
        "HD-BET device:",
        HD_BET_DEVICE
    )


# ============================================================
# REORIENTATION
# ============================================================

def reorient_to_ras(
    input_path: Path,
    output_path: Path
):
    """
    Reorient image to RAS orientation using SimpleITK.
    """

    image = sitk.ReadImage(
        str(input_path)
    )

    # SimpleITK uses DICOM orientation convention internally.
    # DICOMOrientImageFilter allows us to explicitly request RAS.
    orienter = sitk.DICOMOrientImageFilter()

    orienter.SetDesiredCoordinateOrientation(
        "RAS"
    )

    ras_image = orienter.Execute(
        image
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sitk.WriteImage(
        ras_image,
        str(output_path)
    )

    return output_path


# ============================================================
# N4 BIAS FIELD CORRECTION
# ============================================================

def n4_bias_correction(
    input_path: Path,
    output_path: Path
):
    """
    Apply N4 bias-field correction using SimpleITK.

    A foreground mask is generated using the
    SimpleITK OtsuThresholdImageFilter API.
    """

    image = sitk.ReadImage(
        str(input_path)
    )

    # --------------------------------------------------------
    # Generate foreground mask
    # --------------------------------------------------------

    otsu = sitk.OtsuThresholdImageFilter()

    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)

    mask = otsu.Execute(
        image
    )

    # Convert image to floating point.
    image_float = sitk.Cast(
        image,
        sitk.sitkFloat32
    )

    # --------------------------------------------------------
    # N4 correction
    # --------------------------------------------------------

    corrector = (
        sitk.N4BiasFieldCorrectionImageFilter()
    )

    corrected = corrector.Execute(
        image_float,
        mask
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sitk.WriteImage(
        corrected,
        str(output_path)
    )

    return output_path


# ============================================================
# HD-BET SKULL STRIPPING
# ============================================================

def run_hd_bet(
    input_path: Path,
    output_path: Path
):
    """
    Run HD-BET using CUDA when available.

    IMPORTANT:
    HD-BET requires the output filename itself to end
    with .nii.gz.
    """

    if HD_BET is None:

        raise RuntimeError(
            "HD-BET executable was not found."
        )

    # --------------------------------------------------------
    # Ensure correct extension
    # --------------------------------------------------------

    output_path = Path(
        output_path
    )

    if not str(output_path).endswith(
        ".nii.gz"
    ):

        raise ValueError(
            "HD-BET output path must end with .nii.gz"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Build command
    # --------------------------------------------------------

    command = [
        HD_BET,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-device",
        HD_BET_DEVICE,
        "--save_bet_mask",
    ]

    print(
        "HD-BET command:"
    )

    print(
        " ".join(command)
    )

    # --------------------------------------------------------
    # Run HD-BET
    # --------------------------------------------------------

    subprocess.run(
        command,
        check=True
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    if not output_path.exists():

        raise RuntimeError(
            "HD-BET finished but output file was not created."
        )

    return output_path


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(
    input_path: Path,
    output_path: Path
):
    """
    Robust intensity normalization.

    Steps:
        1. Read skull-stripped image.
        2. Select foreground voxels.
        3. Clip to 1st-99th percentile.
        4. Z-score normalize foreground.
        5. Preserve image spatial information.
    """

    image = sitk.ReadImage(
        str(input_path)
    )

    array = sitk.GetArrayFromImage(
        image
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Foreground
    # --------------------------------------------------------

    foreground = array[
        array > 0
    ]

    if foreground.size == 0:

        raise ValueError(
            "No foreground voxels found."
        )

    # --------------------------------------------------------
    # Robust clipping
    # --------------------------------------------------------

    p1 = np.percentile(
        foreground,
        1
    )

    p99 = np.percentile(
        foreground,
        99
    )

    array = np.clip(
        array,
        p1,
        p99
    )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    foreground = array[
        array > 0
    ]

    mean = foreground.mean()

    std = foreground.std()

    if std == 0:

        raise ValueError(
            "Foreground standard deviation is zero."
        )

    mask = array > 0

    array[mask] = (
        array[mask] - mean
    ) / std

    # --------------------------------------------------------
    # Convert back to SimpleITK image
    # --------------------------------------------------------

    normalized = sitk.GetImageFromArray(
        array
    )

    # Preserve original geometry
    normalized.CopyInformation(
        image
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sitk.WriteImage(
        normalized,
        str(output_path)
    )

    return output_path


# ============================================================
# VALIDATION
# ============================================================

def validate_image(
    path: Path
):

    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    spacing = image.GetSpacing()

    nan_count = np.isnan(
        array
    ).sum()

    inf_count = np.isinf(
        array
    ).sum()

    return {

        "shape": tuple(
            reversed(
                array.shape
            )
        ),

        "spacing": spacing,

        "min": float(
            np.min(array)
        ),

        "max": float(
            np.max(array)
        ),

        "mean": float(
            np.mean(array)
        ),

        "std": float(
            np.std(array)
        ),

        "nan": int(
            nan_count
        ),

        "inf": int(
            inf_count
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "NeuroFusion - OpenNeuro sMRI preprocessing"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    print_environment()

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():

        print(
            "\nERROR: Manifest not found:"
        )

        print(
            MANIFEST
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check HD-BET
    # --------------------------------------------------------

    if HD_BET is None:

        print(
            "\nERROR: HD-BET was not found."
        )

        print(
            "Run:"
        )

        print(
            "where hd-bet"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print(
        f"\nManifest : {MANIFEST}"
    )

    print(
        f"HD-BET   : {HD_BET}"
    )

    print(
        f"Device   : "
        f"{'CUDA GPU' if USE_GPU else 'CPU'}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for directory in [

        RAS_DIR,
        N4_DIR,
        BET_DIR,
        NORM_DIR

    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        MANIFEST
    )

    print(
        f"\nSubjects: {len(df)}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    results = []

    # ========================================================
    # PROCESS SUBJECTS
    # ========================================================

    for index, row in df.iterrows():

        subject = str(
            row["subject_id"]
        )

        input_path = Path(
            row["t1_path"]
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"[{index + 1}/{len(df)}] {subject}"
        )

        # ----------------------------------------------------
        # Check input
        # ----------------------------------------------------

        if not input_path.exists():

            print(
                "ERROR: Input file missing"
            )

            results.append({

                "subject_id": subject,

                "label": row.get(
                    "label",
                    ""
                ),

                "status":
                    "MISSING_INPUT",

                "error":
                    "Input file does not exist."

            })

            continue

        try:

            # =================================================
            # STEP 1 — RAS REORIENTATION
            # =================================================

            ras_path = (
                RAS_DIR /
                f"{subject}_T1w_ras.nii.gz"
            )

            if ras_path.exists():

                print(
                    "Reorientation to RAS: "
                    "already exists -> SKIP"
                )

            else:

                print(
                    "Reorientation to RAS: "
                    "running..."
                )

                reorient_to_ras(
                    input_path,
                    ras_path
                )

                print(
                    "Reorientation: DONE"
                )

            # =================================================
            # STEP 2 — N4
            # =================================================

            n4_path = (
                N4_DIR /
                f"{subject}_T1w_ras_n4.nii.gz"
            )

            if n4_path.exists():

                print(
                    "N4: already exists -> SKIP"
                )

            else:

                print(
                    "N4: running..."
                )

                n4_bias_correction(
                    ras_path,
                    n4_path
                )

                print(
                    "N4: DONE"
                )

            # =================================================
            # STEP 3 — HD-BET
            # =================================================

            bet_output = (
                BET_DIR /
                f"{subject}_T1w_ras_n4_bet.nii.gz"
            )

            if bet_output.exists():

                print(
                    "HD-BET: "
                    "already exists -> SKIP"
                )

            else:

                print(
                    "HD-BET: "
                    f"running on "
                    f"{'CUDA GPU' if USE_GPU else 'CPU'}..."
                )

                run_hd_bet(
                    n4_path,
                    bet_output
                )

                print(
                    "HD-BET: DONE"
                )

            # =================================================
            # STEP 4 — NORMALIZATION
            # =================================================

            normalized_path = (
                NORM_DIR /
                f"{subject}_T1w_ras_n4_bet_normalized.nii.gz"
            )

            if normalized_path.exists():

                print(
                    "Normalization: "
                    "already exists -> SKIP"
                )

            else:

                print(
                    "Normalization: running..."
                )

                normalize_image(
                    bet_output,
                    normalized_path
                )

                print(
                    "Normalization: DONE"
                )

            # =================================================
            # STEP 5 — VALIDATION
            # =================================================

            print(
                "Validation: running..."
            )

            stats = validate_image(
                normalized_path
            )

            print(
                f"Validation: "
                f"shape={stats['shape']} "
                f"spacing="
                f"{np.round(stats['spacing'], 3)} "
                f"NaN={stats['nan']} "
                f"Inf={stats['inf']}"
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            results.append({

                "subject_id":
                    subject,

                "label":
                    row.get(
                        "label",
                        ""
                    ),

                "age":
                    row.get(
                        "age",
                        ""
                    ),

                "gender":
                    row.get(
                        "gender",
                        ""
                    ),

                "input_path":
                    str(input_path),

                "ras_path":
                    str(ras_path),

                "n4_path":
                    str(n4_path),

                "skull_stripped_path":
                    str(bet_output),

                "normalized_path":
                    str(normalized_path),

                "shape":
                    str(stats["shape"]),

                "spacing":
                    str(stats["spacing"]),

                "min":
                    stats["min"],

                "max":
                    stats["max"],

                "mean":
                    stats["mean"],

                "std":
                    stats["std"],

                "nan":
                    stats["nan"],

                "inf":
                    stats["inf"],

                "status":
                    "SUCCESS"

            })

            print(
                "STATUS: SUCCESS"
            )

        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except subprocess.CalledProcessError as e:

            print(
                f"HD-BET ERROR: {e}"
            )

            results.append({

                "subject_id":
                    subject,

                "label":
                    row.get(
                        "label",
                        ""
                    ),

                "status":
                    "HD_BET_FAILED",

                "error":
                    str(e)

            })

            continue

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

            results.append({

                "subject_id":
                    subject,

                "label":
                    row.get(
                        "label",
                        ""
                    ),

                "status":
                    "FAILED",

                "error":
                    str(e)

            })

            continue

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = pd.DataFrame(
        results
    )

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nReport:"
    )

    print(
        REPORT_PATH
    )

    if (
        not report.empty
        and "status" in report.columns
    ):

        print(
            "\nStatus:"
        )

        print(
            report["status"]
            .value_counts()
        )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()