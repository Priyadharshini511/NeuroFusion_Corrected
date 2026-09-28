# src\batch_qc_registration.py

import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ================================================================
# MNI REGISTRATION VISUAL QUALITY CONTROL
#
# Generic, manifest-driven visual QC.
#
# Current cohort:
#   mprage_selection_manifest.csv
#
# Visual QC:
#   Echo 1 only
#
# Displays:
#   MNI template
#   Registered MRI
#   Overlay
#
# Planes:
#   Axial
#   Coronal
#   Sagittal
#
# This script is QC ONLY.
# It does not modify registered MRI or transforms.
#
# IMPORTANT:
# Successful PNG generation does NOT mean visual QC has been
# manually completed. The generated images must be inspected.
# ================================================================


print("=" * 70)
print("MNI REGISTRATION VISUAL QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

REGISTERED_DIR = (
    r"processed_data\smri"
    r"\registered"
)

MNI_TEMPLATE = (
    r"processed_data\smri"
    r"\mni"
    r"\mni152_template_2mm.nii.gz"
)

MANIFEST_PATH = (
    r"processed_data\smri"
    r"\mprage_selection"
    r"\mprage_selection_manifest.csv"
)

QC_DIR = (
    r"processed_data\smri"
    r"\qc_registration"
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "registration_visual_qc_summary.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

VISUAL_ECHO = 1

GEOMETRY_ATOL = 1e-5

# Percentiles used only for display windowing.
DISPLAY_PERCENTILES = (1, 99)


# ================================================================
# HELPERS
# ================================================================

def get_selected_subjects():

    if not os.path.isfile(
        MANIFEST_PATH
    ):

        raise FileNotFoundError(
            f"Selection manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "subject_id",
        "selection_status"
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:

        raise ValueError(
            "Manifest is missing required columns: "
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
            "No SELECTED subjects found in manifest."
        )

    subjects = (
        selected["subject_id"]
        .astype(str)
        .str.strip()
    )

    if subjects.eq("").any():

        raise ValueError(
            "Manifest contains empty subject IDs."
        )

    if subjects.duplicated().any():

        duplicates = sorted(
            subjects[
                subjects.duplicated()
            ].unique()
        )

        raise ValueError(
            "Duplicate SELECTED subjects found: "
            f"{duplicates}"
        )

    return sorted(
        subjects.unique()
    )


def read_image(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    image = sitk.ReadImage(
        path,
        sitk.sitkFloat32
    )

    data = sitk.GetArrayFromImage(
        image
    )

    if data.size == 0:

        raise ValueError(
            "Image contains no voxels."
        )

    if not np.all(
        np.isfinite(data)
    ):

        raise ValueError(
            "Image contains NaN or infinite values."
        )

    if np.count_nonzero(
        data
    ) == 0:

        raise ValueError(
            "Image contains no non-zero voxels."
        )

    return image, data


def geometry_matches(
    image_a,
    image_b
):

    return (
        image_a.GetSize()
        == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=GEOMETRY_ATOL
        )
        and np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=GEOMETRY_ATOL
        )
        and np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=GEOMETRY_ATOL
        )
    )


def choose_slices(
    template_data
):

    nonzero = np.argwhere(
        np.abs(template_data) > 0
    )

    if nonzero.size == 0:

        shape = template_data.shape

        return (
            shape[0] // 2,
            shape[1] // 2,
            shape[2] // 2
        )

    z_min, y_min, x_min = (
        nonzero.min(axis=0)
    )

    z_max, y_max, x_max = (
        nonzero.max(axis=0)
    )

    z = int(
        round(
            (z_min + z_max) / 2
        )
    )

    y = int(
        round(
            (y_min + y_max) / 2
        )
    )

    x = int(
        round(
            (x_min + x_max) / 2
        )
    )

    return z, y, x


def robust_window(
    data,
    reference_mask=None
):

    if reference_mask is not None:

        values = data[
            reference_mask
        ]

    else:

        values = data[
            np.isfinite(data)
            & (data != 0)
        ]

    if values.size == 0:

        values = data[
            np.isfinite(data)
        ]

    if values.size == 0:

        return 0.0, 1.0

    low, high = np.percentile(
        values,
        DISPLAY_PERCENTILES
    )

    low = float(low)
    high = float(high)

    if not np.isfinite(low):
        low = float(np.min(values))

    if not np.isfinite(high):
        high = float(np.max(values))

    if high <= low:

        high = low + 1e-6

    return low, high


def create_registration_qc(
    subject_id,
    template_data,
    registered_data,
    output_path
):

    # ------------------------------------------------------------
    # Select central anatomical slices.
    # ------------------------------------------------------------

    z, y, x = choose_slices(
        template_data
    )

    # ------------------------------------------------------------
    # Template mask for robust intensity window.
    # ------------------------------------------------------------

    template_mask = (
        np.abs(template_data) > 0
    )

    template_vmin, template_vmax = (
        robust_window(
            template_data,
            template_mask
        )
    )

    registered_mask = (
        np.abs(registered_data) > 0
    )

    registered_vmin, registered_vmax = (
        robust_window(
            registered_data,
            registered_mask
        )
    )

    # ------------------------------------------------------------
    # Create figure.
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12)
    )

    planes = [
        (
            "Axial",
            template_data[z, :, :],
            registered_data[z, :, :]
        ),
        (
            "Coronal",
            template_data[:, y, :],
            registered_data[:, y, :]
        ),
        (
            "Sagittal",
            template_data[:, :, x],
            registered_data[:, :, x]
        )
    ]

    for row, (
        plane_name,
        template_plane,
        registered_plane
    ) in enumerate(
        planes
    ):

        template_plane = np.rot90(
            template_plane
        )

        registered_plane = np.rot90(
            registered_plane
        )

        # --------------------------------------------------------
        # Template
        # --------------------------------------------------------

        axes[row, 0].imshow(
            template_plane,
            cmap="gray",
            vmin=template_vmin,
            vmax=template_vmax
        )

        axes[row, 0].set_title(
            f"{plane_name}: MNI Template"
        )

        axes[row, 0].axis(
            "off"
        )

        # --------------------------------------------------------
        # Registered MRI
        # --------------------------------------------------------

        axes[row, 1].imshow(
            registered_plane,
            cmap="gray",
            vmin=registered_vmin,
            vmax=registered_vmax
        )

        axes[row, 1].set_title(
            f"{plane_name}: Registered MRI"
        )

        axes[row, 1].axis(
            "off"
        )

        # --------------------------------------------------------
        # Overlay
        #
        # Two anatomical images are displayed together with
        # transparency. The purpose is visual alignment inspection.
        # --------------------------------------------------------

        axes[row, 2].imshow(
            template_plane,
            cmap="gray",
            vmin=template_vmin,
            vmax=template_vmax,
            alpha=0.65
        )

        axes[row, 2].imshow(
            registered_plane,
            cmap="gray",
            vmin=registered_vmin,
            vmax=registered_vmax,
            alpha=0.35
        )

        axes[row, 2].set_title(
            f"{plane_name}: Overlay"
        )

        axes[row, 2].axis(
            "off"
        )

    fig.suptitle(
        f"MNI Registration Visual QC — "
        f"{subject_id} — Echo {VISUAL_ECHO}",
        fontsize=14
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ================================================================
# PATH CHECKS
# ================================================================

if not os.path.isdir(
    REGISTERED_DIR
):

    raise FileNotFoundError(
        f"Registered MRI directory not found:\n"
        f"{REGISTERED_DIR}"
    )


if not os.path.isfile(
    MNI_TEMPLATE
):

    raise FileNotFoundError(
        f"MNI template not found:\n{MNI_TEMPLATE}"
    )


os.makedirs(
    QC_DIR,
    exist_ok=True
)


# ================================================================
# LOAD SUBJECTS
# ================================================================

subjects = get_selected_subjects()

print(
    "\nSelected subjects:",
    len(subjects)
)


# ================================================================
# LOAD MNI TEMPLATE
# ================================================================

print("\n" + "=" * 70)
print("LOADING MNI TEMPLATE")
print("=" * 70)

mni_image, mni_data = read_image(
    MNI_TEMPLATE
)

print(
    "MNI size:",
    mni_image.GetSize()
)

print(
    "MNI spacing:",
    mni_image.GetSpacing()
)

print(
    "MNI origin:",
    mni_image.GetOrigin()
)

print(
    "MNI direction:",
    mni_image.GetDirection()
)


# ================================================================
# PROCESS CURRENT COHORT
# ================================================================

results = []

successful = 0
failed = 0


for index, subject_id in enumerate(
    subjects,
    start=1
):

    print("\n" + "=" * 70)

    print(
        f"VISUAL QC {index}/{len(subjects)} — "
        f"{subject_id}"
    )

    print("=" * 70)

    registered_path = os.path.join(
        REGISTERED_DIR,
        f"{subject_id}_echo{VISUAL_ECHO}_mni.nii.gz"
    )

    output_path = os.path.join(
        QC_DIR,
        f"{subject_id}_registration_qc.png"
    )

    errors = []

    checks = {
        "registered_readable": False,
        "registered_finite": False,
        "registered_nonempty": False,
        "mni_geometry_match": False,
        "qc_image_created": False
    }

    try:

        # --------------------------------------------------------
        # Load registered MRI.
        # --------------------------------------------------------

        registered_image, registered_data = (
            read_image(
                registered_path
            )
        )

        checks[
            "registered_readable"
        ] = True

        checks[
            "registered_finite"
        ] = True

        checks[
            "registered_nonempty"
        ] = True

        print(
            "Registered Echo-1: PASS"
        )

        # --------------------------------------------------------
        # Geometry.
        # --------------------------------------------------------

        if geometry_matches(
            registered_image,
            mni_image
        ):

            checks[
                "mni_geometry_match"
            ] = True

            print(
                "MNI geometry: PASS"
            )

        else:

            errors.append(
                "Registered MRI geometry does not "
                "match MNI template."
            )

            print(
                "MNI geometry: FAIL"
            )

        # --------------------------------------------------------
        # Create visual QC image only when geometry is valid.
        # --------------------------------------------------------

        if checks[
            "mni_geometry_match"
        ]:

            create_registration_qc(
                subject_id,
                mni_data,
                registered_data,
                output_path
            )

            if (
                os.path.isfile(
                    output_path
                )
                and os.path.getsize(
                    output_path
                ) > 0
            ):

                checks[
                    "qc_image_created"
                ] = True

                print(
                    "QC image generation: PASS"
                )

            else:

                errors.append(
                    "QC image was not created."
                )

        else:

            errors.append(
                "Visual QC image not generated "
                "because MNI geometry failed."
            )


    except Exception as e:

        errors.append(
            str(e)
        )

        print(
            "QC processing failed:",
            str(e)
        )


    # ------------------------------------------------------------
    # Status
    # ------------------------------------------------------------

    if len(errors) == 0:

        status = "PASS"
        successful += 1

    else:

        status = "FAIL"
        failed += 1


    results.append(
        {
            "subject_id": subject_id,
            "echo": VISUAL_ECHO,
            "status": status,
            "qc_image": output_path,
            "error_count": len(errors),
            "errors": " | ".join(errors),
            **checks
        }
    )


    print(
        "Automated status:",
        status
    )


# ================================================================
# SAVE SUMMARY
# ================================================================

summary_df = pd.DataFrame(
    results
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("REGISTRATION VISUAL QC SUMMARY")
print("=" * 70)

print(
    "Selected subjects:",
    len(subjects)
)

print(
    "Successful QC image generation:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Summary CSV:",
    SUMMARY_CSV
)

print(
    "QC images:",
    QC_DIR
)


if failed == 0:

    print(
        "\nSTATUS: AUTOMATED QC PASS"
    )

    print(
        "IMPORTANT: Manually inspect the generated "
        "registration QC images."
    )

else:

    print(
        "\nSTATUS: QC FAILED"
    )

print("=" * 70)