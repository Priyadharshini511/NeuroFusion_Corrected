# # src\qc_hdbet.py

# import os
# import numpy as np
# import pandas as pd
# import SimpleITK as sitk
# import matplotlib.pyplot as plt


# # ================================================================
# # HD-BET QUALITY CONTROL
# #
# # Generic, manifest-driven QC for the current sMRI cohort.
# #
# # Checks:
# #   1. Current subjects from selection manifest
# #   2. N4 Echo-1 exists and is valid
# #   3. HD-BET brain image exists and is valid
# #   4. HD-BET mask exists and is valid
# #   5. RAS orientation
# #   6. Geometry consistency
# #   7. Finite/non-empty data
# #   8. Binary mask
# #   9. Non-empty brain mask
# #  10. Brain image is zero outside mask
# #  11. Visual QC image generation
# #
# # This script is READ-ONLY with respect to preprocessing inputs.
# # It does not modify N4, HD-BET brain images, or masks.
# # ================================================================


# print("=" * 70)
# print("HD-BET QUALITY CONTROL")
# print("=" * 70)


# # ================================================================
# # PATHS
# # ================================================================

# N4_DIR = (
#     r"processed_data\smri"
#     r"\n4"
# )

# HDBET_DIR = (
#     r"processed_data\smri"
#     r"\mprage_5e_bet"
# )

# MANIFEST_PATH = (
#     r"processed_data\smri"
#     r"\mprage_selection"
#     r"\mprage_selection_manifest.csv"
# )

# QC_DIR = (
#     r"processed_data\smri"
#     r"\qc_hdbet"
# )

# SUMMARY_CSV = os.path.join(
#     QC_DIR,
#     "hdbet_qc_summary.csv"
# )


# # ================================================================
# # CONFIGURATION
# # ================================================================

# EXPECTED_ECHO = 1

# GEOMETRY_ATOL = 1e-5
# AFFINE_ATOL = 1e-5

# MASK_NONZERO_MIN = 10

# # For visual QC only.
# # The numerical QC does not depend on this threshold.
# VISUAL_PERCENTILES = (1, 99)


# # ================================================================
# # HELPERS
# # ================================================================

# def get_selected_subjects():

#     if not os.path.isfile(MANIFEST_PATH):
#         raise FileNotFoundError(
#             f"Selection manifest not found:\n{MANIFEST_PATH}"
#         )

#     manifest = pd.read_csv(
#         MANIFEST_PATH
#     )

#     required_columns = {
#         "subject_id",
#         "selection_status"
#     }

#     missing = (
#         required_columns
#         - set(manifest.columns)
#     )

#     if missing:
#         raise ValueError(
#             "Manifest is missing required columns: "
#             f"{sorted(missing)}"
#         )

#     selected = manifest[
#         manifest["selection_status"]
#         .astype(str)
#         .str.upper()
#         .eq("SELECTED")
#     ].copy()

#     if selected.empty:
#         raise ValueError(
#             "No SELECTED subjects found in manifest."
#         )

#     subjects = (
#         selected["subject_id"]
#         .astype(str)
#         .str.strip()
#     )

#     if subjects.eq("").any():
#         raise ValueError(
#             "Manifest contains empty subject IDs."
#         )

#     if subjects.duplicated().any():
#         duplicates = sorted(
#             subjects[
#                 subjects.duplicated()
#             ].unique()
#         )

#         raise ValueError(
#             "Duplicate SELECTED subjects found: "
#             f"{duplicates}"
#         )

#     return sorted(
#         subjects.unique()
#     )


# def read_image(path):

#     if not os.path.isfile(path):
#         raise FileNotFoundError(
#             f"File not found: {path}"
#         )

#     image = sitk.ReadImage(path)

#     data = sitk.GetArrayFromImage(
#         image
#     )

#     if data.size == 0:
#         raise ValueError(
#             "Image contains no voxels."
#         )

#     if not np.all(
#         np.isfinite(data)
#     ):
#         raise ValueError(
#             "Image contains NaN or infinite values."
#         )

#     return image, data


# def check_ras(image):

#     direction = image.GetDirection()

#     # SimpleITK direction matrix.
#     # HD-BET output should already be RAS/canonical.
#     #
#     # We additionally use nibabel-style orientation through
#     # an affine constructed from the SimpleITK geometry below.
#     return direction


# def sitk_affine(image):

#     spacing = np.asarray(
#         image.GetSpacing(),
#         dtype=np.float64
#     )

#     origin = np.asarray(
#         image.GetOrigin(),
#         dtype=np.float64
#     )

#     direction = np.asarray(
#         image.GetDirection(),
#         dtype=np.float64
#     ).reshape(3, 3)

#     affine = np.eye(4)

#     affine[:3, :3] = (
#         direction
#         @ np.diag(spacing)
#     )

#     affine[:3, 3] = origin

#     return affine


# def geometry_matches(
#     image_a,
#     image_b
# ):

#     return (
#         image_a.GetSize()
#         == image_b.GetSize()
#         and np.allclose(
#             image_a.GetSpacing(),
#             image_b.GetSpacing(),
#             atol=GEOMETRY_ATOL
#         )
#         and np.allclose(
#             image_a.GetOrigin(),
#             image_b.GetOrigin(),
#             atol=GEOMETRY_ATOL
#         )
#         and np.allclose(
#             image_a.GetDirection(),
#             image_b.GetDirection(),
#             atol=GEOMETRY_ATOL
#         )
#     )


# def is_binary_mask(mask_data):

#     if not np.all(
#         np.isfinite(mask_data)
#     ):
#         return False

#     unique_values = np.unique(
#         mask_data
#     )

#     return np.all(
#         np.isin(
#             unique_values,
#             [0, 1]
#         )
#     )


# def create_qc_image(
#     subject_id,
#     n4_image,
#     brain_image,
#     mask_image,
#     output_path
# ):

#     n4_data = sitk.GetArrayFromImage(
#         n4_image
#     ).astype(np.float32)

#     brain_data = sitk.GetArrayFromImage(
#         brain_image
#     ).astype(np.float32)

#     mask_data = sitk.GetArrayFromImage(
#         mask_image
#     )

#     # ------------------------------------------------------------
#     # Use central slice of the brain-mask bounding box.
#     # ------------------------------------------------------------

#     mask_indices = np.argwhere(
#         mask_data > 0
#     )

#     if mask_indices.size == 0:
#         raise ValueError(
#             "Cannot create QC image: empty mask."
#         )

#     z_min, y_min, x_min = (
#         mask_indices.min(axis=0)
#     )

#     z_max, y_max, x_max = (
#         mask_indices.max(axis=0)
#     )

#     z = int(
#         round(
#             (z_min + z_max) / 2
#         )
#     )

#     y = int(
#         round(
#             (y_min + y_max) / 2
#         )
#     )

#     x = int(
#         round(
#             (x_min + x_max) / 2
#         )
#     )

#     # ------------------------------------------------------------
#     # Robust display range from N4 brain voxels.
#     # ------------------------------------------------------------

#     brain_region = n4_data[
#         mask_data > 0
#     ]

#     if brain_region.size == 0:
#         raise ValueError(
#             "No brain voxels available for visual QC."
#         )

#     vmin, vmax = np.percentile(
#         brain_region,
#         VISUAL_PERCENTILES
#     )

#     if not np.isfinite(vmin):
#         vmin = float(
#             np.min(brain_region)
#         )

#     if not np.isfinite(vmax):
#         vmax = float(
#             np.max(brain_region)
#         )

#     if vmax <= vmin:
#         vmax = vmin + 1e-6

#     # ------------------------------------------------------------
#     # Extract three orthogonal planes.
#     # ------------------------------------------------------------

#     axial_n4 = n4_data[z, :, :]
#     axial_brain = brain_data[z, :, :]
#     axial_mask = mask_data[z, :, :]

#     coronal_n4 = n4_data[:, y, :]
#     coronal_brain = brain_data[:, y, :]
#     coronal_mask = mask_data[:, y, :]

#     sagittal_n4 = n4_data[:, :, x]
#     sagittal_brain = brain_data[:, :, x]
#     sagittal_mask = mask_data[:, :, x]

#     fig, axes = plt.subplots(
#         3,
#         3,
#         figsize=(12, 12)
#     )

#     # ------------------------------------------------------------
#     # Helper for one plane.
#     # ------------------------------------------------------------

#     def draw_plane(
#         ax,
#         background,
#         brain,
#         mask,
#         title
#     ):

#         ax.imshow(
#             np.rot90(background),
#             cmap="gray",
#             vmin=vmin,
#             vmax=vmax
#         )

#         ax.imshow(
#             np.ma.masked_where(
#                 np.rot90(mask) == 0,
#                 np.rot90(brain)
#             ),
#             alpha=0.45
#         )

#         # Contour only when mask has enough structure.
#         rotated_mask = np.rot90(
#             mask.astype(np.float32)
#         )

#         if np.any(
#             rotated_mask > 0
#         ):

#             try:
#                 ax.contour(
#                     rotated_mask,
#                     levels=[0.5],
#                     linewidths=0.8
#                 )
#             except Exception:
#                 pass

#         ax.set_title(
#             title
#         )

#         ax.axis(
#             "off"
#         )

#     # ------------------------------------------------------------
#     # Axial
#     # ------------------------------------------------------------

#     draw_plane(
#         axes[0, 0],
#         axial_n4,
#         axial_n4,
#         axial_mask,
#         "Axial: N4 + Mask"
#     )

#     draw_plane(
#         axes[0, 1],
#         axial_n4,
#         axial_brain,
#         axial_mask,
#         "Axial: Brain + Mask"
#     )

#     axes[0, 2].imshow(
#         np.rot90(axial_n4),
#         cmap="gray",
#         vmin=vmin,
#         vmax=vmax
#     )

#     axes[0, 2].imshow(
#         np.ma.masked_where(
#             np.rot90(axial_mask) == 0,
#             np.rot90(axial_mask)
#         ),
#         alpha=0.35
#     )

#     axes[0, 2].set_title(
#         "Axial: Mask Overlay"
#     )

#     axes[0, 2].axis(
#         "off"
#     )

#     # ------------------------------------------------------------
#     # Coronal
#     # ------------------------------------------------------------

#     draw_plane(
#         axes[1, 0],
#         coronal_n4,
#         coronal_n4,
#         coronal_mask,
#         "Coronal: N4 + Mask"
#     )

#     draw_plane(
#         axes[1, 1],
#         coronal_n4,
#         coronal_brain,
#         coronal_mask,
#         "Coronal: Brain + Mask"
#     )

#     axes[1, 2].imshow(
#         np.rot90(coronal_n4),
#         cmap="gray",
#         vmin=vmin,
#         vmax=vmax
#     )

#     axes[1, 2].imshow(
#         np.ma.masked_where(
#             np.rot90(coronal_mask) == 0,
#             np.rot90(coronal_mask)
#         ),
#         alpha=0.35
#     )

#     axes[1, 2].set_title(
#         "Coronal: Mask Overlay"
#     )

#     axes[1, 2].axis(
#         "off"
#     )

#     # ------------------------------------------------------------
#     # Sagittal
#     # ------------------------------------------------------------

#     draw_plane(
#         axes[2, 0],
#         sagittal_n4,
#         sagittal_n4,
#         sagittal_mask,
#         "Sagittal: N4 + Mask"
#     )

#     draw_plane(
#         axes[2, 1],
#         sagittal_n4,
#         sagittal_brain,
#         sagittal_mask,
#         "Sagittal: Brain + Mask"
#     )

#     axes[2, 2].imshow(
#         np.rot90(sagittal_n4),
#         cmap="gray",
#         vmin=vmin,
#         vmax=vmax
#     )

#     axes[2, 2].imshow(
#         np.ma.masked_where(
#             np.rot90(sagittal_mask) == 0,
#             np.rot90(sagittal_mask)
#         ),
#         alpha=0.35
#     )

#     axes[2, 2].set_title(
#         "Sagittal: Mask Overlay"
#     )

#     axes[2, 2].axis(
#         "off"
#     )

#     fig.suptitle(
#         f"HD-BET QC — {subject_id} — Echo 1",
#         fontsize=14
#     )

#     plt.tight_layout()

#     fig.savefig(
#         output_path,
#         dpi=150,
#         bbox_inches="tight"
#     )

#     plt.close(
#         fig
#     )


# # ================================================================
# # PATH VALIDATION
# # ================================================================

# if not os.path.isdir(N4_DIR):
#     raise FileNotFoundError(
#         f"N4 directory not found:\n{N4_DIR}"
#     )

# if not os.path.isdir(HDBET_DIR):
#     raise FileNotFoundError(
#         f"HD-BET directory not found:\n{HDBET_DIR}"
#     )

# os.makedirs(
#     QC_DIR,
#     exist_ok=True
# )


# # ================================================================
# # LOAD CURRENT SUBJECTS
# # ================================================================

# subjects = get_selected_subjects()

# print("\nSelected subjects:", len(subjects))

# for subject in subjects:
#     print(" ", subject)


# # ================================================================
# # QC
# # ================================================================

# results = []

# passed = 0
# failed = 0


# for index, subject_id in enumerate(
#     subjects,
#     start=1
# ):

#     print("\n" + "=" * 70)
#     print(
#         f"QC {index}/{len(subjects)} — {subject_id}"
#     )
#     print("=" * 70)

#     n4_path = os.path.join(
#         N4_DIR,
#         f"{subject_id}_echo1_n4.nii.gz"
#     )

#     brain_path = os.path.join(
#         HDBET_DIR,
#         f"{subject_id}_echo1_n4_bet.nii.gz"
#     )

#     mask_path = os.path.join(
#         HDBET_DIR,
#         f"{subject_id}_echo1_n4_bet_mask.nii.gz"
#     )

#     qc_path = os.path.join(
#         QC_DIR,
#         f"{subject_id}_hdbet_qc.png"
#     )

#     checks = {
#         "n4_exists": False,
#         "brain_exists": False,
#         "mask_exists": False,
#         "n4_readable": False,
#         "brain_readable": False,
#         "mask_readable": False,
#         "n4_finite": False,
#         "brain_finite": False,
#         "mask_finite": False,
#         "n4_nonempty": False,
#         "brain_nonempty": False,
#         "mask_nonempty": False,
#         "brain_ras": False,
#         "mask_ras": False,
#         "brain_geometry_matches_n4": False,
#         "mask_geometry_matches_n4": False,
#         "brain_mask_affine_match": False,
#         "mask_binary": False,
#         "brain_outside_mask_zero": False,
#         "qc_image_created": False
#     }

#     errors = []

#     n4_image = None
#     brain_image = None
#     mask_image = None

#     n4_data = None
#     brain_data = None
#     mask_data = None

#     # ------------------------------------------------------------
#     # Existence
#     # ------------------------------------------------------------

#     checks["n4_exists"] = os.path.isfile(
#         n4_path
#     )

#     checks["brain_exists"] = os.path.isfile(
#         brain_path
#     )

#     checks["mask_exists"] = os.path.isfile(
#         mask_path
#     )

#     if not checks["n4_exists"]:
#         errors.append(
#             "N4 Echo-1 file missing."
#         )

#     if not checks["brain_exists"]:
#         errors.append(
#             "HD-BET brain image missing."
#         )

#     if not checks["mask_exists"]:
#         errors.append(
#             "HD-BET mask missing."
#         )

#     # ------------------------------------------------------------
#     # Read N4
#     # ------------------------------------------------------------

#     if checks["n4_exists"]:

#         try:

#             n4_image, n4_data = read_image(
#                 n4_path
#             )

#             checks["n4_readable"] = True
#             checks["n4_finite"] = True

#             if np.count_nonzero(
#                 n4_data
#             ) > 0:

#                 checks["n4_nonempty"] = True

#             else:

#                 errors.append(
#                     "N4 Echo-1 is empty."
#                 )

#         except Exception as e:

#             errors.append(
#                 f"N4 validation failed: {e}"
#             )

#     # ------------------------------------------------------------
#     # Read brain image
#     # ------------------------------------------------------------

#     if checks["brain_exists"]:

#         try:

#             brain_image, brain_data = read_image(
#                 brain_path
#             )

#             checks["brain_readable"] = True
#             checks["brain_finite"] = True

#             if np.count_nonzero(
#                 brain_data
#             ) > 0:

#                 checks["brain_nonempty"] = True

#             else:

#                 errors.append(
#                     "HD-BET brain image is empty."
#                 )

#         except Exception as e:

#             errors.append(
#                 f"HD-BET brain validation failed: {e}"
#             )

#     # ------------------------------------------------------------
#     # Read mask
#     # ------------------------------------------------------------

#     if checks["mask_exists"]:

#         try:

#             mask_image, mask_data = read_image(
#                 mask_path
#             )

#             checks["mask_readable"] = True
#             checks["mask_finite"] = True

#             if np.count_nonzero(
#                 mask_data
#             ) >= MASK_NONZERO_MIN:

#                 checks["mask_nonempty"] = True

#             else:

#                 errors.append(
#                     "HD-BET mask is empty or nearly empty."
#                 )

#             if is_binary_mask(
#                 mask_data
#             ):

#                 checks["mask_binary"] = True

#             else:

#                 errors.append(
#                     "HD-BET mask is not binary."
#                 )

#         except Exception as e:

#             errors.append(
#                 f"HD-BET mask validation failed: {e}"
#             )

#     # ------------------------------------------------------------
#     # Geometry and orientation
#     # ------------------------------------------------------------

#     if (
#         n4_image is not None
#         and brain_image is not None
#     ):

#         if geometry_matches(
#             brain_image,
#             n4_image
#         ):

#             checks[
#                 "brain_geometry_matches_n4"
#             ] = True

#         else:

#             errors.append(
#                 "HD-BET brain geometry does not "
#                 "match N4 Echo-1."
#             )

#     if (
#         n4_image is not None
#         and mask_image is not None
#     ):

#         if geometry_matches(
#             mask_image,
#             n4_image
#         ):

#             checks[
#                 "mask_geometry_matches_n4"
#             ] = True

#         else:

#             errors.append(
#                 "HD-BET mask geometry does not "
#                 "match N4 Echo-1."
#             )

#     if (
#         brain_image is not None
#         and mask_image is not None
#     ):

#         if geometry_matches(
#             brain_image,
#             mask_image
#         ):

#             checks[
#                 "brain_mask_affine_match"
#             ] = True

#         else:

#             errors.append(
#                 "Brain image and mask geometry differ."
#             )

#     # ------------------------------------------------------------
#     # RAS orientation
#     #
#     # The preprocessing pipeline uses canonical RAS.
#     # For SimpleITK, direction alone cannot be safely interpreted
#     # as anatomical labels without additional coordinate convention.
#     #
#     # We therefore verify the direction/geometry consistency against
#     # the N4 reference and retain this check as a geometry-consistency
#     # indicator.
#     # ------------------------------------------------------------

#     if n4_image is not None:

#         n4_affine = sitk_affine(
#             n4_image
#         )

#         if np.all(
#             np.isfinite(n4_affine)
#         ):

#             if brain_image is not None:

#                 brain_affine = sitk_affine(
#                     brain_image
#                 )

#                 if np.all(
#                     np.isfinite(brain_affine)
#                 ):

#                     checks["brain_ras"] = True

#             if mask_image is not None:

#                 mask_affine = sitk_affine(
#                     mask_image
#                 )

#                 if np.all(
#                     np.isfinite(mask_affine)
#                 ):

#                     checks["mask_ras"] = True

#     # ------------------------------------------------------------
#     # Brain outside mask must be zero
#     # ------------------------------------------------------------

#     if (
#         brain_data is not None
#         and mask_data is not None
#     ):

#         outside_mask = (
#             mask_data == 0
#         )

#         outside_values = (
#             brain_data[
#                 outside_mask
#             ]
#         )

#         if outside_values.size == 0:

#             checks[
#                 "brain_outside_mask_zero"
#             ] = True

#         else:

#             max_outside = np.max(
#                 np.abs(
#                     outside_values
#                 )
#             )

#             if max_outside <= 1e-6:

#                 checks[
#                     "brain_outside_mask_zero"
#                 ] = True

#             else:

#                 errors.append(
#                     "Brain image contains non-zero "
#                     "values outside HD-BET mask."
#                 )

#     # ------------------------------------------------------------
#     # Visual QC
#     # ------------------------------------------------------------

#     if (
#         n4_image is not None
#         and brain_image is not None
#         and mask_image is not None
#         and checks["brain_geometry_matches_n4"]
#         and checks["mask_geometry_matches_n4"]
#     ):

#         try:

#             create_qc_image(
#                 subject_id,
#                 n4_image,
#                 brain_image,
#                 mask_image,
#                 qc_path
#             )

#             if os.path.isfile(
#                 qc_path
#             ) and os.path.getsize(
#                 qc_path
#             ) > 0:

#                 checks[
#                     "qc_image_created"
#                 ] = True

#             else:

#                 errors.append(
#                     "QC image was not created correctly."
#                 )

#         except Exception as e:

#             errors.append(
#                 f"Visual QC image generation failed: {e}"
#             )

#     # ------------------------------------------------------------
#     # Final status
#     # ------------------------------------------------------------

#     status = (
#         "PASS"
#         if len(errors) == 0
#         else "FAIL"
#     )

#     if status == "PASS":
#         passed += 1
#     else:
#         failed += 1

#     result = {
#         "subject_id": subject_id,
#         "status": status,
#         "n4_path": n4_path,
#         "brain_path": brain_path,
#         "mask_path": mask_path,
#         "qc_image": qc_path,
#         "error_count": len(errors),
#         "errors": " | ".join(errors)
#     }

#     result.update(
#         checks
#     )

#     results.append(
#         result
#     )

#     print(
#         "\nSTATUS:",
#         status
#     )

#     if errors:

#         for error in errors:
#             print(
#                 " -",
#                 error
#             )

#     else:

#         print(
#             "All automated HD-BET checks passed."
#         )


# # ================================================================
# # SAVE SUMMARY
# # ================================================================

# summary_df = pd.DataFrame(
#     results
# )

# summary_df.to_csv(
#     SUMMARY_CSV,
#     index=False
# )


# # ================================================================
# # FINAL SUMMARY
# # ================================================================

# print("\n" + "=" * 70)
# print("HD-BET QC SUMMARY")
# print("=" * 70)

# print(
#     "Selected subjects:",
#     len(subjects)
# )

# print(
#     "Passed:",
#     passed
# )

# print(
#     "Failed:",
#     failed
# )

# print(
#     "Summary CSV:",
#     SUMMARY_CSV
# )

# print(
#     "QC images:",
#     QC_DIR
# )


# # ================================================================
# # IMPORTANT:
# # A PASS here means automated numerical/integrity QC passed.
# #
# # The generated PNG images should still be visually inspected by
# # the researcher before declaring visual QC complete.
# # ================================================================

# if failed == 0:

#     print("\nSTATUS: AUTOMATED QC PASS")

#     print(
#         "IMPORTANT: Review the generated QC PNGs visually."
#     )

# else:

#     print("\nSTATUS: QC FAILED")

#     print(
#         "Review the summary CSV for failed subjects."
#     )


# print("=" * 70)



#src\qc_hdbet.py

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt


# ================================================================
# HD-BET QUALITY CONTROL
#
# Generic, manifest-driven QC for the current sMRI cohort.
#
# Supports:
#   - RMS MPRAGE
#   - 5ECHO MPRAGE
#
# Checks:
#   1. Current subjects from selection manifest
#   2. Acquisition type
#   3. N4 input exists and is valid
#   4. HD-BET brain image exists and is valid
#   5. HD-BET mask exists and is valid
#   6. True RAS orientation
#   7. Geometry consistency
#   8. Finite/non-empty data
#   9. Binary mask
#  10. Non-empty mask
#  11. Mask coverage
#  12. Brain image zero outside mask
#  13. Visual QC image generation
#  14. Current-cohort completeness
#
# READ-ONLY:
#   This script does not modify preprocessing images.
# ================================================================


print("=" * 70)
print("HD-BET QUALITY CONTROL")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

N4_DIR = (
    r"processed_data\smri"
    r"\n4"
)

HDBET_DIR = (
    r"processed_data\smri"
    r"\mprage_5e_bet"
)

MANIFEST_PATH = (
    r"processed_data\smri"
    r"\mprage_selection"
    r"\mprage_selection_manifest.csv"
)

QC_DIR = (
    r"processed_data\smri"
    r"\qc_hdbet"
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "hdbet_qc_summary.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

EXPECTED_ECHO = 1

GEOMETRY_ATOL = 1e-5

MASK_NONZERO_MIN = 10

OUTSIDE_MASK_TOL = 1e-6

VISUAL_PERCENTILES = (1, 99)


# ================================================================
# GET CURRENT SELECTED SUBJECTS
# ================================================================

def get_selected_subjects():

    if not os.path.isfile(MANIFEST_PATH):

        raise FileNotFoundError(
            f"Selection manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Existing project manifest uses "subject", not "subject_id".
    # ------------------------------------------------------------

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
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------------
    # Only SELECTED acquisitions
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Normalize fields
    # ------------------------------------------------------------

    selected["subject"] = (
        selected["subject"]
        .astype(str)
        .str.strip()
    )

    selected["acquisition_type"] = (
        selected["acquisition_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ------------------------------------------------------------
    # Validate subject IDs
    # ------------------------------------------------------------

    if selected["subject"].eq("").any():

        raise ValueError(
            "Manifest contains empty subject IDs."
        )

    if selected["subject"].duplicated().any():

        duplicates = sorted(
            selected.loc[
                selected["subject"].duplicated(),
                "subject"
            ].unique()
        )

        raise ValueError(
            "Duplicate SELECTED subjects found: "
            f"{duplicates}"
        )

    # ------------------------------------------------------------
    # Validate acquisition types
    # ------------------------------------------------------------

    invalid_types = sorted(
        set(
            selected["acquisition_type"]
        )
        - {
            "RMS",
            "5ECHO"
        }
    )

    if invalid_types:

        raise ValueError(
            "Unsupported acquisition_type values: "
            f"{invalid_types}"
        )

    # ------------------------------------------------------------
    # Internally rename "subject" → "subject_id"
    # ------------------------------------------------------------

    selected = selected[
        [
            "subject",
            "acquisition_type"
        ]
    ].rename(
        columns={
            "subject": "subject_id"
        }
    )

    return selected.sort_values(
        "subject_id"
    ).reset_index(
        drop=True
    )


# ================================================================
# READ IMAGE
# ================================================================

def read_image(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    image = sitk.ReadImage(
        path
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

    if np.count_nonzero(data) == 0:

        raise ValueError(
            "Image contains no non-zero voxels."
        )

    return image, data


# ================================================================
# TRUE RAS ORIENTATION
# ================================================================

def get_orientation(path):

    nii = nib.load(
        path
    )

    return tuple(
        nib.aff2axcodes(
            nii.affine
        )
    )


def is_ras(path):

    orientation = get_orientation(
        path
    )

    return orientation == (
        "R",
        "A",
        "S"
    )


# ================================================================
# GEOMETRY MATCH
# ================================================================

def geometry_matches(
    image_a,
    image_b
):

    return (
        image_a.GetSize()
        ==
        image_b.GetSize()

        and

        np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=GEOMETRY_ATOL
        )

        and

        np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=GEOMETRY_ATOL
        )

        and

        np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=GEOMETRY_ATOL
        )
    )


# ================================================================
# BINARY MASK
# ================================================================

def is_binary_mask(mask_data):

    if not np.all(
        np.isfinite(mask_data)
    ):

        return False

    unique_values = np.unique(
        mask_data
    )

    return np.all(
        np.isin(
            unique_values,
            [0, 1]
        )
    )


# ================================================================
# MASK COVERAGE
# ================================================================

def calculate_mask_coverage(mask_data):

    total_voxels = mask_data.size

    mask_voxels = np.count_nonzero(
        mask_data
    )

    if total_voxels == 0:

        return 0.0

    return (
        100.0
        *
        mask_voxels
        /
        total_voxels
    )


# ================================================================
# MASK BOUNDING BOX
# ================================================================

def calculate_mask_bbox(mask_data):

    indices = np.argwhere(
        mask_data > 0
    )

    if indices.size == 0:

        return None

    z_min, y_min, x_min = (
        indices.min(
            axis=0
        )
    )

    z_max, y_max, x_max = (
        indices.max(
            axis=0
        )
    )

    return {
        "z_min": int(z_min),
        "z_max": int(z_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "x_min": int(x_min),
        "x_max": int(x_max)
    }


# ================================================================
# VISUAL QC IMAGE
# ================================================================

def create_qc_image(
    subject_id,
    acquisition_type,
    n4_image,
    brain_image,
    mask_image,
    output_path
):

    n4_data = (
        sitk.GetArrayFromImage(
            n4_image
        )
        .astype(
            np.float32
        )
    )

    brain_data = (
        sitk.GetArrayFromImage(
            brain_image
        )
        .astype(
            np.float32
        )
    )

    mask_data = (
        sitk.GetArrayFromImage(
            mask_image
        )
    )

    # ------------------------------------------------------------
    # Mask bounding box
    # ------------------------------------------------------------

    mask_indices = np.argwhere(
        mask_data > 0
    )

    if mask_indices.size == 0:

        raise ValueError(
            "Cannot create QC image: empty mask."
        )

    z_min, y_min, x_min = (
        mask_indices.min(
            axis=0
        )
    )

    z_max, y_max, x_max = (
        mask_indices.max(
            axis=0
        )
    )

    # Central point of mask bounding box

    z = int(
        round(
            (
                z_min
                +
                z_max
            )
            /
            2
        )
    )

    y = int(
        round(
            (
                y_min
                +
                y_max
            )
            /
            2
        )
    )

    x = int(
        round(
            (
                x_min
                +
                x_max
            )
            /
            2
        )
    )

    # ------------------------------------------------------------
    # Robust display range
    # ------------------------------------------------------------

    brain_region = n4_data[
        mask_data > 0
    ]

    if brain_region.size == 0:

        raise ValueError(
            "No brain voxels available for visual QC."
        )

    vmin, vmax = np.percentile(
        brain_region,
        VISUAL_PERCENTILES
    )

    if not np.isfinite(vmin):

        vmin = float(
            np.min(
                brain_region
            )
        )

    if not np.isfinite(vmax):

        vmax = float(
            np.max(
                brain_region
            )
        )

    if vmax <= vmin:

        vmax = (
            vmin
            +
            1e-6
        )

    # ------------------------------------------------------------
    # Orthogonal slices
    # ------------------------------------------------------------

    axial_n4 = n4_data[
        z,
        :,
        :
    ]

    axial_brain = brain_data[
        z,
        :,
        :
    ]

    axial_mask = mask_data[
        z,
        :,
        :
    ]

    coronal_n4 = n4_data[
        :,
        y,
        :
    ]

    coronal_brain = brain_data[
        :,
        y,
        :
    ]

    coronal_mask = mask_data[
        :,
        y,
        :
    ]

    sagittal_n4 = n4_data[
        :,
        :,
        x
    ]

    sagittal_brain = brain_data[
        :,
        :,
        x
    ]

    sagittal_mask = mask_data[
        :,
        :,
        x
    ]

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            12,
            12
        )
    )

    # ------------------------------------------------------------
    # Plane helper
    # ------------------------------------------------------------

    def draw_plane(
        ax,
        background,
        brain,
        mask,
        title
    ):

        background_rot = np.rot90(
            background
        )

        brain_rot = np.rot90(
            brain
        )

        mask_rot = np.rot90(
            mask
        )

        ax.imshow(
            background_rot,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        ax.imshow(
            np.ma.masked_where(
                mask_rot == 0,
                brain_rot
            ),
            alpha=0.45
        )

        if np.any(
            mask_rot > 0
        ):

            try:

                ax.contour(
                    mask_rot.astype(
                        np.float32
                    ),
                    levels=[
                        0.5
                    ],
                    linewidths=0.8
                )

            except Exception:

                pass

        ax.set_title(
            title
        )

        ax.axis(
            "off"
        )

    # ------------------------------------------------------------
    # AXIAL
    # ------------------------------------------------------------

    draw_plane(
        axes[0, 0],
        axial_n4,
        axial_n4,
        axial_mask,
        "Axial: N4 + Mask"
    )

    draw_plane(
        axes[0, 1],
        axial_n4,
        axial_brain,
        axial_mask,
        "Axial: Brain + Mask"
    )

    axes[0, 2].imshow(
        np.rot90(
            axial_n4
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    axes[0, 2].imshow(
        np.ma.masked_where(
            np.rot90(
                axial_mask
            ) == 0,
            np.rot90(
                axial_mask
            )
        ),
        alpha=0.35
    )

    axes[0, 2].set_title(
        "Axial: Mask Overlay"
    )

    axes[0, 2].axis(
        "off"
    )

    # ------------------------------------------------------------
    # CORONAL
    # ------------------------------------------------------------

    draw_plane(
        axes[1, 0],
        coronal_n4,
        coronal_n4,
        coronal_mask,
        "Coronal: N4 + Mask"
    )

    draw_plane(
        axes[1, 1],
        coronal_n4,
        coronal_brain,
        coronal_mask,
        "Coronal: Brain + Mask"
    )

    axes[1, 2].imshow(
        np.rot90(
            coronal_n4
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    axes[1, 2].imshow(
        np.ma.masked_where(
            np.rot90(
                coronal_mask
            ) == 0,
            np.rot90(
                coronal_mask
            )
        ),
        alpha=0.35
    )

    axes[1, 2].set_title(
        "Coronal: Mask Overlay"
    )

    axes[1, 2].axis(
        "off"
    )

    # ------------------------------------------------------------
    # SAGITTAL
    # ------------------------------------------------------------

    draw_plane(
        axes[2, 0],
        sagittal_n4,
        sagittal_n4,
        sagittal_mask,
        "Sagittal: N4 + Mask"
    )

    draw_plane(
        axes[2, 1],
        sagittal_n4,
        sagittal_brain,
        sagittal_mask,
        "Sagittal: Brain + Mask"
    )

    axes[2, 2].imshow(
        np.rot90(
            sagittal_n4
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    axes[2, 2].imshow(
        np.ma.masked_where(
            np.rot90(
                sagittal_mask
            ) == 0,
            np.rot90(
                sagittal_mask
            )
        ),
        alpha=0.35
    )

    axes[2, 2].set_title(
        "Sagittal: Mask Overlay"
    )

    axes[2, 2].axis(
        "off"
    )

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------

    fig.suptitle(
        (
            f"HD-BET QC — "
            f"{subject_id} — "
            f"{acquisition_type}"
        ),
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
# PATH VALIDATION
# ================================================================

if not os.path.isdir(
    N4_DIR
):

    raise FileNotFoundError(
        f"N4 directory not found:\n"
        f"{N4_DIR}"
    )

if not os.path.isdir(
    HDBET_DIR
):

    raise FileNotFoundError(
        f"HD-BET directory not found:\n"
        f"{HDBET_DIR}"
    )

os.makedirs(
    QC_DIR,
    exist_ok=True
)


# ================================================================
# LOAD CURRENT COHORT
# ================================================================

selected_df = (
    get_selected_subjects()
)

subjects = (
    selected_df[
        "subject_id"
    ]
    .tolist()
)

print(
    "\nSelected subjects:",
    len(subjects)
)

for _, row in selected_df.iterrows():

    print(
        f"  {row['subject_id']} "
        f"-> {row['acquisition_type']}"
    )


# ================================================================
# PROCESS EACH SUBJECT
# ================================================================

results = []

passed = 0
failed = 0


for index, row in selected_df.iterrows():

    subject_id = row[
        "subject_id"
    ]

    acquisition_type = row[
        "acquisition_type"
    ]

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        f"QC {index + 1}/"
        f"{len(selected_df)} "
        f"— {subject_id} "
        f"— {acquisition_type}"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------
    # Determine filenames
    # ------------------------------------------------------------

    if acquisition_type == "5ECHO":

        n4_name = (
            f"{subject_id}_echo"
            f"{EXPECTED_ECHO}_n4.nii.gz"
        )

        brain_name = (
            f"{subject_id}_echo"
            f"{EXPECTED_ECHO}_n4_bet.nii.gz"
        )

        mask_name = (
            f"{subject_id}_echo"
            f"{EXPECTED_ECHO}_n4_bet_bet.nii.gz"
        )

    elif acquisition_type == "RMS":

        n4_name = (
            f"{subject_id}_rms_n4.nii.gz"
        )

        brain_name = (
            f"{subject_id}_rms_n4_bet.nii.gz"
        )

        mask_name = (
            f"{subject_id}_rms_n4_bet_bet.nii.gz"
        )

    else:

        raise ValueError(
            f"Unsupported acquisition type: "
            f"{acquisition_type}"
        )

    n4_path = os.path.join(
        N4_DIR,
        n4_name
    )

    brain_path = os.path.join(
        HDBET_DIR,
        brain_name
    )

    mask_path = os.path.join(
        HDBET_DIR,
        mask_name
    )

    qc_path = os.path.join(
        QC_DIR,
        f"{subject_id}_hdbet_qc.png"
    )

    # ------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------

    checks = {

        "n4_exists": False,
        "brain_exists": False,
        "mask_exists": False,

        "n4_readable": False,
        "brain_readable": False,
        "mask_readable": False,

        "n4_finite": False,
        "brain_finite": False,
        "mask_finite": False,

        "n4_nonempty": False,
        "brain_nonempty": False,
        "mask_nonempty": False,

        "n4_ras": False,
        "brain_ras": False,
        "mask_ras": False,

        "brain_geometry_matches_n4": False,
        "mask_geometry_matches_n4": False,
        "brain_mask_geometry_match": False,

        "mask_binary": False,
        "brain_outside_mask_zero": False,

        "qc_image_created": False
    }

    errors = []

    n4_image = None
    brain_image = None
    mask_image = None

    n4_data = None
    brain_data = None
    mask_data = None

    # ------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------

    checks["n4_exists"] = (
        os.path.isfile(
            n4_path
        )
    )

    checks["brain_exists"] = (
        os.path.isfile(
            brain_path
        )
    )

    checks["mask_exists"] = (
        os.path.isfile(
            mask_path
        )
    )

    if not checks[
        "n4_exists"
    ]:

        errors.append(
            f"N4 file missing: "
            f"{n4_name}"
        )

    if not checks[
        "brain_exists"
    ]:

        errors.append(
            f"HD-BET brain image missing: "
            f"{brain_name}"
        )

    if not checks[
        "mask_exists"
    ]:

        errors.append(
            f"HD-BET mask missing: "
            f"{mask_name}"
        )

    # ------------------------------------------------------------
    # N4 validation
    # ------------------------------------------------------------

    if checks[
        "n4_exists"
    ]:

        try:

            n4_image, n4_data = (
                read_image(
                    n4_path
                )
            )

            checks[
                "n4_readable"
            ] = True

            checks[
                "n4_finite"
            ] = True

            checks[
                "n4_nonempty"
            ] = True

        except Exception as e:

            errors.append(
                f"N4 validation failed: "
                f"{e}"
            )

    # ------------------------------------------------------------
    # Brain image validation
    # ------------------------------------------------------------

    if checks[
        "brain_exists"
    ]:

        try:

            brain_image, brain_data = (
                read_image(
                    brain_path
                )
            )

            checks[
                "brain_readable"
            ] = True

            checks[
                "brain_finite"
            ] = True

            checks[
                "brain_nonempty"
            ] = True

        except Exception as e:

            errors.append(
                f"HD-BET brain validation failed: "
                f"{e}"
            )

    # ------------------------------------------------------------
    # Mask validation
    # ------------------------------------------------------------

    if checks[
        "mask_exists"
    ]:

        try:

            mask_image, mask_data = (
                read_image(
                    mask_path
                )
            )

            checks[
                "mask_readable"
            ] = True

            checks[
                "mask_finite"
            ] = True

            mask_voxels = (
                np.count_nonzero(
                    mask_data
                )
            )

            if (
                mask_voxels
                >= MASK_NONZERO_MIN
            ):

                checks[
                    "mask_nonempty"
                ] = True

            else:

                errors.append(
                    "HD-BET mask is empty "
                    "or nearly empty."
                )

            if is_binary_mask(
                mask_data
            ):

                checks[
                    "mask_binary"
                ] = True

            else:

                errors.append(
                    "HD-BET mask is not binary."
                )

        except Exception as e:

            errors.append(
                f"HD-BET mask validation failed: "
                f"{e}"
            )

    # ------------------------------------------------------------
    # RAS validation
    # ------------------------------------------------------------

    if checks[
        "n4_exists"
    ]:

        try:

            n4_orientation = (
                get_orientation(
                    n4_path
                )
            )

            if n4_orientation == (
                "R",
                "A",
                "S"
            ):

                checks[
                    "n4_ras"
                ] = True

            else:

                errors.append(
                    f"N4 is not RAS. "
                    f"Orientation: "
                    f"{n4_orientation}"
                )

        except Exception as e:

            errors.append(
                f"N4 orientation check failed: "
                f"{e}"
            )

    if checks[
        "brain_exists"
    ]:

        try:

            brain_orientation = (
                get_orientation(
                    brain_path
                )
            )

            if brain_orientation == (
                "R",
                "A",
                "S"
            ):

                checks[
                    "brain_ras"
                ] = True

            else:

                errors.append(
                    f"HD-BET brain is not RAS. "
                    f"Orientation: "
                    f"{brain_orientation}"
                )

        except Exception as e:

            errors.append(
                f"Brain orientation check failed: "
                f"{e}"
            )

    if checks[
        "mask_exists"
    ]:

        try:

            mask_orientation = (
                get_orientation(
                    mask_path
                )
            )

            if mask_orientation == (
                "R",
                "A",
                "S"
            ):

                checks[
                    "mask_ras"
                ] = True

            else:

                errors.append(
                    f"HD-BET mask is not RAS. "
                    f"Orientation: "
                    f"{mask_orientation}"
                )

        except Exception as e:

            errors.append(
                f"Mask orientation check failed: "
                f"{e}"
            )

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------

    if (
        n4_image is not None
        and brain_image is not None
    ):

        if geometry_matches(
            brain_image,
            n4_image
        ):

            checks[
                "brain_geometry_matches_n4"
            ] = True

        else:

            errors.append(
                "HD-BET brain geometry "
                "does not match N4."
            )

    if (
        n4_image is not None
        and mask_image is not None
    ):

        if geometry_matches(
            mask_image,
            n4_image
        ):

            checks[
                "mask_geometry_matches_n4"
            ] = True

        else:

            errors.append(
                "HD-BET mask geometry "
                "does not match N4."
            )

    if (
        brain_image is not None
        and mask_image is not None
    ):

        if geometry_matches(
            brain_image,
            mask_image
        ):

            checks[
                "brain_mask_geometry_match"
            ] = True

        else:

            errors.append(
                "Brain image and mask "
                "geometry differ."
            )

    # ------------------------------------------------------------
    # Mask coverage
    # ------------------------------------------------------------

    mask_voxel_count = 0

    total_voxel_count = 0

    mask_coverage_percent = np.nan

    if mask_data is not None:

        total_voxel_count = (
            mask_data.size
        )

        mask_voxel_count = (
            np.count_nonzero(
                mask_data
            )
        )

        mask_coverage_percent = (
            calculate_mask_coverage(
                mask_data
            )
        )

    # ------------------------------------------------------------
    # Brain outside mask
    # ------------------------------------------------------------

    max_outside_mask = np.nan

    if (
        brain_data is not None
        and mask_data is not None
        and checks[
            "brain_mask_geometry_match"
        ]
    ):

        outside_mask = (
            mask_data == 0
        )

        outside_values = (
            brain_data[
                outside_mask
            ]
        )

        if outside_values.size > 0:

            max_outside_mask = float(
                np.max(
                    np.abs(
                        outside_values
                    )
                )
            )

            if (
                max_outside_mask
                <= OUTSIDE_MASK_TOL
            ):

                checks[
                    "brain_outside_mask_zero"
                ] = True

            else:

                errors.append(
                    "Brain image contains "
                    "non-zero values outside "
                    "HD-BET mask. "
                    f"Maximum outside value: "
                    f"{max_outside_mask:.6g}"
                )

        else:

            checks[
                "brain_outside_mask_zero"
            ] = True

    # ------------------------------------------------------------
    # Intensity statistics
    # ------------------------------------------------------------

    n4_min = np.nan
    n4_max = np.nan
    n4_mean = np.nan

    brain_min = np.nan
    brain_max = np.nan
    brain_mean = np.nan

    if n4_data is not None:

        n4_min = float(
            np.min(
                n4_data
            )
        )

        n4_max = float(
            np.max(
                n4_data
            )
        )

        n4_mean = float(
            np.mean(
                n4_data
            )
        )

    if brain_data is not None:

        brain_min = float(
            np.min(
                brain_data
            )
        )

        brain_max = float(
            np.max(
                brain_data
            )
        )

        brain_mean = float(
            np.mean(
                brain_data
            )
        )

    # ------------------------------------------------------------
    # Mask bounding box
    # ------------------------------------------------------------

    bbox = None

    if mask_data is not None:

        bbox = calculate_mask_bbox(
            mask_data
        )

    if bbox is None:

        bbox = {
            "z_min": np.nan,
            "z_max": np.nan,
            "y_min": np.nan,
            "y_max": np.nan,
            "x_min": np.nan,
            "x_max": np.nan
        }

    # ------------------------------------------------------------
    # Visual QC
    # ------------------------------------------------------------

    if (
        n4_image is not None
        and brain_image is not None
        and mask_image is not None
        and checks[
            "brain_geometry_matches_n4"
        ]
        and checks[
            "mask_geometry_matches_n4"
        ]
    ):

        try:

            create_qc_image(
                subject_id,
                acquisition_type,
                n4_image,
                brain_image,
                mask_image,
                qc_path
            )

            if (
                os.path.isfile(
                    qc_path
                )
                and
                os.path.getsize(
                    qc_path
                ) > 0
            ):

                checks[
                    "qc_image_created"
                ] = True

            else:

                errors.append(
                    "QC image was not created correctly."
                )

        except Exception as e:

            errors.append(
                f"Visual QC image generation failed: "
                f"{e}"
            )

    # ------------------------------------------------------------
    # Status
    # ------------------------------------------------------------

    status = (
        "PASS"
        if len(errors) == 0
        else
        "FAIL"
    )

    if status == "PASS":

        passed += 1

    else:

        failed += 1

    # ------------------------------------------------------------
    # Result row
    # ------------------------------------------------------------

    result = {

        "subject_id":
            subject_id,

        "acquisition_type":
            acquisition_type,

        "status":
            status,

        "n4_path":
            n4_path,

        "brain_path":
            brain_path,

        "mask_path":
            mask_path,

        "qc_image":
            qc_path,

        "mask_voxel_count":
            mask_voxel_count,

        "total_voxel_count":
            total_voxel_count,

        "mask_coverage_percent":
            mask_coverage_percent,

        "max_outside_mask_value":
            max_outside_mask,

        "n4_min":
            n4_min,

        "n4_max":
            n4_max,

        "n4_mean":
            n4_mean,

        "brain_min":
            brain_min,

        "brain_max":
            brain_max,

        "brain_mean":
            brain_mean,

        "bbox_z_min":
            bbox["z_min"],

        "bbox_z_max":
            bbox["z_max"],

        "bbox_y_min":
            bbox["y_min"],

        "bbox_y_max":
            bbox["y_max"],

        "bbox_x_min":
            bbox["x_min"],

        "bbox_x_max":
            bbox["x_max"],

        "error_count":
            len(errors),

        "errors":
            " | ".join(
                errors
            )
    }

    result.update(
        checks
    )

    results.append(
        result
    )

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print(
        "\nAcquisition:",
        acquisition_type
    )

    print(
        "N4 RAS:",
        checks[
            "n4_ras"
        ]
    )

    print(
        "Brain RAS:",
        checks[
            "brain_ras"
        ]
    )

    print(
        "Mask RAS:",
        checks[
            "mask_ras"
        ]
    )

    print(
        "Mask binary:",
        checks[
            "mask_binary"
        ]
    )

    print(
        "Mask voxels:",
        mask_voxel_count
    )

    print(
        "Mask coverage:",
        (
            f"{mask_coverage_percent:.4f}%"
            if np.isfinite(
                mask_coverage_percent
            )
            else
            "N/A"
        )
    )

    print(
        "Max outside-mask value:",
        (
            f"{max_outside_mask:.6g}"
            if np.isfinite(
                max_outside_mask
            )
            else
            "N/A"
        )
    )

    print(
        "QC image:",
        checks[
            "qc_image_created"
        ]
    )

    print(
        "\nSTATUS:",
        status
    )

    if errors:

        for error in errors:

            print(
                " -",
                error
            )

    else:

        print(
            "All automated HD-BET checks passed."
        )


# ================================================================
# CURRENT-COHORT FILE COMPLETENESS
# ================================================================

expected_brain_files = set()

expected_mask_files = set()


for _, row in selected_df.iterrows():

    subject_id = row[
        "subject_id"
    ]

    acquisition_type = row[
        "acquisition_type"
    ]

    if acquisition_type == "5ECHO":

        expected_brain_files.add(
            (
                f"{subject_id}_echo"
                f"{EXPECTED_ECHO}_n4_bet.nii.gz"
            )
        )

        expected_mask_files.add(
            (
                f"{subject_id}_echo"
                f"{EXPECTED_ECHO}_n4_bet_bet.nii.gz"
            )
        )

    elif acquisition_type == "RMS":

        expected_brain_files.add(
            (
                f"{subject_id}_rms_n4_bet.nii.gz"
            )
        )

        expected_mask_files.add(
            (
                f"{subject_id}_rms_n4_bet_bet.nii.gz"
            )
        )


# ------------------------------------------------------------
# Actual brain images
# ------------------------------------------------------------

actual_brain_files = {
    f
    for f in os.listdir(
        HDBET_DIR
    )
    if (
        f.endswith(
            ".nii.gz"
        )
        and
        "_bet.nii.gz" in f
        and
        not f.endswith(
            "_bet_bet.nii.gz"
        )
    )
}


# ------------------------------------------------------------
# Actual masks
# ------------------------------------------------------------

actual_mask_files = {
    f
    for f in os.listdir(
        HDBET_DIR
    )
    if f.endswith(
        "_bet_bet.nii.gz"
    )
}


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

missing_brain_files = (
    expected_brain_files
    -
    actual_brain_files
)

unexpected_brain_files = (
    actual_brain_files
    -
    expected_brain_files
)

missing_mask_files = (
    expected_mask_files
    -
    actual_mask_files
)

unexpected_mask_files = (
    actual_mask_files
    -
    expected_mask_files
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
# FINAL SUMMARY
# ================================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "HD-BET QC SUMMARY"
)

print(
    "=" * 70
)

print(
    "Selected subjects:",
    len(subjects)
)

print(
    "Passed:",
    passed
)

print(
    "Failed:",
    failed
)

print(
    "Expected brain images:",
    len(
        expected_brain_files
    )
)

print(
    "Current brain images found:",
    len(
        actual_brain_files
        &
        expected_brain_files
    )
)

print(
    "Missing brain images:",
    len(
        missing_brain_files
    )
)

print(
    "Unexpected/legacy brain images:",
    len(
        unexpected_brain_files
    )
)

print(
    "Expected masks:",
    len(
        expected_mask_files
    )
)

print(
    "Current masks found:",
    len(
        actual_mask_files
        &
        expected_mask_files
    )
)

print(
    "Missing masks:",
    len(
        missing_mask_files
    )
)

print(
    "Unexpected/legacy masks:",
    len(
        unexpected_mask_files
    )
)

print(
    "Summary CSV:",
    SUMMARY_CSV
)

print(
    "QC images:",
    QC_DIR
)


# ================================================================
# REPORT MISSING FILES
# ================================================================

if missing_brain_files:

    print(
        "\nMissing brain files:"
    )

    for f in sorted(
        missing_brain_files
    ):

        print(
            " -",
            f
        )


if missing_mask_files:

    print(
        "\nMissing mask files:"
    )

    for f in sorted(
        missing_mask_files
    ):

        print(
            " -",
            f
        )


# ================================================================
# REPORT LEGACY / UNEXPECTED FILES
# ================================================================

if unexpected_brain_files:

    print(
        "\nUnexpected/legacy brain files "
        "(not used for current QC):"
    )

    for f in sorted(
        unexpected_brain_files
    ):

        print(
            " -",
            f
        )


if unexpected_mask_files:

    print(
        "\nUnexpected/legacy mask files "
        "(not used for current QC):"
    )

    for f in sorted(
        unexpected_mask_files
    ):

        print(
            " -",
            f
        )


# ================================================================
# FINAL STATUS
# ================================================================

completeness_pass = (
    len(missing_brain_files) == 0
    and
    len(missing_mask_files) == 0
)


if (
    failed == 0
    and
    completeness_pass
):

    print(
        "\nSTATUS: AUTOMATED HD-BET QC PASS"
    )

    print(
        "All current selected subjects passed."
    )

    print(
        "IMPORTANT: Review the generated "
        "QC PNGs visually before proceeding."
    )

else:

    print(
        "\nSTATUS: HD-BET QC FAILED"
    )

    print(
        "Review the summary CSV and errors "
        "before proceeding."
    )


print(
    "=" * 70
)