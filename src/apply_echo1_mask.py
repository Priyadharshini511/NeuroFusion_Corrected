# # src/apply_echo1_mask.py
# from pathlib import Path
# import csv

# import nibabel as nib
# import numpy as np


# # ============================================================
# # COBRE MPRAGE 5-ECHO
# # APPLY ECHO-1 HD-BET MASK TO ALL FIVE ECHOES
# #
# # INPUT:
# #   N4-corrected Echoes 1-5
# #   Echo-1 HD-BET mask
# #
# # OUTPUT:
# #   Masked Echoes 1-5
# #
# # SUBJECT SELECTION:
# #   Uses mprage_selection_manifest.csv
# #
# # IMPORTANT:
# #   The Echo-1 HD-BET mask is applied unchanged to
# #   Echoes 1-5 for each selected subject.
# #
# #   No spatial registration or resampling is performed here.
# # ============================================================


# # ============================================================
# # PATHS
# # ============================================================

# INPUT_FOLDER = Path("processed_data/smri/n4")

# MASK_FOLDER = Path("processed_data/smri/mprage_5e_bet")

# OUTPUT_FOLDER = Path(
#     "processed_data/smri/mprage_5e_masked"
# )

# MANIFEST_FILE = Path(
#     "processed_data/smri/mprage_selection/"
#     "mprage_selection_manifest.csv"
# )

# OUTPUT_FOLDER.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # ============================================================
# # SETTINGS
# # ============================================================

# ECHOES = [1, 2, 3, 4, 5]

# VOXEL_TOLERANCE = 0.01

# AFFINE_TOLERANCE = 1e-3


# # ============================================================
# # HEADER
# # ============================================================

# print("=" * 70)
# print("COBRE MPRAGE 5-ECHO MASK APPLICATION")
# print("=" * 70)

# print("\nN4 input folder:")
# print(f"  {INPUT_FOLDER.resolve()}")

# print("\nHD-BET mask folder:")
# print(f"  {MASK_FOLDER.resolve()}")

# print("\nOutput folder:")
# print(f"  {OUTPUT_FOLDER.resolve()}")

# print("\nSelection manifest:")
# print(f"  {MANIFEST_FILE.resolve()}")


# # ============================================================
# # REQUIRED PATH CHECKS
# # ============================================================

# if not INPUT_FOLDER.exists():
#     raise FileNotFoundError(
#         f"N4 input folder does not exist:\n"
#         f"{INPUT_FOLDER.resolve()}"
#     )

# if not MASK_FOLDER.exists():
#     raise FileNotFoundError(
#         f"HD-BET mask folder does not exist:\n"
#         f"{MASK_FOLDER.resolve()}"
#     )

# if not MANIFEST_FILE.exists():
#     raise FileNotFoundError(
#         f"Selection manifest does not exist:\n"
#         f"{MANIFEST_FILE.resolve()}"
#     )


# # ============================================================
# # LOAD SELECTED SUBJECTS
# # ============================================================

# def load_selected_subjects(manifest_file):
#     """
#     Load subjects whose selection_status is SELECTED.

#     The manifest is the single source of truth for the
#     current preprocessing cohort.
#     """

#     subjects = []

#     with manifest_file.open(
#         "r",
#         newline="",
#         encoding="utf-8-sig"
#     ) as f:

#         reader = csv.DictReader(f)

#         if reader.fieldnames is None:
#             raise ValueError(
#                 "Selection manifest has no header."
#             )

#         required_columns = {
#             "subject",
#             "selection_status"
#         }

#         missing = (
#             required_columns -
#             set(reader.fieldnames)
#         )

#         if missing:
#             raise ValueError(
#                 "Selection manifest is missing required "
#                 "columns: "
#                 + ", ".join(sorted(missing))
#             )

#         for row in reader:

#             status = str(
#                 row.get("selection_status", "")
#             ).strip().upper()

#             if status != "SELECTED":
#                 continue

#             subject = str(
#                 row.get("subject", "")
#             ).strip()

#             if not subject:
#                 raise ValueError(
#                     "A SELECTED manifest row has no "
#                     "subject ID."
#                 )

#             subjects.append(subject)

#     if not subjects:
#         raise ValueError(
#             "No SELECTED subjects found in the "
#             "selection manifest."
#         )

#     if len(subjects) != len(set(subjects)):

#         duplicates = sorted(
#             {
#                 subject
#                 for subject in subjects
#                 if subjects.count(subject) > 1
#             }
#         )

#         raise ValueError(
#             "Duplicate selected subjects found:\n"
#             + "\n".join(duplicates)
#         )

#     return sorted(subjects)


# subjects = load_selected_subjects(
#     MANIFEST_FILE
# )


# # ============================================================
# # DISPLAY CURRENT COHORT
# # ============================================================

# print("\n" + "-" * 70)
# print("SELECTED SUBJECTS FROM MANIFEST")
# print("-" * 70)

# for subject_id in subjects:
#     print(f"  {subject_id}")

# print(f"\nTotal selected subjects: {len(subjects)}")


# # ============================================================
# # GENERAL NIFTI VALIDATION
# # ============================================================

# def validate_nifti(
#     file_path,
#     reference_img=None
# ):
#     """
#     Validate a NIfTI image.

#     If reference_img is supplied, shape, voxel size,
#     affine and orientation are compared against it.
#     """

#     try:

#         img = nib.load(str(file_path))

#         data = img.get_fdata(
#             dtype=np.float32
#         )

#         if data.size == 0:
#             return False, "empty image", None, None

#         if not np.all(np.isfinite(data)):
#             return (
#                 False,
#                 "non-finite voxel values",
#                 None,
#                 None
#             )

#         orientation = nib.aff2axcodes(
#             img.affine
#         )

#         if orientation != ("R", "A", "S"):
#             return (
#                 False,
#                 f"orientation is {orientation}, "
#                 f"expected RAS",
#                 None,
#                 None
#             )

#         voxel_size = np.asarray(
#             img.header.get_zooms()[:3],
#             dtype=float
#         )

#         if not np.allclose(
#             voxel_size,
#             [1.0, 1.0, 1.0],
#             atol=VOXEL_TOLERANCE
#         ):
#             return (
#                 False,
#                 f"voxel size is {voxel_size}, "
#                 f"expected 1mm isotropic",
#                 None,
#                 None
#             )

#         if np.count_nonzero(data) == 0:
#             return (
#                 False,
#                 "all voxels are zero",
#                 None,
#                 None
#             )

#         if reference_img is not None:

#             if img.shape != reference_img.shape:
#                 return (
#                     False,
#                     f"shape mismatch: "
#                     f"{img.shape} != "
#                     f"{reference_img.shape}",
#                     None,
#                     None
#                 )

#             reference_voxel_size = np.asarray(
#                 reference_img.header.get_zooms()[:3],
#                 dtype=float
#             )

#             if not np.allclose(
#                 voxel_size,
#                 reference_voxel_size,
#                 atol=VOXEL_TOLERANCE
#             ):
#                 return (
#                     False,
#                     "voxel size mismatch with reference",
#                     None,
#                     None
#                 )

#             if not np.allclose(
#                 img.affine,
#                 reference_img.affine,
#                 atol=AFFINE_TOLERANCE
#             ):
#                 return (
#                     False,
#                     "affine mismatch with reference",
#                     None,
#                     None
#                 )

#         return (
#             True,
#             "VALID",
#             img,
#             data
#         )

#     except Exception as exc:

#         return (
#             False,
#             f"NIfTI validation error: {exc}",
#             None,
#             None
#         )


# # ============================================================
# # MASK VALIDATION
# # ============================================================

# def validate_mask(
#     mask_file,
#     reference_img
# ):
#     """
#     Validate Echo-1 HD-BET mask.

#     Requirements:
#       - readable
#       - finite
#       - same shape
#       - same affine
#       - same voxel size
#       - RAS
#       - binary 0/1
#       - non-empty
#       - not the entire volume
#     """

#     try:

#         mask_img = nib.load(
#             str(mask_file)
#         )

#         mask_data = mask_img.get_fdata(
#             dtype=np.float32
#         )

#         if mask_data.size == 0:
#             return (
#                 False,
#                 "mask is empty",
#                 None,
#                 None
#             )

#         if not np.all(
#             np.isfinite(mask_data)
#         ):
#             return (
#                 False,
#                 "mask contains non-finite values",
#                 None,
#                 None
#             )

#         # ----------------------------------------------------
#         # Geometry
#         # ----------------------------------------------------

#         if mask_img.shape != reference_img.shape:
#             return (
#                 False,
#                 f"mask shape mismatch: "
#                 f"{mask_img.shape} != "
#                 f"{reference_img.shape}",
#                 None,
#                 None
#             )

#         if not np.allclose(
#             mask_img.affine,
#             reference_img.affine,
#             atol=AFFINE_TOLERANCE
#         ):
#             return (
#                 False,
#                 "mask affine does not match N4 Echo 1",
#                 None,
#                 None
#             )

#         mask_voxel_size = np.asarray(
#             mask_img.header.get_zooms()[:3],
#             dtype=float
#         )

#         reference_voxel_size = np.asarray(
#             reference_img.header.get_zooms()[:3],
#             dtype=float
#         )

#         if not np.allclose(
#             mask_voxel_size,
#             reference_voxel_size,
#             atol=VOXEL_TOLERANCE
#         ):
#             return (
#                 False,
#                 "mask voxel size does not match N4 Echo 1",
#                 None,
#                 None
#             )

#         # ----------------------------------------------------
#         # Orientation
#         # ----------------------------------------------------

#         orientation = nib.aff2axcodes(
#             mask_img.affine
#         )

#         if orientation != ("R", "A", "S"):
#             return (
#                 False,
#                 f"mask orientation is {orientation}, "
#                 f"expected RAS",
#                 None,
#                 None
#             )

#         # ----------------------------------------------------
#         # Binary validation
#         # ----------------------------------------------------

#         unique_values = np.unique(
#             mask_data
#         )

#         if not np.all(
#             np.isin(unique_values, [0.0, 1.0])
#         ):
#             return (
#                 False,
#                 "mask is not binary 0/1",
#                 None,
#                 None
#             )

#         # ----------------------------------------------------
#         # Non-empty
#         # ----------------------------------------------------

#         mask_nonzero = np.count_nonzero(
#             mask_data
#         )

#         if mask_nonzero == 0:
#             return (
#                 False,
#                 "mask contains no brain voxels",
#                 None,
#                 None
#             )

#         # ----------------------------------------------------
#         # Sanity-check coverage
#         # ----------------------------------------------------

#         coverage = (
#             mask_nonzero /
#             mask_data.size
#         )

#         if coverage >= 1.0:
#             return (
#                 False,
#                 "mask covers entire image volume",
#                 None,
#                 None
#             )

#         return (
#             True,
#             "VALID",
#             mask_img,
#             mask_data.astype(bool)
#         )

#     except Exception as exc:

#         return (
#             False,
#             f"mask validation error: {exc}",
#             None,
#             None
#         )


# # ============================================================
# # PROCESSING COUNTERS
# # ============================================================

# processed_echoes = 0
# skipped_echoes = 0
# failed_echoes = 0

# successful_subjects = 0
# failed_subjects = 0

# failure_details = []


# # ============================================================
# # PROCESS EACH SUBJECT
# # ============================================================

# for index, subject_id in enumerate(
#     subjects,
#     start=1
# ):

#     print("\n" + "=" * 70)
#     print(
#         f"SUBJECT {index}/{len(subjects)}: "
#         f"{subject_id}"
#     )
#     print("=" * 70)

#     # --------------------------------------------------------
#     # Echo-1 reference input
#     # --------------------------------------------------------

#     echo1_input = (
#         INPUT_FOLDER /
#         f"{subject_id}_echo1_n4.nii.gz"
#     )

#     if not echo1_input.exists():

#         print(
#             "ERROR: N4 Echo 1 input is missing."
#         )

#         failed_subjects += 1
#         failure_details.append(
#             (
#                 subject_id,
#                 "missing Echo-1 N4 input"
#             )
#         )

#         continue

#     # --------------------------------------------------------
#     # Validate Echo-1 input
#     # --------------------------------------------------------

#     (
#         echo1_valid,
#         echo1_reason,
#         echo1_img,
#         echo1_data
#     ) = validate_nifti(
#         echo1_input
#     )

#     if not echo1_valid:

#         print(
#             "ERROR: Echo-1 N4 validation failed:"
#         )
#         print(f"  {echo1_reason}")

#         failed_subjects += 1
#         failure_details.append(
#             (
#                 subject_id,
#                 f"invalid Echo-1 N4: "
#                 f"{echo1_reason}"
#             )
#         )

#         continue

#     # --------------------------------------------------------
#     # Load and validate HD-BET mask
#     # --------------------------------------------------------

#     mask_file = (
#         MASK_FOLDER /
#         f"{subject_id}_echo1_n4_bet_bet.nii.gz"
#     )

#     print("\nHD-BET mask:")
#     print(f"  {mask_file}")

#     if not mask_file.exists():

#         print("ERROR: HD-BET mask is missing.")

#         failed_subjects += 1
#         failure_details.append(
#             (
#                 subject_id,
#                 "missing HD-BET mask"
#             )
#         )

#         continue

#     (
#         mask_valid,
#         mask_reason,
#         mask_img,
#         mask
#     ) = validate_mask(
#         mask_file,
#         echo1_img
#     )

#     if not mask_valid:

#         print(
#             "ERROR: HD-BET mask validation failed:"
#         )
#         print(f"  {mask_reason}")

#         failed_subjects += 1
#         failure_details.append(
#             (
#                 subject_id,
#                 f"invalid HD-BET mask: "
#                 f"{mask_reason}"
#             )
#         )

#         continue

#     mask_nonzero = np.count_nonzero(mask)

#     print(
#         f"Mask validation: PASS"
#     )

#     print(
#         f"Mask brain voxels: "
#         f"{mask_nonzero}"
#     )

#     print(
#         f"Mask coverage: "
#         f"{mask_nonzero / mask.size:.4%}"
#     )

#     # --------------------------------------------------------
#     # Subject-level status
#     # --------------------------------------------------------

#     subject_failed = False

#     # --------------------------------------------------------
#     # Process all five echoes
#     # --------------------------------------------------------

#     for echo in ECHOES:

#         print("\n" + "-" * 70)
#         print(f"ECHO {echo}")
#         print("-" * 70)

#         input_file = (
#             INPUT_FOLDER /
#             f"{subject_id}_echo{echo}_n4.nii.gz"
#         )

#         output_file = (
#             OUTPUT_FOLDER /
#             f"{subject_id}_echo{echo}_n4_masked.nii.gz"
#         )

#         print("Input:")
#         print(f"  {input_file}")

#         print("Output:")
#         print(f"  {output_file}")

#         # ----------------------------------------------------
#         # Input existence
#         # ----------------------------------------------------

#         if not input_file.exists():

#             print(
#                 "ERROR: N4 input is missing."
#             )

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "missing N4 input"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Validate input
#         # ----------------------------------------------------

#         (
#             input_valid,
#             input_reason,
#             echo_img,
#             echo_data
#         ) = validate_nifti(
#             input_file,
#             reference_img=echo1_img
#         )

#         if not input_valid:

#             print(
#                 "ERROR: N4 input validation failed:"
#             )
#             print(f"  {input_reason}")

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     f"invalid N4 input: "
#                     f"{input_reason}"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Resume-safe existing output validation
#         # ----------------------------------------------------

#         if output_file.exists():

#             (
#                 existing_valid,
#                 existing_reason,
#                 existing_img,
#                 existing_data
#             ) = validate_nifti(
#                 output_file,
#                 reference_img=echo_img
#             )

#             if existing_valid:

#                 # Ensure background is zero.
#                 outside_mask = (
#                     existing_data[~mask]
#                 )

#                 if (
#                     outside_mask.size > 0
#                     and not np.all(
#                         outside_mask == 0
#                     )
#                 ):

#                     print(
#                         "Existing output failed "
#                         "background-zero validation."
#                     )

#                 else:

#                     print(
#                         "Existing masked output "
#                         "is valid."
#                     )
#                     print(
#                         "Skipping this echo."
#                     )

#                     skipped_echoes += 1
#                     continue

#             else:

#                 print(
#                     "Existing output is invalid:"
#                 )
#                 print(
#                     f"  {existing_reason}"
#                 )

#                 print(
#                     "It will be recreated."
#                 )

#             # ------------------------------------------------
#             # Remove invalid existing output
#             # ------------------------------------------------

#             try:

#                 output_file.unlink()

#                 print(
#                     "Removed invalid existing output."
#                 )

#             except Exception as exc:

#                 print(
#                     "ERROR: Could not remove "
#                     "invalid existing output:"
#                 )
#                 print(f"  {exc}")

#                 subject_failed = True
#                 failed_echoes += 1

#                 failure_details.append(
#                     (
#                         f"{subject_id}_echo{echo}",
#                         "could not remove invalid output"
#                     )
#                 )

#                 continue

#         # ----------------------------------------------------
#         # Geometry with mask
#         # ----------------------------------------------------

#         if echo_img.shape != mask_img.shape:

#             print(
#                 "ERROR: Echo and mask shapes do not match."
#             )

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "echo/mask shape mismatch"
#                 )
#             )

#             continue

#         if not np.allclose(
#             echo_img.affine,
#             mask_img.affine,
#             atol=AFFINE_TOLERANCE
#         ):

#             print(
#                 "ERROR: Echo and mask affines "
#                 "do not match."
#             )

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "echo/mask affine mismatch"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Apply same Echo-1 mask
#         # ----------------------------------------------------

#         masked_data = (
#             echo_data * mask
#         ).astype(np.float32)

#         original_nonzero = np.count_nonzero(
#             echo_data
#         )

#         masked_nonzero = np.count_nonzero(
#             masked_data
#         )

#         print(
#             f"Original non-zero voxels: "
#             f"{original_nonzero}"
#         )

#         print(
#             f"Masked non-zero voxels: "
#             f"{masked_nonzero}"
#         )

#         if masked_nonzero == 0:

#             print(
#                 "ERROR: Masked image is empty."
#             )

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "masked output is empty"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Explicit background verification
#         # ----------------------------------------------------

#         outside_mask = (
#             masked_data[~mask]
#         )

#         if (
#             outside_mask.size > 0
#             and not np.all(
#                 outside_mask == 0
#             )
#         ):

#             print(
#                 "ERROR: Voxels outside mask "
#                 "are not zero."
#             )

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "non-zero values outside mask"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Preserve geometry/header
#         # ----------------------------------------------------

#         masked_img = nib.Nifti1Image(
#             masked_data,
#             echo_img.affine,
#             echo_img.header.copy()
#         )

#         # Explicitly ensure float32 datatype.
#         masked_img.set_data_dtype(
#             np.float32
#         )

#         # ----------------------------------------------------
#         # Save
#         # ----------------------------------------------------

#         try:

#             nib.save(
#                 masked_img,
#                 output_file
#             )

#         except Exception as exc:

#             print(
#                 "ERROR saving output:"
#             )
#             print(f"  {exc}")

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     f"save error: {exc}"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Validate saved output
#         # ----------------------------------------------------

#         (
#             saved_valid,
#             saved_reason,
#             saved_img,
#             saved_data
#         ) = validate_nifti(
#             output_file,
#             reference_img=echo_img
#         )

#         if not saved_valid:

#             print(
#                 "ERROR: Saved output validation failed:"
#             )
#             print(f"  {saved_reason}")

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     f"saved output invalid: "
#                     f"{saved_reason}"
#                 )
#             )

#             continue

#         # ----------------------------------------------------
#         # Saved background verification
#         # ----------------------------------------------------

#         saved_outside_mask = (
#             saved_data[~mask]
#         )

#         if (
#             saved_outside_mask.size > 0
#             and not np.all(
#                 saved_outside_mask == 0
#             )
#         ):

#             print(
#                 "ERROR: Saved output contains "
#                 "non-zero background."
#             )

#             subject_failed = True
#             failed_echoes += 1

#             failure_details.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "saved output background not zero"
#                 )
#             )

#             continue

#         print(
#             "Masked output validation: PASS"
#         )

#         print(
#             "Saved successfully."
#         )

#         processed_echoes += 1

#     # --------------------------------------------------------
#     # Subject result
#     # --------------------------------------------------------

#     subject_output_count = sum(
#         (
#             OUTPUT_FOLDER /
#             f"{subject_id}_echo{echo}_n4_masked.nii.gz"
#         ).exists()
#         for echo in ECHOES
#     )

#     if (
#         not subject_failed
#         and subject_output_count == len(ECHOES)
#     ):

#         successful_subjects += 1

#         print()
#         print(
#             f"SUCCESS: {subject_id} "
#             f"— all five echoes are valid."
#         )

#     else:

#         failed_subjects += 1

#         print()
#         print(
#             f"FAILED/PARTIAL: {subject_id}"
#         )

#         print(
#             f"Valid/existing outputs: "
#             f"{subject_output_count}/5"
#         )


# # ============================================================
# # FINAL BATCH VALIDATION
# # ============================================================

# print("\n" + "=" * 70)
# print("FINAL MASKED-ECHO BATCH VALIDATION")
# print("=" * 70)

# final_valid_subjects = []
# final_invalid_subjects = []


# for subject_id in subjects:

#     echo_outputs_valid = True

#     # Echo-1 input is needed as geometry reference.
#     echo1_input = (
#         INPUT_FOLDER /
#         f"{subject_id}_echo1_n4.nii.gz"
#     )

#     mask_file = (
#         MASK_FOLDER /
#         f"{subject_id}_echo1_n4_bet_bet.nii.gz"
#     )

#     if not echo1_input.exists():

#         final_invalid_subjects.append(
#             (
#                 subject_id,
#                 "missing Echo-1 N4 input"
#             )
#         )

#         continue

#     if not mask_file.exists():

#         final_invalid_subjects.append(
#             (
#                 subject_id,
#                 "missing HD-BET mask"
#             )
#         )

#         continue

#     (
#         ref_valid,
#         ref_reason,
#         reference_img,
#         _
#     ) = validate_nifti(
#         echo1_input
#     )

#     if not ref_valid:

#         final_invalid_subjects.append(
#             (
#                 subject_id,
#                 f"invalid Echo-1 N4: "
#                 f"{ref_reason}"
#             )
#         )

#         continue

#     (
#         mask_valid,
#         mask_reason,
#         mask_img,
#         mask
#     ) = validate_mask(
#         mask_file,
#         reference_img
#     )

#     if not mask_valid:

#         final_invalid_subjects.append(
#             (
#                 subject_id,
#                 f"invalid mask: "
#                 f"{mask_reason}"
#             )
#         )

#         continue

#     for echo in ECHOES:

#         output_file = (
#             OUTPUT_FOLDER /
#             f"{subject_id}_echo{echo}_n4_masked.nii.gz"
#         )

#         if not output_file.exists():

#             echo_outputs_valid = False

#             final_invalid_subjects.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "missing masked output"
#                 )
#             )

#             continue

#         (
#             output_valid,
#             output_reason,
#             output_img,
#             output_data
#         ) = validate_nifti(
#             output_file,
#             reference_img=reference_img
#         )

#         if not output_valid:

#             echo_outputs_valid = False

#             final_invalid_subjects.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     f"invalid masked output: "
#                     f"{output_reason}"
#                 )
#             )

#             continue

#         outside_mask = (
#             output_data[~mask]
#         )

#         if (
#             outside_mask.size > 0
#             and not np.all(
#                 outside_mask == 0
#             )
#         ):

#             echo_outputs_valid = False

#             final_invalid_subjects.append(
#                 (
#                     f"{subject_id}_echo{echo}",
#                     "non-zero background"
#                 )
#             )

#     if echo_outputs_valid:
#         final_valid_subjects.append(
#             subject_id
#         )


# # ============================================================
# # FINAL SUMMARY
# # ============================================================

# expected_subjects = len(subjects)
# expected_echoes = expected_subjects * len(ECHOES)

# actual_output_files = 0

# for subject_id in subjects:

#     for echo in ECHOES:

#         output_file = (
#             OUTPUT_FOLDER /
#             f"{subject_id}_echo{echo}_n4_masked.nii.gz"
#         )

#         if output_file.exists():
#             actual_output_files += 1


# print()
# print(
#     f"Expected subjects        : "
#     f"{expected_subjects}"
# )

# print(
#     f"Expected masked echoes   : "
#     f"{expected_echoes}"
# )

# print(
#     f"Existing output files    : "
#     f"{actual_output_files}"
# )

# print(
#     f"Valid complete subjects  : "
#     f"{len(final_valid_subjects)}"
# )

# print(
#     f"Invalid/missing entries  : "
#     f"{len(final_invalid_subjects)}"
# )

# print()
# print(
#     f"Newly processed echoes   : "
#     f"{processed_echoes}"
# )

# print(
#     f"Skipped valid echoes     : "
#     f"{skipped_echoes}"
# )

# print(
#     f"Failed echoes            : "
#     f"{failed_echoes}"
# )


# # ============================================================
# # INVALID DETAILS
# # ============================================================

# if final_invalid_subjects:

#     print()
#     print("FINAL INVALID/MISSING ENTRIES:")

#     for subject, reason in final_invalid_subjects:
#         print(
#             f"  {subject}: {reason}"
#         )


# if failure_details:

#     print()
#     print("PROCESSING FAILURE DETAILS:")

#     for subject, reason in failure_details:
#         print(
#             f"  {subject}: {reason}"
#         )


# # ============================================================
# # OUTPUT
# # ============================================================

# print()
# print("Output folder:")
# print(
#     f"  {OUTPUT_FOLDER.resolve()}"
# )


# # ============================================================
# # FINAL STATUS
# # ============================================================

# print()
# print("=" * 70)

# if (
#     len(final_valid_subjects) == expected_subjects
#     and actual_output_files == expected_echoes
#     and not final_invalid_subjects
# ):

#     print(
#         "STATUS: MASK APPLICATION "
#         "VALIDATION PASSED"
#     )

# else:

#     print(
#         "STATUS: MASK APPLICATION "
#         "VALIDATION FAILED"
#     )

# print("=" * 70)


# src/apply_echo1_mask.py

from pathlib import Path
import csv

import nibabel as nib
import numpy as np


# ============================================================
# GENERIC sMRI MASK APPLICATION
#
# Supports:
#   1. 5-ECHO MPRAGE
#      - Echo-1 HD-BET mask is applied to all 5 echoes
#
#   2. RMS MPRAGE
#      - RMS HD-BET mask is applied to the RMS N4 volume
#
# INPUT:
#   processed_data/smri/n4
#   processed_data/smri/mprage_5e_bet
#
# OUTPUT:
#   processed_data/smri/mprage_5e_masked
#
# SUBJECT SELECTION:
#   mprage_selection_manifest.csv
#
# IMPORTANT:
#   - One selected acquisition per subject.
#   - No spatial registration or resampling here.
#   - Original N4 inputs are never modified.
#   - Existing valid outputs are skipped.
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT_FOLDER = Path(
    "processed_data/smri/n4"
)

MASK_FOLDER = Path(
    "processed_data/smri/mprage_5e_bet"
)

OUTPUT_FOLDER = Path(
    "processed_data/smri/mprage_5e_masked"
)

MANIFEST_FILE = Path(
    "processed_data/smri/mprage_selection/"
    "mprage_selection_manifest.csv"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

ECHOES = [1, 2, 3, 4, 5]

VOXEL_TOLERANCE = 0.01
AFFINE_TOLERANCE = 1e-3


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GENERIC sMRI HD-BET MASK APPLICATION")
print("=" * 70)

print("\nN4 input folder:")
print(f"  {INPUT_FOLDER.resolve()}")

print("\nHD-BET mask folder:")
print(f"  {MASK_FOLDER.resolve()}")

print("\nOutput folder:")
print(f"  {OUTPUT_FOLDER.resolve()}")

print("\nSelection manifest:")
print(f"  {MANIFEST_FILE.resolve()}")


# ============================================================
# REQUIRED PATH CHECKS
# ============================================================

if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        f"N4 input folder does not exist:\n"
        f"{INPUT_FOLDER.resolve()}"
    )

if not MASK_FOLDER.exists():
    raise FileNotFoundError(
        f"HD-BET mask folder does not exist:\n"
        f"{MASK_FOLDER.resolve()}"
    )

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"Selection manifest does not exist:\n"
        f"{MANIFEST_FILE.resolve()}"
    )


# ============================================================
# LOAD SELECTED ACQUISITIONS
# ============================================================

def load_selected_acquisitions(manifest_file):
    """
    Load exactly one selected acquisition per subject.

    Returns:
        list of dictionaries containing:
            subject
            acquisition_type
            session
            series
    """

    selected = []

    with manifest_file.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "Selection manifest has no header."
            )

        required_columns = {
            "subject",
            "selection_status"
        }

        missing = (
            required_columns -
            set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Selection manifest is missing required "
                "columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            status = str(
                row.get("selection_status", "")
            ).strip().upper()

            if status != "SELECTED":
                continue

            subject = str(
                row.get("subject", "")
            ).strip()

            if not subject:
                raise ValueError(
                    "A SELECTED manifest row has no "
                    "subject ID."
                )

            acquisition_type = str(
                row.get("acquisition_type", "")
            ).strip().upper()

            if not acquisition_type:

                # Compatibility with manifests that may use
                # different capitalization.
                acquisition_type = str(
                    row.get("type", "")
                ).strip().upper()

            if acquisition_type not in {
                "5ECHO",
                "RMS"
            }:
                raise ValueError(
                    f"Unsupported acquisition type for "
                    f"{subject}: {acquisition_type}"
                )

            selected.append(
                {
                    "subject": subject,
                    "acquisition_type": acquisition_type,
                    "session": str(
                        row.get("session", "")
                    ).strip(),
                    "series": str(
                        row.get("series", "")
                    ).strip()
                }
            )

    if not selected:
        raise ValueError(
            "No SELECTED acquisitions found in "
            "the selection manifest."
        )

    subjects = [
        item["subject"]
        for item in selected
    ]

    if len(subjects) != len(set(subjects)):

        duplicates = sorted(
            {
                subject
                for subject in subjects
                if subjects.count(subject) > 1
            }
        )

        raise ValueError(
            "More than one SELECTED acquisition "
            "exists for subject(s):\n"
            + "\n".join(duplicates)
        )

    return sorted(
        selected,
        key=lambda x: x["subject"]
    )


selected_acquisitions = (
    load_selected_acquisitions(
        MANIFEST_FILE
    )
)


# ============================================================
# DISPLAY CURRENT COHORT
# ============================================================

print("\n" + "-" * 70)
print("SELECTED ACQUISITIONS FROM MANIFEST")
print("-" * 70)

for item in selected_acquisitions:

    print(
        f"  {item['subject']} | "
        f"{item['acquisition_type']} | "
        f"{item['session']} | "
        f"{item['series']}"
    )

print(
    f"\nTotal selected subjects: "
    f"{len(selected_acquisitions)}"
)


# ============================================================
# ACQUISITION SUMMARY
# ============================================================

type_counts = {
    "5ECHO": 0,
    "RMS": 0
}

for item in selected_acquisitions:
    type_counts[
        item["acquisition_type"]
    ] += 1

print("\nAcquisition types:")
print(
    f"  5ECHO: {type_counts['5ECHO']}"
)
print(
    f"  RMS   : {type_counts['RMS']}"
)


# ============================================================
# GENERAL NIFTI VALIDATION
# ============================================================

def validate_nifti(
    file_path,
    reference_img=None
):
    """
    Validate a NIfTI image.

    Checks:
      - readable
      - non-empty
      - finite values
      - RAS orientation
      - 1mm isotropic voxel size
      - non-zero content

    If reference_img is supplied:
      - shape
      - voxel size
      - affine
    are compared.
    """

    try:

        img = nib.load(
            str(file_path)
        )

        data = img.get_fdata(
            dtype=np.float32
        )

        if data.size == 0:
            return (
                False,
                "empty image",
                None,
                None
            )

        if not np.all(
            np.isfinite(data)
        ):
            return (
                False,
                "non-finite voxel values",
                None,
                None
            )

        orientation = nib.aff2axcodes(
            img.affine
        )

        if orientation != (
            "R",
            "A",
            "S"
        ):
            return (
                False,
                f"orientation is {orientation}, "
                f"expected RAS",
                None,
                None
            )

        voxel_size = np.asarray(
            img.header.get_zooms()[:3],
            dtype=float
        )

        if not np.allclose(
            voxel_size,
            [1.0, 1.0, 1.0],
            atol=VOXEL_TOLERANCE
        ):
            return (
                False,
                f"voxel size is {voxel_size}, "
                f"expected 1mm isotropic",
                None,
                None
            )

        if np.count_nonzero(data) == 0:
            return (
                False,
                "all voxels are zero",
                None,
                None
            )

        if reference_img is not None:

            if img.shape != reference_img.shape:
                return (
                    False,
                    f"shape mismatch: "
                    f"{img.shape} != "
                    f"{reference_img.shape}",
                    None,
                    None
                )

            reference_voxel_size = np.asarray(
                reference_img.header.get_zooms()[:3],
                dtype=float
            )

            if not np.allclose(
                voxel_size,
                reference_voxel_size,
                atol=VOXEL_TOLERANCE
            ):
                return (
                    False,
                    "voxel size mismatch with reference",
                    None,
                    None
                )

            if not np.allclose(
                img.affine,
                reference_img.affine,
                atol=AFFINE_TOLERANCE
            ):
                return (
                    False,
                    "affine mismatch with reference",
                    None,
                    None
                )

        return (
            True,
            "VALID",
            img,
            data
        )

    except Exception as exc:

        return (
            False,
            f"NIfTI validation error: {exc}",
            None,
            None
        )


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_mask(
    mask_file,
    reference_img
):
    """
    Validate an HD-BET mask.

    Requirements:
      - readable
      - finite
      - same shape
      - same affine
      - same voxel size
      - RAS
      - binary 0/1
      - non-empty
      - does not cover entire image
    """

    try:

        mask_img = nib.load(
            str(mask_file)
        )

        mask_data = mask_img.get_fdata(
            dtype=np.float32
        )

        if mask_data.size == 0:
            return (
                False,
                "mask is empty",
                None,
                None
            )

        if not np.all(
            np.isfinite(mask_data)
        ):
            return (
                False,
                "mask contains non-finite values",
                None,
                None
            )

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        if mask_img.shape != reference_img.shape:
            return (
                False,
                f"mask shape mismatch: "
                f"{mask_img.shape} != "
                f"{reference_img.shape}",
                None,
                None
            )

        if not np.allclose(
            mask_img.affine,
            reference_img.affine,
            atol=AFFINE_TOLERANCE
        ):
            return (
                False,
                "mask affine does not match input",
                None,
                None
            )

        mask_voxel_size = np.asarray(
            mask_img.header.get_zooms()[:3],
            dtype=float
        )

        reference_voxel_size = np.asarray(
            reference_img.header.get_zooms()[:3],
            dtype=float
        )

        if not np.allclose(
            mask_voxel_size,
            reference_voxel_size,
            atol=VOXEL_TOLERANCE
        ):
            return (
                False,
                "mask voxel size does not match input",
                None,
                None
            )

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        orientation = nib.aff2axcodes(
            mask_img.affine
        )

        if orientation != (
            "R",
            "A",
            "S"
        ):
            return (
                False,
                f"mask orientation is {orientation}, "
                f"expected RAS",
                None,
                None
            )

        # ----------------------------------------------------
        # Binary validation
        # ----------------------------------------------------

        unique_values = np.unique(
            mask_data
        )

        if not np.all(
            np.isin(
                unique_values,
                [0.0, 1.0]
            )
        ):
            return (
                False,
                "mask is not binary 0/1",
                None,
                None
            )

        # ----------------------------------------------------
        # Non-empty
        # ----------------------------------------------------

        mask_nonzero = np.count_nonzero(
            mask_data
        )

        if mask_nonzero == 0:
            return (
                False,
                "mask contains no brain voxels",
                None,
                None
            )

        # ----------------------------------------------------
        # Coverage sanity check
        # ----------------------------------------------------

        coverage = (
            mask_nonzero /
            mask_data.size
        )

        if coverage >= 1.0:
            return (
                False,
                "mask covers entire image volume",
                None,
                None
            )

        return (
            True,
            "VALID",
            mask_img,
            mask_data.astype(bool)
        )

    except Exception as exc:

        return (
            False,
            f"mask validation error: {exc}",
            None,
            None
        )


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_masked_output(
    output_file,
    reference_img,
    mask
):
    """
    Validate a masked output against its input
    geometry and the supplied brain mask.
    """

    (
        valid,
        reason,
        output_img,
        output_data
    ) = validate_nifti(
        output_file,
        reference_img=reference_img
    )

    if not valid:
        return False, reason

    outside_mask = (
        output_data[~mask]
    )

    if (
        outside_mask.size > 0
        and not np.all(
            outside_mask == 0
        )
    ):
        return (
            False,
            "non-zero values outside mask"
        )

    if np.count_nonzero(
        output_data
    ) == 0:
        return (
            False,
            "masked output is empty"
        )

    return True, "VALID"


# ============================================================
# PROCESSING COUNTERS
# ============================================================

processed_volumes = 0
skipped_volumes = 0
failed_volumes = 0

successful_subjects = 0
failed_subjects = 0

failure_details = []


# ============================================================
# PROCESS ONE VOLUME
# ============================================================

def process_volume(
    subject_id,
    acquisition_type,
    input_file,
    mask_file,
    output_file
):
    """
    Apply the HD-BET mask to one N4 volume.

    Returns:
        True  = successful
        False = failed
        "SKIPPED" = valid existing output
    """

    global processed_volumes
    global skipped_volumes
    global failed_volumes

    print("\n" + "-" * 70)
    print(
        f"{subject_id} | "
        f"{acquisition_type}"
    )
    print("-" * 70)

    print("Input:")
    print(f"  {input_file}")

    print("Mask:")
    print(f"  {mask_file}")

    print("Output:")
    print(f"  {output_file}")

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not input_file.exists():

        print(
            "ERROR: N4 input is missing."
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                "missing N4 input"
            )
        )

        return False

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    (
        input_valid,
        input_reason,
        input_img,
        input_data
    ) = validate_nifti(
        input_file
    )

    if not input_valid:

        print(
            "ERROR: N4 input validation failed:"
        )
        print(
            f"  {input_reason}"
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                f"invalid N4 input: "
                f"{input_reason}"
            )
        )

        return False

    # --------------------------------------------------------
    # Mask existence
    # --------------------------------------------------------

    if not mask_file.exists():

        print(
            "ERROR: HD-BET mask is missing."
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                "missing HD-BET mask"
            )
        )

        return False

    # --------------------------------------------------------
    # Mask validation
    # --------------------------------------------------------

    (
        mask_valid,
        mask_reason,
        mask_img,
        mask
    ) = validate_mask(
        mask_file,
        input_img
    )

    if not mask_valid:

        print(
            "ERROR: HD-BET mask validation failed:"
        )
        print(
            f"  {mask_reason}"
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                f"invalid HD-BET mask: "
                f"{mask_reason}"
            )
        )

        return False

    mask_nonzero = np.count_nonzero(
        mask
    )

    print(
        "Mask validation: PASS"
    )

    print(
        f"Mask brain voxels: "
        f"{mask_nonzero}"
    )

    print(
        f"Mask coverage: "
        f"{mask_nonzero / mask.size:.4%}"
    )

    # --------------------------------------------------------
    # Existing output validation
    # --------------------------------------------------------

    if output_file.exists():

        (
            existing_valid,
            existing_reason
        ) = validate_masked_output(
            output_file,
            input_img,
            mask
        )

        if existing_valid:

            print(
                "Existing masked output is valid."
            )
            print(
                "Skipping this volume."
            )

            skipped_volumes += 1

            return "SKIPPED"

        print(
            "Existing output is invalid:"
        )
        print(
            f"  {existing_reason}"
        )
        print(
            "It will be recreated."
        )

        try:

            output_file.unlink()

            print(
                "Removed invalid existing output."
            )

        except Exception as exc:

            print(
                "ERROR: Could not remove "
                "invalid existing output:"
            )
            print(
                f"  {exc}"
            )

            failed_volumes += 1

            failure_details.append(
                (
                    f"{subject_id}_{acquisition_type}",
                    "could not remove invalid output"
                )
            )

            return False

    # --------------------------------------------------------
    # Apply mask
    # --------------------------------------------------------

    masked_data = (
        input_data * mask
    ).astype(np.float32)

    original_nonzero = np.count_nonzero(
        input_data
    )

    masked_nonzero = np.count_nonzero(
        masked_data
    )

    print(
        f"Original non-zero voxels: "
        f"{original_nonzero}"
    )

    print(
        f"Masked non-zero voxels: "
        f"{masked_nonzero}"
    )

    if masked_nonzero == 0:

        print(
            "ERROR: Masked image is empty."
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                "masked output is empty"
            )
        )

        return False

    # --------------------------------------------------------
    # Explicit background verification
    # --------------------------------------------------------

    outside_mask = (
        masked_data[~mask]
    )

    if (
        outside_mask.size > 0
        and not np.all(
            outside_mask == 0
        )
    ):

        print(
            "ERROR: Voxels outside mask "
            "are not zero."
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                "non-zero values outside mask"
            )
        )

        return False

    # --------------------------------------------------------
    # Preserve geometry/header
    # --------------------------------------------------------

    masked_img = nib.Nifti1Image(
        masked_data,
        input_img.affine,
        input_img.header.copy()
    )

    masked_img.set_data_dtype(
        np.float32
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        nib.save(
            masked_img,
            output_file
        )

    except Exception as exc:

        print(
            "ERROR saving output:"
        )
        print(
            f"  {exc}"
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                f"save error: {exc}"
            )
        )

        return False

    # --------------------------------------------------------
    # Validate saved output
    # --------------------------------------------------------

    (
        saved_valid,
        saved_reason
    ) = validate_masked_output(
        output_file,
        input_img,
        mask
    )

    if not saved_valid:

        print(
            "ERROR: Saved output validation failed:"
        )
        print(
            f"  {saved_reason}"
        )

        failed_volumes += 1

        failure_details.append(
            (
                f"{subject_id}_{acquisition_type}",
                f"saved output invalid: "
                f"{saved_reason}"
            )
        )

        return False

    print(
        "Masked output validation: PASS"
    )

    print(
        "Saved successfully."
    )

    processed_volumes += 1

    return True


# ============================================================
# PROCESS SELECTED SUBJECTS
# ============================================================

for index, item in enumerate(
    selected_acquisitions,
    start=1
):

    subject_id = item["subject"]
    acquisition_type = item[
        "acquisition_type"
    ]

    print("\n" + "=" * 70)
    print(
        f"SUBJECT {index}/"
        f"{len(selected_acquisitions)}: "
        f"{subject_id}"
    )
    print(
        f"Acquisition type: "
        f"{acquisition_type}"
    )
    print("=" * 70)

    subject_success = True

    # ========================================================
    # RMS
    # ========================================================

    if acquisition_type == "RMS":

        input_file = (
            INPUT_FOLDER /
            f"{subject_id}_rms_n4.nii.gz"
        )

        mask_file = (
            MASK_FOLDER /
            f"{subject_id}_rms_n4_bet_bet.nii.gz"
        )

        output_file = (
            OUTPUT_FOLDER /
            f"{subject_id}_rms_n4_masked.nii.gz"
        )

        result = process_volume(
            subject_id,
            acquisition_type,
            input_file,
            mask_file,
            output_file
        )

        if result is False:
            subject_success = False

    # ========================================================
    # 5ECHO
    # ========================================================

    elif acquisition_type == "5ECHO":

        # ----------------------------------------------------
        # Echo-1 is the geometry reference and mask source.
        # ----------------------------------------------------

        echo1_input = (
            INPUT_FOLDER /
            f"{subject_id}_echo1_n4.nii.gz"
        )

        echo1_mask = (
            MASK_FOLDER /
            f"{subject_id}_echo1_n4_bet_bet.nii.gz"
        )

        if not echo1_input.exists():

            print(
                "ERROR: Echo-1 N4 input is missing."
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject_id,
                    "missing Echo-1 N4 input"
                )
            )

            continue

        (
            echo1_valid,
            echo1_reason,
            echo1_img,
            _
        ) = validate_nifti(
            echo1_input
        )

        if not echo1_valid:

            print(
                "ERROR: Echo-1 N4 validation failed:"
            )
            print(
                f"  {echo1_reason}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject_id,
                    f"invalid Echo-1 N4: "
                    f"{echo1_reason}"
                )
            )

            continue

        if not echo1_mask.exists():

            print(
                "ERROR: Echo-1 HD-BET mask is missing."
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject_id,
                    "missing Echo-1 HD-BET mask"
                )
            )

            continue

        (
            mask_valid,
            mask_reason,
            mask_img,
            mask
        ) = validate_mask(
            echo1_mask,
            echo1_img
        )

        if not mask_valid:

            print(
                "ERROR: Echo-1 HD-BET mask "
                "validation failed:"
            )
            print(
                f"  {mask_reason}"
            )

            failed_subjects += 1

            failure_details.append(
                (
                    subject_id,
                    f"invalid Echo-1 mask: "
                    f"{mask_reason}"
                )
            )

            continue

        print(
            "Echo-1 HD-BET mask validation: PASS"
        )

        # ----------------------------------------------------
        # Apply same Echo-1 mask to all five echoes.
        # ----------------------------------------------------

        for echo in ECHOES:

            input_file = (
                INPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4.nii.gz"
            )

            output_file = (
                OUTPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4_masked.nii.gz"
            )

            result = process_volume(
                subject_id,
                f"5ECHO-Echo{echo}",
                input_file,
                echo1_mask,
                output_file
            )

            if result is False:
                subject_success = False

    # ========================================================
    # UNKNOWN TYPE
    # ========================================================

    else:

        print(
            f"ERROR: Unsupported acquisition type: "
            f"{acquisition_type}"
        )

        subject_success = False

        failed_subjects += 1

        failure_details.append(
            (
                subject_id,
                f"unsupported acquisition type: "
                f"{acquisition_type}"
            )
        )

        continue

    # ========================================================
    # SUBJECT RESULT
    # ========================================================

    if subject_success:

        successful_subjects += 1

        print()
        print(
            f"SUCCESS: {subject_id}"
        )

    else:

        failed_subjects += 1

        print()
        print(
            f"FAILED/PARTIAL: {subject_id}"
        )


# ============================================================
# FINAL BATCH VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MASKED sMRI BATCH VALIDATION")
print("=" * 70)

final_valid_subjects = []
final_invalid_entries = []

expected_output_files = 0


for item in selected_acquisitions:

    subject_id = item["subject"]
    acquisition_type = item[
        "acquisition_type"
    ]

    # ========================================================
    # RMS
    # ========================================================

    if acquisition_type == "RMS":

        expected_output_files += 1

        input_file = (
            INPUT_FOLDER /
            f"{subject_id}_rms_n4.nii.gz"
        )

        mask_file = (
            MASK_FOLDER /
            f"{subject_id}_rms_n4_bet_bet.nii.gz"
        )

        output_file = (
            OUTPUT_FOLDER /
            f"{subject_id}_rms_n4_masked.nii.gz"
        )

        if not input_file.exists():

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing RMS N4 input"
                )
            )

            continue

        if not mask_file.exists():

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing RMS HD-BET mask"
                )
            )

            continue

        (
            ref_valid,
            ref_reason,
            reference_img,
            _
        ) = validate_nifti(
            input_file
        )

        if not ref_valid:

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid RMS N4: "
                    f"{ref_reason}"
                )
            )

            continue

        (
            mask_valid,
            mask_reason,
            _,
            mask
        ) = validate_mask(
            mask_file,
            reference_img
        )

        if not mask_valid:

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid RMS mask: "
                    f"{mask_reason}"
                )
            )

            continue

        if not output_file.exists():

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing RMS masked output"
                )
            )

            continue

        (
            output_valid,
            output_reason
        ) = validate_masked_output(
            output_file,
            reference_img,
            mask
        )

        if not output_valid:

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid RMS masked output: "
                    f"{output_reason}"
                )
            )

            continue

        final_valid_subjects.append(
            subject_id
        )

    # ========================================================
    # 5ECHO
    # ========================================================

    elif acquisition_type == "5ECHO":

        expected_output_files += len(ECHOES)

        echo1_input = (
            INPUT_FOLDER /
            f"{subject_id}_echo1_n4.nii.gz"
        )

        mask_file = (
            MASK_FOLDER /
            f"{subject_id}_echo1_n4_bet_bet.nii.gz"
        )

        if not echo1_input.exists():

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing Echo-1 N4 input"
                )
            )

            continue

        if not mask_file.exists():

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing Echo-1 HD-BET mask"
                )
            )

            continue

        (
            ref_valid,
            ref_reason,
            reference_img,
            _
        ) = validate_nifti(
            echo1_input
        )

        if not ref_valid:

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid Echo-1 N4: "
                    f"{ref_reason}"
                )
            )

            continue

        (
            mask_valid,
            mask_reason,
            _,
            mask
        ) = validate_mask(
            mask_file,
            reference_img
        )

        if not mask_valid:

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid Echo-1 mask: "
                    f"{mask_reason}"
                )
            )

            continue

        subject_valid = True

        for echo in ECHOES:

            output_file = (
                OUTPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4_masked.nii.gz"
            )

            if not output_file.exists():

                subject_valid = False

                final_invalid_entries.append(
                    (
                        f"{subject_id}_echo{echo}",
                        "missing masked output"
                    )
                )

                continue

            (
                output_valid,
                output_reason
            ) = validate_masked_output(
                output_file,
                reference_img,
                mask
            )

            if not output_valid:

                subject_valid = False

                final_invalid_entries.append(
                    (
                        f"{subject_id}_echo{echo}",
                        f"invalid masked output: "
                        f"{output_reason}"
                    )
                )

        if subject_valid:

            final_valid_subjects.append(
                subject_id
            )


# ============================================================
# FINAL OUTPUT INVENTORY
# ============================================================

actual_output_files = 0

for item in selected_acquisitions:

    subject_id = item["subject"]
    acquisition_type = item[
        "acquisition_type"
    ]

    if acquisition_type == "RMS":

        output_file = (
            OUTPUT_FOLDER /
            f"{subject_id}_rms_n4_masked.nii.gz"
        )

        if output_file.exists():
            actual_output_files += 1

    elif acquisition_type == "5ECHO":

        for echo in ECHOES:

            output_file = (
                OUTPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4_masked.nii.gz"
            )

            if output_file.exists():
                actual_output_files += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print(
    f"Selected subjects       : "
    f"{len(selected_acquisitions)}"
)

print(
    f"Expected output volumes : "
    f"{expected_output_files}"
)

print(
    f"Existing output files   : "
    f"{actual_output_files}"
)

print(
    f"Valid complete subjects : "
    f"{len(final_valid_subjects)}"
)

print(
    f"Invalid/missing entries : "
    f"{len(final_invalid_entries)}"
)

print()

print(
    f"Newly processed volumes : "
    f"{processed_volumes}"
)

print(
    f"Skipped valid volumes   : "
    f"{skipped_volumes}"
)

print(
    f"Failed volumes          : "
    f"{failed_volumes}"
)


# ============================================================
# INVALID DETAILS
# ============================================================

if final_invalid_entries:

    print()
    print(
        "FINAL INVALID/MISSING ENTRIES:"
    )

    for subject, reason in final_invalid_entries:

        print(
            f"  {subject}: {reason}"
        )


if failure_details:

    print()
    print(
        "PROCESSING FAILURE DETAILS:"
    )

    for subject, reason in failure_details:

        print(
            f"  {subject}: {reason}"
        )


# ============================================================
# OUTPUT
# ============================================================

print()
print("Output folder:")
print(
    f"  {OUTPUT_FOLDER.resolve()}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

if (
    len(final_valid_subjects)
    == len(selected_acquisitions)

    and actual_output_files
    == expected_output_files

    and not final_invalid_entries
):

    print(
        "STATUS: MASK APPLICATION "
        "VALIDATION PASSED"
    )

else:

    print(
        "STATUS: MASK APPLICATION "
        "VALIDATION FAILED"
    )

print("=" * 70)