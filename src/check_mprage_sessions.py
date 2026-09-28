
# # src\check_mprage_sessions.py

# """
# ======================================================================
# COBRE MPRAGE 5-ECHO INSPECTION AND SELECTION
# ======================================================================

# Purpose
# -------
# Inspect an extracted COBRE DICOM dataset, validate all candidate
# 5-echo MPRAGE acquisitions, and select exactly ONE valid MPRAGE
# acquisition per subject for downstream preprocessing.

# This script is GENERIC and is NOT tied to one ZIP file, one subject,
# one session number, or one series number.

# Selection rule
# --------------
# For each subject:

# 1. Find all Session_* directories automatically.
# 2. Find all directories containing "mprage_5e".
# 3. Validate each candidate.
# 4. Prefer the EARLIEST session containing a complete valid
#    5-echo MPRAGE:
#        Session_1 -> Session_2 -> Session_3 -> Session_4 -> ...
# 5. If multiple valid MPRAGE acquisitions exist within the selected
#    session, choose the lowest SeriesNumber.
# 6. If SeriesNumber cannot be determined, use deterministic
#    alphabetical path ordering.
# 7. Select exactly one acquisition per subject.

# Expected MPRAGE properties
# --------------------------
# - 5 echoes
# - EchoNumbers: 1, 2, 3, 4, 5
# - EchoTimes approximately:
#       1.64
#       3.50
#       5.36
#       7.22
#       9.08 ms
# - 192 slices per echo
# - 960 DICOM files total
# - 256 x 256 matrix
# - 1 mm isotropic voxels
# - One SeriesInstanceUID
# - One StudyInstanceUID
# - Consistent geometry across echoes
# - No unreadable/corrupt DICOM files

# Output
# ------
# The script creates:

# processed_data/smri/mprage_selection/
#     mprage_selection_manifest.csv

# The manifest contains the selected MPRAGE series for each subject
# and the validation status of all candidates.

# The original DICOM dataset is NEVER modified.

# Usage
# -----
# Default:
#     python src\check_mprage_sessions.py

# Custom extracted dataset:
#     python src\check_mprage_sessions.py --root "path\to\extracted\dataset"

# Custom tolerance:
#     python src\check_mprage_sessions.py --te-tolerance 0.15
# ======================================================================
# """

# import os
# import csv
# import glob
# import argparse
# from collections import defaultdict

# import numpy as np
# import pydicom


# # ======================================================================
# # CONSTANTS
# # ======================================================================

# EXPECTED_ECHO_NUMBERS = {1, 2, 3, 4, 5}

# EXPECTED_ECHO_TIMES = {
#     1: 1.64,
#     2: 3.50,
#     3: 5.36,
#     4: 7.22,
#     5: 9.08,
# }

# EXPECTED_ROWS = 256
# EXPECTED_COLUMNS = 256

# EXPECTED_SLICES_PER_ECHO = 192
# EXPECTED_TOTAL_FILES = 960

# EXPECTED_PIXEL_SPACING = (1.0, 1.0)
# EXPECTED_SLICE_THICKNESS = 1.0

# TE_TOLERANCE_DEFAULT = 0.15
# GEOMETRY_TOLERANCE = 1e-3


# # ======================================================================
# # COMMAND-LINE ARGUMENTS
# # ======================================================================

# parser = argparse.ArgumentParser(
#     description=(
#         "Inspect and select one valid 5-echo MPRAGE acquisition "
#         "per COBRE subject."
#     )
# )

# parser.add_argument(
#     "--root",
#     type=str,
#     default=None,
#     help=(
#         "Root directory containing COBRE subject folders. "
#         "If omitted, the standard extracted COBRE location is used "
#         "only as a fallback."
#     ),
# )

# parser.add_argument(
#     "--output",
#     type=str,
#     default=None,
#     help=(
#         "Optional output directory for the selection manifest."
#     ),
# )

# parser.add_argument(
#     "--te-tolerance",
#     type=float,
#     default=TE_TOLERANCE_DEFAULT,
#     help=(
#         "Allowed absolute EchoTime difference in milliseconds. "
#         "Default: 0.15 ms."
#     ),
# )

# args = parser.parse_args()


# # ======================================================================
# # DATASET ROOT
# # ======================================================================

# if args.root is not None:

#     ROOT = os.path.abspath(args.root)

# else:

#     # Generic fallback for the currently extracted COBRE dataset.
#     # This is NOT tied to a particular ZIP logically; --root can be
#     # used for any extracted COBRE dataset.
#     ROOT = os.path.abspath(
#         os.path.join(
#             "temp_folder",
#             "human",
#             "dicom",
#             "triotim",
#             "vcalhoun",
#         )
#     )

#     # Automatically discover the COBRE dataset folder below vcalhoun.
#     if os.path.isdir(ROOT):

#         cobre_candidates = sorted(
#             [
#                 os.path.join(ROOT, name)
#                 for name in os.listdir(ROOT)
#                 if os.path.isdir(
#                     os.path.join(ROOT, name)
#                 )
#                 and name.lower().startswith("cobre_")
#             ]
#         )

#         if len(cobre_candidates) == 1:
#             ROOT = cobre_candidates[0]

#         elif len(cobre_candidates) > 1:

#             raise RuntimeError(
#                 "\nMultiple COBRE datasets were found below:\n"
#                 f"{ROOT}\n\n"
#                 "Please specify the dataset explicitly using:\n"
#                 "  --root \"path\\to\\cobre_dataset\""
#             )


# # ======================================================================
# # OUTPUT DIRECTORY
# # ======================================================================

# if args.output is not None:

#     OUTPUT_DIR = os.path.abspath(args.output)

# else:

#     OUTPUT_DIR = os.path.abspath(
#         os.path.join(
#             "processed_data",
#             "smri",
#             "mprage_selection",
#         )
#     )


# MANIFEST_PATH = os.path.join(
#     OUTPUT_DIR,
#     "mprage_selection_manifest.csv",
# )


# # ======================================================================
# # HEADER
# # ======================================================================

# print("=" * 80)
# print("COBRE MPRAGE 5-ECHO INSPECTION AND SELECTION")
# print("=" * 80)

# print("\nDataset root:")
# print(ROOT)

# print("\nSelection rule:")
# print("  Earliest session with a complete valid 5-echo MPRAGE.")
# print("  Multiple valid series in that session -> lowest SeriesNumber.")
# print("  If SeriesNumber is unavailable -> alphabetical path order.")


# # ======================================================================
# # ROOT VALIDATION
# # ======================================================================

# if not os.path.isdir(ROOT):

#     raise FileNotFoundError(
#         f"\nDataset root not found:\n{ROOT}\n\n"
#         "Use --root to specify the extracted COBRE dataset."
#     )


# # ======================================================================
# # SUBJECT DISCOVERY
# # ======================================================================

# subjects = sorted(
#     [
#         name
#         for name in os.listdir(ROOT)
#         if os.path.isdir(
#             os.path.join(ROOT, name)
#         )
#         and name.upper().startswith("A")
#     ]
# )


# if not subjects:

#     raise RuntimeError(
#         "\nNo COBRE subject folders were detected under:\n"
#         f"{ROOT}"
#     )


# print("\nSubjects detected:", len(subjects))


# # ======================================================================
# # HELPER FUNCTIONS
# # ======================================================================

# def natural_session_key(session_name):
#     """
#     Sort Session_1, Session_2, Session_10 numerically.
#     """

#     lower = session_name.lower()

#     if lower.startswith("session_"):

#         suffix = lower.replace(
#             "session_",
#             "",
#             1,
#         )

#         try:
#             return (
#                 0,
#                 int(suffix),
#                 lower,
#             )

#         except ValueError:
#             pass

#     return (
#         1,
#         lower,
#     )


# def safe_float(value):
#     """
#     Convert a DICOM value to float when possible.
#     """

#     try:
#         return float(value)

#     except Exception:
#         return None


# def get_series_number(files):
#     """
#     Obtain the most common SeriesNumber among readable DICOM files.
#     """

#     values = []

#     for path in files:

#         try:

#             dcm = pydicom.dcmread(
#                 path,
#                 stop_before_pixels=True,
#                 force=True,
#             )

#             value = getattr(
#                 dcm,
#                 "SeriesNumber",
#                 None,
#             )

#             if value is not None:

#                 try:
#                     values.append(int(value))

#                 except Exception:
#                     pass

#         except Exception:
#             continue

#     if not values:
#         return None

#     counts = defaultdict(int)

#     for value in values:
#         counts[value] += 1

#     return sorted(
#         counts,
#         key=lambda x: (-counts[x], x),
#     )[0]


# def validate_candidate(series_path):
#     """
#     Validate one candidate MPRAGE 5-echo DICOM series.

#     Returns a dictionary containing validation information.
#     """

#     result = {
#         "valid": False,
#         "reason": "",
#         "series_path": series_path,
#         "file_count": 0,
#         "series_number": "",
#         "study_uid": "",
#         "series_uid": "",
#         "echo_counts": {},
#         "echo_times": {},
#         "rows": "",
#         "columns": "",
#         "slice_counts": {},
#         "pixel_spacing": "",
#         "slice_thickness": "",
#         "corrupt_files": 0,
#     }

#     files = sorted(
#         [
#             f
#             for f in glob.glob(
#                 os.path.join(
#                     series_path,
#                     "*",
#                 )
#             )
#             if os.path.isfile(f)
#         ]
#     )

#     result["file_count"] = len(files)

#     if len(files) == 0:

#         result["reason"] = "EMPTY_SERIES"

#         return result

#     # --------------------------------------------------------------
#     # Basic containers
#     # --------------------------------------------------------------

#     echo_counts = defaultdict(int)
#     echo_times = defaultdict(list)
#     echo_positions = defaultdict(list)

#     series_uids = set()
#     study_uids = set()

#     rows_values = set()
#     columns_values = set()
#     pixel_spacing_values = set()
#     slice_thickness_values = set()

#     readable_files = 0

#     # --------------------------------------------------------------
#     # Read every DICOM header
#     # --------------------------------------------------------------

#     for path in files:

#         try:

#             dcm = pydicom.dcmread(
#                 path,
#                 stop_before_pixels=True,
#                 force=True,
#             )

#             readable_files += 1

#         except Exception:

#             result["corrupt_files"] += 1

#             continue

#         # ----------------------------------------------------------
#         # UIDs
#         # ----------------------------------------------------------

#         series_uid = getattr(
#             dcm,
#             "SeriesInstanceUID",
#             None,
#         )

#         if series_uid is not None:
#             series_uids.add(str(series_uid))

#         study_uid = getattr(
#             dcm,
#             "StudyInstanceUID",
#             None,
#         )

#         if study_uid is not None:
#             study_uids.add(str(study_uid))

#         # ----------------------------------------------------------
#         # Echo number
#         # ----------------------------------------------------------

#         echo_number = getattr(
#             dcm,
#             "EchoNumbers",
#             None,
#         )

#         try:
#             echo_number = int(echo_number)
#         except Exception:
#             echo_number = None

#         if echo_number is None:

#             continue

#         echo_counts[echo_number] += 1

#         # ----------------------------------------------------------
#         # Echo time
#         # ----------------------------------------------------------

#         echo_time = safe_float(
#             getattr(
#                 dcm,
#                 "EchoTime",
#                 None,
#             )
#         )

#         if echo_time is not None:

#             echo_times[echo_number].append(
#                 echo_time
#             )

#         # ----------------------------------------------------------
#         # Geometry
#         # ----------------------------------------------------------

#         rows = getattr(
#             dcm,
#             "Rows",
#             None,
#         )

#         columns = getattr(
#             dcm,
#             "Columns",
#             None,
#         )

#         if rows is not None:
#             rows_values.add(int(rows))

#         if columns is not None:
#             columns_values.add(int(columns))

#         pixel_spacing = getattr(
#             dcm,
#             "PixelSpacing",
#             None,
#         )

#         if pixel_spacing is not None:

#             try:

#                 pixel_spacing_values.add(
#                     (
#                         round(float(pixel_spacing[0]), 6),
#                         round(float(pixel_spacing[1]), 6),
#                     )
#                 )

#             except Exception:
#                 pass

#         slice_thickness = safe_float(
#             getattr(
#                 dcm,
#                 "SliceThickness",
#                 None,
#             )
#         )

#         if slice_thickness is not None:

#             slice_thickness_values.add(
#                 round(
#                     slice_thickness,
#                     6,
#                 )
#             )

#         # ----------------------------------------------------------
#         # Slice position
#         # ----------------------------------------------------------

#         image_position = getattr(
#             dcm,
#             "ImagePositionPatient",
#             None,
#         )

#         if image_position is not None:

#             try:

#                 position = tuple(
#                     round(float(x), 5)
#                     for x in image_position
#                 )

#                 echo_positions[
#                     echo_number
#                 ].append(position)

#             except Exception:
#                 pass

#     # ==================================================================
#     # VALIDATION
#     # ==================================================================

#     if result["corrupt_files"] > 0:

#         result["reason"] = (
#             f"CORRUPT_OR_UNREADABLE_FILES:"
#             f"{result['corrupt_files']}"
#         )

#         return result

#     if readable_files != len(files):

#         result["reason"] = "NOT_ALL_FILES_READABLE"

#         return result

#     if len(files) != EXPECTED_TOTAL_FILES:

#         result["reason"] = (
#             f"EXPECTED_{EXPECTED_TOTAL_FILES}_FILES_"
#             f"FOUND_{len(files)}"
#         )

#         return result

#     # --------------------------------------------------------------
#     # Echo numbers
#     # --------------------------------------------------------------

#     found_echoes = set(
#         echo_counts.keys()
#     )

#     if found_echoes != EXPECTED_ECHO_NUMBERS:

#         result["reason"] = (
#             "INVALID_ECHO_NUMBERS:"
#             f"{sorted(found_echoes)}"
#         )

#         return result

#     # --------------------------------------------------------------
#     # Equal number of slices per echo
#     # --------------------------------------------------------------

#     for echo_number in sorted(
#         EXPECTED_ECHO_NUMBERS
#     ):

#         count = echo_counts[
#             echo_number
#         ]

#         result["slice_counts"][
#             echo_number
#         ] = count

#         if count != EXPECTED_SLICES_PER_ECHO:

#             result["reason"] = (
#                 f"ECHO_{echo_number}_HAS_"
#                 f"{count}_FILES_EXPECTED_"
#                 f"{EXPECTED_SLICES_PER_ECHO}"
#             )

#             return result

#     # --------------------------------------------------------------
#     # Echo times
#     # --------------------------------------------------------------

#     for echo_number in sorted(
#         EXPECTED_ECHO_NUMBERS
#     ):

#         values = echo_times[
#             echo_number
#         ]

#         if not values:

#             result["reason"] = (
#                 f"ECHO_{echo_number}_MISSING_ECHOTIME"
#             )

#             return result

#         minimum = min(values)
#         maximum = max(values)

#         if (
#             abs(minimum - maximum)
#             > args.te_tolerance
#         ):

#             result["reason"] = (
#                 f"ECHO_{echo_number}_INCONSISTENT_ECHOTIME"
#             )

#             return result

#         measured = float(
#             np.median(values)
#         )

#         result["echo_times"][
#             echo_number
#         ] = measured

#         expected = EXPECTED_ECHO_TIMES[
#             echo_number
#         ]

#         if (
#             abs(measured - expected)
#             > args.te_tolerance
#         ):

#             result["reason"] = (
#                 f"ECHO_{echo_number}_TE_"
#                 f"{measured:.3f}_EXPECTED_"
#                 f"{expected:.3f}"
#             )

#             return result

#     # --------------------------------------------------------------
#     # TE ordering
#     # --------------------------------------------------------------

#     measured_te = [
#         result["echo_times"][echo]
#         for echo in sorted(
#             EXPECTED_ECHO_NUMBERS
#         )
#     ]

#     if any(
#         measured_te[i]
#         >= measured_te[i + 1]
#         for i in range(
#             len(measured_te) - 1
#         )
#     ):

#         result["reason"] = (
#             "ECHO_TIMES_NOT_STRICTLY_INCREASING"
#         )

#         return result

#     # --------------------------------------------------------------
#     # Matrix
#     # --------------------------------------------------------------

#     if rows_values != {EXPECTED_ROWS}:

#         result["reason"] = (
#             f"INVALID_ROWS:{sorted(rows_values)}"
#         )

#         return result

#     if columns_values != {EXPECTED_COLUMNS}:

#         result["reason"] = (
#             f"INVALID_COLUMNS:{sorted(columns_values)}"
#         )

#         return result

#     result["rows"] = EXPECTED_ROWS
#     result["columns"] = EXPECTED_COLUMNS

#     # --------------------------------------------------------------
#     # Pixel spacing
#     # --------------------------------------------------------------

#     if len(pixel_spacing_values) != 1:

#         result["reason"] = (
#             "INCONSISTENT_PIXEL_SPACING"
#         )

#         return result

#     pixel_spacing = next(
#         iter(pixel_spacing_values)
#     )

#     result["pixel_spacing"] = (
#         f"{pixel_spacing[0]:.6f},"
#         f"{pixel_spacing[1]:.6f}"
#     )

#     if (
#         abs(pixel_spacing[0] - 1.0)
#         > GEOMETRY_TOLERANCE
#         or
#         abs(pixel_spacing[1] - 1.0)
#         > GEOMETRY_TOLERANCE
#     ):

#         result["reason"] = (
#             f"PIXEL_SPACING_NOT_1MM:"
#             f"{pixel_spacing}"
#         )

#         return result

#     # --------------------------------------------------------------
#     # Slice thickness
#     # --------------------------------------------------------------

#     if len(slice_thickness_values) != 1:

#         result["reason"] = (
#             "INCONSISTENT_SLICE_THICKNESS"
#         )

#         return result

#     slice_thickness = next(
#         iter(slice_thickness_values)
#     )

#     result["slice_thickness"] = (
#         f"{slice_thickness:.6f}"
#     )

#     if (
#         abs(
#             slice_thickness
#             - EXPECTED_SLICE_THICKNESS
#         )
#         > GEOMETRY_TOLERANCE
#     ):

#         result["reason"] = (
#             f"SLICE_THICKNESS_NOT_1MM:"
#             f"{slice_thickness}"
#         )

#         return result

#     # --------------------------------------------------------------
#     # UID consistency
#     # --------------------------------------------------------------

#     if len(series_uids) != 1:

#         result["reason"] = (
#             f"INVALID_SERIES_UID_COUNT:"
#             f"{len(series_uids)}"
#         )

#         return result

#     if len(study_uids) != 1:

#         result["reason"] = (
#             f"INVALID_STUDY_UID_COUNT:"
#             f"{len(study_uids)}"
#         )

#         return result

#     result["series_uid"] = next(
#         iter(series_uids)
#     )

#     result["study_uid"] = next(
#         iter(study_uids)
#     )

#     # --------------------------------------------------------------
#     # Cross-echo spatial consistency
#     # --------------------------------------------------------------

#     position_sets = {}

#     for echo_number in sorted(
#         EXPECTED_ECHO_NUMBERS
#     ):

#         positions = echo_positions[
#             echo_number
#         ]

#         if len(positions) != EXPECTED_SLICES_PER_ECHO:

#             result["reason"] = (
#                 f"ECHO_{echo_number}_INVALID_POSITION_COUNT:"
#                 f"{len(positions)}"
#             )

#             return result

#         position_sets[
#             echo_number
#         ] = set(positions)

#     reference_positions = position_sets[1]

#     if len(reference_positions) != EXPECTED_SLICES_PER_ECHO:

#         result["reason"] = (
#             "ECHO_1_HAS_DUPLICATE_SLICE_POSITIONS"
#         )

#         return result

#     for echo_number in sorted(
#         EXPECTED_ECHO_NUMBERS
#     )[1:]:

#         if (
#             position_sets[echo_number]
#             != reference_positions
#         ):

#             result["reason"] = (
#                 "CROSS_ECHO_SLICE_POSITIONS_MISMATCH"
#             )

#             return result

#     # --------------------------------------------------------------
#     # Series number
#     # --------------------------------------------------------------

#     result["series_number"] = (
#         get_series_number(files)
#         if files
#         else None
#     )

#     # --------------------------------------------------------------
#     # PASS
#     # --------------------------------------------------------------

#     result["valid"] = True
#     result["reason"] = "PASS"

#     return result


# # ======================================================================
# # DISCOVER ALL CANDIDATES
# # ======================================================================

# all_candidates = []
# selected_candidates = []

# subjects_with_mprage = 0
# subjects_without_mprage = 0

# valid_candidates_count = 0
# invalid_candidates_count = 0


# print("\n" + "=" * 80)
# print("INSPECTING SUBJECTS")
# print("=" * 80)


# for subject in subjects:

#     subject_path = os.path.join(
#         ROOT,
#         subject,
#     )

#     print("\n" + "=" * 80)
#     print(f"SUBJECT: {subject}")
#     print("=" * 80)

#     # --------------------------------------------------------------
#     # Find ALL sessions
#     # --------------------------------------------------------------

#     sessions = sorted(
#         [
#             name
#             for name in os.listdir(
#                 subject_path
#             )
#             if os.path.isdir(
#                 os.path.join(
#                     subject_path,
#                     name,
#                 )
#             )
#             and name.lower().startswith(
#                 "session_"
#             )
#         ],
#         key=natural_session_key,
#     )

#     if not sessions:

#         print("No Session_* directories found.")

#         subjects_without_mprage += 1

#         continue

#     print(
#         "Sessions found:",
#         ", ".join(sessions),
#     )

#     subject_candidates = []

#     # --------------------------------------------------------------
#     # Inspect every session
#     # --------------------------------------------------------------

#     for session in sessions:

#         session_path = os.path.join(
#             subject_path,
#             session,
#         )

#         series_candidates = sorted(
#             [
#                 name
#                 for name in os.listdir(
#                     session_path
#                 )
#                 if os.path.isdir(
#                     os.path.join(
#                         session_path,
#                         name,
#                     )
#                 )
#                 and "mprage_5e"
#                 in name.lower()
#             ]
#         )

#         print(
#             f"\n  Session: {session}"
#         )

#         if not series_candidates:

#             print(
#                 "    MPRAGE 5e: NOT FOUND"
#             )

#             continue

#         print(
#             "    MPRAGE 5e candidates:",
#             len(series_candidates),
#         )

#         # ----------------------------------------------------------
#         # Validate every candidate
#         # ----------------------------------------------------------

#         for series in series_candidates:

#             series_path = os.path.join(
#                 session_path,
#                 series,
#             )

#             print(
#                 f"\n    Series: {series}"
#             )

#             result = validate_candidate(
#                 series_path
#             )

#             result["subject"] = subject
#             result["session"] = session
#             result["series"] = series

#             all_candidates.append(
#                 result
#             )

#             if result["valid"]:

#                 valid_candidates_count += 1

#                 subject_candidates.append(
#                     result
#                 )

#                 print(
#                     "      STATUS: PASS"
#                 )

#                 print(
#                     "      Files:",
#                     result["file_count"],
#                 )

#                 print(
#                     "      Series Number:",
#                     result["series_number"],
#                 )

#                 print(
#                     "      Echo Times:",
#                     [
#                         round(
#                             result["echo_times"][e],
#                             3,
#                         )
#                         for e in sorted(
#                             EXPECTED_ECHO_NUMBERS
#                         )
#                     ],
#                 )

#             else:

#                 invalid_candidates_count += 1

#                 print(
#                     "      STATUS: FAIL"
#                 )

#                 print(
#                     "      Reason:",
#                     result["reason"],
#                 )

#     # --------------------------------------------------------------
#     # Select ONE candidate for subject
#     # --------------------------------------------------------------

#     if not subject_candidates:

#         subjects_without_mprage += 1

#         print(
#             "\n  FINAL SUBJECT STATUS: "
#             "NO VALID 5-ECHO MPRAGE"
#         )

#         continue

#     subjects_with_mprage += 1

#     # --------------------------------------------------------------
#     # Earliest session first
#     # --------------------------------------------------------------

#     subject_candidates.sort(
#         key=lambda item: (
#             natural_session_key(
#                 item["session"]
#             ),
#             (
#                 item["series_number"]
#                 if item["series_number"]
#                 is not None
#                 else float("inf")
#             ),
#             item["series"].lower(),
#         )
#     )

#     selected = subject_candidates[0]

#     selected["selection_status"] = "SELECTED"

#     selected_candidates.append(
#         selected
#     )

#     print(
#         "\n  FINAL SELECTION:"
#     )

#     print(
#         "    Session:",
#         selected["session"],
#     )

#     print(
#         "    Series:",
#         selected["series"],
#     )

#     print(
#         "    Series Number:",
#         selected["series_number"],
#     )


# # ======================================================================
# # SAVE MANIFEST
# # ======================================================================

# os.makedirs(
#     OUTPUT_DIR,
#     exist_ok=True,
# )


# fieldnames = [
#     "subject",
#     "session",
#     "series",
#     "series_path",
#     "selection_status",
#     "valid",
#     "reason",
#     "file_count",
#     "series_number",
#     "study_uid",
#     "series_uid",
#     "rows",
#     "columns",
#     "slice_thickness",
#     "pixel_spacing",
#     "corrupt_files",
#     "echo_1_files",
#     "echo_2_files",
#     "echo_3_files",
#     "echo_4_files",
#     "echo_5_files",
#     "echo_1_te_ms",
#     "echo_2_te_ms",
#     "echo_3_te_ms",
#     "echo_4_te_ms",
#     "echo_5_te_ms",
# ]


# with open(
#     MANIFEST_PATH,
#     "w",
#     newline="",
#     encoding="utf-8",
# ) as csv_file:

#     writer = csv.DictWriter(
#         csv_file,
#         fieldnames=fieldnames,
#     )

#     writer.writeheader()

#     # --------------------------------------------------------------
#     # Write ALL inspected candidates
#     # --------------------------------------------------------------

#     selected_keys = {
#         (
#             item["subject"],
#             item["session"],
#             item["series"],
#         )
#         for item in selected_candidates
#     }

#     for item in all_candidates:

#         key = (
#             item["subject"],
#             item["session"],
#             item["series"],
#         )

#         selection_status = (
#             "SELECTED"
#             if key in selected_keys
#             else (
#                 "VALID_NOT_SELECTED"
#                 if item["valid"]
#                 else "INVALID"
#             )
#         )

#         writer.writerow(
#             {
#                 "subject": item.get(
#                     "subject",
#                     "",
#                 ),
#                 "session": item.get(
#                     "session",
#                     "",
#                 ),
#                 "series": item.get(
#                     "series",
#                     "",
#                 ),
#                 "series_path": item.get(
#                     "series_path",
#                     "",
#                 ),
#                 "selection_status": (
#                     selection_status
#                 ),
#                 "valid": item.get(
#                     "valid",
#                     False,
#                 ),
#                 "reason": item.get(
#                     "reason",
#                     "",
#                 ),
#                 "file_count": item.get(
#                     "file_count",
#                     "",
#                 ),
#                 "series_number": item.get(
#                     "series_number",
#                     "",
#                 ),
#                 "study_uid": item.get(
#                     "study_uid",
#                     "",
#                 ),
#                 "series_uid": item.get(
#                     "series_uid",
#                     "",
#                 ),
#                 "rows": item.get(
#                     "rows",
#                     "",
#                 ),
#                 "columns": item.get(
#                     "columns",
#                     "",
#                 ),
#                 "slice_thickness": item.get(
#                     "slice_thickness",
#                     "",
#                 ),
#                 "pixel_spacing": item.get(
#                     "pixel_spacing",
#                     "",
#                 ),
#                 "corrupt_files": item.get(
#                     "corrupt_files",
#                     "",
#                 ),
#                 "echo_1_files": item.get(
#                     "slice_counts",
#                     {},
#                 ).get(1, ""),
#                 "echo_2_files": item.get(
#                     "slice_counts",
#                     {},
#                 ).get(2, ""),
#                 "echo_3_files": item.get(
#                     "slice_counts",
#                     {},
#                 ).get(3, ""),
#                 "echo_4_files": item.get(
#                     "slice_counts",
#                     {},
#                 ).get(4, ""),
#                 "echo_5_files": item.get(
#                     "slice_counts",
#                     {},
#                 ).get(5, ""),
#                 "echo_1_te_ms": item.get(
#                     "echo_times",
#                     {},
#                 ).get(1, ""),
#                 "echo_2_te_ms": item.get(
#                     "echo_times",
#                     {},
#                 ).get(2, ""),
#                 "echo_3_te_ms": item.get(
#                     "echo_times",
#                     {},
#                 ).get(3, ""),
#                 "echo_4_te_ms": item.get(
#                     "echo_times",
#                     {},
#                 ).get(4, ""),
#                 "echo_5_te_ms": item.get(
#                     "echo_times",
#                     {},
#                 ).get(5, ""),
#             }
#         )


# # ======================================================================
# # FINAL SUMMARY
# # ======================================================================

# print("\n" + "=" * 80)
# print("MPRAGE INSPECTION SUMMARY")
# print("=" * 80)

# print(
#     "Total subjects detected       :",
#     len(subjects),
# )

# print(
#     "Subjects with valid MPRAGE    :",
#     subjects_with_mprage,
# )

# print(
#     "Subjects without valid MPRAGE :",
#     subjects_without_mprage,
# )

# print(
#     "Total MPRAGE candidates       :",
#     len(all_candidates),
# )

# print(
#     "Valid MPRAGE candidates       :",
#     valid_candidates_count,
# )

# print(
#     "Invalid MPRAGE candidates     :",
#     invalid_candidates_count,
# )

# print(
#     "Selected acquisitions         :",
#     len(selected_candidates),
# )

# print(
#     "Expected one per selected subject:",
#     len(selected_candidates),
# )

# print(
#     "\nSelection manifest:"
# )

# print(
#     MANIFEST_PATH
# )

# print("\n" + "=" * 80)
# print("CHECK COMPLETED")
# print("=" * 80)

# src/check_mprage_sessions.py

"""
COBRE MPRAGE INSPECTION AND SELECTION

Supports:
    1. 5ECHO MPRAGE
    2. RMS MPRAGE

Selection:
    1. Earliest session containing a valid MPRAGE.
    2. Within that session, prefer 5ECHO over RMS.
    3. Within the same type, choose the lowest SeriesNumber.
    4. Use series name as deterministic tie-breaker.

The raw DICOM dataset is never modified.
"""

import os
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import pydicom


# ======================================================================
# CONSTANTS
# ======================================================================

EXPECTED_ECHO_NUMBERS = {1, 2, 3, 4, 5}

EXPECTED_ECHO_TIMES = {
    1: 1.64,
    2: 3.50,
    3: 5.36,
    4: 7.22,
    5: 9.08,
}

EXPECTED_ROWS = 256
EXPECTED_COLUMNS = 256
EXPECTED_SLICES = 192

EXPECTED_5ECHO_FILES = 960
EXPECTED_RMS_FILES = 192

EXPECTED_PIXEL_SPACING = (1.0, 1.0)
EXPECTED_SLICE_THICKNESS = 1.0

TE_TOLERANCE = 0.15
GEOMETRY_TOLERANCE = 1e-3

ACQUISITION_PRIORITY = {
    "5ECHO": 0,
    "RMS": 1,
}


# ======================================================================
# ARGUMENTS
# ======================================================================

parser = argparse.ArgumentParser(
    description=(
        "Inspect and select one valid COBRE MPRAGE acquisition "
        "per subject."
    )
)

parser.add_argument(
    "--root",
    type=str,
    default=None,
    help="COBRE dataset root.",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for the manifest.",
)

args = parser.parse_args()


# ======================================================================
# DATASET ROOT
# ======================================================================

if args.root:

    ROOT = os.path.abspath(args.root)

else:

    ROOT = os.path.abspath(
        os.path.join(
            "temp_folder",
            "human",
            "dicom",
            "triotim",
            "vcalhoun",
        )
    )

    if os.path.isdir(ROOT):

        cobre_dirs = sorted(
            [
                os.path.join(ROOT, name)
                for name in os.listdir(ROOT)
                if os.path.isdir(
                    os.path.join(ROOT, name)
                )
                and name.lower().startswith("cobre_")
            ]
        )

        if len(cobre_dirs) == 1:

            ROOT = cobre_dirs[0]

        elif len(cobre_dirs) > 1:

            raise RuntimeError(
                "Multiple COBRE datasets found. "
                "Use --root to specify one dataset."
            )


# ======================================================================
# OUTPUT
# ======================================================================

if args.output:

    OUTPUT_DIR = os.path.abspath(args.output)

else:

    OUTPUT_DIR = os.path.abspath(
        os.path.join(
            "processed_data",
            "smri",
            "mprage_selection",
        )
    )

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "mprage_selection_manifest.csv",
)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("COBRE MPRAGE INSPECTION AND SELECTION")
print("=" * 80)

print("\nDataset root:")
print(ROOT)

print("\nSupported acquisition types:")
print("  5ECHO : 960 files, five echoes")
print("  RMS   : 192 files, reconstructed structural volume")

print("\nSelection rule:")
print("  1. Earliest session with a valid MPRAGE.")
print("  2. 5ECHO preferred over RMS within that session.")
print("  3. Lowest SeriesNumber.")
print("  4. Series name as tie-breaker.")
print("  5. Exactly one acquisition per subject.")


# ======================================================================
# ROOT CHECK
# ======================================================================

if not os.path.isdir(ROOT):

    raise FileNotFoundError(
        f"Dataset root does not exist:\n{ROOT}"
    )


# ======================================================================
# SUBJECT DISCOVERY
# ======================================================================

subjects = sorted(
    [
        name
        for name in os.listdir(ROOT)
        if os.path.isdir(
            os.path.join(ROOT, name)
        )
        and name.upper().startswith("A")
    ]
)

if not subjects:

    raise RuntimeError(
        f"No subject folders found under:\n{ROOT}"
    )

print("\nSubjects detected:", len(subjects))


# ======================================================================
# HELPERS
# ======================================================================

def session_sort_key(name):
    """
    Sort Session_1, Session_2, Session_10 numerically.
    """

    lower = name.lower()

    if lower.startswith("session_"):

        value = lower.replace(
            "session_",
            "",
            1,
        )

        try:
            return (
                0,
                int(value),
                lower,
            )
        except ValueError:
            pass

    return (
        1,
        lower,
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def get_series_number(headers):
    """
    Get the most common SeriesNumber.

    Returns None if unavailable.
    """

    values = []

    for dcm in headers:

        value = getattr(
            dcm,
            "SeriesNumber",
            None,
        )

        try:
            values.append(int(value))
        except Exception:
            pass

    if not values:
        return None

    counts = defaultdict(int)

    for value in values:
        counts[value] += 1

    return sorted(
        counts,
        key=lambda x: (-counts[x], x),
    )[0]


def get_files(series_path):
    """
    Return direct files inside the series directory.
    """

    return sorted(
        [
            path
            for path in glob.glob(
                os.path.join(
                    series_path,
                    "*",
                )
            )
            if os.path.isfile(path)
        ]
    )


def read_headers(files):
    """
    Read DICOM headers without loading pixel data.
    """

    headers = []
    corrupt = 0

    for path in files:

        try:

            dcm = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True,
            )

            headers.append(dcm)

        except Exception:

            corrupt += 1

    return headers, corrupt


def get_unique_attribute_values(
    headers,
    attribute,
):
    """
    Return non-empty unique values for a DICOM attribute.
    """

    values = set()

    for dcm in headers:

        value = getattr(
            dcm,
            attribute,
            None,
        )

        if value is not None:

            text = str(value).strip()

            if text:
                values.add(text)

    return values


def validate_geometry(
    headers,
    result,
):
    """
    Validate common 256x256, 1mm geometry.
    """

    rows = set()
    columns = set()
    spacings = set()
    thicknesses = set()

    for dcm in headers:

        try:
            rows.add(
                int(
                    getattr(
                        dcm,
                        "Rows",
                    )
                )
            )
        except Exception:
            pass

        try:
            columns.add(
                int(
                    getattr(
                        dcm,
                        "Columns",
                    )
                )
            )
        except Exception:
            pass

        spacing = getattr(
            dcm,
            "PixelSpacing",
            None,
        )

        if spacing is not None:

            try:
                spacings.add(
                    (
                        round(
                            float(spacing[0]),
                            6,
                        ),
                        round(
                            float(spacing[1]),
                            6,
                        ),
                    )
                )
            except Exception:
                pass

        thickness = safe_float(
            getattr(
                dcm,
                "SliceThickness",
                None,
            )
        )

        if thickness is not None:
            thicknesses.add(
                round(
                    thickness,
                    6,
                )
            )

    if rows != {EXPECTED_ROWS}:

        result["reason"] = (
            f"INVALID_ROWS:{sorted(rows)}"
        )

        return False

    if columns != {EXPECTED_COLUMNS}:

        result["reason"] = (
            f"INVALID_COLUMNS:{sorted(columns)}"
        )

        return False

    if len(spacings) != 1:

        result["reason"] = (
            "INCONSISTENT_PIXEL_SPACING"
        )

        return False

    spacing = next(iter(spacings))

    if (
        abs(
            spacing[0]
            - EXPECTED_PIXEL_SPACING[0]
        )
        > GEOMETRY_TOLERANCE
        or
        abs(
            spacing[1]
            - EXPECTED_PIXEL_SPACING[1]
        )
        > GEOMETRY_TOLERANCE
    ):

        result["reason"] = (
            f"PIXEL_SPACING_NOT_1MM:{spacing}"
        )

        return False

    if len(thicknesses) != 1:

        result["reason"] = (
            "INCONSISTENT_SLICE_THICKNESS"
        )

        return False

    thickness = next(
        iter(thicknesses)
    )

    if (
        abs(
            thickness
            - EXPECTED_SLICE_THICKNESS
        )
        > GEOMETRY_TOLERANCE
    ):

        result["reason"] = (
            f"SLICE_THICKNESS_NOT_1MM:"
            f"{thickness}"
        )

        return False

    result["rows"] = EXPECTED_ROWS
    result["columns"] = EXPECTED_COLUMNS
    result["pixel_spacing"] = (
        f"{spacing[0]:.6f},{spacing[1]:.6f}"
    )
    result["slice_thickness"] = (
        f"{thickness:.6f}"
    )

    return True


def validate_slice_positions(
    headers,
    expected_count,
    result,
):
    """
    Require the expected number of unique
    ImagePositionPatient locations.
    """

    positions = []

    for dcm in headers:

        value = getattr(
            dcm,
            "ImagePositionPatient",
            None,
        )

        if value is None:

            result["reason"] = (
                "MISSING_IMAGE_POSITION"
            )

            return False

        try:

            position = tuple(
                round(
                    float(x),
                    5,
                )
                for x in value
            )

            if len(position) != 3:

                raise ValueError

            positions.append(position)

        except Exception:

            result["reason"] = (
                "INVALID_IMAGE_POSITION"
            )

            return False

    unique_positions = set(
        positions
    )

    if len(unique_positions) != expected_count:

        result["reason"] = (
            "INVALID_UNIQUE_SLICE_POSITION_COUNT:"
            f"{len(unique_positions)}"
        )

        return False

    return True


def validate_orientation(
    headers,
    result,
):
    """
    Ensure all slices have the same ImageOrientationPatient.
    """

    orientations = set()

    for dcm in headers:

        value = getattr(
            dcm,
            "ImageOrientationPatient",
            None,
        )

        if value is None:
            continue

        try:

            orientation = tuple(
                round(
                    float(x),
                    5,
                )
                for x in value
            )

            orientations.add(
                orientation
            )

        except Exception:

            pass

    if len(orientations) > 1:

        result["reason"] = (
            "INCONSISTENT_IMAGE_ORIENTATION"
        )

        return False

    return True


# ======================================================================
# RMS VALIDATION
# ======================================================================

def validate_rms(
    series_path,
    files,
    headers,
    result,
):
    """
    Validate reconstructed RMS MPRAGE.
    """

    result["acquisition_type"] = "RMS"

    # --------------------------------------------------------------
    # File count
    # --------------------------------------------------------------

    if len(files) != EXPECTED_RMS_FILES:

        result["reason"] = (
            f"RMS_EXPECTED_{EXPECTED_RMS_FILES}_FILES_"
            f"FOUND_{len(files)}"
        )

        return False

    # --------------------------------------------------------------
    # Echo number
    # --------------------------------------------------------------

    echo_numbers = set()

    for dcm in headers:

        value = getattr(
            dcm,
            "EchoNumbers",
            None,
        )

        try:
            echo_numbers.add(
                int(value)
            )
        except Exception:

            result["reason"] = (
                "RMS_INVALID_ECHO_NUMBER"
            )

            return False

    if echo_numbers != {1}:

        result["reason"] = (
            "RMS_INVALID_ECHO_NUMBERS:"
            f"{sorted(echo_numbers)}"
        )

        return False

    # --------------------------------------------------------------
    # Echo time
    # --------------------------------------------------------------

    echo_times = []

    for dcm in headers:

        value = safe_float(
            getattr(
                dcm,
                "EchoTime",
                None,
            )
        )

        if value is None:

            result["reason"] = (
                "RMS_MISSING_ECHOTIME"
            )

            return False

        echo_times.append(value)

    minimum = min(echo_times)
    maximum = max(echo_times)

    if (
        abs(
            maximum
            - minimum
        )
        > TE_TOLERANCE
    ):

        result["reason"] = (
            "RMS_INCONSISTENT_ECHOTIME"
        )

        return False

    measured_te = float(
        np.median(echo_times)
    )

    result["echo_1_te_ms"] = measured_te
    result["echo_1_files"] = len(files)

    if (
        abs(
            measured_te
            - EXPECTED_ECHO_TIMES[1]
        )
        > TE_TOLERANCE
    ):

        result["reason"] = (
            f"RMS_TE_{measured_te:.3f}_"
            f"EXPECTED_{EXPECTED_ECHO_TIMES[1]:.3f}"
        )

        return False

    # --------------------------------------------------------------
    # Reconstruction identity
    # --------------------------------------------------------------

    series_name = os.path.basename(
        series_path
    ).upper()

    descriptions = [
        str(
            getattr(
                dcm,
                "SeriesDescription",
                "",
            )
        ).upper()
        for dcm in headers
    ]

    image_types = []

    for dcm in headers:

        value = getattr(
            dcm,
            "ImageType",
            None,
        )

        if value is not None:

            image_types.append(
                [
                    str(x).upper()
                    for x in value
                ]
            )

    has_rms_name = (
        "RMS" in series_name
        or any(
            "RMS" in value
            for value in descriptions
        )
    )

    has_mean = any(
        "MEAN" in image_type
        for image_type in image_types
    )

    if not (
        has_rms_name
        or has_mean
    ):

        result["reason"] = (
            "RMS_RECONSTRUCTION_NOT_CONFIRMED"
        )

        return False

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------

    if not validate_geometry(
        headers,
        result,
    ):

        return False

    # --------------------------------------------------------------
    # Slice geometry
    # --------------------------------------------------------------

    if not validate_slice_positions(
        headers,
        EXPECTED_SLICES,
        result,
    ):

        return False

    if not validate_orientation(
        headers,
        result,
    ):

        return False

    # --------------------------------------------------------------
    # Series / study identity
    #
    # IMPORTANT:
    # UID tags are recorded when present, but their absence does
    # NOT automatically invalidate this COBRE RMS representation.
    # --------------------------------------------------------------

    series_uids = get_unique_attribute_values(
        headers,
        "SeriesInstanceUID",
    )

    study_uids = get_unique_attribute_values(
        headers,
        "StudyInstanceUID",
    )

    if len(series_uids) == 1:

        result["series_uid"] = next(
            iter(series_uids)
        )

    elif len(series_uids) > 1:

        result["reason"] = (
            "INCONSISTENT_SERIES_UIDS"
        )

        return False

    else:

        result["series_uid"] = (
            "NOT_PRESENT"
        )

    if len(study_uids) == 1:

        result["study_uid"] = next(
            iter(study_uids)
        )

    elif len(study_uids) > 1:

        result["reason"] = (
            "INCONSISTENT_STUDY_UIDS"
        )

        return False

    else:

        result["study_uid"] = (
            "NOT_PRESENT"
        )

    return True


# ======================================================================
# 5-ECHO VALIDATION
# ======================================================================

def validate_5echo(
    series_path,
    files,
    headers,
    result,
):
    """
    Validate complete five-echo MPRAGE.
    """

    result["acquisition_type"] = "5ECHO"

    # --------------------------------------------------------------
    # File count
    # --------------------------------------------------------------

    if len(files) != EXPECTED_5ECHO_FILES:

        result["reason"] = (
            f"5ECHO_EXPECTED_{EXPECTED_5ECHO_FILES}_FILES_"
            f"FOUND_{len(files)}"
        )

        return False

    echo_counts = defaultdict(int)
    echo_times = defaultdict(list)
    echo_positions = defaultdict(list)

    # --------------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------------

    for dcm in headers:

        try:

            echo = int(
                getattr(
                    dcm,
                    "EchoNumbers",
                )
            )

        except Exception:

            result["reason"] = (
                "INVALID_ECHO_NUMBER"
            )

            return False

        if echo not in EXPECTED_ECHO_NUMBERS:

            result["reason"] = (
                f"UNEXPECTED_ECHO_NUMBER:{echo}"
            )

            return False

        echo_counts[echo] += 1

        te = safe_float(
            getattr(
                dcm,
                "EchoTime",
                None,
            )
        )

        if te is None:

            result["reason"] = (
                f"ECHO_{echo}_MISSING_ECHOTIME"
            )

            return False

        echo_times[echo].append(te)

        position = getattr(
            dcm,
            "ImagePositionPatient",
            None,
        )

        if position is None:

            result["reason"] = (
                f"ECHO_{echo}_"
                "MISSING_IMAGE_POSITION"
            )

            return False

        try:

            echo_positions[echo].append(
                tuple(
                    round(
                        float(x),
                        5,
                    )
                    for x in position
                )
            )

        except Exception:

            result["reason"] = (
                f"ECHO_{echo}_"
                "INVALID_IMAGE_POSITION"
            )

            return False

    # --------------------------------------------------------------
    # Echo numbers
    # --------------------------------------------------------------

    if set(echo_counts) != EXPECTED_ECHO_NUMBERS:

        result["reason"] = (
            "INVALID_ECHO_NUMBERS:"
            f"{sorted(echo_counts)}"
        )

        return False

    # --------------------------------------------------------------
    # Exactly 192 files per echo
    # --------------------------------------------------------------

    for echo in sorted(
        EXPECTED_ECHO_NUMBERS
    ):

        count = echo_counts[echo]

        result["slice_counts"][echo] = count

        if count != EXPECTED_SLICES:

            result["reason"] = (
                f"ECHO_{echo}_HAS_{count}_FILES_"
                f"EXPECTED_{EXPECTED_SLICES}"
            )

            return False

    # --------------------------------------------------------------
    # Echo times
    # --------------------------------------------------------------

    for echo in sorted(
        EXPECTED_ECHO_NUMBERS
    ):

        values = echo_times[echo]

        if (
            max(values)
            - min(values)
            > TE_TOLERANCE
        ):

            result["reason"] = (
                f"ECHO_{echo}_INCONSISTENT_ECHOTIME"
            )

            return False

        measured = float(
            np.median(values)
        )

        expected = EXPECTED_ECHO_TIMES[echo]

        result["echo_times"][echo] = measured

        if (
            abs(
                measured
                - expected
            )
            > TE_TOLERANCE
        ):

            result["reason"] = (
                f"ECHO_{echo}_TE_{measured:.3f}_"
                f"EXPECTED_{expected:.3f}"
            )

            return False

    # --------------------------------------------------------------
    # Echo times must increase
    # --------------------------------------------------------------

    tes = [
        result["echo_times"][echo]
        for echo in sorted(
            EXPECTED_ECHO_NUMBERS
        )
    ]

    if any(
        tes[i] >= tes[i + 1]
        for i in range(
            len(tes) - 1
        )
    ):

        result["reason"] = (
            "ECHO_TIMES_NOT_INCREASING"
        )

        return False

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------

    if not validate_geometry(
        headers,
        result,
    ):

        return False

    if not validate_orientation(
        headers,
        result,
    ):

        return False

    # --------------------------------------------------------------
    # Slice positions within each echo
    # --------------------------------------------------------------

    position_sets = {}

    for echo in sorted(
        EXPECTED_ECHO_NUMBERS
    ):

        positions = echo_positions[echo]

        if len(
            set(positions)
        ) != EXPECTED_SLICES:

            result["reason"] = (
                f"ECHO_{echo}_"
                "DUPLICATE_SLICE_POSITIONS"
            )

            return False

        position_sets[echo] = set(
            positions
        )

    # --------------------------------------------------------------
    # Cross-echo position consistency
    # --------------------------------------------------------------

    reference = position_sets[1]

    for echo in sorted(
        EXPECTED_ECHO_NUMBERS
    )[1:]:

        if (
            position_sets[echo]
            != reference
        ):

            result["reason"] = (
                "CROSS_ECHO_SLICE_POSITIONS_MISMATCH"
            )

            return False

    # --------------------------------------------------------------
    # UIDs
    #
    # For 5ECHO retain the strict UID consistency check.
    # --------------------------------------------------------------

    series_uids = get_unique_attribute_values(
        headers,
        "SeriesInstanceUID",
    )

    study_uids = get_unique_attribute_values(
        headers,
        "StudyInstanceUID",
    )

    if len(series_uids) != 1:

        result["reason"] = (
            "INVALID_SERIES_UID_COUNT:"
            f"{len(series_uids)}"
        )

        return False

    if len(study_uids) != 1:

        result["reason"] = (
            "INVALID_STUDY_UID_COUNT:"
            f"{len(study_uids)}"
        )

        return False

    result["series_uid"] = next(
        iter(series_uids)
    )

    result["study_uid"] = next(
        iter(study_uids)
    )

    return True


# ======================================================================
# CANDIDATE VALIDATION
# ======================================================================

def validate_candidate(series_path):

    result = {
        "valid": False,
        "reason": "",
        "acquisition_type": "",
        "series_path": series_path,
        "file_count": 0,
        "series_number": "",
        "study_uid": "",
        "series_uid": "",
        "rows": "",
        "columns": "",
        "slice_thickness": "",
        "pixel_spacing": "",
        "corrupt_files": 0,
        "slice_counts": {},
        "echo_times": {},
        "echo_1_files": "",
        "echo_1_te_ms": "",
    }

    files = get_files(
        series_path
    )

    result["file_count"] = len(files)

    if not files:

        result["reason"] = "EMPTY_SERIES"

        return result

    headers, corrupt = read_headers(
        files
    )

    result["corrupt_files"] = corrupt

    if corrupt:

        result["reason"] = (
            f"CORRUPT_OR_UNREADABLE_FILES:{corrupt}"
        )

        return result

    if len(headers) != len(files):

        result["reason"] = (
            "NOT_ALL_FILES_READABLE"
        )

        return result

    result["series_number"] = (
        get_series_number(headers)
    )

    # --------------------------------------------------------------
    # Determine acquisition representation
    # --------------------------------------------------------------

    series_name = os.path.basename(
        series_path
    ).lower()

    descriptions = [
        str(
            getattr(
                dcm,
                "SeriesDescription",
                "",
            )
        ).lower()
        for dcm in headers
    ]

    image_types = []

    for dcm in headers:

        value = getattr(
            dcm,
            "ImageType",
            None,
        )

        if value is not None:

            image_types.extend(
                str(x).lower()
                for x in value
            )

    is_rms = (
        "rms" in series_name
        or any(
            "rms" in x
            for x in descriptions
        )
        or "mean" in image_types
    )

    # --------------------------------------------------------------
    # Validate RMS
    # --------------------------------------------------------------

    if is_rms:

        passed = validate_rms(
            series_path,
            files,
            headers,
            result,
        )

    # --------------------------------------------------------------
    # Validate 5-ECHO
    # --------------------------------------------------------------

    else:

        passed = validate_5echo(
            series_path,
            files,
            headers,
            result,
        )

    result["valid"] = bool(
        passed
    )

    if passed:

        result["reason"] = "PASS"

    return result


# ======================================================================
# INSPECTION
# ======================================================================

all_candidates = []
selected_candidates = []

subjects_with_mprage = 0
subjects_without_mprage = 0

valid_candidates = 0
invalid_candidates = 0

valid_5echo = 0
valid_rms = 0


print("\n" + "=" * 80)
print("INSPECTING SUBJECTS")
print("=" * 80)


for subject in subjects:

    subject_path = os.path.join(
        ROOT,
        subject,
    )

    print("\n" + "=" * 80)
    print(f"SUBJECT: {subject}")
    print("=" * 80)

    sessions = sorted(
        [
            name
            for name in os.listdir(
                subject_path
            )
            if os.path.isdir(
                os.path.join(
                    subject_path,
                    name,
                )
            )
            and name.lower().startswith(
                "session_"
            )
        ],
        key=session_sort_key,
    )

    if not sessions:

        print(
            "No Session_* directories found."
        )

        subjects_without_mprage += 1

        continue

    print(
        "Sessions found:",
        ", ".join(sessions),
    )

    subject_valid = []

    for session in sessions:

        session_path = os.path.join(
            subject_path,
            session,
        )

        candidates = sorted(
            [
                name
                for name in os.listdir(
                    session_path
                )
                if os.path.isdir(
                    os.path.join(
                        session_path,
                        name,
                    )
                )
                and "mprage"
                in name.lower()
            ]
        )

        print(
            f"\n  Session: {session}"
        )

        if not candidates:

            print(
                "    MPRAGE candidates: NOT FOUND"
            )

            continue

        print(
            "    MPRAGE candidates:",
            len(candidates),
        )

        for series in candidates:

            series_path = os.path.join(
                session_path,
                series,
            )

            print(
                f"\n    Series: {series}"
            )

            result = validate_candidate(
                series_path
            )

            result["subject"] = subject
            result["session"] = session
            result["series"] = series

            all_candidates.append(
                result
            )

            if result["valid"]:

                valid_candidates += 1

                if (
                    result["acquisition_type"]
                    == "5ECHO"
                ):

                    valid_5echo += 1

                elif (
                    result["acquisition_type"]
                    == "RMS"
                ):

                    valid_rms += 1

                subject_valid.append(
                    result
                )

                print(
                    "      STATUS: PASS"
                )

                print(
                    "      Acquisition Type:",
                    result[
                        "acquisition_type"
                    ],
                )

                print(
                    "      Files:",
                    result[
                        "file_count"
                    ],
                )

                print(
                    "      Series Number:",
                    result[
                        "series_number"
                    ],
                )

                if (
                    result[
                        "acquisition_type"
                    ]
                    == "RMS"
                ):

                    print(
                        "      Echo Time:",
                        round(
                            result[
                                "echo_1_te_ms"
                            ],
                            3,
                        ),
                        "ms",
                    )

                else:

                    print(
                        "      Echo Times:",
                        [
                            round(
                                result[
                                    "echo_times"
                                ][echo],
                                3,
                            )
                            for echo in sorted(
                                EXPECTED_ECHO_NUMBERS
                            )
                        ],
                    )

            else:

                invalid_candidates += 1

                print(
                    "      STATUS: FAIL"
                )

                print(
                    "      Reason:",
                    result["reason"],
                )

    # --------------------------------------------------------------
    # Select one acquisition
    # --------------------------------------------------------------

    if not subject_valid:

        subjects_without_mprage += 1

        print(
            "\n  FINAL SUBJECT STATUS:"
            " NO VALID MPRAGE"
        )

        continue

    subjects_with_mprage += 1

    subject_valid.sort(
        key=lambda item: (
            session_sort_key(
                item["session"]
            ),
            ACQUISITION_PRIORITY.get(
                item[
                    "acquisition_type"
                ],
                99,
            ),
            (
                item["series_number"]
                if item["series_number"]
                is not None
                else float("inf")
            ),
            item["series"].lower(),
        )
    )

    selected = subject_valid[0]

    selected_candidates.append(
        selected
    )

    print(
        "\n  FINAL SELECTION:"
    )

    print(
        "    Session:",
        selected["session"],
    )

    print(
        "    Series:",
        selected["series"],
    )

    print(
        "    Acquisition Type:",
        selected["acquisition_type"],
    )

    print(
        "    Series Number:",
        selected["series_number"],
    )


# ======================================================================
# MANIFEST
# ======================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


fieldnames = [
    "subject",
    "session",
    "series",
    "series_path",
    "selection_status",
    "valid",
    "reason",
    "acquisition_type",
    "file_count",
    "series_number",
    "study_uid",
    "series_uid",
    "rows",
    "columns",
    "slice_thickness",
    "pixel_spacing",
    "corrupt_files",
    "echo_1_files",
    "echo_2_files",
    "echo_3_files",
    "echo_4_files",
    "echo_5_files",
    "echo_1_te_ms",
    "echo_2_te_ms",
    "echo_3_te_ms",
    "echo_4_te_ms",
    "echo_5_te_ms",
]


selected_keys = {
    (
        item["subject"],
        item["session"],
        item["series"],
    )
    for item in selected_candidates
}


with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for item in all_candidates:

        key = (
            item["subject"],
            item["session"],
            item["series"],
        )

        if key in selected_keys:

            status = "SELECTED"

        elif item["valid"]:

            status = "VALID_NOT_SELECTED"

        else:

            status = "INVALID"

        slice_counts = item.get(
            "slice_counts",
            {},
        )

        echo_times = item.get(
            "echo_times",
            {},
        )

        writer.writerow(
            {
                "subject": item.get(
                    "subject",
                    "",
                ),
                "session": item.get(
                    "session",
                    "",
                ),
                "series": item.get(
                    "series",
                    "",
                ),
                "series_path": item.get(
                    "series_path",
                    "",
                ),
                "selection_status": status,
                "valid": item.get(
                    "valid",
                    False,
                ),
                "reason": item.get(
                    "reason",
                    "",
                ),
                "acquisition_type": item.get(
                    "acquisition_type",
                    "",
                ),
                "file_count": item.get(
                    "file_count",
                    "",
                ),
                "series_number": item.get(
                    "series_number",
                    "",
                ),
                "study_uid": item.get(
                    "study_uid",
                    "",
                ),
                "series_uid": item.get(
                    "series_uid",
                    "",
                ),
                "rows": item.get(
                    "rows",
                    "",
                ),
                "columns": item.get(
                    "columns",
                    "",
                ),
                "slice_thickness": item.get(
                    "slice_thickness",
                    "",
                ),
                "pixel_spacing": item.get(
                    "pixel_spacing",
                    "",
                ),
                "corrupt_files": item.get(
                    "corrupt_files",
                    "",
                ),
                "echo_1_files": slice_counts.get(
                    1,
                    item.get(
                        "echo_1_files",
                        "",
                    ),
                ),
                "echo_2_files": slice_counts.get(
                    2,
                    "",
                ),
                "echo_3_files": slice_counts.get(
                    3,
                    "",
                ),
                "echo_4_files": slice_counts.get(
                    4,
                    "",
                ),
                "echo_5_files": slice_counts.get(
                    5,
                    "",
                ),
                "echo_1_te_ms": echo_times.get(
                    1,
                    item.get(
                        "echo_1_te_ms",
                        "",
                    ),
                ),
                "echo_2_te_ms": echo_times.get(
                    2,
                    "",
                ),
                "echo_3_te_ms": echo_times.get(
                    3,
                    "",
                ),
                "echo_4_te_ms": echo_times.get(
                    4,
                    "",
                ),
                "echo_5_te_ms": echo_times.get(
                    5,
                    "",
                ),
            }
        )


# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("MPRAGE INSPECTION SUMMARY")
print("=" * 80)

print(
    "Total subjects detected       :",
    len(subjects),
)

print(
    "Subjects with valid MPRAGE    :",
    subjects_with_mprage,
)

print(
    "Subjects without valid MPRAGE :",
    subjects_without_mprage,
)

print(
    "Total MPRAGE candidates       :",
    len(all_candidates),
)

print(
    "Valid MPRAGE candidates       :",
    valid_candidates,
)

print(
    "  Valid 5ECHO candidates      :",
    valid_5echo,
)

print(
    "  Valid RMS candidates        :",
    valid_rms,
)

print(
    "Invalid MPRAGE candidates     :",
    invalid_candidates,
)

print(
    "Selected acquisitions         :",
    len(selected_candidates),
)

print(
    "\nSelection manifest:"
)

print(
    MANIFEST_PATH
)

print("\n" + "=" * 80)
print("CHECK COMPLETED")
print("=" * 80)