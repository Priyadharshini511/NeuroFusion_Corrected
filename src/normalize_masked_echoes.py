
# src/normalize_masked_echoes.py

from pathlib import Path
import csv

import nibabel as nib
import numpy as np


# ==============================================================
# GENERIC sMRI INTENSITY NORMALIZATION
#
# Supports:
#   1. 5-ECHO MPRAGE
#      - Normalize each of the five masked echoes independently.
#
#   2. RMS MPRAGE
#      - Normalize the single masked RMS volume.
#
# INPUT:
#   processed_data/smri/mprage_5e_masked
#
# OUTPUT:
#   processed_data/smri/mprage_5e_normalized
#
# SUBJECT SELECTION:
#   mprage_selection_manifest.csv
#
# NORMALIZATION:
#   1. Select positive/non-zero brain voxels
#   2. Calculate 1st percentile
#   3. Calculate 99th percentile
#   4. Clip brain intensities to that range
#   5. Scale brain intensities to [0, 1]
#   6. Preserve background as 0
#
# IMPORTANT:
#   - Each selected volume is normalized independently.
#   - No global z-score normalization.
#   - No spatial resampling.
#   - No registration.
#   - Original masked inputs are never modified.
#   - Existing valid outputs are skipped.
# ==============================================================


# ==============================================================
# PATHS
# ==============================================================

INPUT_FOLDER = Path(
    "processed_data/smri/mprage_5e_masked"
)

OUTPUT_FOLDER = Path(
    "processed_data/smri/mprage_5e_normalized"
)

MANIFEST_FILE = Path(
    "processed_data/smri/mprage_selection/"
    "mprage_selection_manifest.csv"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# SETTINGS
# ==============================================================

ECHOES = [1, 2, 3, 4, 5]

VOXEL_TOLERANCE = 0.01

AFFINE_TOLERANCE = 1e-3

LOWER_PERCENTILE = 1.0

UPPER_PERCENTILE = 99.0

RANGE_TOLERANCE = 1e-6


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("GENERIC sMRI INTENSITY NORMALIZATION")
print("=" * 70)

print("\nInput folder:")
print(
    f"  {INPUT_FOLDER.resolve()}"
)

print("\nOutput folder:")
print(
    f"  {OUTPUT_FOLDER.resolve()}"
)

print("\nSelection manifest:")
print(
    f"  {MANIFEST_FILE.resolve()}"
)


# ==============================================================
# REQUIRED PATH CHECKS
# ==============================================================

if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        f"Masked input folder does not exist:\n"
        f"{INPUT_FOLDER.resolve()}"
    )

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"Selection manifest does not exist:\n"
        f"{MANIFEST_FILE.resolve()}"
    )


# ==============================================================
# LOAD SELECTED ACQUISITIONS
# ==============================================================

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
                row.get(
                    "selection_status",
                    ""
                )
            ).strip().upper()

            if status != "SELECTED":
                continue

            subject = str(
                row.get(
                    "subject",
                    ""
                )
            ).strip()

            if not subject:
                raise ValueError(
                    "A SELECTED manifest row has no "
                    "subject ID."
                )

            acquisition_type = str(
                row.get(
                    "acquisition_type",
                    ""
                )
            ).strip().upper()

            if acquisition_type not in {
                "5ECHO",
                "RMS"
            }:
                raise ValueError(
                    f"Unsupported acquisition type "
                    f"for {subject}: "
                    f"{acquisition_type}"
                )

            selected.append(
                {
                    "subject": subject,
                    "acquisition_type": acquisition_type,
                    "session": str(
                        row.get(
                            "session",
                            ""
                        )
                    ).strip(),
                    "series": str(
                        row.get(
                            "series",
                            ""
                        )
                    ).strip()
                }
            )

    if not selected:
        raise ValueError(
            "No SELECTED acquisitions were found "
            "in the manifest."
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
            "Duplicate selected subjects found:\n"
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


# ==============================================================
# DISPLAY CURRENT COHORT
# ==============================================================

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


# ==============================================================
# ACQUISITION SUMMARY
# ==============================================================

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
    f"  5ECHO: "
    f"{type_counts['5ECHO']}"
)

print(
    f"  RMS   : "
    f"{type_counts['RMS']}"
)


# ==============================================================
# EXPECTED OUTPUT COUNT
# ==============================================================

expected_output_files = 0

for item in selected_acquisitions:

    if item["acquisition_type"] == "RMS":
        expected_output_files += 1

    elif item["acquisition_type"] == "5ECHO":
        expected_output_files += len(ECHOES)


print(
    f"\nExpected normalized volumes: "
    f"{expected_output_files}"
)


# ==============================================================
# NIFTI VALIDATION
# ==============================================================

def validate_nifti(
    file_path,
    reference_img=None
):
    """
    Validate a NIfTI image.

    Checks:
      - readable
      - non-empty
      - finite
      - RAS orientation
      - 1mm isotropic voxel size
      - non-zero content
      - optional reference geometry
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

        # ------------------------------------------------------
        # Reference geometry
        # ------------------------------------------------------

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
                    "voxel size mismatch "
                    "with reference",
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
                    "affine mismatch "
                    "with reference",
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


# ==============================================================
# NORMALIZED OUTPUT VALIDATION
# ==============================================================

def validate_normalized_output(
    file_path,
    reference_img
):
    """
    Validate an already normalized output.

    Checks:
      - readable
      - geometry
      - finite
      - positive brain voxels
      - values within [0,1]
      - background remains zero
    """

    (
        valid,
        reason,
        img,
        data
    ) = validate_nifti(
        file_path,
        reference_img=reference_img
    )

    if not valid:
        return (
            False,
            reason,
            None,
            None
        )

    # ----------------------------------------------------------
    # Brain definition
    # ----------------------------------------------------------

    brain_voxels = data > 0

    if not np.any(brain_voxels):

        return (
            False,
            "normalized image has no "
            "positive brain voxels",
            None,
            None
        )

    brain_values = data[
        brain_voxels
    ]

    # ----------------------------------------------------------
    # Range validation
    # ----------------------------------------------------------

    minimum = float(
        np.min(brain_values)
    )

    maximum = float(
        np.max(brain_values)
    )

    if minimum < -RANGE_TOLERANCE:

        return (
            False,
            f"brain minimum {minimum} "
            f"is below 0",
            None,
            None
        )

    if maximum > (
        1.0 + RANGE_TOLERANCE
    ):

        return (
            False,
            f"brain maximum {maximum} "
            f"exceeds 1",
            None,
            None
        )

    # ----------------------------------------------------------
    # Background validation
    # ----------------------------------------------------------

    background = data[
        ~brain_voxels
    ]

    if (
        background.size > 0
        and not np.all(
            background == 0
        )
    ):

        return (
            False,
            "background contains "
            "non-zero values",
            None,
            None
        )

    return (
        True,
        "VALID",
        img,
        data
    )


# ==============================================================
# PROCESSING COUNTERS
# ==============================================================

processed = 0
skipped = 0
failed = 0

successful_subjects = 0
failed_subjects = 0

failure_details = []


# ==============================================================
# PROCESS ONE VOLUME
# ==============================================================

def process_volume(
    subject_id,
    label,
    input_file,
    output_file
):
    """
    Normalize one masked sMRI volume.

    Returns:
        True       = processed successfully
        False      = failed
        "SKIPPED"  = existing valid output
    """

    global processed
    global skipped
    global failed

    print("\n" + "-" * 70)
    print(
        f"{subject_id} | {label}"
    )
    print("-" * 70)

    print("Input:")
    print(
        f"  {input_file}"
    )

    print("Output:")
    print(
        f"  {output_file}"
    )

    # ----------------------------------------------------------
    # Input existence
    # ----------------------------------------------------------

    if not input_file.exists():

        print(
            "ERROR: Masked input is missing."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "missing masked input"
            )
        )

        return False

    # ----------------------------------------------------------
    # Validate input
    # ----------------------------------------------------------

    (
        input_valid,
        input_reason,
        input_img,
        data
    ) = validate_nifti(
        input_file
    )

    if not input_valid:

        print(
            "ERROR: Masked input validation failed:"
        )

        print(
            f"  {input_reason}"
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                f"invalid masked input: "
                f"{input_reason}"
            )
        )

        return False

    # ----------------------------------------------------------
    # Confirm positive brain voxels
    # ----------------------------------------------------------

    brain_voxels = data > 0

    if not np.any(brain_voxels):

        print(
            "ERROR: Masked input contains no "
            "positive brain voxels."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "no positive brain voxels"
            )
        )

        return False

    # ----------------------------------------------------------
    # Existing output validation
    # ----------------------------------------------------------

    if output_file.exists():

        (
            existing_valid,
            existing_reason,
            _,
            _
        ) = validate_normalized_output(
            output_file,
            reference_img=input_img
        )

        if existing_valid:

            print(
                "Existing normalized output is valid."
            )

            print(
                "Skipping this volume."
            )

            skipped += 1

            return "SKIPPED"

        print(
            "Existing normalized output is invalid:"
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

            failed += 1

            failure_details.append(
                (
                    f"{subject_id}_{label}",
                    "could not remove invalid output"
                )
            )

            return False

    # ----------------------------------------------------------
    # Brain values
    # ----------------------------------------------------------

    brain_values = data[
        brain_voxels
    ]

    # ----------------------------------------------------------
    # Percentiles
    # ----------------------------------------------------------

    lower = np.percentile(
        brain_values,
        LOWER_PERCENTILE
    )

    upper = np.percentile(
        brain_values,
        UPPER_PERCENTILE
    )

    print(
        f"{LOWER_PERCENTILE:g}th percentile : "
        f"{float(lower)}"
    )

    print(
        f"{UPPER_PERCENTILE:g}th percentile: "
        f"{float(upper)}"
    )

    # ----------------------------------------------------------
    # Validate percentile range
    # ----------------------------------------------------------

    if not np.isfinite(lower):

        print(
            "ERROR: Lower percentile is "
            "non-finite."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "lower percentile is non-finite"
            )
        )

        return False

    if not np.isfinite(upper):

        print(
            "ERROR: Upper percentile is "
            "non-finite."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "upper percentile is non-finite"
            )
        )

        return False

    if upper <= lower:

        print(
            "ERROR: Invalid intensity range "
            "for normalization."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "upper percentile <= lower percentile"
            )
        )

        return False

    # ----------------------------------------------------------
    # Clip
    # ----------------------------------------------------------

    brain_clipped = np.clip(
        brain_values,
        lower,
        upper
    )

    # ----------------------------------------------------------
    # Scale to [0,1]
    # ----------------------------------------------------------

    brain_normalized = (
        (brain_clipped - lower)
        / (upper - lower)
    )

    # ----------------------------------------------------------
    # Create normalized image
    # ----------------------------------------------------------

    normalized = np.zeros_like(
        data,
        dtype=np.float32
    )

    normalized[
        brain_voxels
    ] = brain_normalized.astype(
        np.float32
    )

    # ----------------------------------------------------------
    # Generated-value validation
    # ----------------------------------------------------------

    normalized_brain = normalized[
        brain_voxels
    ]

    if not np.all(
        np.isfinite(normalized_brain)
    ):

        print(
            "ERROR: Normalized brain contains "
            "non-finite values."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "generated non-finite values"
            )
        )

        return False

    generated_min = float(
        np.min(normalized_brain)
    )

    generated_max = float(
        np.max(normalized_brain)
    )

    if (
        generated_min < -RANGE_TOLERANCE
        or
        generated_max >
        1.0 + RANGE_TOLERANCE
    ):

        print(
            "ERROR: Generated normalized "
            "values are outside [0,1]."
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                "generated values outside [0,1]"
            )
        )

        return False

    # ----------------------------------------------------------
    # Preserve geometry/header
    # ----------------------------------------------------------

    output_img = nib.Nifti1Image(
        normalized,
        input_img.affine,
        input_img.header.copy()
    )

    output_img.set_data_dtype(
        np.float32
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    try:

        nib.save(
            output_img,
            output_file
        )

    except Exception as exc:

        print(
            "ERROR saving normalized output:"
        )

        print(
            f"  {exc}"
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                f"save error: {exc}"
            )
        )

        return False

    # ----------------------------------------------------------
    # Validate saved output
    # ----------------------------------------------------------

    (
        saved_valid,
        saved_reason,
        _,
        saved_data
    ) = validate_normalized_output(
        output_file,
        reference_img=input_img
    )

    if not saved_valid:

        print(
            "ERROR: Saved normalized output "
            "failed validation:"
        )

        print(
            f"  {saved_reason}"
        )

        failed += 1

        failure_details.append(
            (
                f"{subject_id}_{label}",
                f"saved output invalid: "
                f"{saved_reason}"
            )
        )

        return False

    # ----------------------------------------------------------
    # Display statistics
    # ----------------------------------------------------------

    saved_brain = saved_data[
        saved_data > 0
    ]

    print(
        "Original brain minimum:",
        float(np.min(brain_values))
    )

    print(
        "Original brain maximum:",
        float(np.max(brain_values))
    )

    print(
        "Normalized brain minimum:",
        float(np.min(saved_brain))
    )

    print(
        "Normalized brain maximum:",
        float(np.max(saved_brain))
    )

    print(
        "Saved:"
    )

    print(
        f"  {output_file}"
    )

    print(
        "Output validation: PASS"
    )

    processed += 1

    return True


# ==============================================================
# PROCESS SELECTED ACQUISITIONS
# ==============================================================

for index, item in enumerate(
    selected_acquisitions,
    start=1
):

    subject_id = item[
        "subject"
    ]

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

    subject_failed = False

    # ==========================================================
    # RMS
    # ==========================================================

    if acquisition_type == "RMS":

        input_file = (
            INPUT_FOLDER /
            f"{subject_id}_rms_n4_masked.nii.gz"
        )

        output_file = (
            OUTPUT_FOLDER /
            f"{subject_id}_rms_n4_normalized.nii.gz"
        )

        result = process_volume(
            subject_id,
            "RMS",
            input_file,
            output_file
        )

        if result is False:
            subject_failed = True

        if not subject_failed:

            print()
            print(
                f"SUCCESS: {subject_id} "
                f"— RMS volume normalized."
            )

        else:

            print()
            print(
                f"FAILED/PARTIAL: "
                f"{subject_id}"
            )

    # ==========================================================
    # 5ECHO
    # ==========================================================

    elif acquisition_type == "5ECHO":

        for echo in ECHOES:

            input_file = (
                INPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4_masked.nii.gz"
            )

            output_file = (
                OUTPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4_normalized.nii.gz"
            )

            result = process_volume(
                subject_id,
                f"Echo{echo}",
                input_file,
                output_file
            )

            if result is False:
                subject_failed = True

        if not subject_failed:

            print()
            print(
                f"SUCCESS: {subject_id} "
                f"— all five echoes normalized."
            )

        else:

            print()
            print(
                f"FAILED/PARTIAL: "
                f"{subject_id}"
            )

    # ==========================================================
    # UNKNOWN TYPE
    # ==========================================================

    else:

        print(
            f"ERROR: Unsupported acquisition type: "
            f"{acquisition_type}"
        )

        failed_subjects += 1
        subject_failed = True

        failure_details.append(
            (
                subject_id,
                f"unsupported acquisition type: "
                f"{acquisition_type}"
            )
        )

    # ----------------------------------------------------------
    # Subject counters
    # ----------------------------------------------------------

    if subject_failed:
        failed_subjects += 1

    else:
        successful_subjects += 1


# ==============================================================
# FINAL BATCH VALIDATION
# ==============================================================

print("\n" + "=" * 70)
print("FINAL NORMALIZED sMRI BATCH VALIDATION")
print("=" * 70)

final_valid_subjects = []
final_invalid_entries = []

actual_output_files = 0


for item in selected_acquisitions:

    subject_id = item[
        "subject"
    ]

    acquisition_type = item[
        "acquisition_type"
    ]

    subject_valid = True

    # ==========================================================
    # RMS
    # ==========================================================

    if acquisition_type == "RMS":

        input_file = (
            INPUT_FOLDER /
            f"{subject_id}_rms_n4_masked.nii.gz"
        )

        output_file = (
            OUTPUT_FOLDER /
            f"{subject_id}_rms_n4_normalized.nii.gz"
        )

        if not input_file.exists():

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing RMS masked input"
                )
            )

            continue

        (
            input_valid,
            input_reason,
            reference_img,
            _
        ) = validate_nifti(
            input_file
        )

        if not input_valid:

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid RMS masked input: "
                    f"{input_reason}"
                )
            )

            continue

        if not output_file.exists():

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing RMS normalized output"
                )
            )

            continue

        actual_output_files += 1

        (
            output_valid,
            output_reason,
            _,
            _
        ) = validate_normalized_output(
            output_file,
            reference_img
        )

        if not output_valid:

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid RMS normalized output: "
                    f"{output_reason}"
                )
            )

    # ==========================================================
    # 5ECHO
    # ==========================================================

    elif acquisition_type == "5ECHO":

        # Echo 1 provides the reference geometry.
        reference_file = (
            INPUT_FOLDER /
            f"{subject_id}_echo1_n4_masked.nii.gz"
        )

        if not reference_file.exists():

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject_id,
                    "missing Echo-1 masked reference"
                )
            )

            continue

        (
            reference_valid,
            reference_reason,
            reference_img,
            _
        ) = validate_nifti(
            reference_file
        )

        if not reference_valid:

            subject_valid = False

            final_invalid_entries.append(
                (
                    subject_id,
                    f"invalid Echo-1 reference: "
                    f"{reference_reason}"
                )
            )

            continue

        for echo in ECHOES:

            output_file = (
                OUTPUT_FOLDER /
                f"{subject_id}_echo{echo}_n4_normalized.nii.gz"
            )

            if not output_file.exists():

                subject_valid = False

                final_invalid_entries.append(
                    (
                        f"{subject_id}_echo{echo}",
                        "missing normalized output"
                    )
                )

                continue

            actual_output_files += 1

            (
                output_valid,
                output_reason,
                _,
                _
            ) = validate_normalized_output(
                output_file,
                reference_img
            )

            if not output_valid:

                subject_valid = False

                final_invalid_entries.append(
                    (
                        f"{subject_id}_echo{echo}",
                        f"invalid normalized output: "
                        f"{output_reason}"
                    )
                )

    # ==========================================================
    # COMPLETE SUBJECT
    # ==========================================================

    if subject_valid:

        final_valid_subjects.append(
            subject_id
        )


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()

print(
    f"Selected subjects        : "
    f"{len(selected_acquisitions)}"
)

print(
    f"Expected normalized volumes: "
    f"{expected_output_files}"
)

print(
    f"Existing output files    : "
    f"{actual_output_files}"
)

print(
    f"Valid complete subjects  : "
    f"{len(final_valid_subjects)}"
)

print(
    f"Invalid/missing entries  : "
    f"{len(final_invalid_entries)}"
)

print()

print(
    f"Newly processed volumes  : "
    f"{processed}"
)

print(
    f"Skipped valid volumes    : "
    f"{skipped}"
)

print(
    f"Failed volumes           : "
    f"{failed}"
)

print(
    f"Successful subjects      : "
    f"{successful_subjects}"
)

print(
    f"Failed subjects          : "
    f"{failed_subjects}"
)


# ==============================================================
# INVALID DETAILS
# ==============================================================

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


# ==============================================================
# OUTPUT
# ==============================================================

print()
print("Output folder:")

print(
    f"  {OUTPUT_FOLDER.resolve()}"
)


# ==============================================================
# FINAL STATUS
# ==============================================================

print()
print("=" * 70)

if (
    len(final_valid_subjects)
    == len(selected_acquisitions)

    and actual_output_files
    == expected_output_files

    and not final_invalid_entries

    and failed == 0
):

    print(
        "STATUS: NORMALIZATION "
        "VALIDATION PASSED"
    )

else:

    print(
        "STATUS: NORMALIZATION "
        "VALIDATION FAILED"
    )

print("=" * 70)