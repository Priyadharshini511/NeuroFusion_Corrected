# src/exp03_normalization/smri_zscore_normalization.py

from pathlib import Path
import csv

import nibabel as nib
import numpy as np


# ==============================================================
# EXPERIMENT 03 — ALTERNATIVE sMRI INPUT NORMALIZATION
#
# NORMALIZATION:
#   Brain-aware per-subject z-score normalization
#
#   z = (voxel - brain_mean) / brain_std
#
#   Mean and standard deviation are calculated ONLY from
#   positive brain voxels of the same subject.
#
#   Background remains 0.
#
# IMPORTANT:
#   - No labels are used to calculate normalization statistics.
#   - No statistics are calculated across subjects.
#   - No test-set statistics are used.
#   - Original [0,1] CNN dataset is NEVER modified.
#   - Existing valid Experiment 03 outputs are skipped.
#   - One selected MNI volume is used per subject.
#   - The validated subject manifest determines the input volume.
# ==============================================================


# ==============================================================
# PATHS
# ==============================================================

MANIFEST_FILE = Path(
    "processed_data/smri/dataset/"
    "smri_subject_manifest_labeled.csv"
)

OUTPUT_FOLDER = Path(
    "processed_data/smri/cnn_dataset/"
    "experiments/exp03_normalization/"
    "zscore_images"
)

REPORT_FILE = Path(
    "processed_data/smri/cnn_dataset/"
    "experiments/exp03_normalization/"
    "zscore_normalization_report.csv"
)

VALIDATION_REPORT = Path(
    "processed_data/smri/cnn_dataset/"
    "experiments/exp03_normalization/"
    "zscore_normalization_validation_report.txt"
)


# ==============================================================
# SETTINGS
# ==============================================================

EXPECTED_MNI_SHAPE = (
    99,
    117,
    95
)

EXPECTED_MNI_SPACING = (
    2.0,
    2.0,
    2.0
)

VOXEL_TOLERANCE = 0.01

AFFINE_TOLERANCE = 1e-3

STD_EPSILON = 1e-8

ZERO_TOLERANCE = 1e-7


# ==============================================================
# CREATE OUTPUT FOLDER
# ==============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# HEADER
# ==============================================================

print("=" * 75)
print("EXPERIMENT 03 — sMRI BRAIN-AWARE Z-SCORE NORMALIZATION")
print("=" * 75)

print("\nManifest:")
print(
    f"  {MANIFEST_FILE.resolve()}"
)

print("\nOutput folder:")
print(
    f"  {OUTPUT_FOLDER.resolve()}"
)

print("\nReport:")
print(
    f"  {REPORT_FILE.resolve()}"
)


# ==============================================================
# REQUIRED PATH CHECK
# ==============================================================

if not MANIFEST_FILE.exists():

    raise FileNotFoundError(
        "Labeled sMRI subject manifest does not exist:\n"
        f"{MANIFEST_FILE.resolve()}"
    )


# ==============================================================
# LOAD MANIFEST
# ==============================================================

def load_manifest(manifest_file):
    """
    Load the validated labeled sMRI subject manifest.

    Actual manifest columns:
        subject_id
        mni_volume
        aal_csv
        acquisition_source
        has_rms_representation
        has_legacy_representation
        alternative_volume_not_selected
        subject_type
        label
        label_status
        mni_valid
        aal_valid

    Only the following fields are used here:
        subject_id
        label
        acquisition_source
        mni_volume

    Returns:
        list of dictionaries
    """

    rows = []

    with manifest_file.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:

            raise ValueError(
                "Manifest has no header."
            )

        # ------------------------------------------------------
        # These are the ACTUAL column names in the validated
        # manifest.
        # ------------------------------------------------------

        required_columns = {
            "subject_id",
            "label",
            "acquisition_source",
            "mni_volume"
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                "Manifest is missing required columns:\n"
                + ", ".join(
                    sorted(missing)
                )
            )

        for row in reader:

            subject_id = str(
                row["subject_id"]
            ).strip()

            if not subject_id:

                raise ValueError(
                    "Manifest contains an empty subject_id."
                )

            mni_volume = str(
                row["mni_volume"]
            ).strip()

            if not mni_volume:

                raise ValueError(
                    f"Subject {subject_id} has no mni_volume."
                )

            rows.append(
                {
                    "subject_id": subject_id,

                    "label": str(
                        row["label"]
                    ).strip(),

                    "acquisition_source": str(
                        row["acquisition_source"]
                    ).strip(),

                    "mni_volume": mni_volume
                }
            )

    if not rows:

        raise ValueError(
            "Manifest contains no subjects."
        )

    # ----------------------------------------------------------
    # Duplicate subject check
    # ----------------------------------------------------------

    subject_ids = [
        row["subject_id"]
        for row in rows
    ]

    if len(subject_ids) != len(
        set(subject_ids)
    ):

        duplicates = sorted(
            {
                sid
                for sid in subject_ids
                if subject_ids.count(sid) > 1
            }
        )

        raise ValueError(
            "Duplicate subjects found in manifest:\n"
            + "\n".join(duplicates)
        )

    return sorted(
        rows,
        key=lambda x: x["subject_id"]
    )


manifest_rows = load_manifest(
    MANIFEST_FILE
)


# ==============================================================
# DISPLAY COHORT
# ==============================================================

print("\n" + "-" * 75)
print("LABELED COHORT")
print("-" * 75)

print(
    f"Total subjects: {len(manifest_rows)}"
)


label_counts = {}

source_counts = {}


for row in manifest_rows:

    label = row["label"]

    source = row[
        "acquisition_source"
    ]

    label_counts[label] = (
        label_counts.get(label, 0)
        + 1
    )

    source_counts[source] = (
        source_counts.get(source, 0)
        + 1
    )


print("\nLabels:")

for label, count in sorted(
    label_counts.items()
):

    print(
        f"  {label}: {count}"
    )


print("\nAcquisition sources:")

for source, count in sorted(
    source_counts.items()
):

    print(
        f"  {source}: {count}"
    )


# ==============================================================
# NIFTI VALIDATION
# ==============================================================

def validate_mni_image(
    file_path
):
    """
    Validate a selected MNI sMRI image.

    Checks:
      - file exists
      - readable NIfTI
      - expected shape
      - expected 2mm isotropic spacing
      - RAS orientation
      - finite voxel values
      - non-empty positive content

    Returns:
        valid, reason, image, data
    """

    if not file_path.exists():

        return (
            False,
            "file does not exist",
            None,
            None
        )

    try:

        img = nib.load(
            str(file_path)
        )

        data = img.get_fdata(
            dtype=np.float32
        )

    except Exception as exc:

        return (
            False,
            f"NIfTI read error: {exc}",
            None,
            None
        )

    # ----------------------------------------------------------
    # Shape
    # ----------------------------------------------------------

    if data.shape != EXPECTED_MNI_SHAPE:

        return (
            False,
            f"shape {data.shape} != "
            f"expected {EXPECTED_MNI_SHAPE}",
            None,
            None
        )

    # ----------------------------------------------------------
    # Spacing
    # ----------------------------------------------------------

    spacing = np.asarray(
        img.header.get_zooms()[:3],
        dtype=float
    )

    if not np.allclose(
        spacing,
        EXPECTED_MNI_SPACING,
        atol=VOXEL_TOLERANCE
    ):

        return (
            False,
            f"spacing {spacing} != "
            f"expected {EXPECTED_MNI_SPACING}",
            None,
            None
        )

    # ----------------------------------------------------------
    # Orientation
    # ----------------------------------------------------------

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
            f"orientation {orientation} != RAS",
            None,
            None
        )

    # ----------------------------------------------------------
    # Finite values
    # ----------------------------------------------------------

    if not np.all(
        np.isfinite(data)
    ):

        return (
            False,
            "image contains non-finite values",
            None,
            None
        )

    # ----------------------------------------------------------
    # Positive content
    # ----------------------------------------------------------

    positive = data > 0

    if not np.any(positive):

        return (
            False,
            "image contains no positive voxels",
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
# OUTPUT VALIDATION
# ==============================================================

def validate_zscore_output(
    output_file,
    reference_img
):
    """
    Validate an Experiment 03 z-score output.

    Checks:
      - readable
      - same geometry as source
      - finite
      - background is zero
      - non-zero brain values exist
      - brain mean approximately zero
      - brain standard deviation approximately one
    """

    if not output_file.exists():

        return (
            False,
            "output does not exist",
            None
        )

    try:

        img = nib.load(
            str(output_file)
        )

        data = img.get_fdata(
            dtype=np.float32
        )

    except Exception as exc:

        return (
            False,
            f"output read error: {exc}",
            None
        )

    # ----------------------------------------------------------
    # Shape
    # ----------------------------------------------------------

    if img.shape != reference_img.shape:

        return (
            False,
            "shape mismatch with source image",
            None
        )

    # ----------------------------------------------------------
    # Spacing
    # ----------------------------------------------------------

    spacing = np.asarray(
        img.header.get_zooms()[:3],
        dtype=float
    )

    reference_spacing = np.asarray(
        reference_img.header.get_zooms()[:3],
        dtype=float
    )

    if not np.allclose(
        spacing,
        reference_spacing,
        atol=VOXEL_TOLERANCE
    ):

        return (
            False,
            "spacing mismatch with source image",
            None
        )

    # ----------------------------------------------------------
    # Affine
    # ----------------------------------------------------------

    if not np.allclose(
        img.affine,
        reference_img.affine,
        atol=AFFINE_TOLERANCE
    ):

        return (
            False,
            "affine mismatch with source image",
            None
        )

    # ----------------------------------------------------------
    # Finite
    # ----------------------------------------------------------

    if not np.all(
        np.isfinite(data)
    ):

        return (
            False,
            "output contains non-finite values",
            None
        )

    # ----------------------------------------------------------
    # Brain mask
    #
    # Since z-score values can be positive or negative, we
    # cannot use data > 0 to identify the brain.
    #
    # The source image is already brain-masked. We therefore
    # reconstruct the source mask using the reference image.
    # ----------------------------------------------------------

    reference_data = reference_img.get_fdata(
        dtype=np.float32
    )

    brain_mask = reference_data > 0

    if not np.any(brain_mask):

        return (
            False,
            "reference image contains no positive brain voxels",
            None
        )

    brain_values = data[
        brain_mask
    ]

    # ----------------------------------------------------------
    # Background validation
    # ----------------------------------------------------------

    background = data[
        ~brain_mask
    ]

    if (
        background.size > 0
        and not np.allclose(
            background,
            0.0,
            atol=ZERO_TOLERANCE
        )
    ):

        return (
            False,
            "background is not zero",
            None
        )

    # ----------------------------------------------------------
    # Brain finite validation
    # ----------------------------------------------------------

    if not np.all(
        np.isfinite(
            brain_values
        )
    ):

        return (
            False,
            "brain contains non-finite values",
            None
        )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    mean = float(
        np.mean(
            brain_values
        )
    )

    std = float(
        np.std(
            brain_values
        )
    )

    if not np.isfinite(mean):

        return (
            False,
            "brain mean is non-finite",
            None
        )

    if not np.isfinite(std):

        return (
            False,
            "brain standard deviation is non-finite",
            None
        )

    # ----------------------------------------------------------
    # Z-score validation
    # ----------------------------------------------------------

    if abs(mean) > 1e-4:

        return (
            False,
            f"brain mean {mean} is not approximately zero",
            None
        )

    if abs(
        std - 1.0
    ) > 1e-4:

        return (
            False,
            f"brain std {std} is not approximately one",
            None
        )

    return (
        True,
        "VALID",
        data
    )


# ==============================================================
# NORMALIZE ONE SUBJECT
# ==============================================================

def normalize_subject(
    subject_id,
    input_file,
    output_file
):
    """
    Perform brain-aware per-subject z-score normalization.
    """

    print("\n" + "-" * 75)
    print(
        f"{subject_id}"
    )
    print("-" * 75)

    print(
        f"Input : {input_file}"
    )

    print(
        f"Output: {output_file}"
    )

    # ----------------------------------------------------------
    # Validate source
    # ----------------------------------------------------------

    (
        valid,
        reason,
        input_img,
        data
    ) = validate_mni_image(
        input_file
    )

    if not valid:

        print(
            "ERROR: Source validation failed:"
        )

        print(
            f"  {reason}"
        )

        return {
            "status": "FAILED",
            "reason": reason
        }

    # ----------------------------------------------------------
    # Existing output
    # ----------------------------------------------------------

    if output_file.exists():

        (
            output_valid,
            output_reason,
            _
        ) = validate_zscore_output(
            output_file,
            input_img
        )

        if output_valid:

            print(
                "Existing valid z-score output found."
            )

            print(
                "Skipping."
            )

            return {
                "status": "SKIPPED",
                "reason": "existing output valid"
            }

        print(
            "Existing output is invalid:"
        )

        print(
            f"  {output_reason}"
        )

        print(
            "Recreating output."
        )

        try:

            output_file.unlink()

        except Exception as exc:

            return {
                "status": "FAILED",
                "reason":
                    "could not remove invalid output: "
                    f"{exc}"
            }

    # ----------------------------------------------------------
    # Brain mask
    #
    # The selected MNI images are already brain-masked.
    # Positive source voxels define the brain.
    # ----------------------------------------------------------

    brain_mask = data > 0

    if not np.any(brain_mask):

        return {
            "status": "FAILED",
            "reason": "no positive brain voxels"
        }

    # ----------------------------------------------------------
    # Brain values
    # ----------------------------------------------------------

    brain_values = data[
        brain_mask
    ].astype(
        np.float64
    )

    # ----------------------------------------------------------
    # Per-subject statistics
    # ----------------------------------------------------------

    brain_mean = float(
        np.mean(
            brain_values
        )
    )

    brain_std = float(
        np.std(
            brain_values
        )
    )

    print(
        f"Brain mean: "
        f"{brain_mean:.8f}"
    )

    print(
        f"Brain std : "
        f"{brain_std:.8f}"
    )

    if not np.isfinite(
        brain_mean
    ):

        return {
            "status": "FAILED",
            "reason": "brain mean is non-finite"
        }

    if not np.isfinite(
        brain_std
    ):

        return {
            "status": "FAILED",
            "reason": "brain std is non-finite"
        }

    if brain_std <= STD_EPSILON:

        return {
            "status": "FAILED",
            "reason":
                f"brain std too small: {brain_std}"
        }

    # ----------------------------------------------------------
    # Z-score normalization
    # ----------------------------------------------------------

    normalized = np.zeros_like(
        data,
        dtype=np.float32
    )

    normalized[
        brain_mask
    ] = (
        (
            data[brain_mask]
            - brain_mean
        )
        / brain_std
    ).astype(
        np.float32
    )

    # ----------------------------------------------------------
    # Generated brain statistics
    # ----------------------------------------------------------

    normalized_brain = normalized[
        brain_mask
    ]

    if not np.all(
        np.isfinite(
            normalized_brain
        )
    ):

        return {
            "status": "FAILED",
            "reason":
                "generated z-score contains "
                "non-finite values"
        }

    generated_mean = float(
        np.mean(
            normalized_brain
        )
    )

    generated_std = float(
        np.std(
            normalized_brain
        )
    )

    generated_min = float(
        np.min(
            normalized_brain
        )
    )

    generated_max = float(
        np.max(
            normalized_brain
        )
    )

    # ----------------------------------------------------------
    # Create output image
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

        return {
            "status": "FAILED",
            "reason":
                f"save error: {exc}"
        }

    # ----------------------------------------------------------
    # Validate saved output
    # ----------------------------------------------------------

    (
        saved_valid,
        saved_reason,
        _
    ) = validate_zscore_output(
        output_file,
        input_img
    )

    if not saved_valid:

        return {
            "status": "FAILED",
            "reason":
                f"saved output invalid: "
                f"{saved_reason}"
        }

    # ----------------------------------------------------------
    # Brain fraction
    # ----------------------------------------------------------

    brain_fraction = float(
        np.count_nonzero(
            brain_mask
        )
        / data.size
    )

    # ----------------------------------------------------------
    # Display
    # ----------------------------------------------------------

    print(
        f"Z-score mean: "
        f"{generated_mean:.8f}"
    )

    print(
        f"Z-score std : "
        f"{generated_std:.8f}"
    )

    print(
        f"Z-score min : "
        f"{generated_min:.8f}"
    )

    print(
        f"Z-score max : "
        f"{generated_max:.8f}"
    )

    print(
        f"Brain fraction: "
        f"{brain_fraction:.6f}"
    )

    print(
        "Validation: PASS"
    )

    print(
        f"Saved: {output_file}"
    )

    return {
        "status": "PROCESSED",
        "reason": "success",
        "brain_mean": brain_mean,
        "brain_std": brain_std,
        "normalized_mean": generated_mean,
        "normalized_std": generated_std,
        "normalized_min": generated_min,
        "normalized_max": generated_max,
        "brain_fraction": brain_fraction
    }


# ==============================================================
# PROCESS ALL SUBJECTS
# ==============================================================

results = []

processed = 0
skipped = 0
failed = 0


print("\n" + "=" * 75)
print("STARTING EXPERIMENT 03 NORMALIZATION")
print("=" * 75)


for index, row in enumerate(
    manifest_rows,
    start=1
):

    subject_id = row[
        "subject_id"
    ]

    # ----------------------------------------------------------
    # IMPORTANT:
    # The actual manifest column is mni_volume.
    # ----------------------------------------------------------

    input_file = Path(
        row["mni_volume"]
    )

    output_file = (
        OUTPUT_FOLDER
        / f"{subject_id}_zscore_mni.nii.gz"
    )

    print(
        f"\n[{index}/{len(manifest_rows)}]"
    )

    result = normalize_subject(
        subject_id,
        input_file,
        output_file
    )

    status = result[
        "status"
    ]

    if status == "PROCESSED":

        processed += 1

    elif status == "SKIPPED":

        skipped += 1

    else:

        failed += 1

    results.append(
        {
            "subject_id":
                subject_id,

            "label":
                row["label"],

            "acquisition_source":
                row["acquisition_source"],

            "input_file":
                str(input_file),

            "output_file":
                str(output_file),

            "status":
                status,

            "reason":
                result.get(
                    "reason",
                    ""
                ),

            "brain_mean":
                result.get(
                    "brain_mean",
                    ""
                ),

            "brain_std":
                result.get(
                    "brain_std",
                    ""
                ),

            "normalized_mean":
                result.get(
                    "normalized_mean",
                    ""
                ),

            "normalized_std":
                result.get(
                    "normalized_std",
                    ""
                ),

            "normalized_min":
                result.get(
                    "normalized_min",
                    ""
                ),

            "normalized_max":
                result.get(
                    "normalized_max",
                    ""
                ),

            "brain_fraction":
                result.get(
                    "brain_fraction",
                    ""
                )
        }
    )


# ==============================================================
# WRITE CSV REPORT
# ==============================================================

report_columns = [
    "subject_id",
    "label",
    "acquisition_source",
    "input_file",
    "output_file",
    "status",
    "reason",
    "brain_mean",
    "brain_std",
    "normalized_mean",
    "normalized_std",
    "normalized_min",
    "normalized_max",
    "brain_fraction"
]


with REPORT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=report_columns
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ==============================================================
# FINAL VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("FINAL EXPERIMENT 03 VALIDATION")
print("=" * 75)


final_pass = 0
final_fail = 0

validation_rows = []


for row in manifest_rows:

    subject_id = row[
        "subject_id"
    ]

    input_file = Path(
        row["mni_volume"]
    )

    output_file = (
        OUTPUT_FOLDER
        / f"{subject_id}_zscore_mni.nii.gz"
    )

    (
        input_valid,
        input_reason,
        input_img,
        _
    ) = validate_mni_image(
        input_file
    )

    if not input_valid:

        final_fail += 1

        validation_rows.append(
            (
                subject_id,
                "FAIL",
                f"input: {input_reason}"
            )
        )

        continue

    (
        output_valid,
        output_reason,
        _
    ) = validate_zscore_output(
        output_file,
        input_img
    )

    if output_valid:

        final_pass += 1

        validation_rows.append(
            (
                subject_id,
                "PASS",
                "valid z-score output"
            )
        )

    else:

        final_fail += 1

        validation_rows.append(
            (
                subject_id,
                "FAIL",
                output_reason
            )
        )


# ==============================================================
# WRITE VALIDATION REPORT
# ==============================================================

with VALIDATION_REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 03 — Z-SCORE NORMALIZATION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Expected subjects: "
        f"{len(manifest_rows)}\n"
    )

    f.write(
        f"Processed: {processed}\n"
    )

    f.write(
        f"Skipped: {skipped}\n"
    )

    f.write(
        f"Failed during processing: "
        f"{failed}\n"
    )

    f.write(
        f"Final validation PASS: "
        f"{final_pass}\n"
    )

    f.write(
        f"Final validation FAIL: "
        f"{final_fail}\n"
    )

    f.write(
        "\nNormalization:\n"
    )

    f.write(
        "  Brain-aware per-subject z-score\n"
    )

    f.write(
        "  Mean/std calculated only from "
        "positive brain voxels\n"
    )

    f.write(
        "  Background = 0\n"
    )

    f.write(
        "  No labels used for normalization statistics\n"
    )

    f.write(
        "  No cross-subject statistics\n"
    )

    f.write(
        "  No spatial modification\n"
    )

    f.write(
        "\nSubject validation:\n"
    )

    for subject, status, reason in validation_rows:

        f.write(
            f"{subject} | "
            f"{status} | "
            f"{reason}\n"
        )


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()

print(
    f"Expected subjects          : "
    f"{len(manifest_rows)}"
)

print(
    f"Processed                  : "
    f"{processed}"
)

print(
    f"Skipped valid outputs      : "
    f"{skipped}"
)

print(
    f"Processing failures        : "
    f"{failed}"
)

print(
    f"Final validation PASS      : "
    f"{final_pass}"
)

print(
    f"Final validation FAIL      : "
    f"{final_fail}"
)

print()

print(
    f"CSV report: "
    f"{REPORT_FILE.resolve()}"
)

print(
    f"Validation report: "
    f"{VALIDATION_REPORT.resolve()}"
)


# ==============================================================
# FINAL STATUS
# ==============================================================

print()
print("=" * 75)

if (
    final_pass
    == len(manifest_rows)
    and final_fail == 0
):

    print(
        "FINAL STATUS: EXPERIMENT 03 "
        "NORMALIZATION PASS"
    )

else:

    print(
        "FINAL STATUS: EXPERIMENT 03 "
        "NORMALIZATION FAILED"
    )

print("=" * 75)