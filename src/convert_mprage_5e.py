
# src/convert_mprage_5e.py

"""
======================================================================
COBRE MPRAGE DICOM -> NIFTI CONVERSION
======================================================================

Supports:
    1. 5ECHO MPRAGE
    2. RMS MPRAGE

The acquisition type is obtained from:
    processed_data/smri/mprage_selection/
    mprage_selection_manifest.csv

5ECHO:
    960 DICOM files
    5 echoes
    192 slices per echo

    Output:
        SUBJECT_echo1.nii.gz
        SUBJECT_echo2.nii.gz
        SUBJECT_echo3.nii.gz
        SUBJECT_echo4.nii.gz
        SUBJECT_echo5.nii.gz

RMS:
    192 DICOM files
    single reconstructed structural volume

    Output:
        SUBJECT_rms.nii.gz

Important:
    - Exactly one selected acquisition per subject.
    - No averaging of 5ECHO data.
    - RMS is NOT duplicated into five echoes.
    - Raw DICOM files are never modified.
    - Existing valid outputs are preserved.
    - Invalid/incomplete outputs are regenerated.
    - DICOM LPS -> NIfTI RAS affine is constructed explicitly.
======================================================================
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import pydicom
import nibabel as nib


# ======================================================================
# CONSTANTS
# ======================================================================

EXPECTED_ROWS = 256
EXPECTED_COLUMNS = 256
EXPECTED_SLICES = 192

EXPECTED_5ECHO_FILES = 960
EXPECTED_RMS_FILES = 192

EXPECTED_ECHO_NUMBERS = {1, 2, 3, 4, 5}

EXPECTED_ECHO_TIMES = {
    1: 1.64,
    2: 3.50,
    3: 5.36,
    4: 7.22,
    5: 9.08,
}

EXPECTED_PIXEL_SPACING = 1.0

SLICE_SPACING_TOLERANCE = 0.05
GEOMETRY_TOLERANCE = 0.01


# ======================================================================
# ARGUMENTS
# ======================================================================

parser = argparse.ArgumentParser(
    description=(
        "Convert selected COBRE MPRAGE acquisitions "
        "(5ECHO or RMS) to NIfTI."
    )
)

parser.add_argument(
    "--manifest",
    type=str,
    default=None,
    help="Path to mprage_selection_manifest.csv.",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for NIfTI files.",
)

args = parser.parse_args()


# ======================================================================
# PATHS
# ======================================================================

if args.manifest:

    MANIFEST_PATH = os.path.abspath(
        args.manifest
    )

else:

    MANIFEST_PATH = os.path.abspath(
        os.path.join(
            "processed_data",
            "smri",
            "mprage_selection",
            "mprage_selection_manifest.csv",
        )
    )


if args.output:

    OUTPUT_ROOT = os.path.abspath(
        args.output
    )

else:

    OUTPUT_ROOT = os.path.abspath(
        os.path.join(
            "processed_data",
            "smri",
            "mprage_5e",
        )
    )


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("COBRE MPRAGE DICOM -> NIFTI CONVERSION")
print("=" * 80)

print("\nSelection manifest:")
print(MANIFEST_PATH)

print("\nNIfTI output directory:")
print(OUTPUT_ROOT)


# ======================================================================
# MANIFEST CHECK
# ======================================================================

if not os.path.isfile(
    MANIFEST_PATH
):

    raise FileNotFoundError(
        "\nSelection manifest not found:\n"
        f"{MANIFEST_PATH}\n\n"
        "Run check_mprage_sessions.py first."
    )


# ======================================================================
# LOAD MANIFEST
# ======================================================================

selected_rows = []

with open(
    MANIFEST_PATH,
    "r",
    newline="",
    encoding="utf-8",
) as csv_file:

    reader = csv.DictReader(
        csv_file
    )

    required_columns = {
        "subject",
        "session",
        "series",
        "series_path",
        "selection_status",
        "acquisition_type",
    }

    missing = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing:

        raise RuntimeError(
            "\nManifest is missing required columns:\n"
            + ", ".join(
                sorted(missing)
            )
        )

    for row in reader:

        if (
            row.get(
                "selection_status",
                "",
            )
            .strip()
            .upper()
            == "SELECTED"
        ):

            selected_rows.append(
                row
            )


if not selected_rows:

    raise RuntimeError(
        "\nNo SELECTED acquisitions were found "
        "in the manifest."
    )


# ======================================================================
# ONE SELECTION PER SUBJECT
# ======================================================================

selected_by_subject = {}

for row in selected_rows:

    subject = row[
        "subject"
    ].strip()

    if subject in selected_by_subject:

        raise RuntimeError(
            "\nMore than one selected acquisition "
            f"found for {subject}."
        )

    selected_by_subject[
        subject
    ] = row


selected_rows = sorted(
    selected_rows,
    key=lambda row:
        row["subject"].strip(),
)


print(
    "\nSelected acquisitions:",
    len(selected_rows),
)


for row in selected_rows:

    print(
        f"  {row['subject']} | "
        f"{row['session']} | "
        f"{row['acquisition_type']} | "
        f"{row['series']}"
    )


# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True,
)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def get_dicom_files(
    series_path,
):
    """
    Return all direct files in a DICOM series.
    """

    if not os.path.isdir(
        series_path
    ):

        return []

    files = []

    for name in os.listdir(
        series_path
    ):

        path = os.path.join(
            series_path,
            name,
        )

        if os.path.isfile(
            path
        ):

            files.append(
                path
            )

    return sorted(files)


def read_dicom(
    path,
):
    """
    Read DICOM including pixel data.
    """

    return pydicom.dcmread(
        path,
        stop_before_pixels=False,
    )


def normalize_vector(
    vector,
):
    """
    Normalize a 3-D vector.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm <= 0:

        raise RuntimeError(
            "Zero-length direction vector."
        )

    return vector / norm


def get_orientation(
    ds,
):
    """
    Get row, column and slice directions.
    """

    if not hasattr(
        ds,
        "ImageOrientationPatient",
    ):

        raise RuntimeError(
            "ImageOrientationPatient missing."
        )

    orientation = np.asarray(
        ds.ImageOrientationPatient,
        dtype=np.float64,
    )

    if orientation.size != 6:

        raise RuntimeError(
            "Invalid ImageOrientationPatient."
        )

    # DICOM:
    # first 3 = direction of increasing columns
    # second 3 = direction of increasing rows

    column_direction = normalize_vector(
        orientation[:3]
    )

    row_direction = normalize_vector(
        orientation[3:]
    )

    slice_direction = normalize_vector(
        np.cross(
            column_direction,
            row_direction,
        )
    )

    return (
        row_direction,
        column_direction,
        slice_direction,
    )


def get_slice_coordinate(
    ds,
    slice_direction,
):
    """
    Scalar spatial coordinate used for slice sorting.
    """

    position = np.asarray(
        ds.ImagePositionPatient,
        dtype=np.float64,
    )

    return float(
        np.dot(
            position,
            slice_direction,
        )
    )


def verify_nifti(
    path,
    expected_shape,
):
    """
    Verify an existing NIfTI.
    """

    try:

        image = nib.load(
            path
        )

        data = image.get_fdata(
            dtype=np.float32
        )

        if image.shape != expected_shape:

            return False

        if not np.all(
            np.isfinite(data)
        ):

            return False

        if not np.all(
            np.isfinite(
                image.affine
            )
        ):

            return False

        voxel_sizes = (
            image.header
            .get_zooms()[:3]
        )

        if not np.all(
            np.isfinite(
                voxel_sizes
            )
        ):

            return False

        return True

    except Exception:

        return False


def check_geometry(
    datasets,
):
    """
    Validate common geometry and return
    orientation information.
    """

    if not datasets:

        raise RuntimeError(
            "No DICOM datasets."
        )

    first = datasets[0]

    rows = {
        int(
            getattr(
                ds,
                "Rows",
            )
        )
        for ds in datasets
    }

    columns = {
        int(
            getattr(
                ds,
                "Columns",
            )
        )
        for ds in datasets
    }

    if rows != {
        EXPECTED_ROWS
    }:

        raise RuntimeError(
            f"Invalid Rows values: {rows}"
        )

    if columns != {
        EXPECTED_COLUMNS
    }:

        raise RuntimeError(
            f"Invalid Columns values: {columns}"
        )

    # --------------------------------------------------------------
    # Pixel spacing
    # --------------------------------------------------------------

    spacings = set()

    for ds in datasets:

        if not hasattr(
            ds,
            "PixelSpacing",
        ):

            raise RuntimeError(
                "PixelSpacing missing."
            )

        spacing = ds.PixelSpacing

        if len(spacing) != 2:

            raise RuntimeError(
                "Invalid PixelSpacing."
            )

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

    if len(spacings) != 1:

        raise RuntimeError(
            "Inconsistent PixelSpacing."
        )

    spacing = next(
        iter(spacings)
    )

    if not np.allclose(
        spacing,
        (1.0, 1.0),
        atol=GEOMETRY_TOLERANCE,
    ):

        raise RuntimeError(
            f"Expected 1x1 mm spacing, found {spacing}."
        )

    # --------------------------------------------------------------
    # Slice thickness
    # --------------------------------------------------------------

    thicknesses = set()

    for ds in datasets:

        if not hasattr(
            ds,
            "SliceThickness",
        ):

            raise RuntimeError(
                "SliceThickness missing."
            )

        thicknesses.add(
            round(
                float(
                    ds.SliceThickness
                ),
                6,
            )
        )

    if len(thicknesses) != 1:

        raise RuntimeError(
            "Inconsistent SliceThickness."
        )

    thickness = next(
        iter(thicknesses)
    )

    if not np.isclose(
        thickness,
        1.0,
        atol=GEOMETRY_TOLERANCE,
    ):

        raise RuntimeError(
            f"Expected 1 mm slice thickness, "
            f"found {thickness}."
        )

    # --------------------------------------------------------------
    # Orientation consistency
    # --------------------------------------------------------------

    orientations = set()

    for ds in datasets:

        if not hasattr(
            ds,
            "ImageOrientationPatient",
        ):

            raise RuntimeError(
                "ImageOrientationPatient missing."
            )

        orientation = tuple(
            round(
                float(x),
                6,
            )
            for x in ds.ImageOrientationPatient
        )

        orientations.add(
            orientation
        )

    if len(orientations) != 1:

        raise RuntimeError(
            "Inconsistent ImageOrientationPatient."
        )

    row_direction, column_direction, slice_direction = (
        get_orientation(first)
    )

    return (
        row_direction,
        column_direction,
        slice_direction,
        spacing,
        thickness,
    )


def calculate_slice_spacing(
    datasets,
    slice_direction,
):
    """
    Calculate and validate inter-slice spacing.
    """

    coordinates = np.asarray(
        [
            get_slice_coordinate(
                ds,
                slice_direction,
            )
            for ds in datasets
        ],
        dtype=np.float64,
    )

    coordinates = np.sort(
        coordinates
    )

    differences = np.diff(
        coordinates
    )

    if len(differences) == 0:

        raise RuntimeError(
            "Unable to calculate slice spacing."
        )

    spacing = float(
        np.median(
            np.abs(
                differences
            )
        )
    )

    deviation = float(
        np.max(
            np.abs(
                np.abs(
                    differences
                )
                - spacing
            )
        )
    )

    if spacing <= 0:

        raise RuntimeError(
            "Invalid slice spacing."
        )

    if deviation > SLICE_SPACING_TOLERANCE:

        raise RuntimeError(
            f"Inconsistent slice spacing. "
            f"Median={spacing:.4f} mm, "
            f"max deviation={deviation:.4f} mm."
        )

    return spacing


def build_ras_affine(
    first_ds,
    row_direction,
    column_direction,
    slice_direction,
    row_spacing,
    column_spacing,
    slice_spacing,
):
    """
    Construct DICOM LPS -> NIfTI RAS affine.

    Numpy volume axes:
        axis 0 = rows
        axis 1 = columns
        axis 2 = slices
    """

    position = np.asarray(
        first_ds.ImagePositionPatient,
        dtype=np.float64,
    )

    lps_affine = np.eye(
        4,
        dtype=np.float64,
    )

    lps_affine[:3, 0] = (
        row_direction
        * row_spacing
    )

    lps_affine[:3, 1] = (
        column_direction
        * column_spacing
    )

    lps_affine[:3, 2] = (
        slice_direction
        * slice_spacing
    )

    lps_affine[:3, 3] = (
        position
    )

    # DICOM LPS -> NIfTI RAS

    lps_to_ras = np.diag(
        [
            -1.0,
            -1.0,
             1.0,
             1.0,
        ]
    )

    ras_affine = (
        lps_to_ras
        @
        lps_affine
    )

    if not np.all(
        np.isfinite(
            ras_affine
        )
    ):

        raise RuntimeError(
            "Generated affine contains NaN/Inf."
        )

    return ras_affine


def create_volume(
    datasets,
    slice_direction,
):
    """
    Sort DICOM slices spatially and create
    one 3-D volume.
    """

    ordered = sorted(
        datasets,
        key=lambda ds:
            get_slice_coordinate(
                ds,
                slice_direction,
            ),
    )

    positions = [
        np.asarray(
            ds.ImagePositionPatient,
            dtype=np.float64,
        )
        for ds in ordered
    ]

    rounded_positions = {
        tuple(
            np.round(
                position,
                5,
            )
        )
        for position in positions
    }

    if len(rounded_positions) != len(
        ordered
    ):

        raise RuntimeError(
            "Duplicate slice positions detected."
        )

    spacing = calculate_slice_spacing(
        ordered,
        slice_direction,
    )

    slices = []

    for index, ds in enumerate(
        ordered,
        start=1,
    ):

        try:

            pixels = ds.pixel_array

        except Exception as e:

            raise RuntimeError(
                f"Pixel data failed at slice "
                f"{index}: {e}"
            )

        if pixels.shape != (
            EXPECTED_ROWS,
            EXPECTED_COLUMNS,
        ):

            raise RuntimeError(
                f"Unexpected pixel shape at "
                f"slice {index}: {pixels.shape}"
            )

        slope = float(
            getattr(
                ds,
                "RescaleSlope",
                1.0,
            )
        )

        intercept = float(
            getattr(
                ds,
                "RescaleIntercept",
                0.0,
            )
        )

        pixels = (
            pixels.astype(
                np.float32
            )
            * slope
            + intercept
        )

        if not np.all(
            np.isfinite(
                pixels
            )
        ):

            raise RuntimeError(
                f"NaN/Inf detected at slice {index}."
            )

        slices.append(
            pixels
        )

    volume = np.stack(
        slices,
        axis=2,
    )

    if volume.shape != (
        EXPECTED_ROWS,
        EXPECTED_COLUMNS,
        EXPECTED_SLICES,
    ):

        raise RuntimeError(
            f"Unexpected volume shape: "
            f"{volume.shape}"
        )

    return (
        volume,
        ordered,
        spacing,
    )


def save_and_verify(
    volume,
    affine,
    output_path,
    description,
):
    """
    Save NIfTI and verify after reload.
    """

    image = nib.Nifti1Image(
        volume,
        affine,
    )

    image.header.set_xyzt_units(
        "mm"
    )

    image.header["descrip"] = (
        description
    )

    nib.save(
        image,
        output_path,
    )

    if not verify_nifti(
        output_path,
        volume.shape,
    ):

        raise RuntimeError(
            f"Saved NIfTI failed verification: "
            f"{output_path}"
        )

    check = nib.load(
        output_path
    )

    voxel_sizes = (
        check.header
        .get_zooms()[:3]
    )

    if not np.all(
        np.isfinite(
            voxel_sizes
        )
    ):

        raise RuntimeError(
            "Saved NIfTI has invalid voxel sizes."
        )

    print(
        "Saved NIfTI verification: PASS"
    )

    print(
        "  Shape:",
        check.shape,
    )

    print(
        "  Voxel size:",
        voxel_sizes,
    )


# ======================================================================
# PROCESSING COUNTERS
# ======================================================================

successful_subjects = []
skipped_subjects = []
failed_subjects = []

total_outputs = 0
total_5echo_outputs = 0
total_rms_outputs = 0


# ======================================================================
# PROCESS SUBJECTS
# ======================================================================

print("\n" + "=" * 80)
print("STARTING CONVERSION")
print("=" * 80)


for row in selected_rows:

    subject = row[
        "subject"
    ].strip()

    session = row[
        "session"
    ].strip()

    series = row[
        "series"
    ].strip()

    acquisition_type = (
        row[
            "acquisition_type"
        ]
        .strip()
        .upper()
    )

    series_path = os.path.abspath(
        row[
            "series_path"
        ].strip()
    )

    print("\n" + "=" * 80)
    print(f"SUBJECT : {subject}")
    print(f"SESSION : {session}")
    print(f"TYPE    : {acquisition_type}")
    print(f"SERIES  : {series}")
    print("=" * 80)

    if acquisition_type not in {
        "5ECHO",
        "RMS",
    }:

        print(
            "ERROR: Unsupported acquisition type:",
            acquisition_type,
        )

        failed_subjects.append(
            subject
        )

        continue

    if not os.path.isdir(
        series_path
    ):

        print(
            "ERROR: Selected DICOM directory does not exist."
        )

        failed_subjects.append(
            subject
        )

        continue

    dicom_files = get_dicom_files(
        series_path
    )

    print(
        "\nDICOM files found:",
        len(dicom_files),
    )

    expected_files = (
        EXPECTED_5ECHO_FILES
        if acquisition_type == "5ECHO"
        else EXPECTED_RMS_FILES
    )

    if len(dicom_files) != expected_files:

        print(
            "ERROR: Expected",
            expected_files,
            "files but found",
            len(dicom_files),
        )

        failed_subjects.append(
            subject
        )

        continue

    # ==================================================================
    # DETERMINE EXPECTED OUTPUTS
    # ==================================================================

    if acquisition_type == "5ECHO":

        expected_outputs = [
            os.path.join(
                OUTPUT_ROOT,
                f"{subject}_echo{echo}.nii.gz",
            )
            for echo in range(
                1,
                6,
            )
        ]

    else:

        expected_outputs = [
            os.path.join(
                OUTPUT_ROOT,
                f"{subject}_rms.nii.gz",
            )
        ]

    # ==================================================================
    # EXISTING OUTPUT CHECK
    # ==================================================================

    if all(
        os.path.exists(path)
        for path in expected_outputs
    ):

        expected_shape = (
            EXPECTED_ROWS,
            EXPECTED_COLUMNS,
            EXPECTED_SLICES,
        )

        if all(
            verify_nifti(
                path,
                expected_shape,
            )
            for path in expected_outputs
        ):

            print(
                "\nExisting outputs verified: PASS"
            )

            print(
                "STATUS: SKIPPED"
            )

            skipped_subjects.append(
                subject
            )

            total_outputs += len(
                expected_outputs
            )

            if acquisition_type == "5ECHO":

                total_5echo_outputs += 5

            else:

                total_rms_outputs += 1

            continue

        print(
            "\nExisting output(s) invalid."
        )

        print(
            "They will be regenerated."
        )

    # ==================================================================
    # READ DICOM
    # ==================================================================

    print(
        "\nReading DICOM files..."
    )

    datasets = []

    try:

        for index, path in enumerate(
            dicom_files,
            start=1,
        ):

            ds = read_dicom(
                path
            )

            datasets.append(
                ds
            )

            if index % 100 == 0:

                print(
                    f"  Read {index}/{len(dicom_files)}..."
                )

    except Exception as e:

        print(
            "ERROR reading DICOM:",
            e,
        )

        failed_subjects.append(
            subject
        )

        continue

    print(
        "All DICOM files loaded successfully."
    )

    # ==================================================================
    # BASIC GEOMETRY
    # ==================================================================

    try:

        (
            row_direction,
            column_direction,
            slice_direction,
            pixel_spacing,
            slice_thickness,
        ) = check_geometry(
            datasets
        )

    except Exception as e:

        print(
            "ERROR in geometry validation:",
            e,
        )

        failed_subjects.append(
            subject
        )

        continue

    print(
        "\nGeometry verification: PASS"
    )

    print(
        "  Matrix:",
        EXPECTED_ROWS,
        "x",
        EXPECTED_COLUMNS,
    )

    print(
        "  Pixel spacing:",
        pixel_spacing,
        "mm",
    )

    print(
        "  Slice thickness:",
        slice_thickness,
        "mm",
    )

    # ==================================================================
    # PROCESS RMS
    # ==================================================================

    if acquisition_type == "RMS":

        print(
            "\n" + "-" * 70
        )

        print(
            "RMS CONVERSION"
        )

        print(
            "-" * 70
        )

        try:

            echo_numbers = {
                int(
                    getattr(
                        ds,
                        "EchoNumbers",
                    )
                )
                for ds in datasets
            }

            if echo_numbers != {1}:

                raise RuntimeError(
                    f"RMS expected EchoNumber=1, "
                    f"found {echo_numbers}"
                )

            echo_times = [
                float(
                    getattr(
                        ds,
                        "EchoTime",
                    )
                )
                for ds in datasets
            ]

            if not np.allclose(
                echo_times,
                1.64,
                atol=0.15,
            ):

                raise RuntimeError(
                    "RMS EchoTime is not approximately 1.64 ms."
                )

            # ----------------------------------------------------------
            # Create volume
            # ----------------------------------------------------------

            (
                volume,
                ordered,
                slice_spacing,
            ) = create_volume(
                datasets,
                slice_direction,
            )

            print(
                "Volume shape:",
                volume.shape,
            )

            print(
                "Slice spacing:",
                slice_spacing,
                "mm",
            )

            print(
                "Intensity min:",
                float(volume.min()),
            )

            print(
                "Intensity max:",
                float(volume.max()),
            )

            print(
                "Intensity mean:",
                float(volume.mean()),
            )

            print(
                "Intensity std:",
                float(volume.std()),
            )

            # ----------------------------------------------------------
            # Affine
            # ----------------------------------------------------------

            affine = build_ras_affine(
                ordered[0],
                row_direction,
                column_direction,
                slice_direction,
                pixel_spacing[0],
                pixel_spacing[1],
                slice_spacing,
            )

            output_path = expected_outputs[0]

            # ----------------------------------------------------------
            # Remove invalid existing output
            # ----------------------------------------------------------

            if os.path.exists(
                output_path
            ):

                if not verify_nifti(
                    output_path,
                    volume.shape,
                ):

                    os.remove(
                        output_path
                    )

            # ----------------------------------------------------------
            # Save
            # ----------------------------------------------------------

            print(
                "\nSaving:",
                output_path,
            )

            save_and_verify(
                volume,
                affine,
                output_path,
                (
                    f"COBRE RMS MPRAGE "
                    f"Subject={subject} "
                    f"Session={session} "
                    f"TE=1.64ms"
                ),
            )

            print(
                "\nRMS conversion: PASS"
            )

            successful_subjects.append(
                subject
            )

            total_outputs += 1
            total_rms_outputs += 1

            continue

        except Exception as e:

            print(
                "\nRMS conversion FAILED:"
            )

            print(
                e
            )

            failed_subjects.append(
                subject
            )

            continue

    # ==================================================================
    # PROCESS 5-ECHO
    # ==================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "5-ECHO CONVERSION"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------------
    # Group by echo
    # --------------------------------------------------------------

    echo_groups = defaultdict(
        list
    )

    try:

        for ds in datasets:

            echo = int(
                getattr(
                    ds,
                    "EchoNumbers",
                )
            )

            if echo not in (
                EXPECTED_ECHO_NUMBERS
            ):

                raise RuntimeError(
                    f"Unexpected EchoNumber: {echo}"
                )

            echo_groups[
                echo
            ].append(
                ds
            )

    except Exception as e:

        print(
            "ERROR grouping echoes:",
            e,
        )

        failed_subjects.append(
            subject
        )

        continue

    if set(
        echo_groups.keys()
    ) != EXPECTED_ECHO_NUMBERS:

        print(
            "ERROR: Echo numbers are not exactly 1-5."
        )

        failed_subjects.append(
            subject
        )

        continue

    # --------------------------------------------------------------
    # Validate each echo
    # --------------------------------------------------------------

    echo_times = {}

    echo_validation_failed = False

    for echo in sorted(
        EXPECTED_ECHO_NUMBERS
    ):

        group = echo_groups[
            echo
        ]

        if len(group) != EXPECTED_SLICES:

            print(
                f"ERROR: Echo {echo} contains "
                f"{len(group)} slices."
            )

            echo_validation_failed = True

            break

        values = []

        for ds in group:

            try:

                values.append(
                    float(
                        ds.EchoTime
                    )
                )

            except Exception:

                print(
                    f"ERROR: Echo {echo} "
                    "has invalid EchoTime."
                )

                echo_validation_failed = True

                break

        if echo_validation_failed:

            break

        te = float(
            np.median(
                values
            )
        )

        expected_te = (
            EXPECTED_ECHO_TIMES[
                echo
            ]
        )

        if not np.allclose(
            values,
            te,
            atol=0.15,
        ):

            print(
                f"ERROR: Echo {echo} "
                "has inconsistent TE."
            )

            echo_validation_failed = True

            break

        if abs(
            te
            - expected_te
        ) > 0.15:

            print(
                f"ERROR: Echo {echo} "
                f"TE={te:.3f} ms; "
                f"expected {expected_te:.3f} ms."
            )

            echo_validation_failed = True

            break

        echo_times[
            echo
        ] = te

        print(
            f"Echo {echo}: "
            f"{len(group)} slices, "
            f"TE={te:.3f} ms"
        )

    if echo_validation_failed:

        failed_subjects.append(
            subject
        )

        continue

    # --------------------------------------------------------------
    # TE ordering
    # --------------------------------------------------------------

    tes = [
        echo_times[echo]
        for echo in sorted(
            EXPECTED_ECHO_NUMBERS
        )
    ]

    if not all(
        tes[i] < tes[i + 1]
        for i in range(
            len(tes) - 1
        )
    ):

        print(
            "ERROR: Echo times are not increasing."
        )

        failed_subjects.append(
            subject
        )

        continue

    # --------------------------------------------------------------
    # Reference geometry = Echo 1
    # --------------------------------------------------------------

    try:

        (
            reference_volume,
            reference_ordered,
            reference_spacing,
        ) = create_volume(
            echo_groups[1],
            slice_direction,
        )

    except Exception as e:

        print(
            "ERROR processing Echo 1:",
            e,
        )

        failed_subjects.append(
            subject
        )

        continue

    reference_positions = [
        tuple(
            np.round(
                ds.ImagePositionPatient,
                5,
            )
        )
        for ds in reference_ordered
    ]

    reference_position_set = set(
        reference_positions
    )

    # --------------------------------------------------------------
    # Process all echoes
    # --------------------------------------------------------------

    subject_outputs = []

    subject_success = True

    for echo in sorted(
        EXPECTED_ECHO_NUMBERS
    ):

        print(
            "\nProcessing Echo",
            echo,
        )

        try:

            (
                volume,
                ordered,
                slice_spacing,
            ) = create_volume(
                echo_groups[echo],
                slice_direction,
            )

            positions = [
                tuple(
                    np.round(
                        ds.ImagePositionPatient,
                        5,
                    )
                )
                for ds in ordered
            ]

            if set(
                positions
            ) != reference_position_set:

                raise RuntimeError(
                    f"Echo {echo} spatial positions "
                    "do not match Echo 1."
                )

            if not np.isclose(
                slice_spacing,
                reference_spacing,
                atol=0.01,
            ):

                raise RuntimeError(
                    f"Echo {echo} slice spacing "
                    "does not match Echo 1."
                )

            # ------------------------------------------------------
            # Affine
            # ------------------------------------------------------

            affine = build_ras_affine(
                ordered[0],
                row_direction,
                column_direction,
                slice_direction,
                pixel_spacing[0],
                pixel_spacing[1],
                slice_spacing,
            )

            output_path = (
                expected_outputs[
                    echo - 1
                ]
            )

            # ------------------------------------------------------
            # Existing invalid output
            # ------------------------------------------------------

            if os.path.exists(
                output_path
            ):

                if not verify_nifti(
                    output_path,
                    volume.shape,
                ):

                    os.remove(
                        output_path
                    )

            print(
                "Saving:",
                output_path,
            )

            save_and_verify(
                volume,
                affine,
                output_path,
                (
                    f"COBRE 5ECHO MPRAGE "
                    f"Subject={subject} "
                    f"Session={session} "
                    f"Echo={echo} "
                    f"TE={echo_times[echo]:.3f}ms"
                ),
            )

            subject_outputs.append(
                output_path
            )

        except Exception as e:

            print(
                f"ERROR processing Echo {echo}:",
                e,
            )

            subject_success = False

            break

    # --------------------------------------------------------------
    # Remove partial outputs on failure
    # --------------------------------------------------------------

    if not subject_success:

        print(
            "\n5-ECHO conversion FAILED."
        )

        for path in subject_outputs:

            try:

                if os.path.exists(
                    path
                ):

                    os.remove(
                        path
                    )

            except Exception:

                pass

        failed_subjects.append(
            subject
        )

        continue

    # --------------------------------------------------------------
    # Final verification
    # --------------------------------------------------------------

    if len(
        subject_outputs
    ) != 5:

        print(
            "ERROR: Expected five echo outputs."
        )

        failed_subjects.append(
            subject
        )

        continue

    final_ok = all(
        verify_nifti(
            path,
            (
                EXPECTED_ROWS,
                EXPECTED_COLUMNS,
                EXPECTED_SLICES,
            ),
        )
        for path in subject_outputs
    )

    if not final_ok:

        print(
            "ERROR: Final 5-ECHO verification failed."
        )

        failed_subjects.append(
            subject
        )

        continue

    print(
        "\n5-ECHO conversion: PASS"
    )

    successful_subjects.append(
        subject
    )

    total_outputs += 5
    total_5echo_outputs += 5


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("MPRAGE CONVERSION SUMMARY")
print("=" * 80)

print(
    "Selected subjects             :",
    len(selected_rows),
)

print(
    "Successful subjects           :",
    len(successful_subjects),
)

print(
    "Skipped subjects              :",
    len(skipped_subjects),
)

print(
    "Failed subjects               :",
    len(failed_subjects),
)

print(
    "Total NIfTI outputs generated :",
    total_outputs,
)

print(
    "5-ECHO NIfTI outputs          :",
    total_5echo_outputs,
)

print(
    "RMS NIfTI outputs             :",
    total_rms_outputs,
)


if successful_subjects:

    print(
        "\nSuccessful subjects:"
    )

    for subject in successful_subjects:

        print(
            "  ",
            subject,
        )


if skipped_subjects:

    print(
        "\nSkipped subjects:"
    )

    for subject in skipped_subjects:

        print(
            "  ",
            subject,
        )


if failed_subjects:

    print(
        "\nFailed subjects:"
    )

    for subject in failed_subjects:

        print(
            "  ",
            subject,
        )


print(
    "\nOutput directory:"
)

print(
    OUTPUT_ROOT
)

print(
    "\nManifest used:"
)

print(
    MANIFEST_PATH
)

print(
    "\nRaw DICOM files were not modified."
)

print(
    "RMS acquisitions remain single 3-D volumes."
)

print(
    "5-ECHO acquisitions remain five separate echoes."
)

print("\n" + "=" * 80)

if failed_subjects:

    print(
        "STATUS: CONVERSION COMPLETED WITH FAILURES"
    )

else:

    print(
        "STATUS: CONVERSION COMPLETED"
    )

print("=" * 80)