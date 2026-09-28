"""
Validate the subject-level sMRI volumes before 3D CNN preparation.

Input
-----
processed_data/smri/dataset/smri_subject_manifest_labeled.csv

Validation
----------
- Exactly one row per subject
- Expected labeled cohort size
- Binary class counts
- Selected MNI volume exists
- 3D volume
- Common MNI geometry
- Finite voxel values
- Expected normalized intensity range
- No accidental use of old generic *_mni.nii.gz files

This script DOES NOT modify MRI files.
It only validates and reports their readiness for 3D CNN input preparation.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

MANIFEST = (
    ROOT
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

MNI_TEMPLATE = (
    ROOT
    / "mni"
    / "mni152_template_2mm.nii.gz"
)

REPORT = (
    ROOT
    / "dataset"
    / "smri_cnn_input_validation_report.txt"
)

SUBJECT_PATTERN = re.compile(r"^A\d{8}$")

RMS_PATTERN = re.compile(
    r"^(A\d{8})_rms_n4_mni\.nii\.gz$"
)

LEGACY_PATTERN = re.compile(
    r"^(A\d{8})_legacy_n4_mni\.nii\.gz$"
)

EXPECTED_SUBJECTS = 172
EXPECTED_CONTROL = 93
EXPECTED_SCHIZOPHRENIA = 79

EXPECTED_SIZE = (99, 117, 95)
EXPECTED_SPACING = (2.0, 2.0, 2.0)

GEOMETRY_ATOL = 1e-6

# Current and harmonized sMRI normalization is [0, 1].
INTENSITY_MIN = 0.0
INTENSITY_MAX = 1.0

# Tiny floating-point tolerance for values written as float32.
INTENSITY_ATOL = 1e-5


# ============================================================
# HELPERS
# ============================================================

def geometry_matches(
    image: sitk.Image,
    template: sitk.Image,
) -> tuple[bool, str]:
    """Check that image geometry matches the common MNI template."""

    if image.GetDimension() != 3:
        return False, "not_3d"

    if image.GetSize() != template.GetSize():
        return False, f"size:{image.GetSize()}"

    if not np.allclose(
        image.GetSpacing(),
        template.GetSpacing(),
        atol=GEOMETRY_ATOL,
        rtol=0.0,
    ):
        return False, "spacing"

    if not np.allclose(
        image.GetOrigin(),
        template.GetOrigin(),
        atol=GEOMETRY_ATOL,
        rtol=0.0,
    ):
        return False, "origin"

    if not np.allclose(
        image.GetDirection(),
        template.GetDirection(),
        atol=GEOMETRY_ATOL,
        rtol=0.0,
    ):
        return False, "direction"

    return True, "valid"


def classify_volume_name(path: Path) -> str | None:
    """Return RMS or LEGACY for the exact approved filename patterns."""

    if RMS_PATTERN.fullmatch(path.name):
        return "RMS"

    if LEGACY_PATTERN.fullmatch(path.name):
        return "LEGACY"

    return None


# ============================================================
# MAIN VALIDATION
# ============================================================

def main() -> None:

    print("=" * 72)
    print("NeuroFusion AI - VALIDATE 3D CNN sMRI INPUTS")
    print("=" * 72)
    print()

    errors: list[str] = []

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    df = pd.read_csv(MANIFEST)

    print(f"Manifest rows       : {len(df)}")
    print(f"Expected rows       : {EXPECTED_SUBJECTS}")

    if len(df) != EXPECTED_SUBJECTS:
        errors.append(
            f"Expected {EXPECTED_SUBJECTS} rows; "
            f"found {len(df)}"
        )

    required_columns = {
        "subject_id",
        "mni_volume",
        "acquisition_source",
        "label",
        "mni_valid",
        "aal_valid",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        errors.append(
            "Missing manifest columns: "
            + ", ".join(sorted(missing_columns))
        )

    # --------------------------------------------------------
    # Subject uniqueness
    # --------------------------------------------------------

    duplicate_ids = df[
        df["subject_id"].duplicated(keep=False)
    ]["subject_id"].tolist()

    print(
        f"Duplicate subjects  : "
        f"{len(set(duplicate_ids))}"
    )

    if duplicate_ids:
        errors.append(
            "Duplicate subject IDs: "
            + ", ".join(sorted(set(duplicate_ids)))
        )

    invalid_ids = [
        str(x)
        for x in df["subject_id"]
        if not SUBJECT_PATTERN.fullmatch(str(x))
    ]

    if invalid_ids:
        errors.append(
            "Invalid subject IDs: "
            + ", ".join(invalid_ids)
        )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    label_counts = (
        pd.to_numeric(
            df["label"],
            errors="coerce",
        )
        .value_counts()
        .to_dict()
    )

    control_count = int(label_counts.get(0, 0))
    schizophrenia_count = int(label_counts.get(1, 0))

    print(f"Control subjects    : {control_count}")
    print(f"Schizophrenia       : {schizophrenia_count}")

    if control_count != EXPECTED_CONTROL:
        errors.append(
            f"Expected {EXPECTED_CONTROL} controls; "
            f"found {control_count}"
        )

    if schizophrenia_count != EXPECTED_SCHIZOPHRENIA:
        errors.append(
            f"Expected {EXPECTED_SCHIZOPHRENIA} "
            f"schizophrenia-spectrum subjects; "
            f"found {schizophrenia_count}"
        )

    if not set(df["label"].dropna().unique()).issubset({0, 1}):
        errors.append("Non-binary labels detected.")

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    if not MNI_TEMPLATE.exists():
        raise FileNotFoundError(
            f"MNI template not found: {MNI_TEMPLATE}"
        )

    template = sitk.ReadImage(
        str(MNI_TEMPLATE),
        sitk.sitkFloat32,
    )

    print()
    print("Template geometry:")
    print(f"  Size    : {template.GetSize()}")
    print(f"  Spacing : {template.GetSpacing()}")

    # --------------------------------------------------------
    # Validate each selected MRI
    # --------------------------------------------------------

    records = []

    for index, row in df.iterrows():

        subject = str(row["subject_id"]).strip()
        volume_path = Path(str(row["mni_volume"]))

        record = {
            "subject_id": subject,
            "path": str(volume_path),
            "source": str(row["acquisition_source"]),
            "label": int(row["label"]),
            "exists": False,
            "readable": False,
            "geometry_valid": False,
            "finite": False,
            "range_valid": False,
            "shape": "",
            "spacing": "",
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "status": "FAIL",
            "message": "",
        }

        # ----------------------------------------------------
        # Exact filename lineage
        # ----------------------------------------------------

        filename_source = classify_volume_name(volume_path)

        if filename_source is None:
            record["message"] = (
                "unapproved_filename_pattern"
            )
            errors.append(
                f"{subject}: unapproved MNI filename "
                f"{volume_path.name}"
            )
            records.append(record)
            continue

        if filename_source != row["acquisition_source"]:
            record["message"] = (
                "manifest_source_filename_mismatch"
            )
            errors.append(
                f"{subject}: manifest source "
                f"{row['acquisition_source']} does not match "
                f"filename {volume_path.name}"
            )
            records.append(record)
            continue

        # ----------------------------------------------------
        # Existence
        # ----------------------------------------------------

        if not volume_path.exists():
            record["message"] = "missing_file"
            errors.append(
                f"{subject}: missing MRI file "
                f"{volume_path}"
            )
            records.append(record)
            continue

        record["exists"] = True

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        try:
            image = sitk.ReadImage(
                str(volume_path),
                sitk.sitkFloat32,
            )
        except Exception as exc:
            record["message"] = (
                f"read_error:{type(exc).__name__}"
            )
            errors.append(
                f"{subject}: cannot read MRI "
                f"{volume_path}"
            )
            records.append(record)
            continue

        record["readable"] = True

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        geometry_valid, geometry_message = geometry_matches(
            image,
            template,
        )

        record["geometry_valid"] = geometry_valid

        record["shape"] = str(image.GetSize())
        record["spacing"] = str(image.GetSpacing())

        if not geometry_valid:
            record["message"] = geometry_message
            errors.append(
                f"{subject}: geometry invalid: "
                f"{geometry_message}"
            )
            records.append(record)
            continue

        # ----------------------------------------------------
        # Voxel data
        # ----------------------------------------------------

        data = sitk.GetArrayViewFromImage(image)

        if data.size == 0:
            record["message"] = "empty_volume"
            errors.append(
                f"{subject}: empty volume"
            )
            records.append(record)
            continue

        finite = bool(np.isfinite(data).all())
        record["finite"] = finite

        if not finite:
            record["message"] = "nonfinite_values"
            errors.append(
                f"{subject}: NaN/Inf voxel values"
            )
            records.append(record)
            continue

        minimum = float(np.min(data))
        maximum = float(np.max(data))
        mean = float(np.mean(data))
        std = float(np.std(data))

        record["minimum"] = minimum
        record["maximum"] = maximum
        record["mean"] = mean
        record["std"] = std

        # ----------------------------------------------------
        # Intensity range
        # ----------------------------------------------------

        range_valid = (
            minimum >= INTENSITY_MIN - INTENSITY_ATOL
            and maximum <= INTENSITY_MAX + INTENSITY_ATOL
        )

        record["range_valid"] = range_valid

        if not range_valid:
            record["message"] = (
                f"intensity_range:{minimum:.6f}"
                f"..{maximum:.6f}"
            )
            errors.append(
                f"{subject}: intensity range outside "
                f"[0,1]: {minimum:.6f}..{maximum:.6f}"
            )
            records.append(record)
            continue

        record["status"] = "PASS"
        record["message"] = "valid"

        records.append(record)

        print(
            f"[{index + 1:03d}/{len(df):03d}] "
            f"{subject} | "
            f"{filename_source:<6} | "
            f"label={int(row['label'])} | PASS"
        )

    results = pd.DataFrame(records)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    passed = int(
        (results["status"] == "PASS").sum()
    )

    failed = len(results) - passed

    source_counts = (
        df["acquisition_source"]
        .value_counts()
        .to_dict()
    )

    print()
    print("=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Expected subjects : {EXPECTED_SUBJECTS}")
    print(f"Checked subjects  : {len(results)}")
    print(f"Passed            : {passed}")
    print(f"Failed            : {failed}")
    print(
        f"RMS selected      : "
        f"{int(source_counts.get('RMS', 0))}"
    )
    print(
        f"Legacy selected   : "
        f"{int(source_counts.get('LEGACY', 0))}"
    )

    # --------------------------------------------------------
    # Distribution diagnostics
    # --------------------------------------------------------

    valid_results = results[
        results["status"] == "PASS"
    ]

    if not valid_results.empty:

        print()
        print("Intensity diagnostics across valid volumes:")
        print(
            f"  Minimum voxel : "
            f"{valid_results['minimum'].min():.6f}"
        )
        print(
            f"  Maximum voxel : "
            f"{valid_results['maximum'].max():.6f}"
        )
        print(
            f"  Mean range   : "
            f"{valid_results['mean'].min():.6f} .. "
            f"{valid_results['mean'].max():.6f}"
        )
        print(
            f"  Std range    : "
            f"{valid_results['std'].min():.6f} .. "
            f"{valid_results['std'].max():.6f}"
        )

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = [
        "NeuroFusion AI - 3D CNN sMRI Input Validation",
        "=" * 72,
        f"Manifest                  : {MANIFEST}",
        f"Expected subjects        : {EXPECTED_SUBJECTS}",
        f"Checked subjects         : {len(results)}",
        f"Passed                   : {passed}",
        f"Failed                   : {failed}",
        f"RMS selected             : {int(source_counts.get('RMS', 0))}",
        f"Legacy selected          : {int(source_counts.get('LEGACY', 0))}",
        "",
        "Expected common MNI geometry:",
        f"  Size                   : {EXPECTED_SIZE}",
        f"  Spacing                : {EXPECTED_SPACING}",
        "",
        "Expected intensity convention:",
        "  Range                  : [0, 1]",
        "",
    ]

    if not valid_results.empty:
        report_lines.extend(
            [
                "Intensity diagnostics:",
                f"  Minimum voxel         : {valid_results['minimum'].min():.6f}",
                f"  Maximum voxel         : {valid_results['maximum'].max():.6f}",
                f"  Mean minimum          : {valid_results['mean'].min():.6f}",
                f"  Mean maximum          : {valid_results['mean'].max():.6f}",
                f"  Std minimum           : {valid_results['std'].min():.6f}",
                f"  Std maximum           : {valid_results['std'].max():.6f}",
                "",
            ]
        )

    report_lines.append("Subject-level results:")

    for _, result in results.iterrows():
        report_lines.append(
            f"  {result['subject_id']} | "
            f"{result['source']} | "
            f"label={result['label']} | "
            f"{result['status']} | "
            f"{result['message']}"
        )

    report_lines.extend(
        [
            "",
            "Validation errors:",
        ]
    )

    if errors:
        report_lines.extend(
            f"  - {error}"
            for error in errors
        )
        final_status = "FAIL"
    else:
        report_lines.append("  None")
        final_status = "PASS"

    report_lines.extend(
        [
            "",
            f"FINAL STATUS: {final_status}",
        ]
    )

    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print(f"Report            : {REPORT}")

    if errors:
        print()
        print(
            f"FINAL STATUS: FAIL "
            f"({len(errors)} issue(s))"
        )
        raise SystemExit(1)

    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()