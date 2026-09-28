"""
Build and validate the subject-level sMRI manifest for NeuroFusion AI.

Sources
-------
MRI:
  1. processed_data/smri/registered/*_rms_n4_mni.nii.gz
  2. processed_data/smri/legacy_registered/*_legacy_n4_mni.nii.gz

AAL:
  processed_data/smri/features/aal/

Clinical labels:
  Downloads/assessment_data.zip
  -> 1139_Demographics_20260804.csv

Primary binary labels
---------------------
0 = Control, Young Control, Old Control
1 = Schizophrenia, Chronic Sz, Early Sz
Excluded = Schizoaffective, Bipolar, unknown/unusable

Selection rule
--------------
RMS is preferred whenever an RMS representation exists.
Legacy is used only when RMS is unavailable.

The script never uses the old generic *_mni.nii.gz files from
processed_data/smri/registered/ because those belong to the previous
STANDARD processing lineage.

Outputs
-------
processed_data/smri/dataset/smri_subject_manifest_all.csv
processed_data/smri/dataset/smri_subject_manifest_labeled.csv
processed_data/smri/dataset/smri_manifest_validation_report.txt
"""

from pathlib import Path
import csv
import os
import re
import zipfile

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

CURRENT_REGISTERED_DIR = ROOT / "registered"
LEGACY_REGISTERED_DIR = ROOT / "legacy_registered"
AAL_DIR = ROOT / "features" / "aal"

OUTPUT_DIR = ROOT / "dataset"

ALL_MANIFEST = OUTPUT_DIR / "smri_subject_manifest_all.csv"
LABELED_MANIFEST = OUTPUT_DIR / "smri_subject_manifest_labeled.csv"
VALIDATION_REPORT = OUTPUT_DIR / "smri_manifest_validation_report.txt"

MNI_TEMPLATE = ROOT / "mni" / "mni152_template_2mm.nii.gz"

ASSESSMENT_ZIP = (
    Path(os.path.expandvars(r"%USERPROFILE%"))
    / "Downloads"
    / "assessment_data.zip"
)

DEMOGRAPHICS_MEMBER = "1139_Demographics_20260804.csv"


# ============================================================
# LABEL MAPPING
# ============================================================

CONTROL_TYPES = {
    "Control",
    "Young Control",
    "Old Control",
}

SCHIZOPHRENIA_TYPES = {
    "Schizophrenia",
    "Chronic Sz",
    "Early Sz",
}

EXCLUDED_TYPES = {
    "Schizoaffective",
    "Bipolar",
}


# ============================================================
# FILENAME PATTERNS
# ============================================================

RMS_PATTERN = re.compile(
    r"^(A\d{8})_rms_n4_mni\.nii\.gz$"
)

LEGACY_PATTERN = re.compile(
    r"^(A\d{8})_legacy_n4_mni\.nii\.gz$"
)

RMS_AAL_PATTERN = re.compile(
    r"^(A\d{8})_rms_aal_features\.csv$"
)

LEGACY_AAL_PATTERN = re.compile(
    r"^(A\d{8})_aal_features\.csv$"
)

SUBJECT_PATTERN = re.compile(r"^A\d{8}$")

AAL_FEATURE_PREFIX = "sMRI_AAL_"


# ============================================================
# VALIDATION
# ============================================================

EXPECTED_TOTAL_MRI_SUBJECTS = 193
EXPECTED_LABELED_SUBJECTS = 172
EXPECTED_CONTROL = 93
EXPECTED_SCHIZOPHRENIA = 79
EXPECTED_AAL_FEATURES = 116

GEOMETRY_ATOL = 1e-6


def validate_mni_volume(path: Path, template: sitk.Image) -> tuple[bool, str]:
    """Validate a selected MNI volume against the common template."""
    if not path.exists():
        return False, "missing"

    try:
        image = sitk.ReadImage(str(path))
    except Exception as exc:
        return False, f"read_error:{type(exc).__name__}"

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

    data = sitk.GetArrayViewFromImage(image)

    if data.size == 0:
        return False, "empty"

    if not np.isfinite(data).all():
        return False, "nonfinite"

    return True, "valid"


def read_and_validate_aal(
    path: Path,
    subject_id: str,
) -> tuple[bool, str, pd.DataFrame | None]:
    """Validate one subject-level AAL CSV."""
    if not path.exists():
        return False, "missing", None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"read_error:{type(exc).__name__}", None

    if "subject_id" not in df.columns:
        return False, "missing_subject_id_column", None

    if len(df) != 1:
        return False, f"row_count:{len(df)}", None

    actual_id = str(df.iloc[0]["subject_id"]).strip()

    if actual_id != subject_id:
        return False, f"subject_id_mismatch:{actual_id}", None

    feature_columns = [
        c for c in df.columns
        if str(c).startswith(AAL_FEATURE_PREFIX)
    ]

    if len(feature_columns) != EXPECTED_AAL_FEATURES:
        return False, f"aal_feature_count:{len(feature_columns)}", None

    values = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    if values.isna().any().any():
        return False, "aal_non_numeric_or_nan", None

    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        return False, "aal_nonfinite", None

    return True, "valid", df


def load_demographics() -> pd.DataFrame:
    """Read demographics directly from the untouched assessment ZIP."""
    if not ASSESSMENT_ZIP.exists():
        raise FileNotFoundError(
            f"Assessment ZIP not found: {ASSESSMENT_ZIP}"
        )

    with zipfile.ZipFile(ASSESSMENT_ZIP, "r") as zf:
        names = zf.namelist()

        if DEMOGRAPHICS_MEMBER not in names:
            raise FileNotFoundError(
                f"{DEMOGRAPHICS_MEMBER} not found inside {ASSESSMENT_ZIP}"
            )

        with zf.open(DEMOGRAPHICS_MEMBER) as handle:
            demographics = pd.read_csv(handle)

    required = {"Anonymized ID", "Subject Type"}

    missing = required.difference(demographics.columns)

    if missing:
        raise RuntimeError(
            f"Demographics is missing required columns: {sorted(missing)}"
        )

    demographics = demographics.copy()

    demographics["subject_id"] = (
        demographics["Anonymized ID"]
        .astype(str)
        .str.strip()
    )

    demographics["subject_type"] = (
        demographics["Subject Type"]
        .astype(str)
        .str.strip()
    )

    # Only valid anonymized IDs are relevant to the MRI merge.
    demographics = demographics[
        demographics["subject_id"].map(
            lambda x: bool(SUBJECT_PATTERN.fullmatch(x))
        )
    ].copy()

    # The earlier audit established that no subject has multiple
    # Subject Types. We enforce that here rather than silently resolving
    # conflicting diagnostic labels.
    conflicts = (
        demographics.groupby("subject_id")["subject_type"]
        .nunique(dropna=False)
    )

    conflicting_ids = conflicts[conflicts > 1].index.tolist()

    if conflicting_ids:
        raise RuntimeError(
            "Multiple Subject Types found for subjects: "
            + ", ".join(conflicting_ids)
        )

    demographics = (
        demographics[
            ["subject_id", "subject_type"]
        ]
        .drop_duplicates(subset=["subject_id"])
        .reset_index(drop=True)
    )

    def make_label(subject_type: str):
        if subject_type in CONTROL_TYPES:
            return 0
        if subject_type in SCHIZOPHRENIA_TYPES:
            return 1
        return np.nan

    demographics["label"] = demographics["subject_type"].map(make_label)

    return demographics


def discover_mri_subjects() -> dict[str, dict]:
    """Discover RMS and legacy registered volumes using exact patterns."""
    subjects: dict[str, dict] = {}

    # Current RMS branch.
    for path in sorted(CURRENT_REGISTERED_DIR.glob("*.nii.gz")):
        match = RMS_PATTERN.fullmatch(path.name)
        if not match:
            continue

        subject = match.group(1)

        subjects.setdefault(
            subject,
            {
                "rms_volume": None,
                "legacy_volume": None,
            },
        )

        if subjects[subject]["rms_volume"] is not None:
            raise RuntimeError(
                f"Duplicate RMS MNI volume for {subject}"
            )

        subjects[subject]["rms_volume"] = path

    # Legacy branch.
    for path in sorted(LEGACY_REGISTERED_DIR.glob("*.nii.gz")):
        match = LEGACY_PATTERN.fullmatch(path.name)
        if not match:
            continue

        subject = match.group(1)

        subjects.setdefault(
            subject,
            {
                "rms_volume": None,
                "legacy_volume": None,
            },
        )

        if subjects[subject]["legacy_volume"] is not None:
            raise RuntimeError(
                f"Duplicate legacy MNI volume for {subject}"
            )

        subjects[subject]["legacy_volume"] = path

    return subjects


def discover_aal_files() -> dict[str, dict]:
    """
    Discover RMS and legacy AAL files.

    RMS AAL has priority for the overlap subject.
    """
    aal: dict[str, dict] = {}

    for path in sorted(AAL_DIR.glob("*.csv")):
        match_rms = RMS_AAL_PATTERN.fullmatch(path.name)

        if match_rms:
            subject = match_rms.group(1)

            aal.setdefault(
                subject,
                {
                    "rms_aal": None,
                    "legacy_aal": None,
                },
            )

            if aal[subject]["rms_aal"] is not None:
                raise RuntimeError(
                    f"Duplicate RMS AAL file for {subject}"
                )

            aal[subject]["rms_aal"] = path
            continue

        match_legacy = LEGACY_AAL_PATTERN.fullmatch(path.name)

        if match_legacy:
            subject = match_legacy.group(1)

            aal.setdefault(
                subject,
                {
                    "rms_aal": None,
                    "legacy_aal": None,
                },
            )

            if aal[subject]["legacy_aal"] is not None:
                raise RuntimeError(
                    f"Duplicate legacy AAL file for {subject}"
                )

            aal[subject]["legacy_aal"] = path

    return aal


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build all-subject and labeled manifests and collect validation notes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    template = sitk.ReadImage(
        str(MNI_TEMPLATE),
        sitk.sitkFloat32,
    )

    demographics = load_demographics()
    mri = discover_mri_subjects()
    aal = discover_aal_files()

    validation_errors: list[str] = []

    rows = []

    for subject in sorted(mri):
        rms_volume = mri[subject]["rms_volume"]
        legacy_volume = mri[subject]["legacy_volume"]

        # ----------------------------------------------------
        # Explicit RMS priority
        # ----------------------------------------------------
        if rms_volume is not None:
            selected_volume = rms_volume
            acquisition_source = "RMS"
            alternative_volume = legacy_volume
        elif legacy_volume is not None:
            selected_volume = legacy_volume
            acquisition_source = "LEGACY"
            alternative_volume = None
        else:
            validation_errors.append(
                f"{subject}: no selectable MNI volume"
            )
            continue

        # ----------------------------------------------------
        # Validate selected MNI volume
        # ----------------------------------------------------
        volume_valid, volume_message = validate_mni_volume(
            selected_volume,
            template,
        )

        if not volume_valid:
            validation_errors.append(
                f"{subject}: selected volume invalid: {volume_message}"
            )

        # If both representations exist, retain the fact in the
        # manifest for traceability, but never create a duplicate row.
        has_rms = rms_volume is not None
        has_legacy = legacy_volume is not None

        # ----------------------------------------------------
        # Select matching AAL representation
        # ----------------------------------------------------
        aal_info = aal.get(
            subject,
            {
                "rms_aal": None,
                "legacy_aal": None,
            },
        )

        if acquisition_source == "RMS":
            selected_aal = aal_info["rms_aal"]
            if selected_aal is None:
                validation_errors.append(
                    f"{subject}: RMS volume selected but RMS AAL missing"
                )
        else:
            selected_aal = aal_info["legacy_aal"]
            if selected_aal is None:
                validation_errors.append(
                    f"{subject}: legacy volume selected but legacy AAL missing"
                )

        aal_valid = False
        aal_message = "not_checked"

        if selected_aal is not None:
            aal_valid, aal_message, _ = read_and_validate_aal(
                selected_aal,
                subject,
            )

            if not aal_valid:
                validation_errors.append(
                    f"{subject}: selected AAL invalid: {aal_message}"
                )

        # ----------------------------------------------------
        # Clinical label
        # ----------------------------------------------------
        clinical = demographics[
            demographics["subject_id"] == subject
        ]

        if len(clinical) == 0:
            subject_type = ""
            label = np.nan
            label_status = "NO_ASSESSMENT_MATCH"
        elif len(clinical) == 1:
            subject_type = str(clinical.iloc[0]["subject_type"])
            label_value = clinical.iloc[0]["label"]

            if pd.isna(label_value):
                label = np.nan
                if subject_type in EXCLUDED_TYPES:
                    label_status = "EXCLUDED_DIAGNOSTIC_GROUP"
                else:
                    label_status = "UNUSABLE_LABEL"
            else:
                label = int(label_value)
                label_status = "USABLE_BINARY_LABEL"
        else:
            raise RuntimeError(
                f"Unexpected duplicate demographics rows for {subject}"
            )

        rows.append(
            {
                "subject_id": subject,
                "mni_volume": str(selected_volume),
                "aal_csv": (
                    str(selected_aal)
                    if selected_aal is not None
                    else ""
                ),
                "acquisition_source": acquisition_source,
                "has_rms_representation": int(has_rms),
                "has_legacy_representation": int(has_legacy),
                "alternative_volume_not_selected": (
                    str(alternative_volume)
                    if alternative_volume is not None
                    else ""
                ),
                "subject_type": subject_type,
                "label": label,
                "label_status": label_status,
                "mni_valid": int(volume_valid),
                "aal_valid": int(aal_valid),
            }
        )

    all_df = pd.DataFrame(rows)

    if all_df.empty:
        raise RuntimeError("No MRI subjects were discovered.")

    # --------------------------------------------------------
    # Strict subject uniqueness
    # --------------------------------------------------------
    duplicate_ids = all_df[
        all_df["subject_id"].duplicated(keep=False)
    ]["subject_id"].tolist()

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate subject IDs in manifest: "
            + ", ".join(sorted(set(duplicate_ids)))
        )

    # --------------------------------------------------------
    # Expected total MRI cohort
    # --------------------------------------------------------
    if len(all_df) != EXPECTED_TOTAL_MRI_SUBJECTS:
        validation_errors.append(
            f"Expected {EXPECTED_TOTAL_MRI_SUBJECTS} unique MRI subjects; "
            f"found {len(all_df)}"
        )

    # --------------------------------------------------------
    # Labeled cohort
    # --------------------------------------------------------
    labeled_df = all_df[
        all_df["label_status"] == "USABLE_BINARY_LABEL"
    ].copy()

    labeled_df["label"] = labeled_df["label"].astype(int)

    if len(labeled_df) != EXPECTED_LABELED_SUBJECTS:
        validation_errors.append(
            f"Expected {EXPECTED_LABELED_SUBJECTS} labeled subjects; "
            f"found {len(labeled_df)}"
        )

    class_counts = labeled_df["label"].value_counts().to_dict()

    control_count = int(class_counts.get(0, 0))
    schizophrenia_count = int(class_counts.get(1, 0))

    if control_count != EXPECTED_CONTROL:
        validation_errors.append(
            f"Expected {EXPECTED_CONTROL} controls; "
            f"found {control_count}"
        )

    if schizophrenia_count != EXPECTED_SCHIZOPHRENIA:
        validation_errors.append(
            f"Expected {EXPECTED_SCHIZOPHRENIA} schizophrenia-spectrum; "
            f"found {schizophrenia_count}"
        )

    # --------------------------------------------------------
    # Label integrity
    # --------------------------------------------------------
    invalid_labeled = labeled_df[
        ~labeled_df["label"].isin([0, 1])
    ]

    if not invalid_labeled.empty:
        validation_errors.append(
            "Non-binary labels found for: "
            + ", ".join(invalid_labeled["subject_id"].tolist())
        )

    # --------------------------------------------------------
    # Selected representation integrity
    # --------------------------------------------------------
    bad_sources = all_df[
        ~all_df["acquisition_source"].isin(["RMS", "LEGACY"])
    ]

    if not bad_sources.empty:
        validation_errors.append(
            "Invalid acquisition source(s): "
            + ", ".join(
                sorted(bad_sources["acquisition_source"].unique())
            )
        )

    # A subject with both representations must select RMS.
    overlap = all_df[
        (all_df["has_rms_representation"] == 1)
        & (all_df["has_legacy_representation"] == 1)
    ]

    non_rms_overlap = overlap[
        overlap["acquisition_source"] != "RMS"
    ]

    if not non_rms_overlap.empty:
        validation_errors.append(
            "RMS priority violated for: "
            + ", ".join(non_rms_overlap["subject_id"].tolist())
        )

    # --------------------------------------------------------
    # Validate every labeled row has valid selected inputs
    # --------------------------------------------------------
    invalid_labeled_inputs = labeled_df[
        (labeled_df["mni_valid"] != 1)
        | (labeled_df["aal_valid"] != 1)
    ]

    if not invalid_labeled_inputs.empty:
        validation_errors.append(
            "Labeled subjects with invalid MRI/AAL inputs: "
            + ", ".join(
                invalid_labeled_inputs["subject_id"].tolist()
            )
        )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------
    all_columns = [
        "subject_id",
        "mni_volume",
        "aal_csv",
        "acquisition_source",
        "has_rms_representation",
        "has_legacy_representation",
        "alternative_volume_not_selected",
        "subject_type",
        "label",
        "label_status",
        "mni_valid",
        "aal_valid",
    ]

    all_df = all_df[all_columns].sort_values(
        "subject_id"
    ).reset_index(drop=True)

    labeled_df = labeled_df[all_columns].sort_values(
        "subject_id"
    ).reset_index(drop=True)

    all_df.to_csv(
        ALL_MANIFEST,
        index=False,
        encoding="utf-8",
    )

    labeled_df.to_csv(
        LABELED_MANIFEST,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------
    overlap_ids = overlap["subject_id"].tolist()

    legacy_only = all_df[
        all_df["acquisition_source"] == "LEGACY"
    ]

    rms_selected = all_df[
        all_df["acquisition_source"] == "RMS"
    ]

    report_lines = [
        "NeuroFusion AI - sMRI Subject-Level Manifest Validation",
        "=" * 68,
        f"Unique MRI subjects discovered : {len(all_df)}",
        f"Expected MRI subjects          : {EXPECTED_TOTAL_MRI_SUBJECTS}",
        f"RMS-selected subjects          : {len(rms_selected)}",
        f"Legacy-selected subjects       : {len(legacy_only)}",
        f"Subjects with both reps        : {len(overlap)}",
        f"Labeled subjects               : {len(labeled_df)}",
        f"Expected labeled subjects      : {EXPECTED_LABELED_SUBJECTS}",
        f"Control (label 0)              : {control_count}",
        f"Schizophrenia-spectrum (1)     : {schizophrenia_count}",
        "",
        "Overlap subjects:",
        ", ".join(overlap_ids) if overlap_ids else "None",
        "",
        "Label-status counts:",
    ]

    for status, count in (
        all_df["label_status"]
        .value_counts()
        .sort_index()
        .items()
    ):
        report_lines.append(f"  {status}: {count}")

    report_lines.extend(
        [
            "",
            "Validation errors:",
        ]
    )

    if validation_errors:
        report_lines.extend(
            f"  - {error}" for error in validation_errors
        )
        final_status = "FAIL"
    else:
        report_lines.append("  None")
        final_status = "PASS"

    report_lines.extend(
        [
            "",
            f"FINAL STATUS: {final_status}",
            "",
            f"All-subject manifest : {ALL_MANIFEST}",
            f"Labeled manifest     : {LABELED_MANIFEST}",
        ]
    )

    VALIDATION_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    return all_df, labeled_df, validation_errors


def main() -> None:
    print("=" * 68)
    print("NeuroFusion AI - BUILD sMRI SUBJECT MANIFEST")
    print("=" * 68)
    print()

    all_df, labeled_df, errors = build_manifest()

    print(f"Unique MRI subjects : {len(all_df)}")
    print(f"Labeled subjects    : {len(labeled_df)}")
    print(
        "Class counts        : "
        f"0={int((labeled_df['label'] == 0).sum())}, "
        f"1={int((labeled_df['label'] == 1).sum())}"
    )

    print()
    print(f"All manifest        : {ALL_MANIFEST}")
    print(f"Labeled manifest    : {LABELED_MANIFEST}")
    print(f"Validation report   : {VALIDATION_REPORT}")

    print()

    if errors:
        print(f"FINAL STATUS: FAIL ({len(errors)} validation issue(s))")
        raise SystemExit(1)

    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()
