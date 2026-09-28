"""
NeuroFusion AI
RMS sMRI CNN Input Pipeline Audit

Purpose
-------
Independent, READ-ONLY audit of the existing 119-subject RMS sMRI cohort.

This script:
    1. Audits the RMS labeled cohort from the existing manifest.
    2. Verifies subject inventory and diagnosis distribution.
    3. Verifies manifest -> file linkage.
    4. Audits the existing preprocessing lineage:
           reoriented
           -> N4
           -> HD-BET
           -> masked
           -> normalized
           -> MNI registered
           -> CNN .npy
    5. Audits NIfTI geometry, affine, orientation, finite values,
       intensity statistics, brain-mask plausibility and tensor integrity.
    6. Independently compares registered MNI NIfTI against the existing
       CNN .npy tensor.
    7. Audits the existing RMS nested-CV assignments.
    8. Produces subject-level and summary audit reports.

IMPORTANT:
    - This script NEVER modifies existing project files.
    - It only creates NEW files under:
          processed_data/smri/audit/rms_input_audit
    - It does NOT regenerate preprocessing.
    - It does NOT regenerate the CNN dataset.
    - It does NOT regenerate CV splits.
    - It does NOT overwrite model experiment outputs.

Expected project root:
    C:/Users/User/Documents/neurofusion
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit(
        "ERROR: nibabel is required. Install it in the active environment."
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SMRI_ROOT = PROJECT_ROOT / "processed_data" / "smri"

MANIFEST_PATH = (
    SMRI_ROOT / "dataset" / "smri_subject_manifest_labeled.csv"
)

ALL_MANIFEST_PATH = (
    SMRI_ROOT / "dataset" / "smri_subject_manifest_all.csv"
)

CNN_ROOT = SMRI_ROOT / "cnn_dataset"
CNN_IMAGES_DIR = CNN_ROOT / "images"

RMS_CV_DIR = (
    SMRI_ROOT
    / "cnn_dataset"
    / "cross_validation"
    / "nested"
)

# Existing preprocessing directories.
REORIENTED_DIR = SMRI_ROOT / "reoriented"
N4_DIR = SMRI_ROOT / "n4"
BET_DIR = SMRI_ROOT / "mprage_5e_bet"
MASKED_DIR = SMRI_ROOT / "mprage_5e_masked"
NORMALIZED_DIR = SMRI_ROOT / "mprage_5e_normalized"
REGISTERED_DIR = SMRI_ROOT / "registered"
TRANSFORMS_DIR = SMRI_ROOT / "transforms"
SELECTION_DIR = SMRI_ROOT / "mprage_selection"

AUDIT_ROOT = SMRI_ROOT / "audit" / "rms_input_audit"

SUBJECT_PATTERN = re.compile(r"^A\d{8}$")

EXPECTED_SUBJECTS = 119
EXPECTED_CONTROLS = 40
EXPECTED_SCHIZOPHRENIA = 79

EXPECTED_SHAPE = (95, 117, 99)
EXPECTED_SPACING = (2.0, 2.0, 2.0)

# Numerical tolerances.
SPACING_TOL = 0.05
AFFINE_TOL = 1e-3

# CNN <-> MNI comparison tolerances.
# Exact equality is preferred. Small floating point differences are allowed.
CNN_EXACT_ATOL = 1e-6
CNN_CLOSE_ATOL = 1e-4
CNN_CORRELATION_WARNING = 0.999999

# Plausibility thresholds.
MIN_NONFINITE_FREE_FRACTION = 1.0
MIN_NONZERO_FRACTION_WARNING = 0.005
MIN_NONZERO_FRACTION_FAIL = 1e-6

# Mask fraction is intentionally broad because MNI volumes may contain
# interpolation/background differences.
MIN_BRAIN_FRACTION_WARNING = 0.01
MAX_BRAIN_FRACTION_WARNING = 0.95


# ============================================================
# OUTPUT PATHS
# ============================================================

SUBJECT_REPORT_PATH = AUDIT_ROOT / "rms_subject_level_audit.csv"
SUMMARY_REPORT_PATH = AUDIT_ROOT / "rms_audit_summary.csv"
TEXT_REPORT_PATH = AUDIT_ROOT / "rms_input_audit_report.txt"
CV_REPORT_PATH = AUDIT_ROOT / "rms_cv_audit.csv"
LINEAGE_REPORT_PATH = AUDIT_ROOT / "rms_lineage_inventory.csv"
DIAGNOSTIC_REPORT_PATH = AUDIT_ROOT / "rms_audit_diagnostics.csv"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_subject_id(value) -> str:
    value = safe_str(value).upper()

    # Accept A00012345 directly.
    if SUBJECT_PATTERN.fullmatch(value):
        return value

    # Extract a valid subject ID if embedded in a path/name.
    match = re.search(r"(A\d{8})", value)
    if match:
        return match.group(1)

    return value


def relative_or_absolute_path(value: str) -> Optional[Path]:
    """
    Resolve a manifest path without modifying anything.

    Handles:
      - absolute Windows paths
      - project-relative paths
      - paths containing processed_data/smri
    """
    if not value:
        return None

    raw = Path(value)

    if raw.is_absolute():
        return raw

    candidate1 = PROJECT_ROOT / raw

    if candidate1.exists():
        return candidate1

    # Some manifests may store a path beginning at processed_data.
    candidate2 = PROJECT_ROOT / Path(str(raw).replace("\\", "/"))

    if candidate2.exists():
        return candidate2

    return candidate1


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate MD5 without modifying the source file.
    Used only for audit traceability.
    """
    digest = hashlib.md5()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def finite_stats(data: np.ndarray) -> Dict[str, float]:
    result = {
        "finite": False,
        "min": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "median": np.nan,
        "p01": np.nan,
        "p05": np.nan,
        "p25": np.nan,
        "p75": np.nan,
        "p95": np.nan,
        "p99": np.nan,
        "nonzero_fraction": np.nan,
        "negative_fraction": np.nan,
        "positive_fraction": np.nan,
        "zero_fraction": np.nan,
    }

    arr = np.asarray(data)

    if arr.size == 0:
        return result

    finite_mask = np.isfinite(arr)

    if not np.all(finite_mask):
        return result

    flat = arr.astype(np.float64, copy=False).ravel()

    result["finite"] = True
    result["min"] = float(np.min(flat))
    result["max"] = float(np.max(flat))
    result["mean"] = float(np.mean(flat))
    result["std"] = float(np.std(flat))
    result["median"] = float(np.median(flat))

    percentiles = np.percentile(
        flat,
        [1, 5, 25, 75, 95, 99]
    )

    result["p01"] = float(percentiles[0])
    result["p05"] = float(percentiles[1])
    result["p25"] = float(percentiles[2])
    result["p75"] = float(percentiles[3])
    result["p95"] = float(percentiles[4])
    result["p99"] = float(percentiles[5])

    result["nonzero_fraction"] = float(np.mean(flat != 0))
    result["negative_fraction"] = float(np.mean(flat < 0))
    result["positive_fraction"] = float(np.mean(flat > 0))
    result["zero_fraction"] = float(np.mean(flat == 0))

    return result


def geometry_from_nifti(path: Path) -> Dict[str, object]:
    result = {
        "exists": path.exists(),
        "readable": False,
        "shape": None,
        "dtype": None,
        "spacing": None,
        "orientation": None,
        "affine": None,
        "finite": False,
        "error": "",
    }

    if not path.exists():
        return result

    try:
        image = nib.load(str(path))

        result["readable"] = True
        result["shape"] = tuple(int(x) for x in image.shape)
        result["dtype"] = str(image.get_data_dtype())
        result["spacing"] = tuple(
            float(x) for x in image.header.get_zooms()[:3]
        )
        result["orientation"] = "".join(
            nib.aff2axcodes(image.affine)
        )
        result["affine"] = image.affine.tolist()

        data = np.asanyarray(image.dataobj)

        result["finite"] = bool(np.all(np.isfinite(data)))

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def load_nifti_data(path: Path) -> Tuple[Optional[nib.Nifti1Image], Optional[np.ndarray], str]:
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)

        if not np.all(np.isfinite(data)):
            return image, None, "NIfTI contains non-finite values."

        return image, np.asarray(data), ""

    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def check_shape(shape, expected_shape=EXPECTED_SHAPE) -> str:
    if shape is None:
        return "FAIL"

    try:
        shape_tuple = tuple(int(x) for x in shape)
    except Exception:
        return "FAIL"

    return ("PASS" if shape_tuple == expected_shape or shape_tuple == tuple(reversed(expected_shape)) else "WARNING")


def check_spacing(spacing) -> str:
    if spacing is None:
        return "FAIL"

    try:
        values = np.asarray(spacing, dtype=float)
    except Exception:
        return "FAIL"

    expected = np.asarray(EXPECTED_SPACING, dtype=float)

    if values.shape != expected.shape:
        return "FAIL"

    return (
        "PASS"
        if np.allclose(values, expected, atol=SPACING_TOL, rtol=0)
        else "WARNING"
    )


def check_affine(affine) -> str:
    if affine is None:
        return "FAIL"

    try:
        matrix = np.asarray(affine, dtype=float)
    except Exception:
        return "FAIL"

    if matrix.shape != (4, 4):
        return "FAIL"

    if not np.all(np.isfinite(matrix)):
        return "FAIL"

    if abs(np.linalg.det(matrix[:3, :3])) < 1e-8:
        return "FAIL"

    return "PASS"


def constant_status(stats: Dict[str, float]) -> str:
    if not stats["finite"]:
        return "FAIL"

    std = stats["std"]

    if not np.isfinite(std):
        return "FAIL"

    if std == 0:
        return "FAIL"

    return "PASS"


def nonzero_status(stats: Dict[str, float]) -> str:
    if not stats["finite"]:
        return "FAIL"

    fraction = stats["nonzero_fraction"]

    if fraction <= MIN_NONZERO_FRACTION_FAIL:
        return "FAIL"

    if fraction < MIN_NONZERO_FRACTION_WARNING:
        return "WARNING"

    return "PASS"


def overall_status(statuses: List[str]) -> str:
    """
    Priority:
       FAIL > WARNING > PASS
    """
    if "FAIL" in statuses:
        return "FAIL"

    if "WARNING" in statuses:
        return "WARNING"

    return "PASS"


def compare_arrays(
    mni_data: np.ndarray,
    cnn_data: np.ndarray,
) -> Dict[str, object]:

    result = {
        "cnn_mni_shape_match": False,
        "cnn_mni_exact_match": False,
        "cnn_mni_close_match": False,
        "cnn_mni_correlation": np.nan,
        "cnn_mni_mean_abs_diff": np.nan,
        "cnn_mni_max_abs_diff": np.nan,
        "cnn_mni_rmse": np.nan,
        "cnn_mni_status": "FAIL",
        "cnn_mni_error": "",
    }

    try:
        mni = np.asarray(mni_data)
        cnn = np.asarray(cnn_data)

        # Expected CNN shape is (1, X, Y, Z).
        if cnn.ndim == 4 and cnn.shape[0] == 1:
            cnn_3d = cnn[0]
        else:
            cnn_3d = cnn

        if mni.ndim == 3 and cnn_3d.ndim == 3 and tuple(cnn_3d.shape) == tuple(reversed(mni.shape)):
            mni = np.transpose(mni, (2, 1, 0))
            result["cnn_mni_axis_conversion"] = "NiBabel XYZ - ZYX"
        else:
            result["cnn_mni_axis_conversion"] = "none"

        result["cnn_mni_shape_match"] = (
            tuple(cnn_3d.shape) == tuple(mni.shape)
        )

        if not result["cnn_mni_shape_match"]:
            result["cnn_mni_error"] = (
                f"MNI shape={mni.shape}, CNN shape={cnn.shape}, "
                f"CNN 3D shape={cnn_3d.shape}"
            )
            result["cnn_mni_status"] = "FAIL"
            return result

        mni_float = mni.astype(np.float64, copy=False)
        cnn_float = cnn_3d.astype(np.float64, copy=False)

        if not np.all(np.isfinite(mni_float)):
            result["cnn_mni_error"] = "MNI contains non-finite values."
            return result

        if not np.all(np.isfinite(cnn_float)):
            result["cnn_mni_error"] = "CNN tensor contains non-finite values."
            return result

        difference = cnn_float - mni_float

        abs_diff = np.abs(difference)

        result["cnn_mni_mean_abs_diff"] = float(np.mean(abs_diff))
        result["cnn_mni_max_abs_diff"] = float(np.max(abs_diff))
        result["cnn_mni_rmse"] = float(
            np.sqrt(np.mean(np.square(difference)))
        )

        result["cnn_mni_exact_match"] = bool(
            np.array_equal(cnn_float, mni_float)
        )

        result["cnn_mni_close_match"] = bool(
            np.allclose(
                cnn_float,
                mni_float,
                atol=CNN_CLOSE_ATOL,
                rtol=1e-5,
            )
        )

        x = mni_float.ravel()
        y = cnn_float.ravel()

        x_std = float(np.std(x))
        y_std = float(np.std(y))

        if x_std > 0 and y_std > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            result["cnn_mni_correlation"] = float(correlation)

        if result["cnn_mni_exact_match"]:
            result["cnn_mni_status"] = "PASS"

        elif (
            result["cnn_mni_close_match"]
            and np.isfinite(result["cnn_mni_correlation"])
            and result["cnn_mni_correlation"] >= CNN_CORRELATION_WARNING
        ):
            result["cnn_mni_status"] = "PASS"

        elif (
            np.isfinite(result["cnn_mni_correlation"])
            and result["cnn_mni_correlation"] >= 0.999
        ):
            result["cnn_mni_status"] = "WARNING"

        else:
            result["cnn_mni_status"] = "FAIL"

    except Exception as exc:
        result["cnn_mni_error"] = f"{type(exc).__name__}: {exc}"
        result["cnn_mni_status"] = "FAIL"

    return result


def expected_rms_paths(subject_id: str) -> Dict[str, Path]:
    return {
        "reoriented": REORIENTED_DIR / f"{subject_id}_rms_reoriented.nii.gz",
        "n4": N4_DIR / f"{subject_id}_rms_n4.nii.gz",
        "bet": BET_DIR / f"{subject_id}_rms_n4_bet.nii.gz",
        "bet_duplicate": BET_DIR / f"{subject_id}_rms_n4_bet_bet.nii.gz",
        "masked": MASKED_DIR / f"{subject_id}_rms_n4_masked.nii.gz",
        "normalized": NORMALIZED_DIR / f"{subject_id}_rms_n4_normalized.nii.gz",
        "registered_expected": REGISTERED_DIR / f"{subject_id}_rms_n4_mni.nii.gz",
        "transform": TRANSFORMS_DIR / f"{subject_id}_to_mni.tfm",
        "cnn": CNN_IMAGES_DIR / f"{subject_id}.npy",
    }


def find_registered_candidates(subject_id: str) -> List[Path]:
    """
    Find all MNI registered candidates for the subject.

    This does not choose silently; the audit records all candidates.
    """
    patterns = [
        f"{subject_id}_rms_n4_mni.nii.gz",
        f"{subject_id}_mni.nii.gz",
    ]

    candidates = []

    for pattern in patterns:
        candidate = REGISTERED_DIR / pattern
        if candidate.exists():
            candidates.append(candidate)

    return candidates


# ============================================================
# MANIFEST AUDIT
# ============================================================

def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Labeled manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = [
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

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            "Manifest is missing expected columns: "
            + ", ".join(missing)
        )

    return manifest


def filter_rms_cohort(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    RMS cohort definition is based on acquisition_source and/or
    RMS representation flags.

    We require exactly 119 final RMS subjects.
    """

    source = (
        manifest["acquisition_source"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    rms_flag = (
        manifest["has_rms_representation"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    rms = manifest.loc[
        (source == "RMS") | rms_flag
    ].copy()

    return rms


# ============================================================
# SUBJECT-LEVEL AUDIT
# ============================================================

def audit_subject(
    row: pd.Series,
) -> Dict[str, object]:

    subject_id = normalize_subject_id(row["subject_id"])

    paths = expected_rms_paths(subject_id)

    statuses = []

    record = {
        "subject_id": subject_id,
        "label": row.get("label"),
        "subject_type": safe_str(row.get("subject_type")),
        "label_status": safe_str(row.get("label_status")),
        "acquisition_source": safe_str(row.get("acquisition_source")),
        "has_rms_representation": row.get("has_rms_representation"),
        "has_legacy_representation": row.get("has_legacy_representation"),

        "manifest_mni_volume": safe_str(row.get("mni_volume")),
        "manifest_aal_csv": safe_str(row.get("aal_csv")),

        "subject_id_status": "PASS",
        "duplicate_subject_id": False,

        # File existence
        "reoriented_exists": False,
        "n4_exists": False,
        "bet_exists": False,
        "bet_duplicate_exists": False,
        "masked_exists": False,
        "normalized_exists": False,
        "registered_expected_exists": False,
        "transform_exists": False,
        "cnn_exists": False,

        # Manifest selected MNI
        "manifest_mni_exists": False,
        "manifest_mni_resolved": "",
        "manifest_mni_subject_match": False,

        # NIfTI geometry/status
        "reoriented_readable": False,
        "reoriented_finite": False,
        "n4_readable": False,
        "n4_finite": False,
        "bet_readable": False,
        "bet_finite": False,
        "masked_readable": False,
        "masked_finite": False,
        "normalized_readable": False,
        "normalized_finite": False,
        "registered_readable": False,
        "registered_finite": False,

        # Geometry
        "registered_shape": None,
        "registered_spacing": None,
        "registered_orientation": None,
        "registered_affine_status": "FAIL",

        # Statistics
        "registered_min": np.nan,
        "registered_max": np.nan,
        "registered_mean": np.nan,
        "registered_std": np.nan,
        "registered_median": np.nan,
        "registered_nonzero_fraction": np.nan,
        "registered_zero_fraction": np.nan,

        "cnn_dtype": "",
        "cnn_shape": None,
        "cnn_finite": False,
        "cnn_min": np.nan,
        "cnn_max": np.nan,
        "cnn_mean": np.nan,
        "cnn_std": np.nan,
        "cnn_median": np.nan,
        "cnn_nonzero_fraction": np.nan,
        "cnn_zero_fraction": np.nan,

        # MNI <-> CNN
        "cnn_mni_shape_match": False,
        "cnn_mni_exact_match": False,
        "cnn_mni_close_match": False,
        "cnn_mni_correlation": np.nan,
        "cnn_mni_mean_abs_diff": np.nan,
        "cnn_mni_max_abs_diff": np.nan,
        "cnn_mni_rmse": np.nan,
        "cnn_mni_status": "FAIL",
        "cnn_mni_error": "",

        # Brain mask
        "brain_mask_fraction": np.nan,
        "brain_mask_status": "WARNING",

        # Lineage
        "lineage_status": "FAIL",

        # Overall
        "status": "FAIL",
        "issues": "",
    }

    # --------------------------------------------------------
    # Basic subject identity
    # --------------------------------------------------------

    if not SUBJECT_PATTERN.fullmatch(subject_id):
        record["subject_id_status"] = "FAIL"
        statuses.append("FAIL")
        record["issues"] = "Invalid subject ID format."

        return record

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    for key in [
        "reoriented",
        "n4",
        "bet",
        "masked",
        "normalized",
        "registered_expected",
        "transform",
        "cnn",
    ]:
        exists = paths[key].exists()
        record[f"{key}_exists"] = exists

        if not exists:
            statuses.append("FAIL")

    record["bet_duplicate_exists"] = paths["bet_duplicate"].exists()

    # --------------------------------------------------------
    # Manifest MNI path
    # --------------------------------------------------------

    manifest_mni_value = safe_str(row.get("mni_volume"))

    resolved_manifest_mni = relative_or_absolute_path(
        manifest_mni_value
    )

    if resolved_manifest_mni is not None:
        record["manifest_mni_resolved"] = str(
            resolved_manifest_mni
        )

        record["manifest_mni_exists"] = (
            resolved_manifest_mni.exists()
        )

        manifest_name = resolved_manifest_mni.name

        record["manifest_mni_subject_match"] = (
            subject_id in manifest_name
        )

    if not record["manifest_mni_exists"]:
        statuses.append("WARNING")

    if not record["manifest_mni_subject_match"]:
        statuses.append("FAIL")

    # --------------------------------------------------------
    # Audit expected registered file
    # --------------------------------------------------------

    registered_path = paths["registered_expected"]

    # If expected file is absent, search candidates so the audit can
    # distinguish a naming/path mismatch from genuinely missing data.
    registered_candidates = find_registered_candidates(subject_id)

    if not registered_path.exists() and registered_candidates:
        registered_path = registered_candidates[0]
        statuses.append("WARNING")

    # --------------------------------------------------------
    # Read and audit preprocessing lineage
    # --------------------------------------------------------

    lineage_files = {
        "reoriented": paths["reoriented"],
        "n4": paths["n4"],
        "bet": paths["bet"],
        "masked": paths["masked"],
        "normalized": paths["normalized"],
        "registered": registered_path,
    }

    loaded_data = {}

    for stage, path in lineage_files.items():

        if not path.exists():
            continue

        image, data, error = load_nifti_data(path)

        readable_key = f"{stage}_readable"
        finite_key = f"{stage}_finite"

        record[readable_key] = image is not None
        record[finite_key] = data is not None

        if image is None:
            statuses.append("FAIL")
            continue

        if data is None:
            statuses.append("FAIL")
            continue

        loaded_data[stage] = {
            "image": image,
            "data": data,
            "path": path,
        }

    # --------------------------------------------------------
    # Registered geometry
    # --------------------------------------------------------

    if "registered" in loaded_data:

        image = loaded_data["registered"]["image"]
        data = loaded_data["registered"]["data"]

        record["registered_shape"] = tuple(
            int(x) for x in image.shape
        )

        record["registered_spacing"] = tuple(
            float(x) for x in image.header.get_zooms()[:3]
        )

        record["registered_orientation"] = "".join(
            nib.aff2axcodes(image.affine)
        )

        shape_status = check_shape(
            record["registered_shape"]
        )

        spacing_status = check_spacing(
            record["registered_spacing"]
        )

        affine_status = check_affine(
            image.affine
        )

        record["registered_affine_status"] = affine_status

        statuses.extend([
            shape_status,
            spacing_status,
            affine_status,
        ])

        stats = finite_stats(data)

        for key in [
            "min",
            "max",
            "mean",
            "std",
            "median",
            "nonzero_fraction",
            "zero_fraction",
        ]:
            record[f"registered_{key}"] = stats[key]

        statuses.append(
            constant_status(stats)
        )

        statuses.append(
            nonzero_status(stats)
        )

    # --------------------------------------------------------
    # Brain-mask plausibility
    # --------------------------------------------------------

    if "masked" in loaded_data:

        masked = loaded_data["masked"]["data"]

        # Since masked volumes should contain background zeros,
        # estimate the fraction of nonzero voxels.
        mask_fraction = float(
            np.mean(np.abs(masked) > 0)
        )

        record["brain_mask_fraction"] = mask_fraction

        if mask_fraction <= 0:
            record["brain_mask_status"] = "FAIL"
            statuses.append("FAIL")

        elif (
            mask_fraction < MIN_BRAIN_FRACTION_WARNING
            or mask_fraction > MAX_BRAIN_FRACTION_WARNING
        ):
            record["brain_mask_status"] = "WARNING"
            statuses.append("WARNING")

        else:
            record["brain_mask_status"] = "PASS"

    else:
        record["brain_mask_status"] = "FAIL"
        statuses.append("FAIL")

    # --------------------------------------------------------
    # CNN .npy
    # --------------------------------------------------------

    cnn_path = paths["cnn"]

    if cnn_path.exists():

        try:
            cnn = np.load(
                cnn_path,
                allow_pickle=False,
            )

            record["cnn_dtype"] = str(cnn.dtype)
            record["cnn_shape"] = tuple(
                int(x) for x in cnn.shape
            )

            record["cnn_finite"] = bool(
                np.all(np.isfinite(cnn))
            )

            if not record["cnn_finite"]:
                statuses.append("FAIL")

            if tuple(cnn.shape) != (1,) + EXPECTED_SHAPE:
                statuses.append("FAIL")

            if cnn.dtype != np.float32:
                statuses.append("WARNING")

            cnn_stats = finite_stats(cnn)

            for key in [
                "min",
                "max",
                "mean",
                "std",
                "median",
                "nonzero_fraction",
                "zero_fraction",
            ]:
                record[f"cnn_{key}"] = cnn_stats[key]

            statuses.append(
                constant_status(cnn_stats)
            )

            statuses.append(
                nonzero_status(cnn_stats)
            )

            # ------------------------------------------------
            # MNI -> CNN independent consistency
            # ------------------------------------------------

            if "registered" in loaded_data:

                comparison = compare_arrays(
                    loaded_data["registered"]["data"],
                    cnn,
                )

                for key, value in comparison.items():
                    record[key] = value

                statuses.append(
                    comparison["cnn_mni_status"]
                )

        except Exception as exc:

            record["cnn_finite"] = False
            record["cnn_mni_status"] = "FAIL"
            record["cnn_mni_error"] = (
                f"CNN load failed: {type(exc).__name__}: {exc}"
            )

            statuses.append("FAIL")

    # --------------------------------------------------------
    # Lineage consistency
    # --------------------------------------------------------

    required_lineage = [
        record["reoriented_exists"],
        record["n4_exists"],
        record["bet_exists"],
        record["masked_exists"],
        record["normalized_exists"],
        record["registered_expected_exists"],
        record["transform_exists"],
        record["cnn_exists"],
    ]

    if all(required_lineage):
        record["lineage_status"] = "PASS"
    elif any(required_lineage):
        record["lineage_status"] = "WARNING"
        statuses.append("WARNING")
    else:
        record["lineage_status"] = "FAIL"
        statuses.append("FAIL")

    # --------------------------------------------------------
    # Final subject status
    # --------------------------------------------------------

    record["status"] = overall_status(statuses)

    # Deduplicate issue labels.
    unique_issues = []

    for status in statuses:
        if status not in unique_issues:
            unique_issues.append(status)

    # Add human-readable specific issues.
    specific = []

    if not record["manifest_mni_exists"]:
        specific.append("manifest MNI path missing")

    if not record["manifest_mni_subject_match"]:
        specific.append("manifest MNI subject mismatch")

    if not record["reoriented_exists"]:
        specific.append("missing reoriented file")

    if not record["n4_exists"]:
        specific.append("missing N4 file")

    if not record["bet_exists"]:
        specific.append("missing expected BET file")

    if not record["masked_exists"]:
        specific.append("missing masked file")

    if not record["normalized_exists"]:
        specific.append("missing normalized file")

    if not record["registered_expected_exists"]:
        specific.append("expected registered RMS file missing")

    if not record["transform_exists"]:
        specific.append("missing transform")

    if not record["cnn_exists"]:
        specific.append("missing CNN tensor")

    if record["cnn_mni_status"] == "FAIL":
        specific.append("CNN/MNI consistency failure")

    elif record["cnn_mni_status"] == "WARNING":
        specific.append("CNN/MNI consistency warning")

    if (
        record["brain_mask_status"] == "FAIL"
    ):
        specific.append("implausible/empty brain mask")

    elif (
        record["brain_mask_status"] == "WARNING"
    ):
        specific.append("unusual brain-mask fraction")

    record["issues"] = "; ".join(specific)

    return record


# ============================================================
# CV AUDIT
# ============================================================

def audit_cv(rms_subjects: set) -> Tuple[pd.DataFrame, Dict[str, object]]:

    result_rows = []

    summary = {
        "cv_directory_exists": RMS_CV_DIR.exists(),
        "assignments_exists": False,
        "summary_exists": False,
        "validation_report_exists": False,
        "assignment_rows": 0,
        "unique_subjects": 0,
        "expected_subjects": len(rms_subjects),
        "duplicate_subject_role_rows": 0,
        "subjects_with_wrong_appearance_count": 0,
        "subjects_not_in_rms_cohort": 0,
        "subjects_missing_from_cv": 0,
        "outer_folds": 0,
        "inner_folds": 0,
        "cv_status": "FAIL",
        "issues": "",
    }

    if not RMS_CV_DIR.exists():
        summary["issues"] = "RMS CV directory missing."
        return pd.DataFrame(), summary

    assignments_path = RMS_CV_DIR / "nested_cv_assignments.csv"
    summary_path = RMS_CV_DIR / "nested_cv_summary.csv"
    validation_path = RMS_CV_DIR / "nested_cv_validation_report.txt"

    summary["assignments_exists"] = assignments_path.exists()
    summary["summary_exists"] = summary_path.exists()
    summary["validation_report_exists"] = validation_path.exists()

    if not assignments_path.exists():
        summary["issues"] = "nested_cv_assignments.csv missing."
        return pd.DataFrame(), summary

    try:
        cv = pd.read_csv(assignments_path)
    except Exception as exc:
        summary["issues"] = (
            f"Cannot read CV assignments: {type(exc).__name__}: {exc}"
        )
        return pd.DataFrame(), summary

    required = [
        "subject_id",
        "outer_fold",
        "inner_fold",
        "role",
    ]

    missing = [x for x in required if x not in cv.columns]

    if missing:
        summary["issues"] = (
            "CV assignment columns missing: " + ", ".join(missing)
        )
        return pd.DataFrame(), summary

    cv["subject_id"] = cv["subject_id"].map(normalize_subject_id)

    summary["assignment_rows"] = len(cv)
    summary["unique_subjects"] = cv["subject_id"].nunique()

    summary["outer_folds"] = cv["outer_fold"].nunique()

    inner_values = cv["inner_fold"].dropna()
    summary["inner_folds"] = inner_values.nunique()

    # Unexpected subjects.
    unexpected = sorted(
        set(cv["subject_id"]) - rms_subjects
    )

    missing_subjects = sorted(
        rms_subjects - set(cv["subject_id"])
    )

    summary["subjects_not_in_rms_cohort"] = len(unexpected)
    summary["subjects_missing_from_cv"] = len(missing_subjects)

    # Duplicate subject/role/fold combinations.
    duplicate_count = int(
        cv.duplicated(
            subset=[
                "subject_id",
                "outer_fold",
                "inner_fold",
                "role",
            ]
        ).sum()
    )

    summary["duplicate_subject_role_rows"] = duplicate_count

    # Each subject should appear 20 times in 5x4 nested structure:
    # 4 inner-role appearances + outer-test per outer fold.
    counts = cv.groupby("subject_id").size()

    wrong_counts = counts[counts != 20]

    summary["subjects_with_wrong_appearance_count"] = len(
        wrong_counts
    )

    for subject_id, group in cv.groupby("subject_id"):

        roles = group["role"].astype(str).str.upper().str.strip()

        outer_test_count = int(
            np.sum(roles == "OUTER_TEST")
        )

        inner_train_count = int(
            np.sum(roles == "INNER_TRAIN")
        )

        inner_validation_count = int(
            np.sum(roles == "INNER_VALIDATION")
        )

        result_rows.append(
            {
                "subject_id": subject_id,
                "rows": len(group),
                "outer_test_count": outer_test_count,
                "inner_train_count": inner_train_count,
                "inner_validation_count": inner_validation_count,
                "outer_folds": group["outer_fold"].nunique(),
                "inner_folds": group["inner_fold"].nunique(),
                "is_rms_subject": subject_id in rms_subjects,
                "expected_20_rows": len(group) == 20,
            }
        )

    problems = []

    if summary["assignment_rows"] != len(rms_subjects) * 20:
        problems.append(
            f"expected 2380 rows, found {summary['assignment_rows']}"
        )

    if summary["unique_subjects"] != len(rms_subjects):
        problems.append(
            f"expected {len(rms_subjects)} unique subjects, "
            f"found {summary['unique_subjects']}"
        )

    if summary["subjects_not_in_rms_cohort"] > 0:
        problems.append(
            f"{summary['subjects_not_in_rms_cohort']} unexpected subjects"
        )

    if summary["subjects_missing_from_cv"] > 0:
        problems.append(
            f"{summary['subjects_missing_from_cv']} RMS subjects missing"
        )

    if duplicate_count > 0:
        problems.append(
            f"{duplicate_count} duplicate CV assignment rows"
        )

    if summary["subjects_with_wrong_appearance_count"] > 0:
        problems.append(
            f"{summary['subjects_with_wrong_appearance_count']} "
            "subjects do not appear exactly 20 times"
        )

    # Explicit outer-test uniqueness.
    outer_test = cv[
        cv["role"].astype(str).str.upper().str.strip()
        == "OUTER_TEST"
    ]

    outer_test_duplicates = outer_test.duplicated(
        subset=["subject_id", "outer_fold", "inner_fold"]
    ).sum()

    if outer_test_duplicates > 0:
        problems.append(
            f"{outer_test_duplicates} duplicate OUTER_TEST assignments"
        )

    if not problems:
        summary["cv_status"] = "PASS"
    else:
        summary["cv_status"] = "FAIL"
        summary["issues"] = "; ".join(problems)

    return pd.DataFrame(result_rows), summary


# ============================================================
# SUMMARY / LINEAGE INVENTORY
# ============================================================

def build_lineage_inventory(
    subject_ids: List[str],
) -> pd.DataFrame:

    rows = []

    for subject_id in sorted(subject_ids):

        paths = expected_rms_paths(subject_id)

        for stage, path in paths.items():

            if stage == "bet_duplicate":
                continue

            rows.append(
                {
                    "subject_id": subject_id,
                    "stage": stage,
                    "path": str(path),
                    "exists": path.exists(),
                    "file_size_bytes": (
                        path.stat().st_size
                        if path.exists()
                        else np.nan
                    ),
                    "md5": (
                        md5_file(path)
                        if path.exists() and path.is_file()
                        else ""
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 80)
    print("NeuroFusion AI - RMS sMRI CNN Input Pipeline Audit")
    print("=" * 80)
    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"sMRI root    : {SMRI_ROOT}")
    print(f"Audit output : {AUDIT_ROOT}")
    print()
    print("READ-ONLY MODE: existing project datasets will NOT be modified.")
    print()

    # --------------------------------------------------------
    # Create ONLY the audit output directory.
    # --------------------------------------------------------

    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    print(
        f"Labeled manifest rows: {len(manifest)}"
    )

    rms = filter_rms_cohort(manifest)

    print(
        f"RMS cohort rows:       {len(rms)}"
    )

    # --------------------------------------------------------
    # Subject inventory
    # --------------------------------------------------------

    rms["subject_id"] = rms["subject_id"].map(
        normalize_subject_id
    )

    duplicate_subjects = (
        rms["subject_id"]
        .duplicated(keep=False)
    )

    duplicate_count = int(
        duplicate_subjects.sum()
    )

    unique_subjects = sorted(
        rms["subject_id"].unique()
    )

    labels_numeric = pd.to_numeric(
        rms["label"],
        errors="coerce",
    )

    controls = int(
        np.sum(labels_numeric == 0)
    )

    schizophrenia = int(
        np.sum(labels_numeric == 1)
    )

    invalid_labels = int(
        labels_numeric.isna().sum()
    )

    unexpected_labels = sorted(
        set(labels_numeric.dropna().unique())
        - {0, 1}
    )

    print(
        f"Unique RMS subjects:   {len(unique_subjects)}"
    )

    print(
        f"Controls:              {controls}"
    )

    print(
        f"Schizophrenia:         {schizophrenia}"
    )

    print(
        f"Duplicate subject rows: {duplicate_count}"
    )

    print(
        f"Invalid labels:         {invalid_labels}"
    )

    print()

    # --------------------------------------------------------
    # Subject-level audit
    # --------------------------------------------------------

    records = []

    for index, row in rms.iterrows():

        subject_id = normalize_subject_id(
            row["subject_id"]
        )

        print(
            f"[{len(records)+1:03d}/{len(rms):03d}] "
            f"Auditing {subject_id}...",
            end="\r",
        )

        record = audit_subject(row)

        record["manifest_row_index"] = int(index)

        records.append(record)

    print()
    print()

    subject_df = pd.DataFrame(records)

    # Mark duplicate IDs.
    duplicate_ids = set(
        subject_df.loc[
            subject_df["subject_id"].duplicated(keep=False),
            "subject_id",
        ]
    )

    subject_df["duplicate_subject_id"] = (
        subject_df["subject_id"].isin(duplicate_ids)
    )

    if duplicate_ids:
        subject_df.loc[
            subject_df["duplicate_subject_id"],
            "status"
        ] = "FAIL"

        subject_df.loc[
            subject_df["duplicate_subject_id"],
            "issues"
        ] = (
            subject_df.loc[
                subject_df["duplicate_subject_id"],
                "issues",
            ].fillna("")
            + "; duplicate subject ID"
        ).str.strip("; ")

    # --------------------------------------------------------
    # CV audit
    # --------------------------------------------------------

    all_cv_subjects = set(manifest["subject_id"].astype(str).map(normalize_subject_id))
    cv_subject_df, cv_summary = audit_cv(all_cv_subjects)

    if not cv_subject_df.empty:
        cv_subject_df.to_csv(
            CV_REPORT_PATH,
            index=False,
        )

    # --------------------------------------------------------
    # Lineage inventory
    # --------------------------------------------------------

    lineage_df = build_lineage_inventory(
        unique_subjects
    )

    lineage_df.to_csv(
        LINEAGE_REPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Subject report
    # --------------------------------------------------------

    subject_df.to_csv(
        SUBJECT_REPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    status_counts = (
        subject_df["status"]
        .value_counts()
        .to_dict()
    )

    cnn_mni_counts = (
        subject_df["cnn_mni_status"]
        .value_counts()
        .to_dict()
    )

    lineage_counts = (
        subject_df["lineage_status"]
        .value_counts()
        .to_dict()
    )

    brain_mask_counts = (
        subject_df["brain_mask_status"]
        .value_counts()
        .to_dict()
    )

    summary_rows = [
        {
            "category": "SUBJECT_INVENTORY",
            "metric": "manifest_rows",
            "value": len(manifest),
            "status": "PASS",
            "details": "",
        },
        {
            "category": "SUBJECT_INVENTORY",
            "metric": "rms_rows",
            "value": len(rms),
            "status": (
                "PASS"
                if len(rms) == EXPECTED_SUBJECTS
                else "FAIL"
            ),
            "details": (
                f"Expected {EXPECTED_SUBJECTS}"
            ),
        },
        {
            "category": "SUBJECT_INVENTORY",
            "metric": "unique_rms_subjects",
            "value": len(unique_subjects),
            "status": (
                "PASS"
                if len(unique_subjects) == EXPECTED_SUBJECTS
                else "FAIL"
            ),
            "details": (
                f"Expected {EXPECTED_SUBJECTS}"
            ),
        },
        {
            "category": "LABELS",
            "metric": "controls",
            "value": controls,
            "status": (
                "PASS"
                if controls == EXPECTED_CONTROLS
                else "FAIL"
            ),
            "details": (
                f"Expected {EXPECTED_CONTROLS}"
            ),
        },
        {
            "category": "LABELS",
            "metric": "schizophrenia",
            "value": schizophrenia,
            "status": (
                "PASS"
                if schizophrenia == EXPECTED_SCHIZOPHRENIA
                else "FAIL"
            ),
            "details": (
                f"Expected {EXPECTED_SCHIZOPHRENIA}"
            ),
        },
        {
            "category": "LABELS",
            "metric": "invalid_labels",
            "value": invalid_labels,
            "status": (
                "PASS"
                if invalid_labels == 0
                else "FAIL"
            ),
            "details": "",
        },
        {
            "category": "SUBJECT_INVENTORY",
            "metric": "duplicate_subject_rows",
            "value": duplicate_count,
            "status": (
                "PASS"
                if duplicate_count == 0
                else "FAIL"
            ),
            "details": "",
        },
        {
            "category": "SUBJECT_AUDIT",
            "metric": "subject_pass",
            "value": status_counts.get("PASS", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "SUBJECT_AUDIT",
            "metric": "subject_warning",
            "value": status_counts.get("WARNING", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "SUBJECT_AUDIT",
            "metric": "subject_fail",
            "value": status_counts.get("FAIL", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "LINEAGE",
            "metric": "lineage_pass",
            "value": lineage_counts.get("PASS", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "LINEAGE",
            "metric": "lineage_warning",
            "value": lineage_counts.get("WARNING", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "CNN_MNI",
            "metric": "cnn_mni_pass",
            "value": cnn_mni_counts.get("PASS", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "CNN_MNI",
            "metric": "cnn_mni_warning",
            "value": cnn_mni_counts.get("WARNING", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "CNN_MNI",
            "metric": "cnn_mni_fail",
            "value": cnn_mni_counts.get("FAIL", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "BRAIN_MASK",
            "metric": "brain_mask_pass",
            "value": brain_mask_counts.get("PASS", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "BRAIN_MASK",
            "metric": "brain_mask_warning",
            "value": brain_mask_counts.get("WARNING", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "BRAIN_MASK",
            "metric": "brain_mask_fail",
            "value": brain_mask_counts.get("FAIL", 0),
            "status": "INFO",
            "details": "",
        },
        {
            "category": "CV",
            "metric": "cv_status",
            "value": cv_summary["cv_status"],
            "status": cv_summary["cv_status"],
            "details": cv_summary["issues"],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        SUMMARY_REPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Diagnostics: only WARNING/FAIL subjects
    # --------------------------------------------------------

    diagnostic_df = subject_df.loc[
        subject_df["status"].isin(
            ["WARNING", "FAIL"]
        )
    ].copy()

    diagnostic_df.to_csv(
        DIAGNOSTIC_REPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Overall audit determination
    # --------------------------------------------------------

    critical_failures = []

    if len(rms) != EXPECTED_SUBJECTS:
        critical_failures.append(
            f"RMS cohort size is {len(rms)}, expected "
            f"{EXPECTED_SUBJECTS}"
        )

    if len(unique_subjects) != EXPECTED_SUBJECTS:
        critical_failures.append(
            f"Unique RMS subjects are {len(unique_subjects)}, "
            f"expected {EXPECTED_SUBJECTS}"
        )

    if controls != EXPECTED_CONTROLS:
        critical_failures.append(
            f"Controls are {controls}, expected {EXPECTED_CONTROLS}"
        )

    if schizophrenia != EXPECTED_SCHIZOPHRENIA:
        critical_failures.append(
            f"Schizophrenia subjects are {schizophrenia}, "
            f"expected {EXPECTED_SCHIZOPHRENIA}"
        )

    if invalid_labels > 0:
        critical_failures.append(
            f"{invalid_labels} invalid/missing labels"
        )

    if duplicate_count > 0:
        critical_failures.append(
            f"{duplicate_count} duplicate subject rows"
        )

    if cv_summary["cv_status"] == "FAIL":
        critical_failures.append(
            "RMS nested CV audit failed"
        )

    subject_fail_count = int(
        np.sum(subject_df["status"] == "FAIL")
    )

    cnn_mni_fail_count = int(
        np.sum(subject_df["cnn_mni_status"] == "FAIL")
    )

    if subject_fail_count > 0:
        critical_failures.append(
            f"{subject_fail_count} subjects have FAIL status"
        )

    if cnn_mni_fail_count > 0:
        critical_failures.append(
            f"{cnn_mni_fail_count} CNN/MNI consistency failures"
        )

    if critical_failures:
        overall_audit_status = "FAIL"
    elif np.sum(subject_df["status"] == "WARNING") > 0:
        overall_audit_status = "WARNING"
    else:
        overall_audit_status = "PASS"

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    with TEXT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "NeuroFusion AI\n"
        )
        report.write(
            "RMS sMRI CNN Input Pipeline Audit\n"
        )
        report.write(
            "=" * 80 + "\n\n"
        )

        report.write(
            "AUDIT MODE\n"
        )
        report.write(
            "READ-ONLY. Existing preprocessing/model datasets were not "
            "modified by this audit.\n\n"
        )

        report.write(
            "PROJECT PATHS\n"
        )
        report.write(
            f"Project root: {PROJECT_ROOT}\n"
        )
        report.write(
            f"sMRI root: {SMRI_ROOT}\n"
        )
        report.write(
            f"Audit output: {AUDIT_ROOT}\n\n"
        )

        report.write(
            "SUBJECT INVENTORY\n"
        )
        report.write(
            f"Labeled manifest rows: {len(manifest)}\n"
        )
        report.write(
            f"RMS rows: {len(rms)}\n"
        )
        report.write(
            f"Unique RMS subjects: {len(unique_subjects)}\n"
        )
        report.write(
            f"Controls: {controls}\n"
        )
        report.write(
            f"Schizophrenia: {schizophrenia}\n"
        )
        report.write(
            f"Invalid labels: {invalid_labels}\n"
        )
        report.write(
            f"Duplicate rows: {duplicate_count}\n\n"
        )

        report.write(
            "EXPECTED CNN GEOMETRY\n"
        )
        report.write(
            f"Shape: {EXPECTED_SHAPE}\n"
        )
        report.write(
            f"Spacing: {EXPECTED_SPACING}\n\n"
        )

        report.write(
            "SUBJECT AUDIT STATUS\n"
        )

        for status in ["PASS", "WARNING", "FAIL"]:
            report.write(
                f"{status}: {status_counts.get(status, 0)}\n"
            )

        report.write("\n")

        report.write(
            "CNN <-> MNI CONSISTENCY\n"
        )

        for status in ["PASS", "WARNING", "FAIL"]:
            report.write(
                f"{status}: {cnn_mni_counts.get(status, 0)}\n"
            )

        report.write("\n")

        report.write(
            "BRAIN MASK STATUS\n"
        )

        for status in ["PASS", "WARNING", "FAIL"]:
            report.write(
                f"{status}: {brain_mask_counts.get(status, 0)}\n"
            )

        report.write("\n")

        report.write(
            "CV AUDIT\n"
        )

        for key, value in cv_summary.items():
            report.write(
                f"{key}: {value}\n"
            )

        report.write("\n")

        report.write(
            "OVERALL AUDIT STATUS\n"
        )
        report.write(
            f"{overall_audit_status}\n\n"
        )

        if critical_failures:

            report.write(
                "CRITICAL FINDINGS\n"
            )

            for failure in critical_failures:
                report.write(
                    f"- {failure}\n"
                )

            report.write("\n")

        report.write(
            "SUBJECTS REQUIRING INVESTIGATION\n"
        )

        if diagnostic_df.empty:

            report.write(
                "None.\n"
            )

        else:

            for _, diagnostic in diagnostic_df.iterrows():

                report.write(
                    f"- {diagnostic['subject_id']}: "
                    f"{diagnostic['status']}"
                )

                issues = safe_str(
                    diagnostic["issues"]
                )

                if issues:
                    report.write(
                        f" | {issues}"
                    )

                report.write("\n")

        report.write("\n")

        report.write(
            "OUTPUT FILES\n"
        )
        report.write(
            f"- {SUBJECT_REPORT_PATH.name}\n"
        )
        report.write(
            f"- {SUMMARY_REPORT_PATH.name}\n"
        )
        report.write(
            f"- {TEXT_REPORT_PATH.name}\n"
        )
        report.write(
            f"- {CV_REPORT_PATH.name}\n"
        )
        report.write(
            f"- {LINEAGE_REPORT_PATH.name}\n"
        )
        report.write(
            f"- {DIAGNOSTIC_REPORT_PATH.name}\n"
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print()
    print(
        f"Overall status : {overall_audit_status}"
    )
    print(
        f"Subjects       : {len(unique_subjects)}"
    )
    print(
        f"Controls       : {controls}"
    )
    print(
        f"Schizophrenia  : {schizophrenia}"
    )
    print()
    print(
        f"PASS subjects  : {status_counts.get('PASS', 0)}"
    )
    print(
        f"WARNING        : {status_counts.get('WARNING', 0)}"
    )
    print(
        f"FAIL           : {status_counts.get('FAIL', 0)}"
    )
    print()
    print(
        "CNN <-> MNI:"
    )
    print(
        f"  PASS        : {cnn_mni_counts.get('PASS', 0)}"
    )
    print(
        f"  WARNING     : {cnn_mni_counts.get('WARNING', 0)}"
    )
    print(
        f"  FAIL        : {cnn_mni_counts.get('FAIL', 0)}"
    )
    print()
    print(
        f"CV status     : {cv_summary['cv_status']}"
    )

    if cv_summary["issues"]:
        print(
            f"CV issues     : {cv_summary['issues']}"
        )

    print()
    print(
        f"Subject report: {SUBJECT_REPORT_PATH}"
    )
    print(
        f"Summary       : {SUMMARY_REPORT_PATH}"
    )
    print(
        f"Text report   : {TEXT_REPORT_PATH}"
    )
    print(
        f"Diagnostics   : {DIAGNOSTIC_REPORT_PATH}"
    )
    print()

    return 0


if __name__ == "__main__":

    try:
        sys.exit(main())

    except Exception as exc:

        print()
        print("=" * 80)
        print("AUDIT SCRIPT FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        sys.exit(1)