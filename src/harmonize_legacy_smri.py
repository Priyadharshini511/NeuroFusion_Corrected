from pathlib import Path
import csv
import numpy as np
import SimpleITK as sitk


# ============================================================
# Configuration
# ============================================================

ROOT = Path("processed_data/smri")

N4_DIR = ROOT / "n4"
MASK_DIR = ROOT / "hd_bet"

OUTPUT_DIR = ROOT / "legacy_normalized"
REPORT_FILE = OUTPUT_DIR / "legacy_normalization_report.csv"

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0

# Explicit legacy STANDARD-only cohort.
# These are the subjects with a standard *_mni.nii.gz output
# and no current *_rms_n4_mni.nii.gz output.
LEGACY_SUBJECTS = [
    "A00000300",
    "A00002198",
    "A00003150",
    "A00004087",
    "A00007409",
    "A00009946",
    "A00010150",
    "A00010684",
    "A00011265",
    "A00011725",
    "A00012995",
    "A00013140",
    "A00013363",
    "A00013816",
    "A00014120",
    "A00014225",
    "A00014522",
    "A00014839",
    "A00014898",
    "A00015759",
    "A00015826",
    "A00017294",
    "A00022729",
    "A00022773",
    "A00022810",
    "A00022835",
    "A00022837",
    "A00022915",
    "A00023095",
    "A00023120",
    "A00023131",
    "A00023143",
    "A00023330",
    "A00023337",
    "A00023730",
    "A00023800",
    "A00023848",
    "A00023866",
    "A00024160",
    "A00024301",
    "A00024372",
    "A00024446",
    "A00024535",
    "A00027487",
    "A00028406",
    "A00028409",
    "A00029226",
    "A00029452",
    "A00031478",
    "A00031764",
    "A00033214",
    "A00037564",
    "A00037665",
]


# ============================================================
# Helpers
# ============================================================

def geometry_matches(image, mask, tolerance=1e-6):
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(
            image.GetSpacing(),
            mask.GetSpacing(),
            atol=tolerance,
        )
        and np.allclose(
            image.GetOrigin(),
            mask.GetOrigin(),
            atol=tolerance,
        )
        and np.allclose(
            image.GetDirection(),
            mask.GetDirection(),
            atol=tolerance,
        )
    )


def validate_output(image, data, mask):
    brain = data[mask]

    if data.dtype != np.float32:
        return False, "output dtype is not float32"

    if not np.all(np.isfinite(data)):
        return False, "output contains non-finite values"

    if brain.size == 0:
        return False, "output brain is empty"

    if float(brain.min()) < -1e-6:
        return False, "brain values below 0"

    if float(brain.max()) > 1.0 + 1e-6:
        return False, "brain values above 1"

    background = data[~mask]

    if background.size > 0 and not np.allclose(
        background,
        0.0,
        atol=1e-7,
    ):
        return False, "background is not zero"

    return True, "PASS"


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("LEGACY sMRI INTENSITY HARMONIZATION")
    print("=" * 75)
    print(f"Subjects: {len(LEGACY_SUBJECTS)}")
    print(f"Lower percentile: {LOWER_PERCENTILE}")
    print(f"Upper percentile: {UPPER_PERCENTILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    rows = []

    passed = 0
    skipped = 0
    failed = 0

    for subject in LEGACY_SUBJECTS:

        print(f"\n[{subject}]")

        n4_path = N4_DIR / f"{subject}_n4.nii.gz"
        mask_path = MASK_DIR / f"{subject}_brain_bet.nii.gz"
        output_path = (
            OUTPUT_DIR /
            f"{subject}_legacy_n4_normalized.nii.gz"
        )

        # ----------------------------------------------------
        # Input existence
        # ----------------------------------------------------

        if not n4_path.exists():
            print("ERROR: N4 image missing.")
            failed += 1
            rows.append(
                [subject, "", "", "", "", "", "", "", "FAILED",
                 "N4 image missing"]
            )
            continue

        if not mask_path.exists():
            print("ERROR: HD-BET mask missing.")
            failed += 1
            rows.append(
                [subject, "", "", "", "", "", "", "", "FAILED",
                 "HD-BET mask missing"]
            )
            continue

        # ----------------------------------------------------
        # Preserve valid existing output
        # ----------------------------------------------------

        if output_path.exists():
            try:
                existing = sitk.ReadImage(str(output_path))
                existing_data = sitk.GetArrayFromImage(existing)
                mask_image = sitk.ReadImage(str(mask_path))
                mask_data = (
                    sitk.GetArrayFromImage(mask_image) > 0
                )

                if geometry_matches(
                    existing,
                    mask_image,
                ):
                    valid, reason = validate_output(
                        existing,
                        existing_data,
                        mask_data,
                    )

                    if valid:
                        print(
                            "Existing valid output found — SKIP."
                        )
                        skipped += 1
                        rows.append(
                            [
                                subject,
                                "",
                                "",
                                int(mask_data.sum()),
                                float(existing_data.min()),
                                float(existing_data.max()),
                                float(existing_data[mask_data].mean()),
                                float(existing_data[mask_data].std()),
                                "SKIPPED",
                                "Existing valid output",
                            ]
                        )
                        continue

            except Exception:
                pass

        # ----------------------------------------------------
        # Read inputs
        # ----------------------------------------------------

        image = sitk.ReadImage(str(n4_path))
        mask_image = sitk.ReadImage(str(mask_path))

        data = sitk.GetArrayFromImage(image).astype(
            np.float32,
            copy=False,
        )

        mask = (
            sitk.GetArrayFromImage(mask_image) > 0
        )

        # ----------------------------------------------------
        # Geometry validation
        # ----------------------------------------------------

        if not geometry_matches(
            image,
            mask_image,
        ):
            print(
                "ERROR: N4 and HD-BET mask geometry mismatch."
            )
            failed += 1
            rows.append(
                [subject, "", "", "", "", "", "", "", "FAILED",
                 "N4/mask geometry mismatch"]
            )
            continue

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if not np.all(np.isfinite(data)):
            print("ERROR: N4 contains non-finite values.")
            failed += 1
            rows.append(
                [subject, "", "", "", "", "", "", "", "FAILED",
                 "N4 contains non-finite values"]
            )
            continue

        brain_values = data[mask]

        if brain_values.size == 0:
            print("ERROR: Brain mask contains no voxels.")
            failed += 1
            rows.append(
                [subject, "", "", "", "", "", "", "", "FAILED",
                 "Empty brain mask"]
            )
            continue

        # ----------------------------------------------------
        # Percentiles — same convention as current pipeline
        # ----------------------------------------------------

        lower = np.percentile(
            brain_values,
            LOWER_PERCENTILE,
        )

        upper = np.percentile(
            brain_values,
            UPPER_PERCENTILE,
        )

        print(f"1st percentile : {float(lower)}")
        print(f"99th percentile: {float(upper)}")

        if not np.isfinite(lower):
            print("ERROR: Lower percentile is non-finite.")
            failed += 1
            rows.append(
                [subject, lower, upper, "", "", "", "", "",
                 "FAILED", "Lower percentile non-finite"]
            )
            continue

        if not np.isfinite(upper):
            print("ERROR: Upper percentile is non-finite.")
            failed += 1
            rows.append(
                [subject, lower, upper, "", "", "", "", "",
                 "FAILED", "Upper percentile non-finite"]
            )
            continue

        if upper <= lower:
            print("ERROR: Invalid percentile range.")
            failed += 1
            rows.append(
                [subject, lower, upper, "", "", "", "", "",
                 "FAILED", "Upper percentile <= lower percentile"]
            )
            continue

        # ----------------------------------------------------
        # Clip and scale to [0,1]
        # ----------------------------------------------------

        brain_clipped = np.clip(
            brain_values,
            lower,
            upper,
        )

        brain_normalized = (
            (brain_clipped - lower)
            / (upper - lower)
        )

        normalized = np.zeros_like(
            data,
            dtype=np.float32,
        )

        normalized[mask] = brain_normalized.astype(
            np.float32,
        )

        # ----------------------------------------------------
        # Validate generated output
        # ----------------------------------------------------

        valid, reason = validate_output(
            image,
            normalized,
            mask,
        )

        if not valid:
            print(f"ERROR: {reason}")
            failed += 1
            rows.append(
                [
                    subject,
                    lower,
                    upper,
                    int(mask.sum()),
                    float(normalized.min()),
                    float(normalized.max()),
                    float(normalized[mask].mean()),
                    float(normalized[mask].std()),
                    "FAILED",
                    reason,
                ]
            )
            continue

        # ----------------------------------------------------
        # Save while preserving geometry
        # ----------------------------------------------------

        output_image = sitk.GetImageFromArray(
            normalized,
        )

        output_image.CopyInformation(image)

        sitk.WriteImage(
            output_image,
            str(output_path),
            useCompression=True,
        )

        # ----------------------------------------------------
        # Reload validation
        # ----------------------------------------------------

        saved = sitk.ReadImage(str(output_path))
        saved_data = sitk.GetArrayFromImage(saved)

        if not geometry_matches(
            saved,
            mask_image,
        ):
            print(
                "ERROR: Saved output geometry mismatch."
            )
            failed += 1
            rows.append(
                [
                    subject,
                    lower,
                    upper,
                    int(mask.sum()),
                    float(normalized.min()),
                    float(normalized.max()),
                    float(normalized[mask].mean()),
                    float(normalized[mask].std()),
                    "FAILED",
                    "Saved geometry mismatch",
                ]
            )
            continue

        valid, reason = validate_output(
            saved,
            saved_data,
            mask,
        )

        if not valid:
            print(
                f"ERROR: Saved output validation failed: {reason}"
            )
            failed += 1
            rows.append(
                [
                    subject,
                    lower,
                    upper,
                    int(mask.sum()),
                    float(saved_data.min()),
                    float(saved_data.max()),
                    float(saved_data[mask].mean()),
                    float(saved_data[mask].std()),
                    "FAILED",
                    reason,
                ]
            )
            continue

        print(
            "PASS:",
            f"brain range [{saved_data[mask].min():.6f}, "
            f"{saved_data[mask].max():.6f}]",
        )

        passed += 1

        rows.append(
            [
                subject,
                lower,
                upper,
                int(mask.sum()),
                float(saved_data.min()),
                float(saved_data.max()),
                float(saved_data[mask].mean()),
                float(saved_data[mask].std()),
                "PASS",
                "",
            ]
        )

    # ========================================================
    # Report
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "subject_id",
                "lower_percentile",
                "upper_percentile",
                "brain_voxels",
                "output_min",
                "output_max",
                "brain_mean",
                "brain_std",
                "status",
                "message",
            ]
        )

        writer.writerows(rows)

    print()
    print("=" * 75)
    print("LEGACY HARMONIZATION SUMMARY")
    print("=" * 75)
    print("Expected:", len(LEGACY_SUBJECTS))
    print("Passed:", passed)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Report:", REPORT_FILE)
    print("=" * 75)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()