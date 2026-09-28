from pathlib import Path
import SimpleITK as sitk
import numpy as np


ROOT = Path("processed_data/smri")

N4_DIR = ROOT / "n4"
OUTPUT_DIR = ROOT / "legacy_normalized"


def geometry_matches(image_a, image_b, tolerance=1e-6):
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=tolerance,
        )
        and np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=tolerance,
        )
        and np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=tolerance,
        )
    )


def main():
    files = sorted(
        OUTPUT_DIR.glob(
            "*_legacy_n4_normalized.nii.gz"
        )
    )

    print("=" * 75)
    print("LEGACY NORMALIZED VS N4 GEOMETRY VALIDATION")
    print("=" * 75)

    print("Files:", len(files))
    print("Expected:", 53)

    bad = []
    passed = 0

    for output_path in files:

        subject = output_path.name.replace(
            "_legacy_n4_normalized.nii.gz",
            "",
        )

        n4_path = (
            N4_DIR /
            f"{subject}_n4.nii.gz"
        )

        if not n4_path.exists():
            bad.append(
                (subject, "missing N4")
            )
            continue

        try:
            n4 = sitk.ReadImage(
                str(n4_path)
            )

            output = sitk.ReadImage(
                str(output_path)
            )

            data = sitk.GetArrayFromImage(
                output
            )

            if not geometry_matches(
                n4,
                output,
            ):
                bad.append(
                    (subject, "geometry")
                )
                continue

            if not np.all(
                np.isfinite(data)
            ):
                bad.append(
                    (subject, "nonfinite")
                )
                continue

            if (
                float(data.min()) < -1e-6
                or float(data.max()) > 1.000001
            ):
                bad.append(
                    (subject, "range")
                )
                continue

            # Background is defined by zero-valued
            # voxels in the output. Verify there are no
            # non-zero values outside the original brain mask.
            mask = sitk.ReadImage(
                str(
                    ROOT
                    / "hd_bet"
                    / f"{subject}_brain_bet.nii.gz"
                )
            )

            mask_data = (
                sitk.GetArrayFromImage(mask) > 0
            )

            if not np.allclose(
                data[~mask_data],
                0.0,
                atol=1e-7,
            ):
                bad.append(
                    (subject, "background")
                )
                continue

            passed += 1

        except Exception as exc:
            bad.append(
                (subject, f"exception: {exc}")
            )

    print("Passed:", passed)
    print("Invalid:", len(bad))
    print("Invalid details:", bad)

    if len(files) != 53:
        raise SystemExit(
            "ERROR: Expected exactly 53 output files."
        )

    if bad:
        raise SystemExit(
            "ERROR: Legacy normalized validation failed."
        )

    print("=" * 75)
    print("FINAL STATUS: PASS")
    print("=" * 75)


if __name__ == "__main__":
    main()