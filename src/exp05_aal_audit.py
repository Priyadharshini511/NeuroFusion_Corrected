from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path(
    r"processed_data\smri\dataset\smri_subject_manifest_labeled.csv"
)


def main():
    manifest = pd.read_csv(MANIFEST)

    rows = []
    bad = []
    reference_columns = None
    sources = []

    for _, row in manifest.iterrows():
        subject_id = row["subject_id"]
        aal_path = Path(row["aal_csv"])

        try:
            df = pd.read_csv(aal_path)

            if len(df) != 1:
                bad.append(
                    (subject_id, "row_count", len(df))
                )
                continue

            feature_df = df.drop(
                columns=["subject_id", "acquisition_type", "volume"],
                errors="ignore",
            )

            columns = list(feature_df.columns)

            if reference_columns is None:
                reference_columns = columns

            if columns != reference_columns:
                bad.append(
                    (subject_id, "columns_mismatch", len(columns))
                )
                continue

            values = feature_df.iloc[0].to_numpy(dtype=np.float32)

            if len(values) != 116:
                bad.append(
                    (subject_id, "feature_count", len(values))
                )
                continue

            if not np.isfinite(values).all():
                bad.append(
                    (subject_id, "non_finite", "")
                )
                continue

            rows.append(values)
            sources.append(row["acquisition_source"])

        except Exception as exc:
            bad.append(
                (subject_id, "read_error", str(exc))
            )

    X = np.vstack(rows)

    print("=" * 60)
    print("EXPERIMENT 05 — AAL FEATURE AUDIT")
    print("=" * 60)

    print(f"Manifest subjects: {len(manifest)}")
    print(f"Subjects loaded: {len(rows)}")
    print(f"AAL feature count: {X.shape[1]}")
    print(f"All finite: {np.isfinite(X).all()}")
    print(f"Global minimum: {X.min():.6f}")
    print(f"Global maximum: {X.max():.6f}")
    print(f"Global mean: {X.mean():.6f}")
    print(f"Global std: {X.std():.6f}")

    print(
        "Constant features:",
        int(np.sum(np.std(X, axis=0) == 0)),
    )

    print(
        "RMS loaded:",
        sources.count("RMS"),
    )

    print(
        "LEGACY loaded:",
        sources.count("LEGACY"),
    )

    print(f"Bad files: {len(bad)}")

    if bad:
        print("\nBad-file details:")
        for item in bad[:20]:
            print(item)

    print("\nFirst 10 AAL features:")
    print(reference_columns[:10])

    print("\nLast 10 AAL features:")
    print(reference_columns[-10:])

    expected = (
        len(manifest) == 172
        and len(rows) == 172
        and X.shape == (172, 116)
        and np.isfinite(X).all()
        and len(bad) == 0
        and sources.count("RMS") == 119
        and sources.count("LEGACY") == 53
        and np.all(np.std(X, axis=0) > 0)
    )

    print("\n" + "=" * 60)

    if expected:
        print("FINAL STATUS: EXPERIMENT 05 AAL AUDIT PASS")
    else:
        print("FINAL STATUS: EXPERIMENT 05 AAL AUDIT REQUIRES INVESTIGATION")

    print("=" * 60)


if __name__ == "__main__":
    main()