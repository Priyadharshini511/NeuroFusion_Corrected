from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# NeuroFusion AI
# Analyze Full 5x4 Nested-CV Training Runs
#
# READ-ONLY DIAGNOSTIC
# Does not retrain models.
# Does not modify dataset/model files.
# ============================================================

BASE_DIR = Path("processed_data/smri/cnn_dataset/cross_validation")

TRAINING_DIR = BASE_DIR / "training_full_5x4"
OUTPUT_DIR = BASE_DIR / "audit" / "training_runs"

OOF_FILE = TRAINING_DIR / "nested_cv_oof_predictions.csv"
FOLD_METRICS_FILE = TRAINING_DIR / "nested_cv_fold_metrics.csv"
SUMMARY_FILE = TRAINING_DIR / "nested_cv_summary.txt"


def find_training_history_files():
    """
    Recursively find CSV files that look like training histories.
    Excludes final OOF predictions and fold-level summary tables.
    """
    files = []

    for path in TRAINING_DIR.rglob("*.csv"):
        name = path.name.lower()

        if name in {
            "nested_cv_oof_predictions.csv",
            "nested_cv_fold_metrics.csv",
        }:
            continue

        # Training history files generally contain epoch information.
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        cols = {str(c).lower() for c in df.columns}

        if "epoch" in cols:
            files.append(path)

    return sorted(files)


def normalize_columns(df):
    """
    Normalize common training-history column names.
    """
    mapping = {}

    for col in df.columns:
        low = str(col).strip().lower()

        if low == "epoch":
            mapping[col] = "epoch"

        elif low in {"train_loss", "training_loss", "loss_train"}:
            mapping[col] = "train_loss"

        elif low in {"val_loss", "validation_loss", "loss_val"}:
            mapping[col] = "val_loss"

        elif low in {"train_auc", "training_auc", "auc_train"}:
            mapping[col] = "train_auc"

        elif low in {"val_auc", "validation_auc", "auc_val"}:
            mapping[col] = "val_auc"

        elif low in {"learning_rate", "lr"}:
            mapping[col] = "learning_rate"

    return df.rename(columns=mapping)


def infer_outer_inner(path):
    """
    Try to infer outer/inner fold numbers from path components.
    """
    outer = None
    inner = None

    parts = [p.lower() for p in path.parts]

    for part in parts:
        if "outer" in part:
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                outer = int(digits)

        if "inner" in part:
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                inner = int(digits)

    # Also inspect filename.
    name = path.stem.lower()

    import re

    m = re.search(r"outer[_-]?(\d+)", name)
    if m:
        outer = int(m.group(1))

    m = re.search(r"inner[_-]?(\d+)", name)
    if m:
        inner = int(m.group(1))

    return outer, inner


def main():

    print("=" * 72)
    print("NeuroFusion AI - Nested CV Training Run Analysis")
    print("=" * 72)

    print("\nInput directory:")
    print(f"  {TRAINING_DIR}")

    if not TRAINING_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAINING_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Find training histories
    # --------------------------------------------------------

    print("\n[1] Finding inner-CV training histories...")

    history_files = find_training_history_files()

    print(f"Training-history CSV files found: {len(history_files)}")

    if len(history_files) == 0:
        print("\nNo training-history CSV files were found.")
        print(
            "The completed nested-CV training may have stored "
            "training information in another format."
        )
        print("\nFiles currently present:")

        for path in sorted(TRAINING_DIR.rglob("*")):
            if path.is_file():
                print(f"  {path}")

        return

    for path in history_files:
        print(f"  {path}")

    # --------------------------------------------------------
    # 2. Read histories
    # --------------------------------------------------------

    print("\n[2] Reading training histories...")

    records = []
    detailed = []

    for path in history_files:

        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)

        except Exception as exc:
            print(f"  WARNING: Could not read {path}: {exc}")
            continue

        if "epoch" not in df.columns:
            print(f"  WARNING: No epoch column: {path}")
            continue

        outer, inner = infer_outer_inner(path)

        df["outer_fold"] = outer
        df["inner_fold"] = inner
        df["history_file"] = str(path)

        # Numeric conversion.
        for col in [
            "epoch",
            "train_loss",
            "val_loss",
            "train_auc",
            "val_auc",
            "learning_rate",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        detailed.append(df)

        # ----------------------------------------------------
        # Determine best validation AUC.
        # ----------------------------------------------------

        if "val_auc" in df.columns:

            valid_auc = df.dropna(subset=["val_auc"])

            if len(valid_auc) > 0:

                best_idx = valid_auc["val_auc"].idxmax()
                best = valid_auc.loc[best_idx]

                record = {
                    "history_file": str(path),
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "epochs_completed": int(
                        pd.to_numeric(df["epoch"]).max()
                    ),
                    "best_epoch_by_val_auc": int(best["epoch"]),
                    "best_val_auc": float(best["val_auc"]),
                }

                if "train_auc" in df.columns:
                    record["train_auc_at_best_val_auc"] = float(
                        best["train_auc"]
                    )

                if "train_loss" in df.columns:
                    record["train_loss_at_best_val_auc"] = float(
                        best["train_loss"]
                    )

                if "val_loss" in df.columns:
                    record["val_loss_at_best_val_auc"] = float(
                        best["val_loss"]
                    )

                if "learning_rate" in df.columns:
                    record["learning_rate_at_best_val_auc"] = float(
                        best["learning_rate"]
                    )

                # Best validation loss.
                if "val_loss" in df.columns:
                    valid_loss = df.dropna(subset=["val_loss"])

                    if len(valid_loss) > 0:
                        loss_idx = valid_loss["val_loss"].idxmin()
                        best_loss = valid_loss.loc[loss_idx]

                        record["best_epoch_by_val_loss"] = int(
                            best_loss["epoch"]
                        )
                        record["best_val_loss"] = float(
                            best_loss["val_loss"]
                        )
                        record["val_auc_at_best_val_loss"] = float(
                            best_loss["val_auc"]
                        ) if pd.notna(
                            best_loss.get("val_auc", np.nan)
                        ) else np.nan

                records.append(record)

    if not detailed:
        print("\nNo usable training histories found.")
        return

    history_df = pd.concat(detailed, ignore_index=True)

    summary_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # 3. Basic summary
    # --------------------------------------------------------

    print("\n[3] Training-run summary...")

    print(f"Usable training runs: {len(summary_df)}")

    if "outer_fold" in summary_df.columns:
        print(
            "Outer folds detected:",
            sorted(
                summary_df["outer_fold"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    if "inner_fold" in summary_df.columns:
        print(
            "Inner folds detected:",
            sorted(
                summary_df["inner_fold"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    # --------------------------------------------------------
    # 4. Print individual runs
    # --------------------------------------------------------

    print("\n[4] Individual inner-CV runs...")

    display_cols = [
        "outer_fold",
        "inner_fold",
        "epochs_completed",
        "best_epoch_by_val_auc",
        "best_val_auc",
        "train_auc_at_best_val_auc",
        "best_epoch_by_val_loss",
        "best_val_loss",
        "val_auc_at_best_val_loss",
    ]

    display_cols = [
        c for c in display_cols if c in summary_df.columns
    ]

    if display_cols:
        print(
            summary_df[
                display_cols
            ].sort_values(
                ["outer_fold", "inner_fold"],
                na_position="last"
            ).to_string(index=False)
        )

    # --------------------------------------------------------
    # 5. Outer-fold aggregation
    # --------------------------------------------------------

    print("\n[5] Aggregating inner runs by outer fold...")

    if "outer_fold" in summary_df.columns:

        outer_summary = (
            summary_df
            .groupby("outer_fold", dropna=False)
            .agg(
                inner_runs=("best_val_auc", "count"),
                mean_best_val_auc=("best_val_auc", "mean"),
                std_best_val_auc=("best_val_auc", "std"),
                min_best_val_auc=("best_val_auc", "min"),
                max_best_val_auc=("best_val_auc", "max"),
                mean_selected_epoch=(
                    "best_epoch_by_val_auc",
                    "mean"
                ),
                min_selected_epoch=(
                    "best_epoch_by_val_auc",
                    "min"
                ),
                max_selected_epoch=(
                    "best_epoch_by_val_auc",
                    "max"
                ),
            )
            .reset_index()
        )

        print(outer_summary.to_string(index=False))

    else:
        outer_summary = pd.DataFrame()

    # --------------------------------------------------------
    # 6. Train-vs-validation behavior
    # --------------------------------------------------------

    print("\n[6] Train-vs-validation AUC behavior...")

    if (
        "train_auc_at_best_val_auc" in summary_df.columns
        and "best_val_auc" in summary_df.columns
    ):

        summary_df["train_val_auc_gap"] = (
            summary_df["train_auc_at_best_val_auc"]
            - summary_df["best_val_auc"]
        )

        print(
            summary_df[
                [
                    "outer_fold",
                    "inner_fold",
                    "train_auc_at_best_val_auc",
                    "best_val_auc",
                    "train_val_auc_gap",
                ]
            ].sort_values(
                ["outer_fold", "inner_fold"]
            ).to_string(index=False)
        )

        print(
            "\nMean train-validation AUC gap:",
            f"{summary_df['train_val_auc_gap'].mean():.6f}"
        )

        print(
            "Maximum train-validation AUC gap:",
            f"{summary_df['train_val_auc_gap'].max():.6f}"
        )

    # --------------------------------------------------------
    # 7. Learning-rate behavior
    # --------------------------------------------------------

    print("\n[7] Learning-rate behavior...")

    if "learning_rate" in history_df.columns:

        lr_values = (
            history_df["learning_rate"]
            .dropna()
            .unique()
        )

        print(
            "Learning rates observed:",
            sorted(lr_values.tolist())
        )

        lr_summary = (
            history_df
            .groupby("history_file")["learning_rate"]
            .agg(
                min_lr="min",
                max_lr="max",
                unique_lr_values="nunique",
            )
            .reset_index()
        )

        print(lr_summary.to_string(index=False))

    else:
        lr_summary = pd.DataFrame()

    # --------------------------------------------------------
    # 8. Epoch stability
    # --------------------------------------------------------

    print("\n[8] Selected-epoch stability...")

    if "best_epoch_by_val_auc" in summary_df.columns:

        epochs = summary_df[
            "best_epoch_by_val_auc"
        ].dropna()

        if len(epochs) > 0:

            print(
                f"Mean selected epoch: {epochs.mean():.2f}"
            )
            print(
                f"Median selected epoch: {epochs.median():.2f}"
            )
            print(
                f"Minimum selected epoch: {epochs.min():.0f}"
            )
            print(
                f"Maximum selected epoch: {epochs.max():.0f}"
            )
            print(
                f"Epoch standard deviation: {epochs.std(ddof=0):.2f}"
            )

    # --------------------------------------------------------
    # 9. Validation AUC stability
    # --------------------------------------------------------

    print("\n[9] Validation-AUC stability...")

    aucs = summary_df["best_val_auc"].dropna()

    if len(aucs) > 0:

        print(f"Mean best inner val AUC: {aucs.mean():.6f}")
        print(f"Median: {aucs.median():.6f}")
        print(f"Std: {aucs.std(ddof=0):.6f}")
        print(f"Min: {aucs.min():.6f}")
        print(f"Max: {aucs.max():.6f}")

    # --------------------------------------------------------
    # 10. Compare with outer-fold performance
    # --------------------------------------------------------

    print("\n[10] Comparing selected epochs with outer-fold performance...")

    if FOLD_METRICS_FILE.exists():

        try:
            fold_df = pd.read_csv(FOLD_METRICS_FILE)

            print("\nOuter-fold metrics file:")
            print(
                fold_df.to_string(index=False)
            )

        except Exception as exc:
            print(
                f"WARNING: Could not read outer-fold metrics: {exc}"
            )

    else:
        fold_df = pd.DataFrame()

    # --------------------------------------------------------
    # 11. Diagnostic interpretation
    # --------------------------------------------------------

    print("\n[11] Diagnostic interpretation...")

    observations = []

    if len(summary_df) < 20:
        observations.append(
            "Fewer than 20 usable inner-CV histories were detected; "
            "verify the training-output structure."
        )
    else:
        observations.append(
            "The expected 20 inner-CV training runs are available."
        )

    if len(aucs) > 0:

        auc_std = float(aucs.std(ddof=0))
        auc_range = float(aucs.max() - aucs.min())

        if auc_std >= 0.15:
            observations.append(
                "Inner validation AUC is highly variable across runs."
            )
        elif auc_std >= 0.08:
            observations.append(
                "Inner validation AUC shows noticeable run-to-run variation."
            )
        else:
            observations.append(
                "Inner validation AUC is relatively stable across runs."
            )

        if auc_range >= 0.30:
            observations.append(
                "The best and worst inner validation AUCs differ substantially."
            )

    if "train_val_auc_gap" in summary_df.columns:

        gap = summary_df["train_val_auc_gap"].dropna()

        if len(gap) > 0:

            if gap.mean() > 0.15:
                observations.append(
                    "Average train-vs-validation AUC gap suggests "
                    "possible overfitting."
                )
            elif gap.mean() < -0.10:
                observations.append(
                    "Validation AUC frequently exceeds training AUC; "
                    "this can indicate high validation variance, "
                    "rather than conventional overfitting."
                )
            else:
                observations.append(
                    "Average train-vs-validation AUC gap is not strongly "
                    "indicative of conventional overfitting."
                )

    if "best_epoch_by_val_auc" in summary_df.columns:

        epoch_std = float(
            summary_df["best_epoch_by_val_auc"]
            .dropna()
            .std(ddof=0)
        )

        if epoch_std >= 8:
            observations.append(
                "Selected epochs vary considerably across inner folds, "
                "suggesting unstable training/validation behavior."
            )
        else:
            observations.append(
                "Selected epochs are comparatively consistent."
            )

    if "learning_rate" in history_df.columns:

        lr_values = history_df[
            "learning_rate"
        ].dropna().unique()

        if len(lr_values) > 1:
            observations.append(
                "The scheduler changed the learning rate during training "
                "in at least some runs."
            )

    for item in observations:
        print(f"- {item}")

    # --------------------------------------------------------
    # 12. Save outputs
    # --------------------------------------------------------

    print("\n[12] Saving analysis outputs...")

    summary_path = OUTPUT_DIR / "training_run_summary.csv"
    history_path = OUTPUT_DIR / "all_inner_training_history.csv"
    outer_path = OUTPUT_DIR / "outer_fold_training_summary.csv"
    lr_path = OUTPUT_DIR / "learning_rate_summary.csv"
    report_path = OUTPUT_DIR / "training_run_analysis_report.txt"

    summary_df.to_csv(
        summary_path,
        index=False
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    outer_summary.to_csv(
        outer_path,
        index=False
    )

    lr_summary.to_csv(
        lr_path,
        index=False
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NeuroFusion AI - Nested CV Training Run Analysis\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(
            f"Training directory: {TRAINING_DIR}\n"
        )
        f.write(
            f"Training histories found: {len(history_files)}\n"
        )
        f.write(
            f"Usable training histories: {len(summary_df)}\n\n"
        )

        f.write("INDIVIDUAL RUN SUMMARY\n")
        f.write("-" * 72 + "\n")

        if display_cols:
            f.write(
                summary_df[
                    display_cols
                ]
                .sort_values(
                    ["outer_fold", "inner_fold"],
                    na_position="last"
                )
                .to_string(index=False)
            )

        f.write("\n\nOUTER-FOLD SUMMARY\n")
        f.write("-" * 72 + "\n")

        if not outer_summary.empty:
            f.write(
                outer_summary.to_string(index=False)
            )

        f.write("\n\nDIAGNOSTIC OBSERVATIONS\n")
        f.write("-" * 72 + "\n")

        for item in observations:
            f.write(f"- {item}\n")

        f.write("\n\nIMPORTANT\n")
        f.write("-" * 72 + "\n")
        f.write(
            "This analysis is descriptive only.\n"
        )
        f.write(
            "No model was retrained.\n"
        )
        f.write(
            "No dataset files were modified.\n"
        )
        f.write(
            "No outer test predictions were used for tuning.\n"
        )

    print("\n" + "=" * 72)
    print("FINAL STATUS: PASS")
    print("=" * 72)

    print("\nOutput files:")
    print(f"  {summary_path}")
    print(f"  {history_path}")
    print(f"  {outer_path}")
    print(f"  {lr_path}")
    print(f"  {report_path}")

    print("\nNo training or dataset files were modified.")


if __name__ == "__main__":
    main()