"""
NeuroFusion AI
sMRI Branch - 3D CNN Training History Analysis

Purpose
-------
Analyze the existing training history of the baseline sMRI 3D CNN.

This script DOES NOT:
    - retrain the model
    - modify the checkpoint
    - modify the dataset
    - modify train/validation/test splits
    - modify existing evaluation results

It analyzes:
    1. Training history integrity
    2. Loss progression
    3. Accuracy progression
    4. Training AUC progression
    5. Validation AUC progression
    6. Validation loss progression
    7. Best epoch
    8. Overfitting pattern
    9. Learning-rate progression
   10. Early-stopping behavior
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "processed_data/smri/cnn_dataset"
)

TRAINING_DIR = (
    BASE_DIR / "training"
)

HISTORY_FILE = (
    TRAINING_DIR / "training_history.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "diagnostics"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):

    print()

    print(
        "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )


def find_column(
    dataframe,
    candidates,
):

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    return None


def safe_float(value):

    try:

        return float(value)

    except Exception:

        return np.nan


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "NeuroFusion AI - sMRI 3D CNN Training History Analysis"
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 1. CHECK HISTORY FILE
    # ========================================================

    print_section(
        "1. Loading Training History"
    )

    if not HISTORY_FILE.exists():

        raise FileNotFoundError(
            f"Training history file not found:\n"
            f"{HISTORY_FILE}"
        )

    history = pd.read_csv(
        HISTORY_FILE
    )

    print(
        f"History file: "
        f"{HISTORY_FILE}"
    )

    print(
        f"Rows: "
        f"{len(history)}"
    )

    print()

    print(
        "Columns:"
    )

    for column in history.columns:

        print(
            f"  - {column}"
        )


    # ========================================================
    # 2. BASIC VALIDATION
    # ========================================================

    print_section(
        "2. Training History Integrity"
    )

    if len(history) == 0:

        raise ValueError(
            "Training history is empty."
        )

    print(
        f"Rows found: "
        f"{len(history)}"
    )

    duplicate_epochs = (
        history[
            "epoch"
        ]
        .duplicated()
        .sum()
        if "epoch" in history.columns
        else 0
    )

    print(
        f"Duplicate epochs: "
        f"{duplicate_epochs}"
    )

    if "epoch" in history.columns:

        epochs = (
            history[
                "epoch"
            ]
            .astype(int)
            .tolist()
        )

        print(
            f"Epoch range: "
            f"{min(epochs)} -> "
            f"{max(epochs)}"
        )

        print(
            f"Epoch sequence: "
            f"{epochs}"
        )


    # ========================================================
    # 3. IDENTIFY METRIC COLUMNS
    # ========================================================

    print_section(
        "3. Identifying Training Metrics"
    )

    train_loss_col = find_column(
        history,
        [
            "train_loss",
            "training_loss",
            "loss_train",
        ]
    )

    val_loss_col = find_column(
        history,
        [
            "validation_loss",
            "val_loss",
            "valid_loss",
        ]
    )

    train_auc_col = find_column(
        history,
        [
            "train_auc",
            "training_auc",
        ]
    )

    val_auc_col = find_column(
        history,
        [
            "validation_auc",
            "val_auc",
            "valid_auc",
        ]
    )

    train_accuracy_col = find_column(
        history,
        [
            "train_accuracy",
            "training_accuracy",
            "train_acc",
        ]
    )

    val_accuracy_col = find_column(
        history,
        [
            "validation_accuracy",
            "val_accuracy",
            "valid_accuracy",
            "val_acc",
        ]
    )

    learning_rate_col = find_column(
        history,
        [
            "learning_rate",
            "lr",
        ]
    )

    print(
        f"Training loss column     : "
        f"{train_loss_col}"
    )

    print(
        f"Validation loss column   : "
        f"{val_loss_col}"
    )

    print(
        f"Training AUC column      : "
        f"{train_auc_col}"
    )

    print(
        f"Validation AUC column    : "
        f"{val_auc_col}"
    )

    print(
        f"Training accuracy column : "
        f"{train_accuracy_col}"
    )

    print(
        f"Validation accuracy col. : "
        f"{val_accuracy_col}"
    )

    print(
        f"Learning-rate column     : "
        f"{learning_rate_col}"
    )


    # ========================================================
    # 4. PRINT COMPLETE HISTORY
    # ========================================================

    print_section(
        "4. Complete Training History"
    )

    print(
        history.to_string(
            index=False
        )
    )


    # ========================================================
    # 5. BEST VALIDATION AUC
    # ========================================================

    print_section(
        "5. Best Validation AUC"
    )

    best_val_auc_epoch = None
    best_val_auc = np.nan

    if val_auc_col is not None:

        history[
            val_auc_col
        ] = pd.to_numeric(
            history[
                val_auc_col
            ],
            errors="coerce"
        )

        valid_auc = history[
            history[
                val_auc_col
            ].notna()
        ]

        if len(valid_auc) > 0:

            best_index = (
                valid_auc[
                    val_auc_col
                ]
                .idxmax()
            )

            best_row = (
                valid_auc.loc[
                    best_index
                ]
            )

            best_val_auc = float(
                best_row[
                    val_auc_col
                ]
            )

            best_val_auc_epoch = int(
                best_row[
                    "epoch"
                ]
            )

            print(
                f"Best validation AUC: "
                f"{best_val_auc:.6f}"
            )

            print(
                f"Best epoch: "
                f"{best_val_auc_epoch}"
            )


    # ========================================================
    # 6. BEST VALIDATION LOSS
    # ========================================================

    print_section(
        "6. Best Validation Loss"
    )

    best_val_loss_epoch = None
    best_val_loss = np.nan

    if val_loss_col is not None:

        history[
            val_loss_col
        ] = pd.to_numeric(
            history[
                val_loss_col
            ],
            errors="coerce"
        )

        valid_loss = history[
            history[
                val_loss_col
            ].notna()
        ]

        if len(valid_loss) > 0:

            best_loss_index = (
                valid_loss[
                    val_loss_col
                ]
                .idxmin()
            )

            best_loss_row = (
                valid_loss.loc[
                    best_loss_index
                ]
            )

            best_val_loss = float(
                best_loss_row[
                    val_loss_col
                ]
            )

            best_val_loss_epoch = int(
                best_loss_row[
                    "epoch"
                ]
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.6f}"
            )

            print(
                f"Best loss epoch: "
                f"{best_val_loss_epoch}"
            )


    # ========================================================
    # 7. FINAL EPOCH
    # ========================================================

    print_section(
        "7. Final Recorded Epoch"
    )

    final_row = (
        history.iloc[-1]
    )

    final_epoch = int(
        final_row[
            "epoch"
        ]
    )

    print(
        f"Final epoch: "
        f"{final_epoch}"
    )

    if train_loss_col is not None:

        print(
            f"Final training loss: "
            f"{safe_float(final_row[train_loss_col]):.6f}"
        )

    if val_loss_col is not None:

        print(
            f"Final validation loss: "
            f"{safe_float(final_row[val_loss_col]):.6f}"
        )

    if train_auc_col is not None:

        print(
            f"Final training AUC: "
            f"{safe_float(final_row[train_auc_col]):.6f}"
        )

    if val_auc_col is not None:

        print(
            f"Final validation AUC: "
            f"{safe_float(final_row[val_auc_col]):.6f}"
        )

    if train_accuracy_col is not None:

        print(
            f"Final training accuracy: "
            f"{safe_float(final_row[train_accuracy_col]):.6f}"
        )

    if val_accuracy_col is not None:

        print(
            f"Final validation accuracy: "
            f"{safe_float(final_row[val_accuracy_col]):.6f}"
        )

    if learning_rate_col is not None:

        print(
            f"Final learning rate: "
            f"{safe_float(final_row[learning_rate_col]):.8f}"
        )


    # ========================================================
    # 8. OVERFITTING ANALYSIS
    # ========================================================

    print_section(
        "8. Overfitting Analysis"
    )

    overfitting_flags = []

    if (
        train_auc_col is not None
        and val_auc_col is not None
    ):

        history[
            train_auc_col
        ] = pd.to_numeric(
            history[
                train_auc_col
            ],
            errors="coerce"
        )

        train_auc_at_best = (
            history.loc[
                history["epoch"]
                == best_val_auc_epoch,
                train_auc_col
            ]
        )

        if len(
            train_auc_at_best
        ) > 0:

            train_auc_best_epoch = (
                float(
                    train_auc_at_best.iloc[0]
                )
            )

            auc_gap = (
                train_auc_best_epoch
                - best_val_auc
            )

            print(
                f"Training AUC at best "
                f"validation epoch: "
                f"{train_auc_best_epoch:.6f}"
            )

            print(
                f"Train-validation AUC gap: "
                f"{auc_gap:.6f}"
            )

            if auc_gap > 0.10:

                overfitting_flags.append(
                    "Large train-validation AUC gap (>0.10)"
                )


    if (
        train_loss_col is not None
        and val_loss_col is not None
    ):

        history[
            train_loss_col
        ] = pd.to_numeric(
            history[
                train_loss_col
            ],
            errors="coerce"
        )

        train_loss_at_best = (
            history.loc[
                history["epoch"]
                == best_val_auc_epoch,
                train_loss_col
            ]
        )

        val_loss_at_best = (
            history.loc[
                history["epoch"]
                == best_val_auc_epoch,
                val_loss_col
            ]
        )

        if (
            len(train_loss_at_best)
            > 0
            and len(val_loss_at_best)
            > 0
        ):

            train_loss_value = float(
                train_loss_at_best.iloc[0]
            )

            val_loss_value = float(
                val_loss_at_best.iloc[0]
            )

            print(
                f"Training loss at best "
                f"validation epoch: "
                f"{train_loss_value:.6f}"
            )

            print(
                f"Validation loss at best "
                f"validation epoch: "
                f"{val_loss_value:.6f}"
            )


    if len(
        overfitting_flags
    ) == 0:

        print(
            "No strong overfitting flag "
            "detected by the simple thresholds."
        )

    else:

        print(
            "Potential overfitting indicators:"
        )

        for flag in overfitting_flags:

            print(
                f"  - {flag}"
            )


    # ========================================================
    # 9. VALIDATION AUC TRAJECTORY
    # ========================================================

    print_section(
        "9. Validation AUC Trajectory"
    )

    if val_auc_col is not None:

        for _, row in history.iterrows():

            epoch = int(
                row["epoch"]
            )

            auc = safe_float(
                row[
                    val_auc_col
                ]
            )

            print(
                f"Epoch {epoch:02d}: "
                f"Validation AUC = "
                f"{auc:.6f}"
            )


    # ========================================================
    # 10. VALIDATION LOSS TRAJECTORY
    # ========================================================

    print_section(
        "10. Validation Loss Trajectory"
    )

    if val_loss_col is not None:

        for _, row in history.iterrows():

            epoch = int(
                row["epoch"]
            )

            loss = safe_float(
                row[
                    val_loss_col
                ]
            )

            print(
                f"Epoch {epoch:02d}: "
                f"Validation Loss = "
                f"{loss:.6f}"
            )


    # ========================================================
    # 11. LEARNING RATE TRAJECTORY
    # ========================================================

    print_section(
        "11. Learning Rate Trajectory"
    )

    if learning_rate_col is not None:

        history[
            learning_rate_col
        ] = pd.to_numeric(
            history[
                learning_rate_col
            ],
            errors="coerce"
        )

        for _, row in history.iterrows():

            epoch = int(
                row["epoch"]
            )

            lr = safe_float(
                row[
                    learning_rate_col
                ]
            )

            print(
                f"Epoch {epoch:02d}: "
                f"Learning Rate = "
                f"{lr:.10f}"
            )

        unique_lrs = (
            history[
                learning_rate_col
            ]
            .dropna()
            .unique()
        )

        print()

        print(
            f"Number of unique learning "
            f"rates: {len(unique_lrs)}"
        )


    # ========================================================
    # 12. AUC IMPROVEMENT / DECLINE
    # ========================================================

    print_section(
        "12. Validation AUC Changes"
    )

    if val_auc_col is not None:

        auc_values = (
            history[
                val_auc_col
            ]
            .dropna()
            .values
        )

        if len(auc_values) > 1:

            changes = np.diff(
                auc_values
            )

            positive_changes = int(
                np.sum(
                    changes > 0
                )
            )

            negative_changes = int(
                np.sum(
                    changes < 0
                )
            )

            unchanged = int(
                np.sum(
                    changes == 0
                )
            )

            print(
                f"Improving epochs: "
                f"{positive_changes}"
            )

            print(
                f"Declining epochs: "
                f"{negative_changes}"
            )

            print(
                f"Unchanged epochs: "
                f"{unchanged}"
            )

            largest_drop = (
                float(
                    changes.min()
                )
            )

            largest_gain = (
                float(
                    changes.max()
                )
            )

            print(
                f"Largest AUC gain: "
                f"{largest_gain:.6f}"
            )

            print(
                f"Largest AUC drop: "
                f"{largest_drop:.6f}"
            )


    # ========================================================
    # 13. CREATE ANALYSIS TABLE
    # ========================================================

    print_section(
        "13. Creating Analysis Table"
    )

    analysis_rows = []

    for _, row in history.iterrows():

        analysis_row = {

            "epoch":
                int(
                    row["epoch"]
                ),
        }

        if train_loss_col is not None:

            analysis_row[
                "train_loss"
            ] = safe_float(
                row[
                    train_loss_col
                ]
            )

        if val_loss_col is not None:

            analysis_row[
                "validation_loss"
            ] = safe_float(
                row[
                    val_loss_col
                ]
            )

        if train_auc_col is not None:

            analysis_row[
                "train_auc"
            ] = safe_float(
                row[
                    train_auc_col
                ]
            )

        if val_auc_col is not None:

            analysis_row[
                "validation_auc"
            ] = safe_float(
                row[
                    val_auc_col
                ]
            )

        if train_accuracy_col is not None:

            analysis_row[
                "train_accuracy"
            ] = safe_float(
                row[
                    train_accuracy_col
                ]
            )

        if val_accuracy_col is not None:

            analysis_row[
                "validation_accuracy"
            ] = safe_float(
                row[
                    val_accuracy_col
                ]
            )

        if learning_rate_col is not None:

            analysis_row[
                "learning_rate"
            ] = safe_float(
                row[
                    learning_rate_col
                ]
            )

        analysis_rows.append(
            analysis_row
        )

    analysis_df = pd.DataFrame(
        analysis_rows
    )

    analysis_path = (
        OUTPUT_DIR
        / "training_history_analysis.csv"
    )

    analysis_df.to_csv(
        analysis_path,
        index=False
    )

    print(
        f"Analysis table: "
        f"{analysis_path}"
    )


    # ========================================================
    # 14. CREATE REPORT
    # ========================================================

    print_section(
        "14. Creating Training History Report"
    )

    report_lines = []

    report_lines.append(
        "NeuroFusion AI"
    )

    report_lines.append(
        "sMRI 3D CNN Training History Analysis"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        "This is a diagnostic analysis of the "
        "existing baseline training run."
    )

    report_lines.append(
        "No retraining was performed."
    )

    report_lines.append("")

    report_lines.append(
        "DATASET / TRAINING RUN"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"History rows: "
        f"{len(history)}"
    )

    report_lines.append(
        f"Epoch range: "
        f"{min(epochs)} -> {max(epochs)}"
    )

    report_lines.append(
        f"Duplicate epochs: "
        f"{duplicate_epochs}"
    )

    report_lines.append("")

    report_lines.append(
        "BEST VALIDATION PERFORMANCE"
    )

    report_lines.append(
        "-" * 40
    )

    if not np.isnan(
        best_val_auc
    ):

        report_lines.append(
            f"Best validation AUC: "
            f"{best_val_auc:.6f}"
        )

        report_lines.append(
            f"Best validation AUC epoch: "
            f"{best_val_auc_epoch}"
        )

    if not np.isnan(
        best_val_loss
    ):

        report_lines.append(
            f"Best validation loss: "
            f"{best_val_loss:.6f}"
        )

        report_lines.append(
            f"Best validation loss epoch: "
            f"{best_val_loss_epoch}"
        )

    report_lines.append("")

    report_lines.append(
        "FINAL RECORDED EPOCH"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Final epoch: "
        f"{final_epoch}"
    )

    if train_auc_col is not None:

        report_lines.append(
            f"Final training AUC: "
            f"{safe_float(final_row[train_auc_col]):.6f}"
        )

    if val_auc_col is not None:

        report_lines.append(
            f"Final validation AUC: "
            f"{safe_float(final_row[val_auc_col]):.6f}"
        )

    if train_loss_col is not None:

        report_lines.append(
            f"Final training loss: "
            f"{safe_float(final_row[train_loss_col]):.6f}"
        )

    if val_loss_col is not None:

        report_lines.append(
            f"Final validation loss: "
            f"{safe_float(final_row[val_loss_col]):.6f}"
        )

    report_lines.append("")

    report_lines.append(
        "OVERFITTING FLAGS"
    )

    report_lines.append(
        "-" * 40
    )

    if len(
        overfitting_flags
    ) == 0:

        report_lines.append(
            "No strong simple-threshold "
            "overfitting flag detected."
        )

    else:

        for flag in overfitting_flags:

            report_lines.append(
                flag
            )

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        "The validation AUC was used for "
        "checkpoint selection in the original run."
    )

    report_lines.append(
        "The held-out test set was not used by "
        "this analysis."
    )

    report_lines.append(
        "The poor test performance observed "
        "previously should not be explained away "
        "using validation performance."
    )

    report_lines.append(
        "This report is diagnostic and does not "
        "constitute a final clinical performance claim."
    )

    report_path = (
        OUTPUT_DIR
        / "smri_training_history_analysis_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_section(
        "TRAINING HISTORY ANALYSIS SUMMARY"
    )

    print(
        f"Epochs analyzed      : "
        f"{len(history)}"
    )

    print(
        f"Best validation AUC : "
        f"{best_val_auc:.6f}"
    )

    print(
        f"Best AUC epoch      : "
        f"{best_val_auc_epoch}"
    )

    if not np.isnan(
        best_val_loss
    ):

        print(
            f"Best validation loss: "
            f"{best_val_loss:.6f}"
        )

        print(
            f"Best loss epoch     : "
            f"{best_val_loss_epoch}"
        )

    print()

    print(
        f"Analysis CSV: "
        f"{analysis_path}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )

    print()

    print(
        "No model or dataset files were modified."
    )

    print(
        "FINAL STATUS: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()