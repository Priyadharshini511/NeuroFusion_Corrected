"""
NeuroFusion AI
sMRI 3D CNN Final Test Evaluation

Purpose
-------
Evaluate the fixed best sMRI 3D CNN checkpoint on the
held-out test set.

IMPORTANT
---------
- The test set is NEVER used for training.
- The test set is NEVER used for model selection.
- The checkpoint was selected using validation AUC.
- This script performs final subject-level evaluation.
- Each subject contributes exactly one MRI tensor.

Outputs
-------
processed_data/smri/cnn_dataset/evaluation/
    test_predictions.csv
    confusion_matrix.csv
    test_metrics.csv
    test_evaluation_report.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from smri_3d_cnn import SMRI3DCNN


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

CNN_DATASET_DIR = (
    ROOT / "cnn_dataset"
)

IMAGE_DIR = (
    CNN_DATASET_DIR / "images"
)

SPLIT_DIR = (
    CNN_DATASET_DIR / "splits"
)

TRAINING_DIR = (
    CNN_DATASET_DIR / "training"
)

EVALUATION_DIR = (
    CNN_DATASET_DIR / "evaluation"
)

TEST_FILE = (
    SPLIT_DIR / "test.csv"
)

BEST_MODEL_FILE = (
    TRAINING_DIR / "best_model.pt"
)

PREDICTIONS_FILE = (
    EVALUATION_DIR / "test_predictions.csv"
)

CONFUSION_FILE = (
    EVALUATION_DIR / "confusion_matrix.csv"
)

METRICS_FILE = (
    EVALUATION_DIR / "test_metrics.csv"
)

REPORT_FILE = (
    EVALUATION_DIR / "test_evaluation_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_TEST_SUBJECTS = 26

EXPECTED_TENSOR_SHAPE = (
    1,
    95,
    117,
    99,
)

THRESHOLD = 0.50

RANDOM_SEED = 42


# ============================================================
# TEST DATASET
# ============================================================

class SMRITestDataset:
    """
    Subject-level test dataset.

    No augmentation or shuffling is performed.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
    ):
        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ):
        row = self.dataframe.iloc[index]

        subject_id = str(
            row["subject_id"]
        )

        label = int(
            row["label"]
        )

        image_path = (
            self.image_dir
            / f"{subject_id}.npy"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Tensor not found: "
                f"{image_path}"
            )

        image = np.load(
            image_path
        ).astype(
            np.float32,
            copy=False,
        )

        if tuple(image.shape) != (
            EXPECTED_TENSOR_SHAPE
        ):
            raise RuntimeError(
                f"Unexpected tensor shape for "
                f"{subject_id}: "
                f"{image.shape}; expected "
                f"{EXPECTED_TENSOR_SHAPE}"
            )

        image_tensor = torch.from_numpy(
            image
        )

        return (
            image_tensor,
            label,
            subject_id,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("NeuroFusion AI - sMRI 3D CNN FINAL TEST EVALUATION")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device              : {device}"
    )

    # --------------------------------------------------------
    # Verify required input files
    # --------------------------------------------------------

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test split not found: {TEST_FILE}"
        )

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found: "
            f"{BEST_MODEL_FILE}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Create evaluation output directory
    #
    # This must happen BEFORE any output CSV/report is saved.
    # --------------------------------------------------------

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read test split
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test subjects       : {len(test_df)}"
    )

    if len(test_df) != EXPECTED_TEST_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SUBJECTS} "
            f"test subjects; found {len(test_df)}"
        )

    required_columns = {
        "subject_id",
        "label",
    }

    missing_columns = (
        required_columns
        - set(test_df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing test columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    if test_df[
        "subject_id"
    ].duplicated().any():

        duplicates = (
            test_df.loc[
                test_df[
                    "subject_id"
                ].duplicated(
                    keep=False
                ),
                "subject_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate subject IDs detected "
            "in test set: "
            + ", ".join(
                sorted(duplicates)
            )
        )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    test_df["label"] = pd.to_numeric(
        test_df["label"],
        errors="coerce",
    )

    if test_df["label"].isna().any():

        raise RuntimeError(
            "Missing or invalid labels detected "
            "in test set."
        )

    test_df["label"] = (
        test_df["label"]
        .astype(int)
    )

    if not set(
        test_df["label"].unique()
    ).issubset({0, 1}):

        raise RuntimeError(
            "Test set contains labels other "
            "than 0 and 1."
        )

    # --------------------------------------------------------
    # Class counts
    # --------------------------------------------------------

    test_labels = (
        test_df["label"]
        .astype(int)
        .to_numpy()
    )

    test_controls = int(
        np.sum(
            test_labels == 0
        )
    )

    test_cases = int(
        np.sum(
            test_labels == 1
        )
    )

    print(
        f"Test controls       : {test_controls}"
    )

    print(
        f"Test schizophrenia  : {test_cases}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = SMRI3DCNN(
        dropout=0.30
    ).to(device)

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Check checkpoint metadata
    # --------------------------------------------------------

    checkpoint_epoch = int(
        checkpoint["epoch"]
    )

    checkpoint_auc = float(
        checkpoint["validation_auc"]
    )

    checkpoint_seed = int(
        checkpoint["random_seed"]
    )

    checkpoint_shape = tuple(
        checkpoint["input_shape"]
    )

    if checkpoint_shape != (
        EXPECTED_TENSOR_SHAPE
    ):

        raise RuntimeError(
            "Checkpoint input shape does not "
            "match expected CNN input shape: "
            f"{checkpoint_shape}"
        )

    if checkpoint_seed != RANDOM_SEED:

        raise RuntimeError(
            "Checkpoint random seed does not "
            f"match expected seed {RANDOM_SEED}: "
            f"{checkpoint_seed}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Checkpoint epoch    : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Validation AUC      : "
        f"{checkpoint_auc:.6f}"
    )

    print(
        f"Random seed         : "
        f"{checkpoint_seed}"
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    dataset = SMRITestDataset(
        test_df,
        IMAGE_DIR,
    )

    # --------------------------------------------------------
    # Final inference
    # --------------------------------------------------------

    labels = []

    probabilities = []

    subject_ids = []

    print()
    print("=" * 72)
    print("FINAL TEST INFERENCE")
    print("=" * 72)

    with torch.no_grad():

        for index in range(
            len(dataset)
        ):

            image, label, subject_id = (
                dataset[index]
            )

            # Add batch dimension:
            #
            # (1, 95, 117, 99)
            # →
            # (1, 1, 95, 117, 99)

            image = image.unsqueeze(
                0
            ).to(device)

            logits = model(
                image
            )

            probability = (
                torch.sigmoid(
                    logits
                ).item()
            )

            labels.append(
                int(label)
            )

            probabilities.append(
                float(probability)
            )

            subject_ids.append(
                subject_id
            )

            print(
                f"[{index + 1:02d}/"
                f"{len(dataset):02d}] "
                f"{subject_id} | "
                f"Label={label} | "
                f"P_sMRI={probability:.6f}"
            )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):

        raise RuntimeError(
            "Non-finite test probabilities detected."
        )

    if np.any(
        probabilities < 0.0
    ) or np.any(
        probabilities > 1.0
    ):

        raise RuntimeError(
            "Test probabilities are outside [0,1]."
        )

    # --------------------------------------------------------
    # Threshold predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(
        np.int64
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    sensitivity = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    # Test set contains both classes, therefore
    # ROC-AUC is valid.

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    # --------------------------------------------------------
    # Confusion matrix
    #
    # sklearn binary layout:
    #
    # [[TN, FP],
    #  [FN, TP]]
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    if (
        tn + fp
    ) > 0:

        specificity = (
            tn
            / (tn + fp)
        )

    else:

        specificity = float(
            "nan"
        )

    # --------------------------------------------------------
    # ROC curve / Youden threshold
    #
    # Diagnostic information only.
    #
    # IMPORTANT:
    # The primary test classification metrics continue
    # to use the pre-defined 0.50 threshold.
    # --------------------------------------------------------

    fpr, tpr, thresholds = (
        roc_curve(
            labels,
            probabilities,
        )
    )

    finite_mask = np.isfinite(
        thresholds
    )

    if np.any(
        finite_mask
    ):

        finite_indices = np.where(
            finite_mask
        )[0]

        youden_values = (
            tpr[finite_indices]
            - fpr[finite_indices]
        )

        best_youden_position = int(
            np.argmax(
                youden_values
            )
        )

        best_youden_index = (
            finite_indices[
                best_youden_position
            ]
        )

        youden_threshold = float(
            thresholds[
                best_youden_index
            ]
        )

        youden_sensitivity = float(
            tpr[
                best_youden_index
            ]
        )

        youden_specificity = float(
            1.0
            - fpr[
                best_youden_index
            ]
        )

    else:

        youden_threshold = float(
            "nan"
        )

        youden_sensitivity = float(
            "nan"
        )

        youden_specificity = float(
            "nan"
        )

    # --------------------------------------------------------
    # Predictions table
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "subject_id":
                subject_ids,

            "true_label":
                labels,

            "p_smri":
                probabilities,

            "predicted_label":
                predictions,
        }
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Confusion matrix table
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "Actual_Control",
            "Actual_Schizophrenia",
        ],
        columns=[
            "Predicted_Control",
            "Predicted_Schizophrenia",
        ],
    )

    confusion_df.to_csv(
        CONFUSION_FILE,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Metrics table
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "metric": "ROC_AUC",
                "value": auc,
            },
            {
                "metric": "Accuracy",
                "value": accuracy,
            },
            {
                "metric": "Balanced_Accuracy",
                "value": balanced_accuracy,
            },
            {
                "metric": "Precision",
                "value": precision,
            },
            {
                "metric": "Sensitivity_Recall",
                "value": sensitivity,
            },
            {
                "metric": "Specificity",
                "value": specificity,
            },
            {
                "metric": "F1",
                "value": f1,
            },
        ]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    classification_text = (
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[
                "Control",
                "Schizophrenia",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Probability diagnostics
    # --------------------------------------------------------

    probability_min = float(
        probabilities.min()
    )

    probability_max = float(
        probabilities.max()
    )

    probability_mean = float(
        probabilities.mean()
    )

    probability_std = float(
        probabilities.std()
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = [
        "NeuroFusion AI - sMRI 3D CNN",
        "Final Held-Out Test Evaluation",
        "=" * 72,
        "",
        "Evaluation protocol:",
        "  Model checkpoint selected using validation AUC.",
        "  Test set was not used during training.",
        "  Test set was not used for model selection.",
        "  One subject = one MRI tensor.",
        "",
        f"Checkpoint epoch       : "
        f"{checkpoint_epoch}",
        f"Validation AUC         : "
        f"{checkpoint_auc:.6f}",
        f"Random seed            : "
        f"{checkpoint_seed}",
        "",
        f"Test subjects          : "
        f"{len(labels)}",
        f"Test controls          : "
        f"{test_controls}",
        f"Test schizophrenia     : "
        f"{test_cases}",
        f"Decision threshold     : "
        f"{THRESHOLD:.2f}",
        "",
        "Final test metrics:",
        f"  ROC-AUC              : "
        f"{auc:.6f}",
        f"  Accuracy             : "
        f"{accuracy:.6f}",
        f"  Balanced Accuracy    : "
        f"{balanced_accuracy:.6f}",
        f"  Precision            : "
        f"{precision:.6f}",
        f"  Sensitivity/Recall   : "
        f"{sensitivity:.6f}",
        f"  Specificity          : "
        f"{specificity:.6f}",
        f"  F1                   : "
        f"{f1:.6f}",
        "",
        "Confusion matrix:",
        "  Rows    = actual",
        "  Columns = predicted",
        "",
        str(confusion_df),
        "",
        "Classification report:",
        classification_text,
        "",
        "Probability diagnostics:",
        f"  Minimum              : "
        f"{probability_min:.6f}",
        f"  Maximum              : "
        f"{probability_max:.6f}",
        f"  Mean                 : "
        f"{probability_mean:.6f}",
        f"  Standard deviation   : "
        f"{probability_std:.6f}",
        "",
        "Diagnostic ROC information:",
        f"  Youden threshold     : "
        f"{youden_threshold:.6f}",
        f"  Youden sensitivity   : "
        f"{youden_sensitivity:.6f}",
        f"  Youden specificity   : "
        f"{youden_specificity:.6f}",
        "",
        "NOTE:",
        "The primary reported test classification metrics",
        "use the pre-defined 0.50 threshold.",
        "The Youden threshold is provided only as diagnostic",
        "information and must not be used to retrospectively",
        "optimize the held-out test result.",
        "",
        f"Subject predictions   : "
        f"{PREDICTIONS_FILE}",
        f"Confusion matrix      : "
        f"{CONFUSION_FILE}",
        f"Metrics               : "
        f"{METRICS_FILE}",
        f"Report                : "
        f"{REPORT_FILE}",
        "",
        "FINAL STATUS: PASS",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)

    print(
        f"ROC-AUC             : "
        f"{auc:.6f}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"Precision           : "
        f"{precision:.6f}"
    )

    print(
        f"Sensitivity/Recall  : "
        f"{sensitivity:.6f}"
    )

    print(
        f"Specificity         : "
        f"{specificity:.6f}"
    )

    print(
        f"F1                  : "
        f"{f1:.6f}"
    )

    print()
    print("Confusion matrix:")
    print(confusion_df)

    print()
    print(
        "Probability diagnostics:"
    )

    print(
        f"  Min : {probability_min:.6f}"
    )

    print(
        f"  Max : {probability_max:.6f}"
    )

    print(
        f"  Mean: {probability_mean:.6f}"
    )

    print(
        f"  Std : {probability_std:.6f}"
    )

    print()
    print(
        f"Predictions : "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Metrics     : "
        f"{METRICS_FILE}"
    )

    print(
        f"Report      : "
        f"{REPORT_FILE}"
    )

    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()