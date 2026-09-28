"""
NeuroFusion AI
sMRI 3D CNN Training Pipeline

Purpose
-------
Train the subject-level sMRI 3D CNN using the fixed
train/validation/test split created in Step 21.

Important
---------
- Subject-level split is fixed and never recreated here.
- Test data is NOT used for training or model selection.
- Best model is selected using validation performance.
- Test evaluation will be performed separately after training.
- MRI tensors are loaded from the already validated .npy files.

Input
-----
processed_data/smri/cnn_dataset/images/
processed_data/smri/cnn_dataset/splits/train.csv
processed_data/smri/cnn_dataset/splits/validation.csv
processed_data/smri/cnn_dataset/splits/test.csv

Output
------
processed_data/smri/cnn_dataset/training/
    best_model.pt
    training_history.csv
    training_summary.txt
"""


from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

TRAIN_FILE = (
    SPLIT_DIR / "train.csv"
)

VALIDATION_FILE = (
    SPLIT_DIR / "validation.csv"
)

TEST_FILE = (
    SPLIT_DIR / "test.csv"
)

TRAINING_DIR = (
    CNN_DATASET_DIR / "training"
)

BEST_MODEL_FILE = (
    TRAINING_DIR / "best_model.pt"
)

HISTORY_FILE = (
    TRAINING_DIR / "training_history.csv"
)

SUMMARY_FILE = (
    TRAINING_DIR / "training_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

EXPECTED_TENSOR_SHAPE = (
    1,
    95,
    117,
    99,
)

BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

EARLY_STOPPING_PATIENCE = 10

NUM_WORKERS = 0

THRESHOLD = 0.50


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferred for this
    # small research dataset.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SMRIDataset(Dataset):
    """
    Subject-level sMRI dataset.

    Each subject has exactly one validated .npy tensor.
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

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ):
        row = self.dataframe.iloc[index]

        subject_id = str(
            row["subject_id"]
        )

        label = float(
            row["label"]
        )

        image_path = (
            self.image_dir
            / f"{subject_id}.npy"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Tensor not found: {image_path}"
            )

        image = np.load(
            image_path
        ).astype(
            np.float32,
            copy=False,
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if tuple(image.shape) != (
            EXPECTED_TENSOR_SHAPE
        ):
            raise RuntimeError(
                f"Unexpected tensor shape for "
                f"{subject_id}: "
                f"{image.shape}; expected "
                f"{EXPECTED_TENSOR_SHAPE}"
            )

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image
        )

        label_tensor = torch.tensor(
            [label],
            dtype=torch.float32,
        )

        return (
            image_tensor,
            label_tensor,
            subject_id,
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
):
    """
    Calculate classification metrics from
    subject-level probabilities.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int64)

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

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    # ROC-AUC requires both classes.
    if len(
        np.unique(labels)
    ) == 2:

        auc = roc_auc_score(
            labels,
            probabilities,
        )

    else:

        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


# ============================================================
# ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train for one epoch.
    """

    model.train()

    running_loss = 0.0

    labels_all = []

    probabilities_all = []

    subject_count = 0

    for images, labels, _ in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        subject_count += batch_size

        probabilities = torch.sigmoid(
            logits
        )

        labels_all.extend(
            labels.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        probabilities_all.extend(
            probabilities.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

    epoch_loss = (
        running_loss
        / subject_count
    )

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the model without gradient computation.
    """

    model.eval()

    running_loss = 0.0

    labels_all = []

    probabilities_all = []

    subject_count = 0

    with torch.no_grad():

        for images, labels, _ in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            batch_size = (
                images.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            subject_count += batch_size

            probabilities = (
                torch.sigmoid(logits)
            )

            labels_all.extend(
                labels.cpu()
                .numpy()
                .reshape(-1)
                .tolist()
            )

            probabilities_all.extend(
                probabilities.cpu()
                .numpy()
                .reshape(-1)
                .tolist()
            )

    epoch_loss = (
        running_loss
        / subject_count
    )

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
    )

    return epoch_loss, metrics


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_split_dataframe(
    dataframe: pd.DataFrame,
    expected_name: str,
):
    """
    Validate one split before training.
    """

    required_columns = {
        "subject_id",
        "label",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise RuntimeError(
            f"{expected_name} is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if dataframe[
        "subject_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{expected_name} contains duplicate "
            "subject IDs."
        )

    labels = set(
        dataframe["label"]
        .astype(int)
        .unique()
    )

    if not labels.issubset({0, 1}):

        raise RuntimeError(
            f"{expected_name} contains invalid labels: "
            f"{labels}"
        )


def validate_no_split_overlap(
    train,
    validation,
    test,
):
    """
    Confirm no subject appears in more than one split.
    """

    train_ids = set(
        train["subject_id"]
        .astype(str)
    )

    validation_ids = set(
        validation["subject_id"]
        .astype(str)
    )

    test_ids = set(
        test["subject_id"]
        .astype(str)
    )

    if train_ids & validation_ids:
        raise RuntimeError(
            "Train/validation subject overlap detected."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test subject overlap detected."
        )

    if validation_ids & test_ids:
        raise RuntimeError(
            "Validation/test subject overlap detected."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("NeuroFusion AI - sMRI 3D CNN TRAINING")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device          : {device}"
    )

    # --------------------------------------------------------
    # Read fixed splits
    # --------------------------------------------------------

    for file_path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required split file not found: "
                f"{file_path}"
            )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Validate splits
    # --------------------------------------------------------

    validate_split_dataframe(
        train_df,
        "Train split",
    )

    validate_split_dataframe(
        validation_df,
        "Validation split",
    )

    validate_split_dataframe(
        test_df,
        "Test split",
    )

    validate_no_split_overlap(
        train_df,
        validation_df,
        test_df,
    )

    print(
        f"Train subjects       : {len(train_df)}"
    )

    print(
        f"Validation subjects  : {len(validation_df)}"
    )

    print(
        f"Test subjects        : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = SMRIDataset(
        train_df,
        IMAGE_DIR,
    )

    validation_dataset = SMRIDataset(
        validation_df,
        IMAGE_DIR,
    )

    test_dataset = SMRIDataset(
        test_df,
        IMAGE_DIR,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SMRI3DCNN(
        dropout=DROPOUT
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters   : {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    train_labels = (
        train_df["label"]
        .astype(int)
        .to_numpy()
    )

    negative_count = np.sum(
        train_labels == 0
    )

    positive_count = np.sum(
        train_labels == 1
    )

    # Positive-class weight:
    #
    #   negative_count / positive_count
    #
    # With 65 controls and 55 cases this is
    # approximately 1.18.
    #
    # The imbalance is mild, but the weighting is
    # retained explicitly and recorded for reproducibility.

    pos_weight_value = (
        negative_count
        / positive_count
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"Training controls   : {negative_count}"
    )

    print(
        f"Training cases      : {positive_count}"
    )

    print(
        f"Positive class weight: "
        f"{pos_weight_value:.6f}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training output directory
    # --------------------------------------------------------

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    history = []

    best_validation_auc = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING START")
    print("=" * 72)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_loss, validation_metrics = (
            evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        validation_auc = (
            validation_metrics["auc"]
        )

        scheduler.step(
            validation_auc
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,

            "train_loss": train_loss,
            "train_accuracy":
                train_metrics["accuracy"],
            "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],
            "train_precision":
                train_metrics["precision"],
            "train_recall":
                train_metrics["recall"],
            "train_f1":
                train_metrics["f1"],
            "train_auc":
                train_metrics["auc"],

            "validation_loss":
                validation_loss,
            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],
            "validation_balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],
            "validation_precision":
                validation_metrics[
                    "precision"
                ],
            "validation_recall":
                validation_metrics[
                    "recall"
                ],
            "validation_f1":
                validation_metrics[
                    "f1"
                ],
            "validation_auc":
                validation_metrics[
                    "auc"
                ],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {validation_loss:.4f} | "
            f"Train AUC {train_metrics['auc']:.4f} | "
            f"Val AUC {validation_auc:.4f} | "
            f"Val F1 {validation_metrics['f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model using validation AUC
        # ----------------------------------------------------

        if (
            np.isfinite(validation_auc)
            and validation_auc
            > best_validation_auc
        ):

            best_validation_auc = (
                validation_auc
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_auc":
                        validation_auc,
                    "random_seed":
                        RANDOM_SEED,
                    "input_shape":
                        EXPECTED_TENSOR_SHAPE,
                    "dropout":
                        DROPOUT,
                    "batch_size":
                        BATCH_SIZE,
                    "learning_rate":
                        LEARNING_RATE,
                    "weight_decay":
                        WEIGHT_DECAY,
                    "pos_weight":
                        pos_weight_value,
                },
                BEST_MODEL_FILE,
            )

            print(
                f"  -> Best model saved "
                f"(validation AUC = "
                f"{validation_auc:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    if not BEST_MODEL_FILE.exists():

        raise RuntimeError(
            "Best model was not saved. "
            "Training cannot be considered successful."
        )

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Test loader is intentionally NOT evaluated here.
    #
    # Test evaluation must be performed only after
    # all training/model-selection decisions are complete.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    summary_lines = [
        "NeuroFusion AI - sMRI 3D CNN Training",
        "=" * 72,
        "",
        f"Random seed             : {RANDOM_SEED}",
        f"Device                  : {device}",
        f"Input tensor shape      : "
        f"{EXPECTED_TENSOR_SHAPE}",
        "",
        f"Train subjects          : {len(train_df)}",
        f"Validation subjects     : {len(validation_df)}",
        f"Test subjects           : {len(test_df)}",
        "",
        f"Batch size              : {BATCH_SIZE}",
        f"Maximum epochs          : {NUM_EPOCHS}",
        f"Learning rate           : {LEARNING_RATE}",
        f"Weight decay            : {WEIGHT_DECAY}",
        f"Dropout                 : {DROPOUT}",
        f"Early stopping patience : "
        f"{EARLY_STOPPING_PATIENCE}",
        "",
        f"Training controls       : "
        f"{negative_count}",
        f"Training cases          : "
        f"{positive_count}",
        f"Positive class weight   : "
        f"{pos_weight_value:.6f}",
        "",
        f"Best epoch              : {best_epoch}",
        f"Best validation AUC     : "
        f"{best_validation_auc:.6f}",
        f"Epochs completed        : "
        f"{len(history)}",
        "",
        "Test set status:",
        "  NOT evaluated during training.",
        "  Test evaluation must be performed separately",
        "  after model selection is complete.",
        "",
        f"Best model              : "
        f"{BEST_MODEL_FILE}",
        f"Training history        : "
        f"{HISTORY_FILE}",
        "",
        "FINAL STATUS: PASS",
    ]

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 72)

    print(
        f"Best epoch          : {best_epoch}"
    )

    print(
        f"Best validation AUC : "
        f"{best_validation_auc:.6f}"
    )

    print(
        f"Best model          : "
        f"{BEST_MODEL_FILE}"
    )

    print(
        f"History             : "
        f"{HISTORY_FILE}"
    )

    print(
        f"Summary             : "
        f"{SUMMARY_FILE}"
    )

    print()
    print(
        "Test set was NOT evaluated."
    )

    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()