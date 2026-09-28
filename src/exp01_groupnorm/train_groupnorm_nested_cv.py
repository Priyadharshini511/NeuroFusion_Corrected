"""
NeuroFusion AI
Experiment 01 - GroupNorm sMRI 3D CNN
Full 5x4 Nested Cross-Validation Training

Purpose
-------
Controlled experiment against the validated baseline sMRI 3D CNN.

ONLY intended model change:
    BatchNorm3d -> GroupNorm

Everything else is kept consistent with the validated
full 5x4 nested-CV methodology.

This script does NOT modify:
    - baseline model
    - baseline training results
    - CV assignments
    - nested CV assignments
    - CNN tensors
    - baseline experiment outputs
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

from smri_3d_cnn_groupnorm import SMRI3DCNNGroupNorm


# ============================================================
# PATHS
# ============================================================

ROOT = Path("processed_data") / "smri"

CNN_DATASET_DIR = ROOT / "cnn_dataset"

IMAGE_DIR = CNN_DATASET_DIR / "images"

NESTED_CV_DIR = (
    CNN_DATASET_DIR
    / "cross_validation"
    / "nested"
)

# ------------------------------------------------------------
# Experiment-specific output directory.
# ------------------------------------------------------------

EXPERIMENT_DIR = (
    CNN_DATASET_DIR
    / "experiments"
    / "exp01_groupnorm"
)

TRAINING_DIR = (
    EXPERIMENT_DIR
    / "training_full_5x4"
)

NESTED_ASSIGNMENTS_FILE = (
    NESTED_CV_DIR
    / "nested_cv_assignments.csv"
)

OOF_PREDICTIONS_FILE = (
    TRAINING_DIR
    / "nested_cv_oof_predictions.csv"
)

FOLD_METRICS_FILE = (
    TRAINING_DIR
    / "nested_cv_fold_metrics.csv"
)

INNER_RESULTS_FILE = (
    TRAINING_DIR
    / "inner_cv_results.csv"
)

EPOCH_SELECTION_FILE = (
    TRAINING_DIR
    / "epoch_selection.csv"
)

CV_SUMMARY_FILE = (
    TRAINING_DIR
    / "nested_cv_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

OUTER_FOLDS = 5

INNER_FOLDS = 4

EXPECTED_SUBJECTS = 172

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

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SMRIDataset(Dataset):

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

    def __getitem__(self, index):

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

        if tuple(image.shape) != (
            EXPECTED_TENSOR_SHAPE
        ):

            raise RuntimeError(
                f"Unexpected tensor shape for "
                f"{subject_id}: {image.shape}; "
                f"expected {EXPECTED_TENSOR_SHAPE}"
            )

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
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "auc": float(
            auc
        ),
    }


# ============================================================
# CREATE LOSS
# ============================================================

def create_criterion(
    dataframe: pd.DataFrame,
    device: torch.device,
):

    labels = (
        dataframe[
            "label"
        ]
        .astype(int)
        .to_numpy()
    )

    negative_count = int(
        np.sum(
            labels == 0
        )
    )

    positive_count = int(
        np.sum(
            labels == 1
        )
    )

    if (
        negative_count == 0
        or positive_count == 0
    ):

        raise RuntimeError(
            "Training partition does not contain both classes."
        )

    pos_weight_value = (
        negative_count
        / positive_count
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    return (
        criterion,
        pos_weight_value,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

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

        batch_size = images.size(0)

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

    if subject_count == 0:

        raise RuntimeError(
            "Training loader is empty."
        )

    epoch_loss = (
        running_loss
        / subject_count
    )

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
    )

    return (
        epoch_loss,
        metrics,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    labels_all = []

    probabilities_all = []

    subjects_all = []

    subject_count = 0

    with torch.no_grad():

        for images, labels, subjects in loader:

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

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            subject_count += batch_size

            probabilities = torch.sigmoid(
                logits
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

            subjects_all.extend(
                list(subjects)
            )

    if subject_count == 0:

        raise RuntimeError(
            "Evaluation loader is empty."
        )

    epoch_loss = (
        running_loss
        / subject_count
    )

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
    )

    return (
        epoch_loss,
        metrics,
        labels_all,
        probabilities_all,
        subjects_all,
    )


# ============================================================
# CREATE LOADER
# ============================================================

def create_loader(
    dataframe,
    shuffle,
):

    dataset = SMRIDataset(
        dataframe,
        IMAGE_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# ============================================================
# TRAIN ONE INNER FOLD
# ============================================================

def train_inner_fold(
    outer_fold: int,
    inner_fold: int,
    outer_df: pd.DataFrame,
    device: torch.device,
):

    print()
    print("-" * 72)
    print(
        f"OUTER {outer_fold} | "
        f"INNER {inner_fold}"
    )
    print("-" * 72)

    split_df = outer_df[
        outer_df["inner_fold"]
        == inner_fold
    ].copy()

    train_df = split_df[
        split_df["role"]
        == "INNER_TRAIN"
    ].copy()

    validation_df = split_df[
        split_df["role"]
        == "INNER_VALIDATION"
    ].copy()

    test_df = split_df[
        split_df["role"]
        == "OUTER_TEST"
    ].copy()

    train_ids = set(
        train_df[
            "subject_id"
        ].astype(str)
    )

    validation_ids = set(
        validation_df[
            "subject_id"
        ].astype(str)
    )

    test_ids = set(
        test_df[
            "subject_id"
        ].astype(str)
    )

    if train_ids.intersection(
        validation_ids
    ):

        raise RuntimeError(
            f"Outer {outer_fold}, Inner {inner_fold}: "
            "train/validation overlap."
        )

    if train_ids.intersection(
        test_ids
    ):

        raise RuntimeError(
            f"Outer {outer_fold}, Inner {inner_fold}: "
            "train/outer-test overlap."
        )

    if validation_ids.intersection(
        test_ids
    ):

        raise RuntimeError(
            f"Outer {outer_fold}, Inner {inner_fold}: "
            "validation/outer-test overlap."
        )

    expected_outer_subjects = set(
        outer_df[
            "subject_id"
        ].astype(str)
    )

    actual_subjects = (
        train_ids
        | validation_ids
        | test_ids
    )

    if actual_subjects != (
        expected_outer_subjects
    ):

        raise RuntimeError(
            f"Outer {outer_fold}, Inner {inner_fold}: "
            "partition does not cover the outer fold."
        )

    for name, dataframe in [
        ("inner train", train_df),
        ("inner validation", validation_df),
        ("outer test", test_df),
    ]:

        classes = set(
            dataframe[
                "label"
            ].astype(int)
        )

        if classes != {0, 1}:

            raise RuntimeError(
                f"Outer {outer_fold}, Inner {inner_fold}: "
                f"{name} does not contain both classes."
            )

    print(
        f"Train: {len(train_df)} "
        f"(0={(train_df['label'] == 0).sum()}, "
        f"1={(train_df['label'] == 1).sum()})"
    )

    print(
        f"Validation: {len(validation_df)} "
        f"(0={(validation_df['label'] == 0).sum()}, "
        f"1={(validation_df['label'] == 1).sum()})"
    )

    print(
        f"Outer test: {len(test_df)}"
    )

    run_seed = (
        RANDOM_SEED
        + outer_fold * 100
        + inner_fold
    )

    set_seed(
        run_seed
    )

    train_loader = create_loader(
        train_df,
        shuffle=True,
    )

    validation_loader = create_loader(
        validation_df,
        shuffle=False,
    )

    # --------------------------------------------------------
    # EXPERIMENT 01 MODEL
    # --------------------------------------------------------

    model = SMRI3DCNNGroupNorm(
        dropout=DROPOUT
    ).to(device)

    criterion, pos_weight_value = (
        create_criterion(
            train_df,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
    )

    best_auc = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

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

        (
            validation_loss,
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
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

        history.append(
            {
                "outer_fold":
                    outer_fold,

                "inner_fold":
                    inner_fold,

                "epoch":
                    epoch,

                "learning_rate":
                    current_lr,

                "train_loss":
                    train_loss,

                "train_auc":
                    train_metrics["auc"],

                "validation_loss":
                    validation_loss,

                "validation_auc":
                    validation_auc,
            }
        )

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train AUC {train_metrics['auc']:.4f} | "
            f"Val AUC {validation_auc:.4f}"
        )

        if validation_auc > best_auc:

            best_auc = validation_auc

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    print(
        f"Best inner validation AUC: "
        f"{best_auc:.6f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    result = {
        "outer_fold":
            outer_fold,

        "inner_fold":
            inner_fold,

        "best_epoch":
            best_epoch,

        "best_validation_auc":
            best_auc,

        "best_validation_loss":
            float(
                min(
                    row[
                        "validation_loss"
                    ]
                    for row in history
                )
            ),

        "train_subjects":
            len(train_df),

        "validation_subjects":
            len(validation_df),

        "pos_weight":
            pos_weight_value,

        "random_seed":
            run_seed,
    }

    history_df = pd.DataFrame(
        history
    )

    return (
        result,
        history_df,
    )


# ============================================================
# SELECT EPOCH FROM ALL FOUR INNER FOLDS
# ============================================================

def select_epoch(
    inner_results,
    outer_fold,
):

    fold_results = [
        result
        for result in inner_results
        if result[
            "outer_fold"
        ] == outer_fold
    ]

    if len(fold_results) != (
        INNER_FOLDS
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"expected {INNER_FOLDS} inner results, "
            f"found {len(fold_results)}."
        )

    best_epochs = [
        result[
            "best_epoch"
        ]
        for result in fold_results
    ]

    selected_epoch = int(
        round(
            np.median(
                best_epochs
            )
        )
    )

    selected_epoch = max(
        1,
        min(
            selected_epoch,
            NUM_EPOCHS,
        ),
    )

    return selected_epoch


# ============================================================
# TRAIN FINAL OUTER MODEL
# ============================================================

def train_final_outer_model(
    outer_fold,
    outer_df,
    selected_epoch,
    device,
):

    print()
    print("=" * 72)
    print(
        f"FINAL OUTER MODEL {outer_fold}"
    )
    print("=" * 72)

    outer_test_df = (
        outer_df[
            outer_df["role"]
            == "OUTER_TEST"
        ]
        .drop_duplicates(
            subset=["subject_id"]
        )
        .copy()
    )

    outer_train_df = (
        outer_df[
            outer_df["role"]
            != "OUTER_TEST"
        ]
        .drop_duplicates(
            subset=["subject_id"]
        )
        .copy()
    )

    train_ids = set(
        outer_train_df[
            "subject_id"
        ].astype(str)
    )

    test_ids = set(
        outer_test_df[
            "subject_id"
        ].astype(str)
    )

    if train_ids.intersection(
        test_ids
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final train/test overlap."
        )

    if len(
        train_ids
        | test_ids
    ) != len(
        set(
            outer_df[
                "subject_id"
            ].astype(str)
        )
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final train/test partition invalid."
        )

    if set(
        outer_train_df[
            "label"
        ].astype(int)
    ) != {0, 1}:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final outer training does not contain both classes."
        )

    if set(
        outer_test_df[
            "label"
        ].astype(int)
    ) != {0, 1}:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "outer test does not contain both classes."
        )

    print(
        f"Full outer training: "
        f"{len(outer_train_df)}"
    )

    print(
        f"Outer test: "
        f"{len(outer_test_df)}"
    )

    print(
        f"Selected epoch: "
        f"{selected_epoch}"
    )

    final_seed = (
        RANDOM_SEED
        + outer_fold * 1000
    )

    set_seed(
        final_seed
    )

    train_loader = create_loader(
        outer_train_df,
        shuffle=True,
    )

    test_loader = create_loader(
        outer_test_df,
        shuffle=False,
    )

    # --------------------------------------------------------
    # EXPERIMENT 01 MODEL
    # --------------------------------------------------------

    model = SMRI3DCNNGroupNorm(
        dropout=DROPOUT
    ).to(device)

    criterion, pos_weight_value = (
        create_criterion(
            outer_train_df,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
    )

    history = []

    for epoch in range(
        1,
        selected_epoch + 1,
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

        scheduler.step(
            train_loss
        )

        history.append(
            {
                "outer_fold":
                    outer_fold,

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_auc":
                    train_metrics["auc"],

                "learning_rate":
                    optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{selected_epoch} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train AUC {train_metrics['auc']:.4f}"
        )

    (
        test_loss,
        test_metrics,
        test_labels,
        test_probabilities,
        test_subjects,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print(
        f"OUTER FOLD {outer_fold} TEST"
    )

    print(
        f"AUC                 : "
        f"{test_metrics['auc']:.6f}"
    )

    print(
        f"Accuracy            : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{test_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Sensitivity         : "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"F1                  : "
        f"{test_metrics['f1']:.6f}"
    )

    source_lookup = (
        outer_test_df
        .set_index(
            "subject_id"
        )[
            "acquisition_source"
        ]
    )

    source_values = [
        source_lookup.loc[
            subject_id
        ]
        for subject_id in test_subjects
    ]

    prediction_df = pd.DataFrame(
        {
            "subject_id":
                test_subjects,

            "label":
                test_labels,

            "probability":
                test_probabilities,

            "prediction":
                (
                    np.asarray(
                        test_probabilities
                    )
                    >= THRESHOLD
                ).astype(int),

            "outer_fold":
                outer_fold,

            "selected_epoch":
                selected_epoch,

            "acquisition_source":
                source_values,
        }
    )

    if prediction_df[
        "subject_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "duplicate test subjects."
        )

    fold_dir = (
        TRAINING_DIR
        / f"outer_fold_{outer_fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "experiment":
                "exp01_groupnorm",

            "model_type":
                "SMRI3DCNNGroupNorm",

            "outer_fold":
                outer_fold,

            "selected_epoch":
                selected_epoch,

            "random_seed":
                final_seed,

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

            "test_auc":
                test_metrics["auc"],
        },

        fold_dir
        / "final_outer_model.pt",
    )

    pd.DataFrame(
        history
    ).to_csv(
        fold_dir
        / "final_outer_training_history.csv",
        index=False,
    )

    prediction_df.to_csv(
        fold_dir
        / "outer_test_predictions.csv",
        index=False,
    )

    fold_metrics = {
        "outer_fold":
            outer_fold,

        "outer_train_subjects":
            len(outer_train_df),

        "outer_test_subjects":
            len(outer_test_df),

        "selected_epoch":
            selected_epoch,

        "test_loss":
            test_loss,

        "test_auc":
            test_metrics["auc"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],

        "test_precision":
            test_metrics["precision"],

        "test_recall":
            test_metrics["recall"],

        "test_f1":
            test_metrics["f1"],

        "pos_weight":
            pos_weight_value,

        "random_seed":
            final_seed,
    }

    return (
        prediction_df,
        fold_metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("NeuroFusion AI - EXPERIMENT 01")
    print("sMRI 3D CNN - GROUPNORM")
    print("FULL 5x4 NESTED CROSS-VALIDATION")
    print("=" * 72)

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load nested assignments.
    # --------------------------------------------------------

    if not NESTED_ASSIGNMENTS_FILE.exists():

        raise FileNotFoundError(
            f"Nested assignments not found: "
            f"{NESTED_ASSIGNMENTS_FILE}"
        )

    assignments_df = pd.read_csv(
        NESTED_ASSIGNMENTS_FILE
    )

    print(
        f"Assignment rows: "
        f"{len(assignments_df)}"
    )

    # --------------------------------------------------------
    # Validate expected assignment size.
    # --------------------------------------------------------

    expected_rows = (
        EXPECTED_SUBJECTS
        * OUTER_FOLDS
        * INNER_FOLDS
    )

    if len(assignments_df) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} nested assignment rows; "
            f"found {len(assignments_df)}."
        )

    # --------------------------------------------------------
    # Validate required columns.
    # --------------------------------------------------------

    required_columns = {
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "outer_fold",
        "inner_fold",
        "role",
    }

    missing = (
        required_columns
        - set(assignments_df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing nested assignment columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Basic assignment validation.
    # --------------------------------------------------------

    if assignments_df["label"].isna().any():

        raise RuntimeError(
            "Nested assignments contain missing labels."
        )

    if not assignments_df[
        "label"
    ].isin([0, 1]).all():

        raise RuntimeError(
            "Nested assignments contain invalid labels."
        )

    unique_subjects = set(
        assignments_df[
            "subject_id"
        ].astype(str)
    )

    if len(unique_subjects) != EXPECTED_SUBJECTS:

        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} unique subjects; "
            f"found {len(unique_subjects)}."
        )

    # --------------------------------------------------------
    # Full nested assignment validation.
    # --------------------------------------------------------

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        outer_df = assignments_df[
            assignments_df["outer_fold"]
            == outer_fold
        ].copy()

        if outer_df.empty:

            raise RuntimeError(
                f"Outer fold {outer_fold} is empty."
            )

        outer_test_all = outer_df[
            outer_df["role"]
            == "OUTER_TEST"
        ]

        outer_test_counts = (
            outer_test_all
            .groupby(
                "subject_id"
            )
            .size()
        )

        if len(outer_test_counts) == 0:

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "no outer test subjects."
            )

        if not (
            outer_test_counts
            == INNER_FOLDS
        ).all():

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "outer test subjects do not repeat "
                f"{INNER_FOLDS} times."
            )

        for inner_fold in range(
            1,
            INNER_FOLDS + 1,
        ):

            split_df = outer_df[
                outer_df["inner_fold"]
                == inner_fold
            ].copy()

            train_ids = set(
                split_df.loc[
                    split_df["role"]
                    == "INNER_TRAIN",
                    "subject_id",
                ].astype(str)
            )

            validation_ids = set(
                split_df.loc[
                    split_df["role"]
                    == "INNER_VALIDATION",
                    "subject_id",
                ].astype(str)
            )

            test_ids = set(
                split_df.loc[
                    split_df["role"]
                    == "OUTER_TEST",
                    "subject_id",
                ].astype(str)
            )

            if train_ids.intersection(
                validation_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"Inner {inner_fold}: "
                    "train/validation overlap."
                )

            if train_ids.intersection(
                test_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"Inner {inner_fold}: "
                    "train/test overlap."
                )

            if validation_ids.intersection(
                test_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"Inner {inner_fold}: "
                    "validation/test overlap."
                )

            if set(
                split_df["subject_id"]
                .astype(str)
            ) != (
                train_ids
                | validation_ids
                | test_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, "
                    f"Inner {inner_fold}: "
                    "invalid partition."
                )

            for role in [
                "INNER_TRAIN",
                "INNER_VALIDATION",
                "OUTER_TEST",
            ]:

                role_labels = (
                    split_df.loc[
                        split_df["role"]
                        == role,
                        "label",
                    ]
                    .astype(int)
                )

                if set(role_labels) != {0, 1}:

                    raise RuntimeError(
                        f"Outer {outer_fold}, "
                        f"Inner {inner_fold}: "
                        f"{role} does not contain both classes."
                    )

    print(
        "Nested assignment validation: PASS"
    )

    # --------------------------------------------------------
    # Run all 4 inner folds for every outer fold.
    # --------------------------------------------------------

    all_inner_results = []

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        outer_df = assignments_df[
            assignments_df["outer_fold"]
            == outer_fold
        ].copy()

        for inner_fold in range(
            1,
            INNER_FOLDS + 1,
        ):

            result, history_df = (
                train_inner_fold(
                    outer_fold,
                    inner_fold,
                    outer_df,
                    device,
                )
            )

            all_inner_results.append(
                result
            )

            # Save every inner history separately so this
            # experiment remains fully auditable.
            inner_dir = (
                TRAINING_DIR
                / f"outer_fold_{outer_fold}"
                / f"inner_fold_{inner_fold}"
            )

            inner_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            history_df.to_csv(
                inner_dir
                / "inner_training_history.csv",
                index=False,
            )

    # --------------------------------------------------------
    # Save inner results.
    # --------------------------------------------------------

    inner_results_df = pd.DataFrame(
        all_inner_results
    )

    if len(inner_results_df) != (
        OUTER_FOLDS * INNER_FOLDS
    ):

        raise RuntimeError(
            "Expected exactly 20 inner-CV results."
        )

    inner_results_df.to_csv(
        INNER_RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Select epoch for every outer fold.
    # --------------------------------------------------------

    epoch_selection_rows = []

    selected_epochs = {}

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        fold_results = [
            result
            for result in all_inner_results
            if result[
                "outer_fold"
            ] == outer_fold
        ]

        best_epochs = [
            result[
                "best_epoch"
            ]
            for result in fold_results
        ]

        best_aucs = [
            result[
                "best_validation_auc"
            ]
            for result in fold_results
        ]

        selected_epoch = select_epoch(
            all_inner_results,
            outer_fold,
        )

        selected_epochs[
            outer_fold
        ] = selected_epoch

        epoch_selection_rows.append(
            {
                "outer_fold":
                    outer_fold,

                "inner_best_epochs":
                    ",".join(
                        map(
                            str,
                            best_epochs,
                        )
                    ),

                "inner_best_auc":
                    ",".join(
                        f"{value:.6f}"
                        for value in best_aucs
                    ),

                "selected_epoch":
                    selected_epoch,
            }
        )

        print()
        print(
            f"Outer fold {outer_fold}: "
            f"inner epochs = {best_epochs} "
            f"-> selected epoch = {selected_epoch}"
        )

    epoch_selection_df = pd.DataFrame(
        epoch_selection_rows
    )

    epoch_selection_df.to_csv(
        EPOCH_SELECTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final outer models.
    # --------------------------------------------------------

    all_predictions = []

    all_fold_metrics = []

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        outer_df = assignments_df[
            assignments_df["outer_fold"]
            == outer_fold
        ].copy()

        prediction_df, fold_metrics = (
            train_final_outer_model(
                outer_fold,
                outer_df,
                selected_epochs[
                    outer_fold
                ],
                device,
            )
        )

        all_predictions.append(
            prediction_df
        )

        all_fold_metrics.append(
            fold_metrics
        )

    # --------------------------------------------------------
    # Combine OOF predictions.
    # --------------------------------------------------------

    oof_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    if len(oof_df) != EXPECTED_SUBJECTS:

        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} OOF predictions; "
            f"found {len(oof_df)}."
        )

    if oof_df[
        "subject_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate subjects detected in OOF predictions."
        )

    if set(
        oof_df[
            "subject_id"
        ].astype(str)
    ) != unique_subjects:

        raise RuntimeError(
            "OOF predictions do not cover exactly the "
            "expected subjects."
        )

    # --------------------------------------------------------
    # OOF metrics.
    # --------------------------------------------------------

    oof_metrics = calculate_metrics(
        oof_df["label"].tolist(),
        oof_df["probability"].tolist(),
    )

    fold_metrics_df = pd.DataFrame(
        all_fold_metrics
    )

    oof_df.to_csv(
        OOF_PREDICTIONS_FILE,
        index=False,
    )

    fold_metrics_df.to_csv(
        FOLD_METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    fold_auc_mean = (
        fold_metrics_df[
            "test_auc"
        ]
        .mean()
    )

    fold_auc_std = (
        fold_metrics_df[
            "test_auc"
        ]
        .std(
            ddof=0
        )
    )

    summary_lines = [
        "NeuroFusion AI",
        "Experiment 01 - GroupNorm sMRI 3D CNN",
        "Full 5x4 Nested Cross-Validation",
        "",
        f"Device: {device}",
        f"Subjects evaluated: {len(oof_df)}",
        f"Controls: {(oof_df['label'] == 0).sum()}",
        f"Schizophrenia: {(oof_df['label'] == 1).sum()}",
        "",
        "MODEL",
        "SMRI3DCNNGroupNorm",
        "GroupNorm instead of BatchNorm3d",
        f"Dropout: {DROPOUT}",
        f"Batch size: {BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        "",
        "NESTED CV",
        f"Outer folds: {OUTER_FOLDS}",
        f"Inner folds: {INNER_FOLDS}",
        f"Total inner runs: {len(inner_results_df)}",
        "",
        "OUTER FOLD AUC",
        f"Mean: {fold_auc_mean:.6f}",
        f"Population SD: {fold_auc_std:.6f}",
        "",
        "OOF RESULTS",
        f"ROC-AUC: {oof_metrics['auc']:.6f}",
        f"Accuracy: {oof_metrics['accuracy']:.6f}",
        f"Balanced Accuracy: {oof_metrics['balanced_accuracy']:.6f}",
        f"Precision: {oof_metrics['precision']:.6f}",
        f"Sensitivity: {oof_metrics['recall']:.6f}",
        f"F1: {oof_metrics['f1']:.6f}",
        "",
        "STATUS",
        "PASS",
    ]

    CV_SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EXPERIMENT 01 COMPLETE")
    print("=" * 72)

    print(
        f"Subjects evaluated: "
        f"{len(oof_df)}"
    )

    print(
        f"OOF ROC-AUC: "
        f"{oof_metrics['auc']:.6f}"
    )

    print(
        f"Fold AUC mean: "
        f"{fold_auc_mean:.6f}"
    )

    print(
        f"Fold AUC population SD: "
        f"{fold_auc_std:.6f}"
    )

    print(
        "FINAL STATUS: PASS"
    )


if __name__ == "__main__":

    main()