"""
NeuroFusion AI
Experiment 02 - 3D sMRI CNN + Training-Only Spatial Augmentation

Controlled experiment:
    Baseline BatchNorm3D CNN
    + mild spatial augmentation during TRAINING ONLY.

Augmentation:
    - Random left-right flip, probability 0.5
    - Random 3D rotation, +/- 5 degrees
    - Random 3D translation, +/- 3 voxels

Validation and outer-test images are NEVER augmented.

Nested CV:
    5 outer folds
    4 inner folds

The validated nested-CV assignment file is NOT modified.

The baseline training script is NOT modified.
"""

from pathlib import Path
import sys
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


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SRC_DIR = PROJECT_ROOT / "src"

EXP_DIR = (
    PROJECT_ROOT
    / "src"
    / "exp02_augmentation"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )

if str(EXP_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(EXP_DIR)
    )


from smri_3d_cnn import SMRI3DCNN

from smri_augmentation import (
    apply_training_augmentation,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_NAME = (
    "exp02_augmentation"
)

RANDOM_SEED = 42

OUTER_FOLDS = 5

INNER_FOLDS = 4

BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

EARLY_STOPPING_PATIENCE = 10

THRESHOLD = 0.50

NUM_WORKERS = 0

EXPECTED_TENSOR_SHAPE = (
    1,
    95,
    117,
    99,
)


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "images"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

NESTED_ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "cross_validation"
    / "nested"
    / "nested_cv_assignments.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / EXPERIMENT_NAME
    / "training_full_5x4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SMRIDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        augment=False,
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.image_dir = Path(
            image_dir
        )

        self.augment = bool(
            augment
        )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.dataframe.iloc[
            index
        ]

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
                f"Tensor not found: "
                f"{image_path}"
            )

        image = np.load(
            image_path
        ).astype(
            np.float32,
            copy=False,
        )

        if tuple(
            image.shape
        ) != (
            EXPECTED_TENSOR_SHAPE
        ):

            raise RuntimeError(
                f"Unexpected tensor shape "
                f"for {subject_id}: "
                f"{image.shape}; "
                f"expected "
                f"{EXPECTED_TENSOR_SHAPE}"
            )

        # ----------------------------------------------------
        # TRAINING-ONLY AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            image = (
                apply_training_augmentation(
                    image
                )
            )

        # ----------------------------------------------------
        # Validate augmented image
        # ----------------------------------------------------

        if tuple(
            image.shape
        ) != (
            EXPECTED_TENSOR_SHAPE
        ):

            raise RuntimeError(
                f"Augmentation changed "
                f"shape for {subject_id}: "
                f"{image.shape}"
            )

        if not np.isfinite(
            image
        ).all():

            raise RuntimeError(
                f"Non-finite values for "
                f"{subject_id}"
            )

        if image.min() < 0.0:

            raise RuntimeError(
                f"Negative intensity for "
                f"{subject_id}: "
                f"{image.min()}"
            )

        if image.max() > 1.0:

            raise RuntimeError(
                f"Intensity > 1 for "
                f"{subject_id}: "
                f"{image.max()}"
            )

        image_tensor = (
            torch.from_numpy(
                image
            )
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
    ).astype(
        np.int64
    )

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

        auc = float(
            "nan"
        )

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
# LOSS
# ============================================================

def create_criterion(
    dataframe,
    device,
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
            "Training partition does not "
            "contain both classes."
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

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
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

    for (
        images,
        labels,
        _,
    ) in loader:

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

        subject_count += (
            batch_size
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
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

        for (
            images,
            labels,
            subjects,
        ) in loader:

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

            subject_count += (
                batch_size
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
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
    augment=False,
):

    dataset = SMRIDataset(
        dataframe,
        IMAGE_DIR,
        augment=augment,
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
    outer_fold,
    inner_fold,
    outer_df,
    device,
):

    print()
    print(
        "-" * 72
    )

    print(
        f"OUTER {outer_fold} | "
        f"INNER {inner_fold}"
    )

    print(
        "-" * 72
    )

    split_df = outer_df[
        outer_df[
            "inner_fold"
        ].astype(int)
        == inner_fold
    ].copy()

    train_df = split_df[
        split_df[
            "role"
        ]
        == "INNER_TRAIN"
    ].copy()

    validation_df = split_df[
        split_df[
            "role"
        ]
        == "INNER_VALIDATION"
    ].copy()

    test_df = split_df[
        split_df[
            "role"
        ]
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

    # --------------------------------------------------------
    # Partition validation
    # --------------------------------------------------------

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
            "train/outer-test overlap."
        )

    if validation_ids.intersection(
        test_ids
    ):

        raise RuntimeError(
            f"Outer {outer_fold}, "
            f"Inner {inner_fold}: "
            "validation/outer-test overlap."
        )

    union_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    if (
        len(train_ids)
        + len(validation_ids)
        + len(test_ids)
        != len(union_ids)
    ):

        raise RuntimeError(
            "Partition union is invalid."
        )

    # --------------------------------------------------------
    # Confirm complete outer partition
    # --------------------------------------------------------

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
            f"Outer {outer_fold}, "
            f"Inner {inner_fold}: "
            "partition does not cover "
            "the outer fold."
        )

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    for name, dataframe in [
        (
            "inner train",
            train_df,
        ),
        (
            "inner validation",
            validation_df,
        ),
        (
            "outer test",
            test_df,
        ),
    ]:

        classes = set(
            dataframe[
                "label"
            ].astype(int)
        )

        if classes != {
            0,
            1,
        }:

            raise RuntimeError(
                f"Outer {outer_fold}, "
                f"Inner {inner_fold}: "
                f"{name} does not contain "
                "both classes."
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

    # --------------------------------------------------------
    # Unique inner-run seed
    # --------------------------------------------------------

    run_seed = (
        RANDOM_SEED
        + outer_fold * 100
        + inner_fold
    )

    set_seed(
        run_seed
    )

    # --------------------------------------------------------
    # Training data:
    # AUGMENTATION ON
    #
    # Validation:
    # AUGMENTATION OFF
    # --------------------------------------------------------

    train_loader = create_loader(
        train_df,
        shuffle=True,
        augment=True,
    )

    validation_loader = create_loader(
        validation_df,
        shuffle=False,
        augment=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SMRI3DCNN(
        dropout=DROPOUT
    ).to(
        device
    )

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

    # --------------------------------------------------------
    # Train inner model
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        (
            train_loss,
            train_metrics,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
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
            validation_metrics[
                "auc"
            ]
        )

        scheduler.step(
            validation_auc
        )

        current_lr = (
            optimizer
            .param_groups[0]
            ["lr"]
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
                    train_metrics[
                        "auc"
                    ],

                "validation_loss":
                    validation_loss,

                "validation_auc":
                    validation_auc,
            }
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{NUM_EPOCHS} | "
            f"Train AUC "
            f"{train_metrics['auc']:.4f} | "
            f"Val AUC "
            f"{validation_auc:.4f}"
        )

        if validation_auc > best_auc:

            best_auc = (
                validation_auc
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break

    if best_epoch <= 0:

        raise RuntimeError(
            f"Outer {outer_fold}, "
            f"Inner {inner_fold}: "
            "no valid best epoch."
        )

    print(
        f"Best inner validation AUC: "
        f"{best_auc:.6f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    result = {
        "outer_fold":
            outer_fold,

        "inner_fold":
            inner_fold,

        "best_epoch":
            best_epoch,

        "best_validation_auc":
            float(
                best_auc
            ),

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
            float(
                pos_weight_value
            ),

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
# SELECT EPOCH FROM FOUR INNER FOLDS
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

    if len(
        fold_results
    ) != INNER_FOLDS:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"expected {INNER_FOLDS} "
            f"inner results, found "
            f"{len(fold_results)}."
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
    print(
        "=" * 72
    )

    print(
        f"FINAL OUTER MODEL "
        f"{outer_fold}"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Correctly identify OUTER TEST
    # --------------------------------------------------------

    outer_test_df = (
        outer_df[
            outer_df[
                "role"
            ]
            == "OUTER_TEST"
        ]
        .drop_duplicates(
            subset=[
                "subject_id"
            ]
        )
        .copy()
    )

    outer_train_df = (
        outer_df[
            outer_df[
                "role"
            ]
            != "OUTER_TEST"
        ]
        .drop_duplicates(
            subset=[
                "subject_id"
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Validate partition
    # --------------------------------------------------------

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

    expected_ids = set(
        outer_df[
            "subject_id"
        ].astype(str)
    )

    if (
        train_ids
        | test_ids
    ) != expected_ids:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final train/test partition "
            "does not cover all subjects."
        )

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    if set(
        outer_train_df[
            "label"
        ].astype(int)
    ) != {
        0,
        1,
    }:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final outer training does "
            "not contain both classes."
        )

    if set(
        outer_test_df[
            "label"
        ].astype(int)
    ) != {
        0,
        1,
    }:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "outer test does not contain "
            "both classes."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # outer_df contains 172 subjects x 4 inner folds.
    #
    # Therefore len(outer_df) is NOT the outer fold size.
    #
    # outer_train_df and outer_test_df are the actual
    # subject-level partitions.
    # --------------------------------------------------------

    if len(
        outer_test_df
    ) not in (
        34,
        35,
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"unexpected outer test size: "
            f"{len(outer_test_df)}"
        )

    if len(
        outer_train_df
    ) not in (
        137,
        138,
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"unexpected outer training size: "
            f"{len(outer_train_df)}"
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

    # --------------------------------------------------------
    # Final seed
    # --------------------------------------------------------

    final_seed = (
        RANDOM_SEED
        + outer_fold * 1000
    )

    set_seed(
        final_seed
    )

    # --------------------------------------------------------
    # DATA
    #
    # Final outer training = augmentation ON
    # Outer test = augmentation OFF
    # --------------------------------------------------------

    train_loader = create_loader(
        outer_train_df,
        shuffle=True,
        augment=True,
    )

    test_loader = create_loader(
        outer_test_df,
        shuffle=False,
        augment=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SMRI3DCNN(
        dropout=DROPOUT
    ).to(
        device
    )

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

    # --------------------------------------------------------
    # Final outer training
    # --------------------------------------------------------

    for epoch in range(
        1,
        selected_epoch + 1,
    ):

        (
            train_loss,
            train_metrics,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
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
                    train_metrics[
                        "auc"
                    ],

                "learning_rate":
                    optimizer
                    .param_groups[0]
                    ["lr"],
            }
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{selected_epoch} | "
            f"Train Loss "
            f"{train_loss:.4f} | "
            f"Train AUC "
            f"{train_metrics['auc']:.4f}"
        )

    # --------------------------------------------------------
    # Outer test evaluation
    # --------------------------------------------------------

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
        f"OUTER FOLD "
        f"{outer_fold} TEST"
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

    # --------------------------------------------------------
    # Source information
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

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
                ).astype(
                    int
                ),

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

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    fold_dir = (
        OUTPUT_DIR
        / f"outer_fold_{outer_fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "outer_fold":
                outer_fold,

            "selected_epoch":
                selected_epoch,

            "dropout":
                DROPOUT,

            "input_shape":
                EXPECTED_TENSOR_SHAPE,

            "random_seed":
                final_seed,

            "pos_weight":
                pos_weight_value,

            "experiment":
                EXPERIMENT_NAME,

            "augmentation":
                {
                    "random_flip_probability":
                        0.5,

                    "rotation_degrees":
                        5.0,

                    "translation_voxels":
                        3,
                },
        },
        fold_dir
        / "final_outer_model.pt",
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        fold_dir
        / "final_outer_training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df.to_csv(
        fold_dir
        / "outer_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "outer_fold":
                    outer_fold,

                "test_subjects":
                    len(
                        test_subjects
                    ),

                "test_auc":
                    test_metrics[
                        "auc"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "test_balanced_accuracy":
                    test_metrics[
                        "balanced_accuracy"
                    ],

                "test_precision":
                    test_metrics[
                        "precision"
                    ],

                "test_sensitivity":
                    test_metrics[
                        "recall"
                    ],

                "test_f1":
                    test_metrics[
                        "f1"
                    ],

                "selected_epoch":
                    selected_epoch,

                "pos_weight":
                    pos_weight_value,

                "final_seed":
                    final_seed,
            }
        ]
    )

    metrics_df.to_csv(
        fold_dir
        / "outer_test_metrics.csv",
        index=False,
    )

    return (
        test_metrics,
        prediction_df,
    )


# ============================================================
# LOAD AND VALIDATE ASSIGNMENTS
# ============================================================

def load_assignments():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found: "
            f"{MANIFEST_PATH}"
        )

    if not NESTED_ASSIGNMENT_PATH.exists():

        raise FileNotFoundError(
            f"Nested assignment not found: "
            f"{NESTED_ASSIGNMENT_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    assignments = pd.read_csv(
        NESTED_ASSIGNMENT_PATH
    )

    manifest["subject_id"] = (
        manifest[
            "subject_id"
        ].astype(str)
    )

    assignments["subject_id"] = (
        assignments[
            "subject_id"
        ].astype(str)
    )

    # --------------------------------------------------------
    # Manifest uniqueness
    # --------------------------------------------------------

    if manifest[
        "subject_id"
    ].duplicated().any():

        raise RuntimeError(
            "Manifest contains duplicate "
            "subject IDs."
        )

    manifest_subjects = set(
        manifest[
            "subject_id"
        ]
    )

    # --------------------------------------------------------
    # Assignment row count
    # --------------------------------------------------------

    expected_rows = (
        len(manifest)
        * OUTER_FOLDS
        * INNER_FOLDS
    )

    if len(assignments) != (
        expected_rows
    ):

        raise RuntimeError(
            f"Expected {expected_rows} "
            f"nested assignment rows, "
            f"found {len(assignments)}."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "subject_id",
        "outer_fold",
        "inner_fold",
        "role",
    }

    missing_columns = (
        required_columns
        - set(assignments.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Nested assignment is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Assignment subjects
    # --------------------------------------------------------

    assignment_subjects = set(
        assignments[
            "subject_id"
        ]
    )

    if assignment_subjects != (
        manifest_subjects
    ):

        raise RuntimeError(
            "Assignment subjects do not "
            "match labeled manifest."
        )

    # --------------------------------------------------------
    # Outer fold values
    # --------------------------------------------------------

    actual_outer_folds = set(
        assignments[
            "outer_fold"
        ].astype(int)
    )

    expected_outer_folds = set(
        range(
            1,
            OUTER_FOLDS + 1,
        )
    )

    if actual_outer_folds != (
        expected_outer_folds
    ):

        raise RuntimeError(
            "Invalid outer fold values: "
            f"{sorted(actual_outer_folds)}"
        )

    # --------------------------------------------------------
    # Inner fold values
    # --------------------------------------------------------

    actual_inner_folds = set(
        assignments[
            "inner_fold"
        ].astype(int)
    )

    expected_inner_folds = set(
        range(
            1,
            INNER_FOLDS + 1,
        )
    )

    if actual_inner_folds != (
        expected_inner_folds
    ):

        raise RuntimeError(
            "Invalid inner fold values: "
            f"{sorted(actual_inner_folds)}"
        )

    # --------------------------------------------------------
    # Duplicate nested assignments
    # --------------------------------------------------------

    duplicate_count = (
        assignments[
            [
                "subject_id",
                "outer_fold",
                "inner_fold",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count:

        raise RuntimeError(
            "Duplicate subject/outer-fold/"
            "inner-fold assignments: "
            f"{duplicate_count}"
        )

    # --------------------------------------------------------
    # Role validation
    # --------------------------------------------------------

    valid_roles = {
        "INNER_TRAIN",
        "INNER_VALIDATION",
        "OUTER_TEST",
    }

    actual_roles = set(
        assignments[
            "role"
        ].astype(str)
    )

    if not actual_roles.issubset(
        valid_roles
    ):

        raise RuntimeError(
            "Unexpected assignment roles: "
            f"{sorted(actual_roles - valid_roles)}"
        )

    # --------------------------------------------------------
    # Every outer fold must cover all 172 subjects
    #
    # This is the critical correction.
    #
    # The assignment file contains:
    #
    #   172 subjects
    #   x 5 outer folds
    #   x 4 inner folds
    #
    # For a selected outer fold, outer_df therefore has
    # 172 x 4 = 688 rows.
    #
    # The actual outer-test size is obtained from role.
    # --------------------------------------------------------

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        outer_df = assignments[
            assignments[
                "outer_fold"
            ].astype(int)
            == outer_fold
        ]

        outer_subjects = set(
            outer_df[
                "subject_id"
            ]
        )

        if outer_subjects != (
            manifest_subjects
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "does not cover all "
                "manifest subjects."
            )

        # ----------------------------------------------------
        # Each subject must occur exactly once per inner fold
        # within the selected outer fold.
        # ----------------------------------------------------

        counts = (
            outer_df[
                [
                    "subject_id",
                    "inner_fold",
                ]
            ]
            .drop_duplicates()
            .groupby(
                "subject_id"
            )[
                "inner_fold"
            ]
            .nunique()
        )

        if len(counts) != (
            len(manifest)
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "incorrect subject coverage "
                "across inner folds."
            )

        if not (
            counts
            == INNER_FOLDS
        ).all():

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "each subject must appear "
                f"exactly {INNER_FOLDS} "
                "times across inner folds."
            )

        # ----------------------------------------------------
        # OUTER TEST SUBJECTS
        #
        # The same outer-test subjects are repeated across
        # all four inner-fold definitions.
        #
        # Therefore each OUTER_TEST subject legitimately has
        # INNER_FOLDS rows.
        # ----------------------------------------------------

        outer_test = outer_df[
            outer_df[
                "role"
            ]
            == "OUTER_TEST"
        ]

        outer_test_subjects = set(
            outer_test[
                "subject_id"
            ]
        )

        if len(
            outer_test_subjects
        ) not in (
            34,
            35,
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"unexpected outer-test "
                f"size "
                f"{len(outer_test_subjects)}"
            )

        # ----------------------------------------------------
        # Each outer-test subject must appear once for every
        # inner-fold definition.
        #
        # Example:
        #     35 test subjects x 4 inner folds = 140 rows
        # ----------------------------------------------------

        test_counts = (
            outer_test[
                [
                    "subject_id",
                    "inner_fold",
                ]
            ]
            .drop_duplicates()
            .groupby(
                "subject_id"
            )[
                "inner_fold"
            ]
            .nunique()
        )

        if not (
            test_counts
            == INNER_FOLDS
        ).all():

            invalid_subjects = (
                test_counts[
                    test_counts
                    != INNER_FOLDS
                ]
                .index
                .tolist()
            )

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "OUTER_TEST subjects do not "
                f"appear exactly {INNER_FOLDS} "
                "times across inner folds. "
                f"Invalid subjects: "
                f"{invalid_subjects[:10]}"
            )

        # ----------------------------------------------------
        # Inner train/validation assignments must exclude
        # the outer-test subjects.
        # ----------------------------------------------------

        inner_rows = outer_df[
            outer_df[
                "role"
            ]
            != "OUTER_TEST"
        ]

        overlap = set(
            inner_rows[
                "subject_id"
            ]
        ).intersection(
            outer_test_subjects
        )

        if overlap:

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                "outer-test subjects appear "
                "in inner partitions."
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print(
        f"Assignment rows: "
        f"{len(assignments)}"
    )

    print(
        f"Subjects: "
        f"{len(manifest)}"
    )

    print(
        f"Outer folds: "
        f"{OUTER_FOLDS}"
    )

    print(
        f"Inner folds: "
        f"{INNER_FOLDS}"
    )

    print(
        "Nested assignment "
        "validation: PASS"
    )

    return (
        manifest,
        assignments,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "NeuroFusion AI - "
        "EXPERIMENT 02"
    )

    print(
        "3D sMRI CNN + "
        "Training-Only Spatial Augmentation"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print()

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Load assignments
    # --------------------------------------------------------

    (
        manifest,
        assignments,
    ) = load_assignments()

    # --------------------------------------------------------
    # Authoritative manifest metadata
    #
    # Remove any assignment-side copies first.
    # --------------------------------------------------------

    manifest_subset = (
        manifest[
            [
                "subject_id",
                "label",
                "acquisition_source",
            ]
        ]
        .drop_duplicates(
            subset=[
                "subject_id"
            ]
        )
        .copy()
    )

    assignments = assignments.drop(
        columns=[
            "label",
            "acquisition_source",
        ],
        errors="ignore",
    )

    assignments = assignments.merge(
        manifest_subset,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Metadata validation
    # --------------------------------------------------------

    if "label" not in (
        assignments.columns
    ):

        raise RuntimeError(
            "Label column was not created "
            "after manifest merge."
        )

    if "acquisition_source" not in (
        assignments.columns
    ):

        raise RuntimeError(
            "Acquisition source column was "
            "not created after manifest merge."
        )

    if assignments[
        "label"
    ].isna().any():

        raise RuntimeError(
            "Missing labels after "
            "assignment merge."
        )

    if assignments[
        "acquisition_source"
    ].isna().any():

        raise RuntimeError(
            "Missing acquisition source "
            "after assignment merge."
        )

    # --------------------------------------------------------
    # Image validation
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found: "
            f"{IMAGE_DIR}"
        )

    missing_images = []

    for subject_id in manifest[
        "subject_id"
    ].astype(str):

        image_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not image_path.exists():

            missing_images.append(
                subject_id
            )

    if missing_images:

        raise RuntimeError(
            f"Missing CNN tensors: "
            f"{missing_images[:10]}"
        )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print(
        f"Subjects: "
        f"{manifest['subject_id'].nunique()}"
    )

    print(
        f"Controls: "
        f"{(
            manifest['label']
            .astype(int)
            == 0
        ).sum()}"
    )

    print(
        f"Schizophrenia: "
        f"{(
            manifest['label']
            .astype(int)
            == 1
        ).sum()}"
    )

    print()

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    config_text = f"""
NeuroFusion AI
Experiment: {EXPERIMENT_NAME}

Architecture:
    Baseline BatchNorm3D SMRI3DCNN

Input:
    {EXPECTED_TENSOR_SHAPE}

Nested CV:
    Outer folds: {OUTER_FOLDS}
    Inner folds: {INNER_FOLDS}

Training:
    Batch size: {BATCH_SIZE}
    Maximum epochs: {NUM_EPOCHS}
    Learning rate: {LEARNING_RATE}
    Weight decay: {WEIGHT_DECAY}
    Dropout: {DROPOUT}
    Early stopping patience:
        {EARLY_STOPPING_PATIENCE}

Augmentation:
    Training only: YES
    Validation augmentation: NO
    Outer-test augmentation: NO
    Random left-right flip: p=0.5
    Random rotation: +/-5 degrees
    Random translation: +/-3 voxels

Intensity:
    Existing [0,1] normalized inputs.
    No intensity augmentation.

Random seed:
    {RANDOM_SEED}
"""

    (
        OUTPUT_DIR
        / "experiment_config.txt"
    ).write_text(
        config_text.strip()
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # INNER CV
    # ========================================================

    all_inner_results = []

    all_inner_histories = []

    epoch_records = []

    outer_prediction_frames = []

    outer_metric_frames = []

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        print()
        print(
            "#" * 72
        )

        print(
            f"OUTER FOLD {outer_fold}"
        )

        print(
            "#" * 72
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This contains all 172 subjects for the selected
        # outer fold, repeated across 4 inner definitions.
        #
        # It is therefore 688 rows, not 34/35 rows.
        # ----------------------------------------------------

        outer_df = (
            assignments[
                assignments[
                    "outer_fold"
                ].astype(int)
                == outer_fold
            ]
            .copy()
        )

        outer_subject_count = (
            outer_df[
                "subject_id"
            ]
            .nunique()
        )

        if outer_subject_count != (
            len(manifest)
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"expected "
                f"{len(manifest)} "
                f"unique subjects, found "
                f"{outer_subject_count}."
            )

        actual_outer_test_count = (
            outer_df[
                outer_df[
                    "role"
                ]
                == "OUTER_TEST"
            ][
                "subject_id"
            ]
            .nunique()
        )

        if actual_outer_test_count not in (
            34,
            35,
        ):

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"unexpected outer-test "
                f"size "
                f"{actual_outer_test_count}."
            )

        print(
            f"Outer definition rows: "
            f"{len(outer_df)}"
        )

        print(
            f"Outer subjects: "
            f"{outer_subject_count}"
        )

        print(
            f"Outer test subjects: "
            f"{actual_outer_test_count}"
        )

        outer_inner_results = []

        # ----------------------------------------------------
        # FOUR INNER FOLDS
        # ----------------------------------------------------

        for inner_fold in range(
            1,
            INNER_FOLDS + 1,
        ):

            (
                result,
                history_df,
            ) = train_inner_fold(
                outer_fold,
                inner_fold,
                outer_df,
                DEVICE,
            )

            outer_inner_results.append(
                result
            )

            all_inner_results.append(
                result
            )

            all_inner_histories.append(
                history_df
            )

        # ----------------------------------------------------
        # Epoch selection
        # ----------------------------------------------------

        selected_epoch = select_epoch(
            outer_inner_results,
            outer_fold,
        )

        print()
        print(
            f"Outer {outer_fold} "
            f"selected epoch: "
            f"{selected_epoch}"
        )

        epoch_records.append(
            {
                "outer_fold":
                    outer_fold,

                "inner_best_epochs":
                    ",".join(
                        str(
                            result[
                                "best_epoch"
                            ]
                        )
                        for result
                        in outer_inner_results
                    ),

                "inner_best_auc":
                    ",".join(
                        f"{result['best_validation_auc']:.6f}"
                        for result
                        in outer_inner_results
                    ),

                "selected_epoch":
                    selected_epoch,
            }
        )

        # ----------------------------------------------------
        # Final outer model
        # ----------------------------------------------------

        (
            test_metrics,
            prediction_df,
        ) = train_final_outer_model(
            outer_fold,
            outer_df,
            selected_epoch,
            DEVICE,
        )

        outer_prediction_frames.append(
            prediction_df
        )

        outer_metric_frames.append(
            pd.DataFrame(
                [
                    {
                        "outer_fold":
                            outer_fold,

                        "test_subjects":
                            len(
                                prediction_df
                            ),

                        "test_auc":
                            test_metrics[
                                "auc"
                            ],

                        "test_accuracy":
                            test_metrics[
                                "accuracy"
                            ],

                        "test_balanced_accuracy":
                            test_metrics[
                                "balanced_accuracy"
                            ],

                        "test_precision":
                            test_metrics[
                                "precision"
                            ],

                        "test_sensitivity":
                            test_metrics[
                                "recall"
                            ],

                        "test_f1":
                            test_metrics[
                                "f1"
                            ],

                        "selected_epoch":
                            selected_epoch,
                    }
                ]
            )
        )

    # ========================================================
    # SAVE INNER RESULTS
    # ========================================================

    inner_results_df = pd.DataFrame(
        all_inner_results
    )

    inner_results_df = (
        inner_results_df
        .sort_values(
            [
                "outer_fold",
                "inner_fold",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    inner_results_df.to_csv(
        OUTPUT_DIR
        / "inner_cv_results.csv",
        index=False,
    )

    # ========================================================
    # SAVE ALL INNER HISTORIES
    # ========================================================

    if all_inner_histories:

        inner_history_df = pd.concat(
            all_inner_histories,
            ignore_index=True,
        )

        inner_history_df.to_csv(
            OUTPUT_DIR
            / "inner_cv_training_histories.csv",
            index=False,
        )

    # ========================================================
    # SAVE EPOCH SELECTION
    # ========================================================

    epoch_selection_df = pd.DataFrame(
        epoch_records
    )

    epoch_selection_df.to_csv(
        OUTPUT_DIR
        / "epoch_selection.csv",
        index=False,
    )

    # ========================================================
    # COLLECT OOF PREDICTIONS
    # ========================================================

    if len(
        outer_prediction_frames
    ) != OUTER_FOLDS:

        raise RuntimeError(
            "Incorrect number of outer "
            "prediction frames."
        )

    oof_predictions = pd.concat(
        outer_prediction_frames,
        ignore_index=True,
    )

    oof_predictions[
        "subject_id"
    ] = (
        oof_predictions[
            "subject_id"
        ].astype(str)
    )

    # --------------------------------------------------------
    # OOF uniqueness
    # --------------------------------------------------------

    if oof_predictions[
        "subject_id"
    ].duplicated().any():

        duplicate_subjects = (
            oof_predictions[
                oof_predictions[
                    "subject_id"
                ].duplicated(
                    keep=False
                )
            ][
                "subject_id"
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate OOF subjects: "
            f"{duplicate_subjects[:10]}"
        )

    # --------------------------------------------------------
    # OOF row count
    # --------------------------------------------------------

    if len(
        oof_predictions
    ) != len(manifest):

        raise RuntimeError(
            f"Expected {len(manifest)} "
            f"OOF predictions, found "
            f"{len(oof_predictions)}."
        )

    # --------------------------------------------------------
    # OOF subject set
    # --------------------------------------------------------

    if set(
        oof_predictions[
            "subject_id"
        ]
    ) != set(
        manifest[
            "subject_id"
        ].astype(str)
    ):

        raise RuntimeError(
            "OOF subjects do not match "
            "the labeled manifest."
        )

    # ========================================================
    # OOF METRICS
    # ========================================================

    oof_metrics = calculate_metrics(
        oof_predictions[
            "label"
        ].to_numpy(),
        oof_predictions[
            "probability"
        ].to_numpy(),
    )

    # --------------------------------------------------------
    # OOF fold count
    # --------------------------------------------------------

    fold_counts = (
        oof_predictions
        .groupby(
            "outer_fold"
        )[
            "subject_id"
        ]
        .nunique()
    )

    if set(
        fold_counts.index.astype(int)
    ) != set(
        range(
            1,
            OUTER_FOLDS + 1,
        )
    ):

        raise RuntimeError(
            "OOF predictions do not contain "
            "all outer folds."
        )

    # ========================================================
    # SAVE OOF OUTPUTS
    # ========================================================

    oof_predictions.to_csv(
        OUTPUT_DIR
        / "nested_cv_oof_predictions.csv",
        index=False,
    )

    fold_metrics_df = pd.concat(
        outer_metric_frames,
        ignore_index=True,
    )

    fold_metrics_df.to_csv(
        OUTPUT_DIR
        / "nested_cv_fold_metrics.csv",
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    fold_auc_mean = float(
        fold_metrics_df[
            "test_auc"
        ].mean()
    )

    fold_auc_std = float(
        fold_metrics_df[
            "test_auc"
        ].std(
            ddof=0
        )
    )

    summary_lines = [
        "NeuroFusion AI",
        "Experiment 02 - 3D sMRI CNN + Training-Only Spatial Augmentation",
        "",
        "FINAL STATUS: PASS",
        "",
        f"Subjects evaluated: {len(oof_predictions)}",
        f"Controls: {(oof_predictions['label'] == 0).sum()}",
        f"Schizophrenia: {(oof_predictions['label'] == 1).sum()}",
        "",
        "Nested CV:",
        "    5 outer folds",
        "    4 inner folds",
        "",
        "Augmentation:",
        "    Random left-right flip: p=0.5",
        "    Random rotation: +/-5 degrees",
        "    Random translation: +/-3 voxels",
        "",
        "Augmentation was applied ONLY to training datasets.",
        "Validation and outer-test datasets were NOT augmented.",
        "",
        f"Overall OOF ROC-AUC: {oof_metrics['auc']:.6f}",
        f"Overall OOF Accuracy: {oof_metrics['accuracy']:.6f}",
        f"Overall OOF Balanced Accuracy: {oof_metrics['balanced_accuracy']:.6f}",
        f"Overall OOF Precision: {oof_metrics['precision']:.6f}",
        f"Overall OOF Sensitivity: {oof_metrics['recall']:.6f}",
        f"Overall OOF F1: {oof_metrics['f1']:.6f}",
        "",
        f"Mean outer-fold ROC-AUC: {fold_auc_mean:.6f}",
        f"Population SD of outer-fold ROC-AUC: {fold_auc_std:.6f}",
        "",
        "Outer-fold AUC values:",
    ]

    for _, row in (
        fold_metrics_df
        .sort_values(
            "outer_fold"
        )
        .iterrows()
    ):

        summary_lines.append(
            f"    Fold "
            f"{int(row['outer_fold'])}: "
            f"{row['test_auc']:.6f}"
        )

    summary_text = (
        "\n".join(
            summary_lines
        )
        + "\n"
    )

    (
        OUTPUT_DIR
        / "nested_cv_summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "EXPERIMENT 02 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Subjects evaluated: "
        f"{len(oof_predictions)}"
    )

    print(
        f"OOF ROC-AUC: "
        f"{oof_metrics['auc']:.6f}"
    )

    print(
        f"OOF Accuracy: "
        f"{oof_metrics['accuracy']:.6f}"
    )

    print(
        f"OOF Balanced Accuracy: "
        f"{oof_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Mean fold AUC: "
        f"{fold_auc_mean:.6f}"
    )

    print(
        f"Fold AUC population SD: "
        f"{fold_auc_std:.6f}"
    )

    print()

    print(
        "Outer-fold AUCs:"
    )

    for _, row in (
        fold_metrics_df
        .sort_values(
            "outer_fold"
        )
        .iterrows()
    ):

        print(
            f"  Fold "
            f"{int(row['outer_fold'])}: "
            f"{row['test_auc']:.6f}"
        )

    print()

    print(
        "FINAL STATUS: PASS"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()