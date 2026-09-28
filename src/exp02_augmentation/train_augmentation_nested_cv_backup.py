"""
NeuroFusion AI
Experiment 02 - 3D sMRI CNN with Mild Spatial Augmentation

Controlled experiment:
    Baseline BatchNorm3D CNN
    + training-only spatial augmentation

Augmentation:
    - Random left-right flip, p=0.5
    - Random 3D rotation, +/- 5 degrees
    - Random 3D translation, +/- 3 voxels

Validation and outer-test data are NEVER augmented.

The baseline training script is intentionally left untouched.
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

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EXP_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(EXP_DIR),
)

from smri_3d_cnn import SMRI3DCNN

from smri_augmentation import (
    apply_training_augmentation,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "exp02_augmentation"

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
# EXPERIMENT SETTINGS
# ============================================================

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

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

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

    def __getitem__(self, index):

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

        if tuple(image.shape) != (
            EXPECTED_TENSOR_SHAPE
        ):

            raise RuntimeError(
                f"Unexpected tensor shape "
                f"for {subject_id}: "
                f"{image.shape}; expected "
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
        # Final validation
        # ----------------------------------------------------

        if tuple(image.shape) != (
            EXPECTED_TENSOR_SHAPE
        ):

            raise RuntimeError(
                f"Augmentation changed shape "
                f"for {subject_id}: "
                f"{image.shape}"
            )

        if not np.isfinite(
            image
        ).all():

            raise RuntimeError(
                f"Non-finite values after "
                f"augmentation for "
                f"{subject_id}"
            )

        if image.min() < 0.0:

            raise RuntimeError(
                f"Negative intensity after "
                f"augmentation for "
                f"{subject_id}"
            )

        if image.max() > 1.0:

            raise RuntimeError(
                f"Intensity > 1 after "
                f"augmentation for "
                f"{subject_id}"
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

        batch_size = images.size(
            0
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

            batch_size = images.size(
                0
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

    if (
        len(train_ids)
        + len(validation_ids)
        + len(test_ids)
        != len(
            train_ids
            | validation_ids
            | test_ids
        )
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

        if classes != {0, 1}:

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
    # DATA
    #
    # IMPORTANT:
    # Training = augmentation ON
    # Validation = augmentation OFF
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
    # MODEL
    # --------------------------------------------------------

    model = SMRI3DCNN(
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

    # --------------------------------------------------------
    # TRAIN INNER MODEL
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
            f"Epoch {epoch:03d}/"
            f"{NUM_EPOCHS} | "
            f"Train AUC "
            f"{train_metrics['auc']:.4f} | "
            f"Val AUC "
            f"{validation_auc:.4f}"
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
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break

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
        f"FINAL OUTER MODEL {outer_fold}"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Outer train/test
    # --------------------------------------------------------

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
        train_ids | test_ids
    ) != len(
        set(
            outer_df[
                "subject_id"
            ].astype(str)
        )
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final train/test partition "
            "invalid."
        )

    if set(
        outer_train_df[
            "label"
        ].astype(int)
    ) != {0, 1}:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "final outer training does "
            "not contain both classes."
        )

    if set(
        outer_test_df[
            "label"
        ].astype(int)
    ) != {0, 1}:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            "outer test does not contain "
            "both classes."
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
    # Final outer seed
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
    # Training = augmentation ON
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
    # MODEL
    # --------------------------------------------------------

    model = SMRI3DCNN(
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

    # --------------------------------------------------------
    # FINAL OUTER TRAINING
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
                    train_metrics["auc"],

                "learning_rate":
                    optimizer.param_groups[0]["lr"],
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
    # OUTER TEST
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
        OUTPUT_DIR
        / f"outer_fold_{outer_fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model_path = (
        fold_dir
        / "final_outer_model.pt"
    )

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
        model_path,
    )

    # --------------------------------------------------------
    # Save history
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

    metrics_df = pd.DataFrame(
        [
            {
                "outer_fold":
                    outer_fold,

                "test_subjects":
                    len(test_subjects),

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

                "test_sensitivity":
                    test_metrics["recall"],

                "test_f1":
                    test_metrics["f1"],

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
    # Assignment subjects
    # --------------------------------------------------------

    if set(
        assignments[
            "subject_id"
        ]
    ) != set(
        manifest[
            "subject_id"
        ]
    ):

        raise RuntimeError(
            "Assignment subjects do not "
            "match labeled manifest."
        )

    # --------------------------------------------------------
    # Outer fold validation
    # --------------------------------------------------------

    outer_counts = (
        assignments[
            [
                "subject_id",
                "outer_fold",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "outer_fold"
        )[
            "subject_id"
        ]
        .nunique()
    )

    if set(
        outer_counts.index
    ) != set(
        range(
            1,
            OUTER_FOLDS + 1,
        )
    ):

        raise RuntimeError(
            "Invalid outer fold values."
        )

    if not (
        outer_counts
        == 1
    ).all():

        raise RuntimeError(
            "Each subject must have "
            "exactly one outer fold."
        )

    print(
        f"Assignment rows: "
        f"{len(assignments)}"
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
    # Merge labels / metadata
    # --------------------------------------------------------

    manifest_columns = [
        "subject_id",
        "label",
        "acquisition_source",
    ]

    manifest_subset = (
        manifest[
            manifest_columns
        ]
        .drop_duplicates(
            subset=["subject_id"]
        )
        .copy()
    )

    assignments = assignments.merge(
        manifest_subset,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    if assignments[
        "label"
    ].isna().any():

        raise RuntimeError(
            "Missing labels after "
            "assignment merge."
        )

    # --------------------------------------------------------
    # Validate image files
    # --------------------------------------------------------

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

    print(
        f"Subjects: "
        f"{manifest['subject_id'].nunique()}"
    )

    print(
        f"Controls: "
        f"{(
            manifest['label'].astype(int) == 0
        ).sum()}"
    )

    print(
        f"Schizophrenia: "
        f"{(
            manifest['label'].astype(int) == 1
        ).sum()}"
    )

    print()

    # --------------------------------------------------------
    # Save experiment configuration
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
    Early stopping patience: {EARLY_STOPPING_PATIENCE}

Augmentation:
    Training only: YES
    Validation augmentation: NO
    Outer test augmentation: NO
    Random left-right flip: p=0.5
    Random rotation: +/-5 degrees
    Random translation: +/-3 voxels

Normalization:
    Existing normalized [0,1] tensors.
    No additional intensity augmentation.

Seed:
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

    # --------------------------------------------------------
    # INNER CV
    # --------------------------------------------------------

    all_inner_results = []

    all_inner_histories = []

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

        outer_df = (
            assignments[
                assignments[
                    "outer_fold"
                ]
                == outer_fold
            ]
            .copy()
        )

        expected_subjects = (
            assignments[
                assignments[
                    "outer_fold"
                ]
                == outer_fold
            ][
                "subject_id"
            ]
            .nunique()
        )

        if expected_subjects not in [
            34,
            35,
        ]:

            raise RuntimeError(
                f"Unexpected outer fold "
                f"{outer_fold} size: "
                f"{expected_subjects}"
            )

        outer_inner_results = []

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

        # ----------------------------------------------------
        # Save epoch-selection record
        # ----------------------------------------------------

        epoch_record = {
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

        epoch_path = (
            OUTPUT_DIR
            / "epoch_selection.csv"
        )

        if epoch_path.exists():

            existing_epochs = (
                pd.read_csv(
                    epoch_path
                )
            )

            existing_epochs = (
                existing_epochs[
                    existing_epochs[
                        "outer_fold"
                    ]
                    != outer_fold
                ]
            )

            existing_epochs = pd.concat(
                [
                    existing_epochs,
                    pd.DataFrame(
                        [epoch_record]
                    ),
                ],
                ignore_index=True,
            )

        else:

            existing_epochs = pd.DataFrame(
                [epoch_record]
            )

        existing_epochs = (
            existing_epochs
            .sort_values(
                "outer_fold"
            )
        )

        existing_epochs.to_csv(
            epoch_path,
            index=False,
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
        .reset_index(drop=True)
    )

    inner_results_df.to_csv(
        OUTPUT_DIR
        / "inner_cv_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save all inner histories
    # --------------------------------------------------------

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
    # COLLECT OUTER PREDICTIONS
    # ========================================================

    prediction_frames = []

    fold_metrics = []

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        prediction_path = (
            OUTPUT_DIR
            / f"outer_fold_{outer_fold}"
            / "outer_test_predictions.csv"
        )

        metrics_path = (
            OUTPUT_DIR
            / f"outer_fold_{outer_fold}"
            / "outer_test_metrics.csv"
        )

        if not prediction_path.exists():

            raise RuntimeError(
                f"Missing predictions: "
                f"{prediction_path}"
            )

        if not metrics_path.exists():

            raise RuntimeError(
                f"Missing metrics: "
                f"{metrics_path}"
            )

        prediction_frames.append(
            pd.read_csv(
                prediction_path
            )
        )

        fold_metrics.append(
            pd.read_csv(
                metrics_path
            )
        )

    oof_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    oof_predictions = (
        oof_predictions
        .sort_values(
            "subject_id"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # OOF uniqueness
    # --------------------------------------------------------

    if oof_predictions[
        "subject_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate OOF subjects."
        )

    if len(
        oof_predictions
    ) != len(manifest):

        raise RuntimeError(
            f"Expected {len(manifest)} "
            f"OOF predictions, found "
            f"{len(oof_predictions)}."
        )

    # --------------------------------------------------------
    # OOF metrics
    # --------------------------------------------------------

    oof_metrics = calculate_metrics(
        oof_predictions[
            "label"
        ].to_numpy(),
        oof_predictions[
            "probability"
        ].to_numpy(),
    )

    oof_predictions.to_csv(
        OUTPUT_DIR
        / "nested_cv_oof_predictions.csv",
        index=False,
    )

    fold_metrics_df = pd.concat(
        fold_metrics,
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

    summary = f"""
NeuroFusion AI
Experiment 02 - 3D sMRI CNN + Training-Only Spatial Augmentation

FINAL STATUS: PASS

Subjects evaluated:
    {len(oof_predictions)}

Controls:
    {(oof_predictions['label'] == 0).sum()}

Schizophrenia:
    {(oof_predictions['label'] == 1).sum()}

Nested CV:
    5 outer folds
    4 inner folds

Augmentation:
    Random left-right flip: p=0.5
    Random rotation: +/-5 degrees
    Random translation: +/-3 voxels

Augmentation was applied ONLY to training datasets.

Validation and outer-test datasets were NOT augmented.

Overall OOF ROC-AUC:
    {oof_metrics['auc']:.6f}

Overall OOF Accuracy:
    {oof_metrics['accuracy']:.6f}

Overall OOF Balanced Accuracy:
    {oof_metrics['balanced_accuracy']:.6f}

Overall OOF Precision:
    {oof_metrics['precision']:.6f}

Overall OOF Sensitivity:
    {oof_metrics['recall']:.6f}

Overall OOF F1:
    {oof_metrics['f1']:.6f}

Mean outer-fold ROC-AUC:
    {fold_auc_mean:.6f}

Population SD of outer-fold ROC-AUC:
    {fold_auc_std:.6f}

Outer-fold AUC values:
"""

    for _, row in fold_metrics_df.iterrows():

        summary += (
            f"\n    Fold "
            f"{int(row['outer_fold'])}: "
            f"{row['test_auc']:.6f}"
        )

    summary += "\n"

    (
        OUTPUT_DIR
        / "nested_cv_summary.txt"
    ).write_text(
        summary.strip()
        + "\n",
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
        "FINAL STATUS: PASS"
    )

    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()