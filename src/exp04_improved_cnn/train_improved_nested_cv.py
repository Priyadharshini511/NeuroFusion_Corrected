from pathlib import Path
import sys
import random
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXPERIMENT 04 MODEL
# ============================================================

from src.exp04_improved_cnn.smri_3d_cnn_improved import (
    SMRI3DImprovedCNN,
)


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
)

LABELS_PATH = (
    DATASET_DIR
    / "labels.csv"
)

METADATA_PATH = (
    DATASET_DIR
    / "metadata.csv"
)

IMAGE_DIR = (
    DATASET_DIR
    / "images"
)

NESTED_ASSIGNMENTS_PATH = (
    DATASET_DIR
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
    / "exp04_improved_cnn"
    / "training_full_5x4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_OUTER_FOLDS = 5
NUM_INNER_FOLDS = 4

EXPECTED_SUBJECTS = 172

BATCH_SIZE = 4

MAX_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

EARLY_STOPPING_PATIENCE = 10

LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 4

EXPECTED_INPUT_SHAPE = (
    1,
    95,
    117,
    99,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SMRIDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        subject_id = str(
            row["subject_id"]
        )

        tensor_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not tensor_path.exists():

            raise FileNotFoundError(
                f"Tensor not found for "
                f"{subject_id}:\n"
                f"{tensor_path}"
            )

        image = np.load(
            tensor_path
        )

        if image.shape != EXPECTED_INPUT_SHAPE:

            raise ValueError(
                f"{subject_id}: expected "
                f"{EXPECTED_INPUT_SHAPE}, "
                f"got {image.shape}"
            )

        if image.dtype != np.float32:

            image = image.astype(
                np.float32
            )

        if not np.isfinite(image).all():

            raise ValueError(
                f"{subject_id}: tensor contains "
                f"NaN or infinite values"
            )

        image_tensor = torch.from_numpy(
            image
        )

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        return (
            image_tensor,
            label,
            subject_id,
        )


# ============================================================
# MODEL
# ============================================================

def create_model(device):

    model = SMRI3DImprovedCNN(
        dropout=DROPOUT
    )

    return model.to(device)


# ============================================================
# AUC
# ============================================================

def calculate_auc(
    labels,
    probabilities,
):

    labels = np.asarray(labels)

    probabilities = np.asarray(
        probabilities
    )

    if len(np.unique(labels)) < 2:

        return float("nan")

    return float(
        roc_auc_score(
            labels,
            probabilities,
        )
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    labels_all = []

    probabilities_all = []

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
        ).view(-1)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        labels_all.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        probabilities_all.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    epoch_auc = calculate_auc(
        labels_all,
        probabilities_all,
    )

    return (
        epoch_loss,
        epoch_auc,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    labels_all = []

    probabilities_all = []

    subjects_all = []

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
        ).view(-1)

        loss = criterion(
            logits,
            labels,
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        probabilities_all.extend(
            probabilities.cpu().numpy()
        )

        subjects_all.extend(
            subjects
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    epoch_auc = calculate_auc(
        labels_all,
        probabilities_all,
    )

    return (
        epoch_loss,
        epoch_auc,
        np.asarray(labels_all),
        np.asarray(probabilities_all),
        subjects_all,
    )


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

def calculate_pos_weight(
    dataframe
):

    negatives = int(
        (
            dataframe["label"] == 0
        ).sum()
    )

    positives = int(
        (
            dataframe["label"] == 1
        ).sum()
    )

    if positives == 0:

        raise ValueError(
            "Training set has no positive subjects."
        )

    return (
        negatives
        / positives
    )


# ============================================================
# LOAD ASSIGNMENTS
# ============================================================

def load_assignments():

    if not NESTED_ASSIGNMENTS_PATH.exists():

        raise FileNotFoundError(
            "Nested CV assignment file "
            "not found:\n"
            f"{NESTED_ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_csv(
        NESTED_ASSIGNMENTS_PATH
    )

    required_columns = [
        "subject_id",
        "label",
        "acquisition_source",
        "cv_stratum",
        "outer_fold",
        "inner_fold",
        "role",
    ]

    missing = [
        column
        for column in required_columns
        if column not in assignments.columns
    ]

    if missing:

        raise ValueError(
            f"Nested assignment file "
            f"missing columns: {missing}"
        )

    expected_rows = (
        EXPECTED_SUBJECTS
        * NUM_OUTER_FOLDS
        * NUM_INNER_FOLDS
    )

    if len(assignments) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} "
            f"assignment rows, found "
            f"{len(assignments)}"
        )

    expected_roles = {
        "INNER_TRAIN",
        "INNER_VALIDATION",
        "OUTER_TEST",
    }

    actual_roles = set(
        assignments["role"].astype(str)
    )

    if actual_roles != expected_roles:

        raise ValueError(
            f"Unexpected role values: "
            f"{sorted(actual_roles)}"
        )

    return assignments


# ============================================================
# TRAIN INNER FOLD
# ============================================================

def train_inner_fold(
    train_df,
    validation_df,
    outer_fold,
    inner_fold,
    device,
):

    pos_weight_value = (
        calculate_pos_weight(
            train_df
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight_value,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    train_dataset = SMRIDataset(
        train_df
    )

    validation_dataset = SMRIDataset(
        validation_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(
        device
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
            factor=LR_REDUCTION_FACTOR,
            patience=LR_REDUCTION_PATIENCE,
        )
    )

    best_auc = -np.inf

    best_epoch = 1

    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        (
            train_loss,
            train_auc,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_loss,
            val_auc,
            _,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "outer_fold":
                    outer_fold,
                "inner_fold":
                    inner_fold,
                "epoch":
                    epoch,
                "train_loss":
                    train_loss,
                "train_auc":
                    train_auc,
                "validation_loss":
                    val_loss,
                "validation_auc":
                    val_auc,
                "learning_rate":
                    current_lr,
            }
        )

        print(
            f"      Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"train_auc={train_auc:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_auc={val_auc:.6f}"
        )

        if (
            np.isfinite(val_auc)
            and val_auc > best_auc
        ):

            best_auc = val_auc

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if np.isfinite(val_auc):

            scheduler.step(
                val_auc
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"      Early stopping "
                f"at epoch {epoch}"
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No valid validation AUC "
            "was obtained."
        )

    return {
        "best_epoch":
            best_epoch,
        "best_validation_auc":
            best_auc,
        "best_state":
            best_state,
        "history":
            history,
        "pos_weight":
            pos_weight_value,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 75)
    print(
        "EXPERIMENT 04 — IMPROVED 3D CNN"
    )
    print(
        "FULL 5×4 NESTED CROSS-VALIDATION"
    )
    print("=" * 75)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if not LABELS_PATH.exists():

        raise FileNotFoundError(
            f"Labels file not found:\n"
            f"{LABELS_PATH}"
        )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}"
        )

    labels_df = pd.read_csv(
        LABELS_PATH
    )

    metadata_df = pd.read_csv(
        METADATA_PATH
    )

    data = labels_df.merge(
        metadata_df,
        on=[
            "subject_id",
            "label",
        ],
        how="inner",
    )

    if len(data) != EXPECTED_SUBJECTS:

        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} "
            f"subjects after merge, found "
            f"{len(data)}"
        )

    if data[
        "subject_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate subject IDs detected."
        )

    print(
        f"Subjects: {len(data)}"
    )

    print(
        f"Controls: "
        f"{(data['label'] == 0).sum()}"
    )

    print(
        f"Schizophrenia: "
        f"{(data['label'] == 1).sum()}"
    )

    # --------------------------------------------------------
    # Verify tensors
    # --------------------------------------------------------

    missing_tensors = []

    for subject_id in data[
        "subject_id"
    ].astype(str):

        tensor_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not tensor_path.exists():

            missing_tensors.append(
                subject_id
            )

    if missing_tensors:

        raise FileNotFoundError(
            f"Missing tensors: "
            f"{missing_tensors[:10]}"
        )

    # --------------------------------------------------------
    # Nested assignments
    # --------------------------------------------------------

    assignments = load_assignments()

    data_subjects = set(
        data["subject_id"].astype(str)
    )

    assignment_subjects = set(
        assignments[
            "subject_id"
        ].astype(str)
    )

    if data_subjects != assignment_subjects:

        raise ValueError(
            "Dataset subjects do not match "
            "nested CV assignment subjects."
        )

    # ========================================================
    # OUTER CV
    # ========================================================

    outer_results = []

    inner_results = []

    all_outer_predictions = []

    for outer_fold in range(
        1,
        NUM_OUTER_FOLDS + 1,
    ):

        print()
        print("=" * 75)
        print(
            f"OUTER FOLD "
            f"{outer_fold}/{NUM_OUTER_FOLDS}"
        )
        print("=" * 75)

        outer_assignment = (
            assignments[
                assignments[
                    "outer_fold"
                ]
                == outer_fold
            ]
        )

        outer_test_ids = set(
            outer_assignment.loc[
                outer_assignment[
                    "role"
                ]
                == "OUTER_TEST",
                "subject_id",
            ].astype(str)
        )

        outer_train_ids = (
            data_subjects
            - outer_test_ids
        )

        if (
            outer_test_ids
            & outer_train_ids
        ):

            raise ValueError(
                f"Outer fold {outer_fold}: "
                "train/test leakage."
            )

        outer_train_df = data[
            data["subject_id"]
            .astype(str)
            .isin(
                outer_train_ids
            )
        ].copy()

        outer_test_df = data[
            data["subject_id"]
            .astype(str)
            .isin(
                outer_test_ids
            )
        ].copy()

        print(
            f"Outer train: "
            f"{len(outer_train_df)}"
        )

        print(
            f"Outer test: "
            f"{len(outer_test_df)}"
        )

        # ====================================================
        # INNER CV
        # ====================================================

        inner_fold_outputs = []

        for inner_fold in range(
            1,
            NUM_INNER_FOLDS + 1,
        ):

            print()
            print(
                f"  INNER FOLD "
                f"{inner_fold}/"
                f"{NUM_INNER_FOLDS}"
            )

            inner_assignment = (
                outer_assignment[
                    outer_assignment[
                        "inner_fold"
                    ]
                    == inner_fold
                ]
            )

            inner_train_ids = set(
                inner_assignment.loc[
                    inner_assignment[
                        "role"
                    ]
                    == "INNER_TRAIN",
                    "subject_id",
                ].astype(str)
            )

            inner_validation_ids = set(
                inner_assignment.loc[
                    inner_assignment[
                        "role"
                    ]
                    == "INNER_VALIDATION",
                    "subject_id",
                ].astype(str)
            )

            inner_outer_test_ids = set(
                inner_assignment.loc[
                    inner_assignment[
                        "role"
                    ]
                    == "OUTER_TEST",
                    "subject_id",
                ].astype(str)
            )

            # ----------------------------------------------
            # Leakage checks
            # ----------------------------------------------

            if (
                inner_train_ids
                & inner_validation_ids
            ):

                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "train/validation overlap."
                )

            if (
                inner_train_ids
                & inner_outer_test_ids
            ):

                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "inner train contains "
                    "outer test subjects."
                )

            if (
                inner_validation_ids
                & inner_outer_test_ids
            ):

                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "inner validation contains "
                    "outer test subjects."
                )

            if (
                inner_train_ids
                | inner_validation_ids
            ) != outer_train_ids:

                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "inner partitions do not "
                    "cover outer training set."
                )

            inner_train_df = data[
                data["subject_id"]
                .astype(str)
                .isin(
                    inner_train_ids
                )
            ].copy()

            inner_validation_df = data[
                data["subject_id"]
                .astype(str)
                .isin(
                    inner_validation_ids
                )
            ].copy()

            print(
                f"    Train: "
                f"{len(inner_train_df)}"
            )

            print(
                f"    Validation: "
                f"{len(inner_validation_df)}"
            )

            result = train_inner_fold(
                inner_train_df,
                inner_validation_df,
                outer_fold,
                inner_fold,
                device,
            )

            inner_fold_outputs.append(
                result
            )

            inner_results.append(
                {
                    "outer_fold":
                        outer_fold,
                    "inner_fold":
                        inner_fold,
                    "best_epoch":
                        result[
                            "best_epoch"
                        ],
                    "best_validation_auc":
                        result[
                            "best_validation_auc"
                        ],
                    "train_subjects":
                        len(
                            inner_train_df
                        ),
                    "validation_subjects":
                        len(
                            inner_validation_df
                        ),
                    "pos_weight":
                        result[
                            "pos_weight"
                        ],
                    "random_seed":
                        SEED,
                }
            )

            pd.DataFrame(
                result["history"]
            ).to_csv(
                OUTPUT_DIR
                / (
                    f"outer{outer_fold}"
                    f"_inner{inner_fold}"
                    f"_history.csv"
                ),
                index=False,
            )

        # ====================================================
        # SELECT EPOCH
        # ====================================================

        best_epochs = [
            result[
                "best_epoch"
            ]
            for result in inner_fold_outputs
        ]

        selected_epoch = int(
            round(
                float(
                    np.median(
                        best_epochs
                    )
                )
            )
        )

        selected_epoch = max(
            1,
            selected_epoch,
        )

        print()

        print(
            f"  Inner best epochs: "
            f"{best_epochs}"
        )

        print(
            f"  Selected outer epoch: "
            f"{selected_epoch}"
        )

        # ====================================================
        # FINAL OUTER TRAINING
        # ====================================================

        outer_pos_weight = (
            calculate_pos_weight(
                outer_train_df
            )
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    outer_pos_weight,
                    dtype=torch.float32,
                    device=device,
                )
            )
        )

        train_dataset = SMRIDataset(
            outer_train_df
        )

        test_dataset = SMRIDataset(
            outer_test_df
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        set_seed(
            SEED + outer_fold
        )

        model = create_model(
            device
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        outer_history = []

        for epoch in range(
            1,
            selected_epoch + 1,
        ):

            (
                train_loss,
                train_auc,
            ) = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )

            outer_history.append(
                {
                    "outer_fold":
                        outer_fold,
                    "epoch":
                        epoch,
                    "train_loss":
                        train_loss,
                    "train_auc":
                        train_auc,
                    "learning_rate":
                        optimizer
                        .param_groups[0][
                            "lr"
                        ],
                }
            )

            print(
                f"    Outer epoch "
                f"{epoch:02d}/"
                f"{selected_epoch:02d} | "
                f"loss={train_loss:.6f} | "
                f"AUC={train_auc:.6f}"
            )

        # ====================================================
        # OUTER TEST
        # ====================================================

        (
            test_loss,
            test_auc,
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
            f"  OUTER FOLD "
            f"{outer_fold} TEST AUC: "
            f"{test_auc:.6f}"
        )

        # ====================================================
        # SAVE MODEL
        # ====================================================

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "outer_fold":
                    outer_fold,
                "selected_epoch":
                    selected_epoch,
                "seed":
                    SEED,
                "input_shape":
                    EXPECTED_INPUT_SHAPE,
                "dropout":
                    DROPOUT,
                "learning_rate":
                    LEARNING_RATE,
                "weight_decay":
                    WEIGHT_DECAY,
                "batch_size":
                    BATCH_SIZE,
                "pos_weight":
                    outer_pos_weight,
                "architecture":
                    "Residual3DImprovedCNN",
                "normalization":
                    "existing brain-aware "
                    "min-max normalized MNI",
            },
            OUTPUT_DIR
            / (
                f"outer_fold_"
                f"{outer_fold}_model.pt"
            ),
        )

        # ====================================================
        # SAVE HISTORY
        # ====================================================

        pd.DataFrame(
            outer_history
        ).to_csv(
            OUTPUT_DIR
            / (
                f"outer_fold_"
                f"{outer_fold}_history.csv"
            ),
            index=False,
        )

        # ====================================================
        # SAVE PREDICTIONS
        # ====================================================

        predictions_df = pd.DataFrame(
            {
                "subject_id":
                    test_subjects,
                "label":
                    test_labels.astype(int),
                "probability":
                    test_probabilities,
                "outer_fold":
                    outer_fold,
            }
        )

        predictions_df.to_csv(
            OUTPUT_DIR
            / (
                f"outer_fold_"
                f"{outer_fold}_predictions.csv"
            ),
            index=False,
        )

        all_outer_predictions.append(
            predictions_df
        )

        outer_results.append(
            {
                "outer_fold":
                    outer_fold,
                "train_subjects":
                    len(
                        outer_train_df
                    ),
                "test_subjects":
                    len(
                        outer_test_df
                    ),
                "selected_epoch":
                    selected_epoch,
                "test_loss":
                    test_loss,
                "test_auc":
                    test_auc,
                "train_controls":
                    int(
                        (
                            outer_train_df[
                                "label"
                            ]
                            == 0
                        ).sum()
                    ),
                "train_cases":
                    int(
                        (
                            outer_train_df[
                                "label"
                            ]
                            == 1
                        ).sum()
                    ),
                "test_controls":
                    int(
                        (
                            outer_test_df[
                                "label"
                            ]
                            == 0
                        ).sum()
                    ),
                "test_cases":
                    int(
                        (
                            outer_test_df[
                                "label"
                            ]
                            == 1
                        ).sum()
                    ),
            }
        )

    # ========================================================
    # OOF PREDICTIONS
    # ========================================================

    oof_predictions = pd.concat(
        all_outer_predictions,
        ignore_index=True,
    )

    if len(oof_predictions) != 172:

        raise ValueError(
            f"Expected 172 OOF predictions, "
            f"found {len(oof_predictions)}"
        )

    if (
        oof_predictions[
            "subject_id"
        ].duplicated().any()
    ):

        raise ValueError(
            "Duplicate OOF predictions."
        )

    oof_auc = calculate_auc(
        oof_predictions[
            "label"
        ],
        oof_predictions[
            "probability"
        ],
    )

    # ========================================================
    # THRESHOLD METRICS
    # ========================================================

    y_true = (
        oof_predictions[
            "label"
        ].to_numpy()
    )

    y_pred = (
        oof_predictions[
            "probability"
        ].to_numpy()
        >= 0.5
    ).astype(int)

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    accuracy = (
        (tp + tn)
        / len(y_true)
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else float("nan")
    )

    balanced_accuracy = (
        sensitivity
        + specificity
    ) / 2

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else float("nan")
    )

    f1 = (
        2
        * precision
        * sensitivity
        / (
            precision
            + sensitivity
        )
        if precision + sensitivity > 0
        else float("nan")
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    inner_results_df = pd.DataFrame(
        inner_results
    )

    outer_results_df = pd.DataFrame(
        outer_results
    )

    inner_results_df.to_csv(
        OUTPUT_DIR
        / "inner_cv_results.csv",
        index=False,
    )

    outer_results_df.to_csv(
        OUTPUT_DIR
        / "outer_fold_results.csv",
        index=False,
    )

    oof_predictions[
        "prediction"
    ] = y_pred

    oof_predictions.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    epoch_summary = (
        inner_results_df
        .groupby(
            "outer_fold"
        )[
            "best_epoch"
        ]
        .agg(
            [
                "min",
                "max",
                "median",
                "mean",
            ]
        )
        .reset_index()
    )

    epoch_summary = epoch_summary.merge(
        outer_results_df[
            [
                "outer_fold",
                "selected_epoch",
            ]
        ],
        on="outer_fold",
        how="left",
    )

    epoch_summary.to_csv(
        OUTPUT_DIR
        / "epoch_selection.csv",
        index=False,
    )

    # ========================================================
    # FOLD STATISTICS
    # ========================================================

    mean_fold_auc = float(
        outer_results_df[
            "test_auc"
        ].mean()
    )

    fold_auc_population_std = float(
        outer_results_df[
            "test_auc"
        ].std(
            ddof=0
        )
    )

    fold_auc_values = (
        outer_results_df[
            "test_auc"
        ].tolist()
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "experiment":
            "Experiment 04 — "
            "Improved Residual 3D CNN",

        "seed":
            SEED,

        "device":
            str(device),

        "subjects":
            EXPECTED_SUBJECTS,

        "controls":
            93,

        "schizophrenia":
            79,

        "outer_folds":
            NUM_OUTER_FOLDS,

        "inner_folds":
            NUM_INNER_FOLDS,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "input_shape":
            EXPECTED_INPUT_SHAPE,

        "architecture":
            "Residual3DImprovedCNN",

        "oof_auc":
            oof_auc,

        "oof_accuracy":
            accuracy,

        "oof_balanced_accuracy":
            balanced_accuracy,

        "oof_precision":
            precision,

        "oof_sensitivity":
            sensitivity,

        "oof_specificity":
            specificity,

        "oof_f1":
            f1,

        "mean_outer_fold_auc":
            mean_fold_auc,

        "outer_fold_auc_population_std":
            fold_auc_population_std,

        "outer_fold_auc":
            fold_auc_values,

        "selected_epochs":
            outer_results_df[
                "selected_epoch"
            ].tolist(),

        "final_status":
            "PASS",
    }

    with open(
        OUTPUT_DIR
        / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================
    # HUMAN-READABLE REPORT
    # ========================================================

    report = []

    report.append(
        "EXPERIMENT 04 — "
        "IMPROVED RESIDUAL 3D CNN"
    )

    report.append(
        "FULL 5×4 NESTED CROSS-VALIDATION"
    )

    report.append(
        "=" * 65
    )

    report.append("")

    report.append(
        f"Subjects: {EXPECTED_SUBJECTS}"
    )

    report.append(
        "Controls: 93"
    )

    report.append(
        "Schizophrenia: 79"
    )

    report.append("")

    report.append(
        f"Seed: {SEED}"
    )

    report.append(
        f"Batch size: {BATCH_SIZE}"
    )

    report.append(
        f"Learning rate: {LEARNING_RATE}"
    )

    report.append(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    report.append(
        f"Dropout: {DROPOUT}"
    )

    report.append(
        f"Input shape: {EXPECTED_INPUT_SHAPE}"
    )

    report.append(
        "Architecture: Residual3DImprovedCNN"
    )

    report.append("")

    report.append(
        "Outer-fold results:"
    )

    for _, row in (
        outer_results_df.iterrows()
    ):

        report.append(
            f"  Fold "
            f"{int(row['outer_fold'])}: "
            f"AUC="
            f"{row['test_auc']:.6f}, "
            f"selected_epoch="
            f"{int(row['selected_epoch'])}"
        )

    report.append("")

    report.append(
        f"Mean outer-fold AUC: "
        f"{mean_fold_auc:.6f}"
    )

    report.append(
        f"Outer-fold AUC population SD: "
        f"{fold_auc_population_std:.6f}"
    )

    report.append("")

    report.append(
        f"OOF ROC-AUC: "
        f"{oof_auc:.6f}"
    )

    report.append(
        f"OOF Accuracy: "
        f"{accuracy:.6f}"
    )

    report.append(
        f"OOF Balanced Accuracy: "
        f"{balanced_accuracy:.6f}"
    )

    report.append(
        f"OOF Precision: "
        f"{precision:.6f}"
    )

    report.append(
        f"OOF Sensitivity: "
        f"{sensitivity:.6f}"
    )

    report.append(
        f"OOF Specificity: "
        f"{specificity:.6f}"
    )

    report.append(
        f"OOF F1: "
        f"{f1:.6f}"
    )

    report.append("")

    report.append(
        "OOF confusion matrix:"
    )

    report.append(
        f"  TN={tn}  FP={fp}"
    )

    report.append(
        f"  FN={fn}  TP={tp}"
    )

    report.append("")

    report.append(
        "FINAL STATUS: "
        "EXPERIMENT 04 NESTED CV PASS"
    )

    (
        OUTPUT_DIR
        / "training_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 75)
    print(
        "EXPERIMENT 04 — "
        "FINAL NESTED CV RESULTS"
    )
    print("=" * 75)

    print(
        f"Outer fold AUCs: "
        f"{fold_auc_values}"
    )

    print(
        f"Mean fold AUC: "
        f"{mean_fold_auc:.6f}"
    )

    print(
        f"Fold AUC population SD: "
        f"{fold_auc_population_std:.6f}"
    )

    print(
        f"OOF ROC-AUC: "
        f"{oof_auc:.6f}"
    )

    print(
        f"OOF Accuracy: "
        f"{accuracy:.6f}"
    )

    print(
        f"OOF Balanced Accuracy: "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"OOF Precision: "
        f"{precision:.6f}"
    )

    print(
        f"OOF Sensitivity: "
        f"{sensitivity:.6f}"
    )

    print(
        f"OOF Specificity: "
        f"{specificity:.6f}"
    )

    print(
        f"OOF F1: "
        f"{f1:.6f}"
    )

    print()

    print(
        "FINAL STATUS: "
        "EXPERIMENT 04 NESTED CV PASS"
    )

    print()

    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()