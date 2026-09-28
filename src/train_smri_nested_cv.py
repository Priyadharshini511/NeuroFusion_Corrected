# """
# NeuroFusion AI
# sMRI 3D CNN Nested Cross-Validation Training Pipeline

# Purpose
# -------
# Train and evaluate the sMRI 3D CNN using the prepared
# subject-level nested cross-validation assignments.

# Nested CV design
# ----------------
# 5 outer folds are already prepared.

# For each outer fold:

#     Outer TRAIN
#         |
#         +-- selected INNER TRAIN
#         |
#         +-- selected INNER VALIDATION
#         |
#         +-- untouched OUTER TEST

# The inner validation set is used for:
#     - early stopping
#     - model selection

# The outer test set is evaluated only after the best
# model has been selected.

# IMPORTANT
# ---------
# The nested assignment file contains repeated rows because
# each outer fold contains four different inner folds.

# Therefore validation is always performed on one specific:

#     outer_fold + inner_fold

# combination.

# This script does NOT modify:
#     - baseline 70/15/15 split
#     - baseline model
#     - baseline training history
#     - outer CV assignments
#     - nested CV assignments

# Input
# -----
# processed_data/smri/cnn_dataset/images/
# processed_data/smri/cnn_dataset/cross_validation/nested/
#     nested_cv_assignments.csv

# Output
# ------
# processed_data/smri/cnn_dataset/cross_validation/training/

#     outer_fold_1/
#         best_model.pt
#         training_history.csv
#         training_summary.txt
#         outer_test_predictions.csv

#     outer_fold_2/
#         ...

#     outer_fold_5/
#         ...

#     nested_cv_oof_predictions.csv
#     nested_cv_fold_metrics.csv
#     nested_cv_summary.txt
# """


# from pathlib import Path
# import random

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn

# from torch.utils.data import Dataset, DataLoader

# from sklearn.metrics import (
#     accuracy_score,
#     balanced_accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
# )

# from smri_3d_cnn import SMRI3DCNN


# # ============================================================
# # PATHS
# # ============================================================

# ROOT = Path("processed_data") / "smri"

# CNN_DATASET_DIR = (
#     ROOT / "cnn_dataset"
# )

# IMAGE_DIR = (
#     CNN_DATASET_DIR / "images"
# )

# NESTED_CV_DIR = (
#     CNN_DATASET_DIR
#     / "cross_validation"
#     / "nested"
# )

# TRAINING_DIR = (
#     CNN_DATASET_DIR
#     / "cross_validation"
#     / "training"
# )

# NESTED_ASSIGNMENTS_FILE = (
#     NESTED_CV_DIR
#     / "nested_cv_assignments.csv"
# )

# OOF_PREDICTIONS_FILE = (
#     TRAINING_DIR
#     / "nested_cv_oof_predictions.csv"
# )

# FOLD_METRICS_FILE = (
#     TRAINING_DIR
#     / "nested_cv_fold_metrics.csv"
# )

# CV_SUMMARY_FILE = (
#     TRAINING_DIR
#     / "nested_cv_summary.txt"
# )


# # ============================================================
# # CONFIGURATION
# # ============================================================

# RANDOM_SEED = 42

# OUTER_FOLDS = 5

# INNER_FOLDS = 4

# # ------------------------------------------------------------
# # For this experiment, use Inner Fold 1 for model selection
# # inside each outer fold.
# #
# # Outer 1 -> Inner 1
# # Outer 2 -> Inner 1
# # ...
# # Outer 5 -> Inner 1
# #
# # The other prepared inner folds remain available for future
# # experiments and are NOT modified.
# # ------------------------------------------------------------

# SELECTED_INNER_FOLD = 1

# EXPECTED_SUBJECTS = 172

# EXPECTED_TENSOR_SHAPE = (
#     1,
#     95,
#     117,
#     99,
# )

# BATCH_SIZE = 4

# NUM_EPOCHS = 50

# LEARNING_RATE = 1e-4

# WEIGHT_DECAY = 1e-4

# DROPOUT = 0.30

# EARLY_STOPPING_PATIENCE = 10

# NUM_WORKERS = 0

# THRESHOLD = 0.50


# # ============================================================
# # REPRODUCIBILITY
# # ============================================================

# def set_seed(seed: int) -> None:
#     """
#     Set random seeds for reproducibility.
#     """

#     random.seed(seed)

#     np.random.seed(seed)

#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # ============================================================
# # DATASET
# # ============================================================

# class SMRIDataset(Dataset):
#     """
#     Subject-level sMRI dataset.

#     Each subject has exactly one validated .npy tensor.
#     """

#     def __init__(
#         self,
#         dataframe: pd.DataFrame,
#         image_dir: Path,
#     ):
#         self.dataframe = (
#             dataframe
#             .reset_index(drop=True)
#             .copy()
#         )

#         self.image_dir = image_dir

#     def __len__(self) -> int:
#         return len(self.dataframe)

#     def __getitem__(
#         self,
#         index: int,
#     ):
#         row = self.dataframe.iloc[index]

#         subject_id = str(
#             row["subject_id"]
#         )

#         label = float(
#             row["label"]
#         )

#         image_path = (
#             self.image_dir
#             / f"{subject_id}.npy"
#         )

#         if not image_path.exists():
#             raise FileNotFoundError(
#                 f"Tensor not found: {image_path}"
#             )

#         image = np.load(
#             image_path
#         ).astype(
#             np.float32,
#             copy=False,
#         )

#         if tuple(image.shape) != (
#             EXPECTED_TENSOR_SHAPE
#         ):
#             raise RuntimeError(
#                 f"Unexpected tensor shape for "
#                 f"{subject_id}: "
#                 f"{image.shape}; expected "
#                 f"{EXPECTED_TENSOR_SHAPE}"
#             )

#         image_tensor = torch.from_numpy(
#             image
#         )

#         label_tensor = torch.tensor(
#             [label],
#             dtype=torch.float32,
#         )

#         return (
#             image_tensor,
#             label_tensor,
#             subject_id,
#         )


# # ============================================================
# # METRICS
# # ============================================================

# def calculate_metrics(
#     labels,
#     probabilities,
# ):
#     """
#     Calculate subject-level classification metrics.
#     """

#     labels = np.asarray(
#         labels,
#         dtype=np.int64,
#     )

#     probabilities = np.asarray(
#         probabilities,
#         dtype=np.float64,
#     )

#     predictions = (
#         probabilities >= THRESHOLD
#     ).astype(np.int64)

#     accuracy = accuracy_score(
#         labels,
#         predictions,
#     )

#     balanced_accuracy = (
#         balanced_accuracy_score(
#             labels,
#             predictions,
#         )
#     )

#     precision = precision_score(
#         labels,
#         predictions,
#         zero_division=0,
#     )

#     recall = recall_score(
#         labels,
#         predictions,
#         zero_division=0,
#     )

#     f1 = f1_score(
#         labels,
#         predictions,
#         zero_division=0,
#     )

#     if len(
#         np.unique(labels)
#     ) == 2:

#         auc = roc_auc_score(
#             labels,
#             probabilities,
#         )

#     else:

#         auc = float("nan")

#     return {
#         "accuracy": float(
#             accuracy
#         ),
#         "balanced_accuracy": float(
#             balanced_accuracy
#         ),
#         "precision": float(
#             precision
#         ),
#         "recall": float(
#             recall
#         ),
#         "f1": float(
#             f1
#         ),
#         "auc": float(
#             auc
#         ),
#     }


# # ============================================================
# # NESTED ASSIGNMENT VALIDATION
# # ============================================================

# def validate_nested_assignments(
#     df: pd.DataFrame,
# ) -> None:
#     """
#     Validate nested CV assignments.

#     IMPORTANT
#     ---------
#     The assignment file contains:

#         172 subjects
#         x 5 outer folds
#         x 4 inner folds

#     = 3440 rows

#     Subjects appear repeatedly because the same outer-test
#     subjects are represented for each inner fold.

#     Therefore all partition checks are performed separately
#     for each outer_fold + inner_fold combination.
#     """

#     required_columns = {
#         "subject_id",
#         "label",
#         "acquisition_source",
#         "cv_stratum",
#         "outer_fold",
#         "inner_fold",
#         "role",
#     }

#     missing = (
#         required_columns
#         - set(df.columns)
#     )

#     if missing:
#         raise RuntimeError(
#             "Nested assignment file is missing columns: "
#             + ", ".join(
#                 sorted(missing)
#             )
#         )

#     if df.empty:
#         raise RuntimeError(
#             "Nested assignment file is empty."
#         )

#     # --------------------------------------------------------
#     # Basic value validation.
#     # --------------------------------------------------------

#     if not df["label"].isin(
#         [0, 1]
#     ).all():

#         raise RuntimeError(
#             "Nested assignment contains non-binary labels."
#         )

#     if not df["outer_fold"].isin(
#         range(
#             1,
#             OUTER_FOLDS + 1,
#         )
#     ).all():

#         raise RuntimeError(
#             "Invalid outer fold detected."
#         )

#     if not df["inner_fold"].isin(
#         range(
#             1,
#             INNER_FOLDS + 1,
#         )
#     ).all():

#         raise RuntimeError(
#             "Invalid inner fold detected."
#         )

#     valid_roles = {
#         "INNER_TRAIN",
#         "INNER_VALIDATION",
#         "OUTER_TEST",
#     }

#     if not set(
#         df["role"].astype(str)
#     ).issubset(
#         valid_roles
#     ):

#         raise RuntimeError(
#             "Invalid nested CV role detected."
#         )

#     # --------------------------------------------------------
#     # Overall unique subject count.
#     # --------------------------------------------------------

#     unique_subjects = set(
#         df["subject_id"]
#         .astype(str)
#     )

#     if len(unique_subjects) != (
#         EXPECTED_SUBJECTS
#     ):

#         raise RuntimeError(
#             f"Expected {EXPECTED_SUBJECTS} "
#             f"unique subjects; found "
#             f"{len(unique_subjects)}."
#         )

#     # --------------------------------------------------------
#     # Validate each outer fold.
#     # --------------------------------------------------------

#     for outer_fold in range(
#         1,
#         OUTER_FOLDS + 1,
#     ):

#         outer_df = df[
#             df["outer_fold"]
#             == outer_fold
#         ].copy()

#         if outer_df.empty:
#             raise RuntimeError(
#                 f"Outer fold {outer_fold} is empty."
#             )

#         # ----------------------------------------------------
#         # Outer test subjects.
#         #
#         # They should appear four times because there are
#         # four inner folds.
#         # ----------------------------------------------------

#         outer_test_all = outer_df[
#             outer_df["role"]
#             == "OUTER_TEST"
#         ]

#         outer_test_ids = set(
#             outer_test_all[
#                 "subject_id"
#             ].astype(str)
#         )

#         if len(
#             outer_test_ids
#         ) == 0:

#             raise RuntimeError(
#                 f"Outer fold {outer_fold}: "
#                 "no outer-test subjects."
#             )

#         outer_test_counts = (
#             outer_test_all
#             .groupby(
#                 "subject_id"
#             )
#             .size()
#         )

#         if not (
#             outer_test_counts
#             == INNER_FOLDS
#         ).all():

#             raise RuntimeError(
#                 f"Outer fold {outer_fold}: "
#                 "each outer-test subject must "
#                 f"appear exactly {INNER_FOLDS} times."
#             )

#         # ----------------------------------------------------
#         # Validate each INNER fold independently.
#         #
#         # This is the critical correction.
#         # ----------------------------------------------------

#         for inner_fold in range(
#             1,
#             INNER_FOLDS + 1,
#         ):

#             split_df = outer_df[
#                 outer_df["inner_fold"]
#                 == inner_fold
#             ].copy()

#             if split_df.empty:

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "no assignments found."
#                 )

#             split_train_ids = set(
#                 split_df.loc[
#                     split_df["role"]
#                     == "INNER_TRAIN",
#                     "subject_id",
#                 ].astype(str)
#             )

#             split_validation_ids = set(
#                 split_df.loc[
#                     split_df["role"]
#                     == "INNER_VALIDATION",
#                     "subject_id",
#                 ].astype(str)
#             )

#             split_test_ids = set(
#                 split_df.loc[
#                     split_df["role"]
#                     == "OUTER_TEST",
#                     "subject_id",
#                 ].astype(str)
#             )

#             # ------------------------------------------------
#             # No overlap.
#             # ------------------------------------------------

#             if split_train_ids.intersection(
#                 split_validation_ids
#             ):

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "train/validation overlap."
#                 )

#             if split_train_ids.intersection(
#                 split_test_ids
#             ):

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "train/outer-test overlap."
#                 )

#             if split_validation_ids.intersection(
#                 split_test_ids
#             ):

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "validation/outer-test overlap."
#                 )

#             # ------------------------------------------------
#             # The three partitions must cover the outer fold
#             # exactly.
#             # ------------------------------------------------

#             outer_subject_ids = set(
#                 outer_df[
#                     "subject_id"
#                 ].astype(str)
#             )

#             partition_ids = (
#                 split_train_ids
#                 | split_validation_ids
#                 | split_test_ids
#             )

#             if partition_ids != (
#                 outer_subject_ids
#             ):

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "partition does not cover exactly "
#                     "the outer-fold subjects."
#                 )

#             # ------------------------------------------------
#             # Inner train + validation must cover the outer
#             # training subjects.
#             # ------------------------------------------------

#             expected_outer_train_ids = (
#                 outer_subject_ids
#                 - split_test_ids
#             )

#             actual_outer_train_ids = (
#                 split_train_ids
#                 | split_validation_ids
#             )

#             if actual_outer_train_ids != (
#                 expected_outer_train_ids
#             ):

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "inner train + validation do not "
#                     "cover the outer training subjects."
#                 )

#             # ------------------------------------------------
#             # Both classes required in each partition.
#             # ------------------------------------------------

#             train_labels = (
#                 split_df.loc[
#                     split_df["role"]
#                     == "INNER_TRAIN",
#                     "label",
#                 ]
#                 .astype(int)
#             )

#             validation_labels = (
#                 split_df.loc[
#                     split_df["role"]
#                     == "INNER_VALIDATION",
#                     "label",
#                 ]
#                 .astype(int)
#             )

#             test_labels = (
#                 split_df.loc[
#                     split_df["role"]
#                     == "OUTER_TEST",
#                     "label",
#                 ]
#                 .astype(int)
#             )

#             if set(train_labels) != {
#                 0,
#                 1,
#             }:

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "inner training does not contain "
#                     "both classes."
#                 )

#             if set(validation_labels) != {
#                 0,
#                 1,
#             }:

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "inner validation does not contain "
#                     "both classes."
#                 )

#             if set(test_labels) != {
#                 0,
#                 1,
#             }:

#                 raise RuntimeError(
#                     f"Outer fold {outer_fold}, "
#                     f"Inner fold {inner_fold}: "
#                     "outer test does not contain "
#                     "both classes."
#                 )


# # ============================================================
# # TRAIN ONE EPOCH
# # ============================================================

# def train_one_epoch(
#     model,
#     loader,
#     criterion,
#     optimizer,
#     device,
# ):
#     """
#     Train one epoch.
#     """

#     model.train()

#     running_loss = 0.0

#     labels_all = []

#     probabilities_all = []

#     subject_count = 0

#     for images, labels, _ in loader:

#         images = images.to(
#             device,
#             non_blocking=True,
#         )

#         labels = labels.to(
#             device,
#             non_blocking=True,
#         )

#         optimizer.zero_grad(
#             set_to_none=True
#         )

#         logits = model(
#             images
#         )

#         loss = criterion(
#             logits,
#             labels,
#         )

#         loss.backward()

#         optimizer.step()

#         batch_size = (
#             images.size(0)
#         )

#         running_loss += (
#             loss.item()
#             * batch_size
#         )

#         subject_count += (
#             batch_size
#         )

#         probabilities = torch.sigmoid(
#             logits
#         )

#         labels_all.extend(
#             labels.detach()
#             .cpu()
#             .numpy()
#             .reshape(-1)
#             .tolist()
#         )

#         probabilities_all.extend(
#             probabilities.detach()
#             .cpu()
#             .numpy()
#             .reshape(-1)
#             .tolist()
#         )

#     if subject_count == 0:

#         raise RuntimeError(
#             "Training loader is empty."
#         )

#     epoch_loss = (
#         running_loss
#         / subject_count
#     )

#     metrics = calculate_metrics(
#         labels_all,
#         probabilities_all,
#     )

#     return (
#         epoch_loss,
#         metrics,
#     )


# # ============================================================
# # EVALUATION
# # ============================================================

# def evaluate(
#     model,
#     loader,
#     criterion,
#     device,
# ):
#     """
#     Evaluate without gradient computation.
#     """

#     model.eval()

#     running_loss = 0.0

#     labels_all = []

#     probabilities_all = []

#     subjects_all = []

#     subject_count = 0

#     with torch.no_grad():

#         for images, labels, subjects in loader:

#             images = images.to(
#                 device,
#                 non_blocking=True,
#             )

#             labels = labels.to(
#                 device,
#                 non_blocking=True,
#             )

#             logits = model(
#                 images
#             )

#             loss = criterion(
#                 logits,
#                 labels,
#             )

#             batch_size = (
#                 images.size(0)
#             )

#             running_loss += (
#                 loss.item()
#                 * batch_size
#             )

#             subject_count += (
#                 batch_size
#             )

#             probabilities = torch.sigmoid(
#                 logits
#             )

#             labels_all.extend(
#                 labels.cpu()
#                 .numpy()
#                 .reshape(-1)
#                 .tolist()
#             )

#             probabilities_all.extend(
#                 probabilities.cpu()
#                 .numpy()
#                 .reshape(-1)
#                 .tolist()
#             )

#             subjects_all.extend(
#                 list(subjects)
#             )

#     if subject_count == 0:

#         raise RuntimeError(
#             "Evaluation loader is empty."
#         )

#     epoch_loss = (
#         running_loss
#         / subject_count
#     )

#     metrics = calculate_metrics(
#         labels_all,
#         probabilities_all,
#     )

#     return (
#         epoch_loss,
#         metrics,
#         labels_all,
#         probabilities_all,
#         subjects_all,
#     )


# # ============================================================
# # TRAIN ONE OUTER FOLD
# # ============================================================

# def train_outer_fold(
#     outer_fold: int,
#     outer_df: pd.DataFrame,
#     device: torch.device,
# ):
#     """
#     Train one outer-fold model.

#     Only the selected inner fold is used for training/model
#     selection.

#     The outer test set is evaluated after model selection.
#     """

#     print()
#     print("=" * 72)
#     print(
#         f"OUTER FOLD {outer_fold}/{OUTER_FOLDS}"
#     )
#     print("=" * 72)

#     # --------------------------------------------------------
#     # Select ONE inner fold.
#     #
#     # This must happen BEFORE partition validation.
#     # --------------------------------------------------------

#     split_df = outer_df[
#         outer_df["inner_fold"]
#         == SELECTED_INNER_FOLD
#     ].copy()

#     if split_df.empty:

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             f"inner fold {SELECTED_INNER_FOLD} "
#             "not found."
#         )

#     inner_train_df = split_df[
#         split_df["role"]
#         == "INNER_TRAIN"
#     ].copy()

#     inner_validation_df = split_df[
#         split_df["role"]
#         == "INNER_VALIDATION"
#     ].copy()

#     outer_test_df = split_df[
#         split_df["role"]
#         == "OUTER_TEST"
#     ].copy()

#     # --------------------------------------------------------
#     # Unique subject IDs.
#     # --------------------------------------------------------

#     train_ids = set(
#         inner_train_df[
#             "subject_id"
#         ].astype(str)
#     )

#     validation_ids = set(
#         inner_validation_df[
#             "subject_id"
#         ].astype(str)
#     )

#     test_ids = set(
#         outer_test_df[
#             "subject_id"
#         ].astype(str)
#     )

#     # --------------------------------------------------------
#     # Partition validation.
#     # --------------------------------------------------------

#     if train_ids.intersection(
#         validation_ids
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "inner train/validation overlap."
#         )

#     if train_ids.intersection(
#         test_ids
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "inner train/outer test overlap."
#         )

#     if validation_ids.intersection(
#         test_ids
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "inner validation/outer test overlap."
#         )

#     expected_outer_subjects = set(
#         outer_df[
#             "subject_id"
#         ].astype(str)
#     )

#     actual_partition_subjects = (
#         train_ids
#         | validation_ids
#         | test_ids
#     )

#     if actual_partition_subjects != (
#         expected_outer_subjects
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "partition does not cover exactly "
#             "the expected unique subjects."
#         )

#     expected_outer_train_subjects = (
#         expected_outer_subjects
#         - test_ids
#     )

#     actual_outer_train_subjects = (
#         train_ids
#         | validation_ids
#     )

#     if actual_outer_train_subjects != (
#         expected_outer_train_subjects
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "inner train + validation do not "
#             "cover the outer training subjects."
#         )

#     # --------------------------------------------------------
#     # Class checks.
#     # --------------------------------------------------------

#     if set(
#         inner_train_df[
#             "label"
#         ].astype(int)
#     ) != {0, 1}:

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "inner training does not contain both classes."
#         )

#     if set(
#         inner_validation_df[
#             "label"
#         ].astype(int)
#     ) != {0, 1}:

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "inner validation does not contain both classes."
#         )

#     if set(
#         outer_test_df[
#             "label"
#         ].astype(int)
#     ) != {0, 1}:

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "outer test does not contain both classes."
#         )

#     # --------------------------------------------------------
#     # Print distribution.
#     # --------------------------------------------------------

#     print(
#         f"Selected inner fold    : "
#         f"{SELECTED_INNER_FOLD}"
#     )

#     print(
#         f"Inner train subjects   : "
#         f"{len(inner_train_df)}"
#     )

#     print(
#         f"Inner validation       : "
#         f"{len(inner_validation_df)}"
#     )

#     print(
#         f"Outer test subjects    : "
#         f"{len(outer_test_df)}"
#     )

#     print(
#         "Inner train classes    : "
#         f"0={(inner_train_df['label'] == 0).sum()}, "
#         f"1={(inner_train_df['label'] == 1).sum()}"
#     )

#     print(
#         "Inner validation       : "
#         f"0={(inner_validation_df['label'] == 0).sum()}, "
#         f"1={(inner_validation_df['label'] == 1).sum()}"
#     )

#     print(
#         "Outer test classes     : "
#         f"0={(outer_test_df['label'] == 0).sum()}, "
#         f"1={(outer_test_df['label'] == 1).sum()}"
#     )

#     # --------------------------------------------------------
#     # Datasets.
#     # --------------------------------------------------------

#     train_dataset = SMRIDataset(
#         inner_train_df,
#         IMAGE_DIR,
#     )

#     validation_dataset = SMRIDataset(
#         inner_validation_df,
#         IMAGE_DIR,
#     )

#     test_dataset = SMRIDataset(
#         outer_test_df,
#         IMAGE_DIR,
#     )

#     # --------------------------------------------------------
#     # DataLoaders.
#     # --------------------------------------------------------

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=torch.cuda.is_available(),
#     )

#     validation_loader = DataLoader(
#         validation_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=torch.cuda.is_available(),
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=torch.cuda.is_available(),
#     )

#     # --------------------------------------------------------
#     # Model.
#     # --------------------------------------------------------

#     model = SMRI3DCNN(
#         dropout=DROPOUT
#     ).to(device)

#     parameter_count = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#     )

#     print(
#         f"Model parameters     : "
#         f"{parameter_count:,}"
#     )

#     # --------------------------------------------------------
#     # Class weighting.
#     # --------------------------------------------------------

#     train_labels = (
#         inner_train_df[
#             "label"
#         ]
#         .astype(int)
#         .to_numpy()
#     )

#     negative_count = int(
#         np.sum(
#             train_labels == 0
#         )
#     )

#     positive_count = int(
#         np.sum(
#             train_labels == 1
#         )
#     )

#     if (
#         negative_count == 0
#         or positive_count == 0
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "invalid training class distribution."
#         )

#     pos_weight_value = (
#         negative_count
#         / positive_count
#     )

#     pos_weight = torch.tensor(
#         [pos_weight_value],
#         dtype=torch.float32,
#         device=device,
#     )

#     print(
#         f"Positive class weight : "
#         f"{pos_weight_value:.6f}"
#     )

#     # --------------------------------------------------------
#     # Loss.
#     # --------------------------------------------------------

#     criterion = nn.BCEWithLogitsLoss(
#         pos_weight=pos_weight
#     )

#     # --------------------------------------------------------
#     # Optimizer.
#     # --------------------------------------------------------

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )

#     # --------------------------------------------------------
#     # Scheduler.
#     # --------------------------------------------------------

#     scheduler = (
#         torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="max",
#             factor=0.5,
#             patience=4,
#             min_lr=1e-6,
#         )
#     )

#     # --------------------------------------------------------
#     # Fold output directory.
#     # --------------------------------------------------------

#     fold_dir = (
#         TRAINING_DIR
#         / f"outer_fold_{outer_fold}"
#     )

#     fold_dir.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     best_model_file = (
#         fold_dir
#         / "best_model.pt"
#     )

#     history_file = (
#         fold_dir
#         / "training_history.csv"
#     )

#     summary_file = (
#         fold_dir
#         / "training_summary.txt"
#     )

#     prediction_file = (
#         fold_dir
#         / "outer_test_predictions.csv"
#     )

#     # --------------------------------------------------------
#     # Training state.
#     # --------------------------------------------------------

#     best_validation_auc = -np.inf

#     best_epoch = 0

#     epochs_without_improvement = 0

#     history = []

#     # --------------------------------------------------------
#     # Training.
#     # --------------------------------------------------------

#     print()
#     print(
#         "TRAINING START"
#     )
#     print()

#     for epoch in range(
#         1,
#         NUM_EPOCHS + 1,
#     ):

#         train_loss, train_metrics = (
#             train_one_epoch(
#                 model,
#                 train_loader,
#                 criterion,
#                 optimizer,
#                 device,
#             )
#         )

#         (
#             validation_loss,
#             validation_metrics,
#             _,
#             _,
#             _,
#         ) = evaluate(
#             model,
#             validation_loader,
#             criterion,
#             device,
#         )

#         validation_auc = (
#             validation_metrics["auc"]
#         )

#         if not np.isfinite(
#             validation_auc
#         ):

#             raise RuntimeError(
#                 f"Outer fold {outer_fold}: "
#                 f"validation AUC is not finite "
#                 f"at epoch {epoch}."
#             )

#         scheduler.step(
#             validation_auc
#         )

#         current_lr = (
#             optimizer.param_groups[0]["lr"]
#         )

#         history_row = {
#             "epoch":
#                 epoch,

#             "learning_rate":
#                 current_lr,

#             "train_loss":
#                 train_loss,

#             "train_accuracy":
#                 train_metrics[
#                     "accuracy"
#                 ],

#             "train_balanced_accuracy":
#                 train_metrics[
#                     "balanced_accuracy"
#                 ],

#             "train_precision":
#                 train_metrics[
#                     "precision"
#                 ],

#             "train_recall":
#                 train_metrics[
#                     "recall"
#                 ],

#             "train_f1":
#                 train_metrics[
#                     "f1"
#                 ],

#             "train_auc":
#                 train_metrics[
#                     "auc"
#                 ],

#             "validation_loss":
#                 validation_loss,

#             "validation_accuracy":
#                 validation_metrics[
#                     "accuracy"
#                 ],

#             "validation_balanced_accuracy":
#                 validation_metrics[
#                     "balanced_accuracy"
#                 ],

#             "validation_precision":
#                 validation_metrics[
#                     "precision"
#                 ],

#             "validation_recall":
#                 validation_metrics[
#                     "recall"
#                 ],

#             "validation_f1":
#                 validation_metrics[
#                     "f1"
#                 ],

#             "validation_auc":
#                 validation_metrics[
#                     "auc"
#                 ],
#         }

#         history.append(
#             history_row
#         )

#         print(
#             f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
#             f"Train Loss {train_loss:.4f} | "
#             f"Val Loss {validation_loss:.4f} | "
#             f"Train AUC {train_metrics['auc']:.4f} | "
#             f"Val AUC {validation_auc:.4f}"
#         )

#         # ----------------------------------------------------
#         # Best model by inner validation AUC.
#         # ----------------------------------------------------

#         if (
#             validation_auc
#             > best_validation_auc
#         ):

#             best_validation_auc = (
#                 validation_auc
#             )

#             best_epoch = epoch

#             epochs_without_improvement = 0

#             torch.save(
#                 {
#                     "epoch":
#                         epoch,

#                     "model_state_dict":
#                         model.state_dict(),

#                     "optimizer_state_dict":
#                         optimizer.state_dict(),

#                     "validation_auc":
#                         validation_auc,

#                     "random_seed":
#                         RANDOM_SEED,

#                     "outer_fold":
#                         outer_fold,

#                     "inner_fold":
#                         SELECTED_INNER_FOLD,

#                     "input_shape":
#                         EXPECTED_TENSOR_SHAPE,

#                     "dropout":
#                         DROPOUT,

#                     "batch_size":
#                         BATCH_SIZE,

#                     "learning_rate":
#                         LEARNING_RATE,

#                     "weight_decay":
#                         WEIGHT_DECAY,

#                     "pos_weight":
#                         pos_weight_value,
#                 },
#                 best_model_file,
#             )

#             print(
#                 f"  -> Best model saved "
#                 f"(Val AUC={validation_auc:.4f})"
#             )

#         else:

#             epochs_without_improvement += 1

#         # ----------------------------------------------------
#         # Early stopping.
#         # ----------------------------------------------------

#         if (
#             epochs_without_improvement
#             >= EARLY_STOPPING_PATIENCE
#         ):

#             print(
#                 "  -> Early stopping."
#             )

#             break

#     # --------------------------------------------------------
#     # Save training history.
#     # --------------------------------------------------------

#     history_df = pd.DataFrame(
#         history
#     )

#     history_df.to_csv(
#         history_file,
#         index=False,
#     )

#     if not best_model_file.exists():

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "best model was not saved."
#         )

#     # --------------------------------------------------------
#     # Load best model.
#     # --------------------------------------------------------

#     checkpoint = torch.load(
#         best_model_file,
#         map_location=device,
#         weights_only=False,
#     )

#     model.load_state_dict(
#         checkpoint[
#             "model_state_dict"
#         ]
#     )

#     # --------------------------------------------------------
#     # OUTER TEST EVALUATION
#     #
#     # Only now is the outer test set evaluated.
#     # --------------------------------------------------------

#     (
#         test_loss,
#         test_metrics,
#         test_labels,
#         test_probabilities,
#         test_subjects,
#     ) = evaluate(
#         model,
#         test_loader,
#         criterion,
#         device,
#     )

#     print()
#     print(
#         f"OUTER FOLD {outer_fold} TEST RESULT"
#     )

#     print(
#         f"  AUC                : "
#         f"{test_metrics['auc']:.6f}"
#     )

#     print(
#         f"  Accuracy           : "
#         f"{test_metrics['accuracy']:.6f}"
#     )

#     print(
#         f"  Balanced Accuracy  : "
#         f"{test_metrics['balanced_accuracy']:.6f}"
#     )

#     print(
#         f"  Precision          : "
#         f"{test_metrics['precision']:.6f}"
#     )

#     print(
#         f"  Sensitivity        : "
#         f"{test_metrics['recall']:.6f}"
#     )

#     print(
#         f"  F1                 : "
#         f"{test_metrics['f1']:.6f}"
#     )

#     # --------------------------------------------------------
#     # Source lookup.
#     # --------------------------------------------------------

#     source_lookup = (
#         outer_test_df
#         .set_index(
#             "subject_id"
#         )[
#             "acquisition_source"
#         ]
#     )

#     source_values = [
#         source_lookup.loc[
#             subject_id
#         ]
#         for subject_id in test_subjects
#     ]

#     # --------------------------------------------------------
#     # Predictions.
#     # --------------------------------------------------------

#     prediction_df = pd.DataFrame(
#         {
#             "subject_id":
#                 test_subjects,

#             "label":
#                 test_labels,

#             "probability":
#                 test_probabilities,

#             "prediction":
#                 (
#                     np.asarray(
#                         test_probabilities
#                     )
#                     >= THRESHOLD
#                 ).astype(int),

#             "outer_fold":
#                 outer_fold,

#             "inner_fold":
#                 SELECTED_INNER_FOLD,

#             "acquisition_source":
#                 source_values,
#         }
#     )

#     # --------------------------------------------------------
#     # Prediction validation.
#     # --------------------------------------------------------

#     if len(prediction_df) != (
#         len(outer_test_df)
#     ):

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "prediction count does not match "
#             "outer test subject count."
#         )

#     if prediction_df[
#         "subject_id"
#     ].duplicated().any():

#         raise RuntimeError(
#             f"Outer fold {outer_fold}: "
#             "duplicate subject IDs in predictions."
#         )

#     prediction_df.to_csv(
#         prediction_file,
#         index=False,
#     )

#     # --------------------------------------------------------
#     # Fold metrics.
#     # --------------------------------------------------------

#     fold_metrics = {
#         "outer_fold":
#             outer_fold,

#         "inner_fold":
#             SELECTED_INNER_FOLD,

#         "inner_train_subjects":
#             len(inner_train_df),

#         "inner_validation_subjects":
#             len(inner_validation_df),

#         "outer_test_subjects":
#             len(outer_test_df),

#         "best_epoch":
#             best_epoch,

#         "best_validation_auc":
#             best_validation_auc,

#         "test_loss":
#             test_loss,

#         "test_auc":
#             test_metrics[
#                 "auc"
#             ],

#         "test_accuracy":
#             test_metrics[
#                 "accuracy"
#             ],

#         "test_balanced_accuracy":
#             test_metrics[
#                 "balanced_accuracy"
#             ],

#         "test_precision":
#             test_metrics[
#                 "precision"
#             ],

#         "test_recall":
#             test_metrics[
#                 "recall"
#             ],

#         "test_f1":
#             test_metrics[
#                 "f1"
#             ],

#         "parameter_count":
#             parameter_count,

#         "pos_weight":
#             pos_weight_value,
#     }

#     # --------------------------------------------------------
#     # Fold summary.
#     # --------------------------------------------------------

#     summary_lines = [
#         (
#             "NeuroFusion AI - sMRI Nested CV "
#             f"Outer Fold {outer_fold}"
#         ),

#         "=" * 72,

#         "",

#         f"Outer fold              : "
#         f"{outer_fold}",

#         f"Selected inner fold     : "
#         f"{SELECTED_INNER_FOLD}",

#         f"Random seed             : "
#         f"{RANDOM_SEED}",

#         f"Device                  : "
#         f"{device}",

#         f"Input tensor shape      : "
#         f"{EXPECTED_TENSOR_SHAPE}",

#         "",

#         f"Inner train subjects    : "
#         f"{len(inner_train_df)}",

#         f"Inner validation        : "
#         f"{len(inner_validation_df)}",

#         f"Outer test subjects     : "
#         f"{len(outer_test_df)}",

#         "",

#         f"Best epoch              : "
#         f"{best_epoch}",

#         f"Best validation AUC     : "
#         f"{best_validation_auc:.6f}",

#         "",

#         f"Outer test loss         : "
#         f"{test_loss:.6f}",

#         f"Outer test AUC          : "
#         f"{test_metrics['auc']:.6f}",

#         f"Outer test accuracy     : "
#         f"{test_metrics['accuracy']:.6f}",

#         f"Outer test balanced acc : "
#         f"{test_metrics['balanced_accuracy']:.6f}",

#         f"Outer test precision    : "
#         f"{test_metrics['precision']:.6f}",

#         f"Outer test sensitivity  : "
#         f"{test_metrics['recall']:.6f}",

#         f"Outer test F1           : "
#         f"{test_metrics['f1']:.6f}",

#         "",

#         "Outer test subjects were not used "
#         "for model selection.",

#         "",

#         "FINAL STATUS: PASS",
#     ]

#     summary_file.write_text(
#         "\n".join(
#             summary_lines
#         ),
#         encoding="utf-8",
#     )

#     return (
#         prediction_df,
#         fold_metrics,
#     )


# # ============================================================
# # MAIN
# # ============================================================

# def main() -> None:

#     print("=" * 72)
#     print(
#         "NeuroFusion AI - sMRI 3D CNN NESTED CV TRAINING"
#     )
#     print("=" * 72)
#     print()

#     # --------------------------------------------------------
#     # Reproducibility.
#     # --------------------------------------------------------

#     set_seed(
#         RANDOM_SEED
#     )

#     # --------------------------------------------------------
#     # Device.
#     # --------------------------------------------------------

#     device = torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "cpu"
#     )

#     print(
#         f"Device       : "
#         f"{device}"
#     )

#     # --------------------------------------------------------
#     # Required files.
#     # --------------------------------------------------------

#     if not NESTED_ASSIGNMENTS_FILE.exists():

#         raise FileNotFoundError(
#             "Nested CV assignments not found:\n"
#             f"{NESTED_ASSIGNMENTS_FILE}"
#         )

#     if not IMAGE_DIR.exists():

#         raise FileNotFoundError(
#             "CNN image directory not found:\n"
#             f"{IMAGE_DIR}"
#         )

#     nested_df = pd.read_csv(
#         NESTED_ASSIGNMENTS_FILE
#     )

#     print(
#         f"Nested assignments : "
#         f"{NESTED_ASSIGNMENTS_FILE}"
#     )

#     print(
#         f"Assignment rows     : "
#         f"{len(nested_df)}"
#     )

#     # --------------------------------------------------------
#     # Validate the complete assignment file.
#     # --------------------------------------------------------

#     validate_nested_assignments(
#         nested_df
#     )

#     print(
#         "Nested assignment validation: PASS"
#     )

#     # --------------------------------------------------------
#     # Training output directory.
#     # --------------------------------------------------------

#     TRAINING_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     # --------------------------------------------------------
#     # Train each outer fold.
#     # --------------------------------------------------------

#     all_predictions = []

#     all_fold_metrics = []

#     for outer_fold in range(
#         1,
#         OUTER_FOLDS + 1,
#     ):

#         outer_df = nested_df[
#             nested_df["outer_fold"]
#             == outer_fold
#         ].copy()

#         if outer_df.empty:

#             raise RuntimeError(
#                 f"Outer fold {outer_fold} "
#                 "is empty."
#             )

#         (
#             prediction_df,
#             fold_metrics,
#         ) = train_outer_fold(
#             outer_fold,
#             outer_df,
#             device,
#         )

#         all_predictions.append(
#             prediction_df
#         )

#         all_fold_metrics.append(
#             fold_metrics
#         )

#     # --------------------------------------------------------
#     # Combine outer-test predictions.
#     #
#     # Every subject must appear exactly once because each
#     # subject belongs to exactly one outer test fold.
#     # --------------------------------------------------------

#     oof_df = pd.concat(
#         all_predictions,
#         ignore_index=True,
#     )

#     duplicate_subjects = (
#         oof_df.loc[
#             oof_df[
#                 "subject_id"
#             ].duplicated(
#                 keep=False
#             ),
#             "subject_id",
#         ]
#         .astype(str)
#         .unique()
#         .tolist()
#     )

#     if duplicate_subjects:

#         raise RuntimeError(
#             "Duplicate subjects in OOF predictions: "
#             + ", ".join(
#                 sorted(
#                     duplicate_subjects
#                 )
#             )
#         )

#     if len(oof_df) != (
#         EXPECTED_SUBJECTS
#     ):

#         raise RuntimeError(
#             f"Expected {EXPECTED_SUBJECTS} "
#             f"OOF subjects; found "
#             f"{len(oof_df)}"
#         )

#     # --------------------------------------------------------
#     # Overall OOF metrics.
#     # --------------------------------------------------------

#     overall_metrics = calculate_metrics(
#         oof_df[
#             "label"
#         ].to_numpy(),

#         oof_df[
#             "probability"
#         ].to_numpy(),
#     )

#     # --------------------------------------------------------
#     # Fold metrics.
#     # --------------------------------------------------------

#     fold_metrics_df = pd.DataFrame(
#         all_fold_metrics
#     )

#     # --------------------------------------------------------
#     # Fold AUC statistics.
#     # --------------------------------------------------------

#     fold_auc_mean = (
#         fold_metrics_df[
#             "test_auc"
#         ].mean()
#     )

#     fold_auc_std = (
#         fold_metrics_df[
#             "test_auc"
#         ].std(
#             ddof=1
#         )
#     )

#     # --------------------------------------------------------
#     # Save OOF predictions.
#     # --------------------------------------------------------

#     oof_df = oof_df.sort_values(
#         "subject_id"
#     ).reset_index(
#         drop=True
#     )

#     oof_df.to_csv(
#         OOF_PREDICTIONS_FILE,
#         index=False,
#     )

#     # --------------------------------------------------------
#     # Save fold metrics.
#     # --------------------------------------------------------

#     fold_metrics_df.to_csv(
#         FOLD_METRICS_FILE,
#         index=False,
#     )

#     # --------------------------------------------------------
#     # Overall summary.
#     # --------------------------------------------------------

#     summary_lines = [
#         "NeuroFusion AI - sMRI Nested CV Summary",

#         "=" * 72,

#         "",

#         f"Subjects evaluated      : "
#         f"{len(oof_df)}",

#         f"Outer folds             : "
#         f"{OUTER_FOLDS}",

#         f"Inner folds prepared    : "
#         f"{INNER_FOLDS}",

#         f"Selected inner fold     : "
#         f"{SELECTED_INNER_FOLD}",

#         f"Random seed             : "
#         f"{RANDOM_SEED}",

#         f"Device                  : "
#         f"{device}",

#         "",

#         "Overall out-of-fold metrics:",

#         f"  AUC                  : "
#         f"{overall_metrics['auc']:.6f}",

#         f"  Accuracy             : "
#         f"{overall_metrics['accuracy']:.6f}",

#         f"  Balanced Accuracy    : "
#         f"{overall_metrics['balanced_accuracy']:.6f}",

#         f"  Precision            : "
#         f"{overall_metrics['precision']:.6f}",

#         f"  Sensitivity/Recall   : "
#         f"{overall_metrics['recall']:.6f}",

#         f"  F1                   : "
#         f"{overall_metrics['f1']:.6f}",

#         "",

#         "Outer-fold AUC:",

#         f"  Mean                 : "
#         f"{fold_auc_mean:.6f}",

#         f"  Std                  : "
#         f"{fold_auc_std:.6f}",

#         "",

#         "Fold results:",
#     ]

#     for _, row in (
#         fold_metrics_df
#         .sort_values(
#             "outer_fold"
#         )
#         .iterrows()
#     ):

#         summary_lines.append(
#             f"  Fold "
#             f"{int(row['outer_fold'])}: "
#             f"AUC="
#             f"{row['test_auc']:.6f}, "
#             f"balanced_acc="
#             f"{row['test_balanced_accuracy']:.6f}, "
#             f"best_epoch="
#             f"{int(row['best_epoch'])}"
#         )

#     summary_lines.extend(
#         [
#             "",

#             "Evaluation protocol:",

#             "  Subject-level nested CV.",

#             "  Outer test subjects were never used "
#             "for model selection.",

#             "  Inner validation AUC determined "
#             "the best epoch.",

#             "  All 172 subjects were evaluated exactly "
#             "once in an outer test fold.",

#             "",

#             "Baseline experiment protection:",

#             "  Existing 70/15/15 split: UNMODIFIED.",

#             "  Existing baseline model: UNMODIFIED.",

#             "  Existing outer CV assignments: READ ONLY.",

#             "  Existing nested CV assignments: READ ONLY.",

#             "",

#             f"OOF predictions : "
#             f"{OOF_PREDICTIONS_FILE}",

#             f"Fold metrics    : "
#             f"{FOLD_METRICS_FILE}",

#             f"Summary         : "
#             f"{CV_SUMMARY_FILE}",

#             "",

#             "FINAL STATUS: PASS",
#         ]
#     )

#     CV_SUMMARY_FILE.write_text(
#         "\n".join(
#             summary_lines
#         ),
#         encoding="utf-8",
#     )

#     # --------------------------------------------------------
#     # Final console output.
#     # --------------------------------------------------------

#     print()
#     print("=" * 72)
#     print(
#         "NESTED CV TRAINING COMPLETE"
#     )
#     print("=" * 72)
#     print()

#     print(
#         f"Subjects evaluated : "
#         f"{len(oof_df)}"
#     )

#     print(
#         f"Overall OOF AUC    : "
#         f"{overall_metrics['auc']:.6f}"
#     )

#     print(
#         f"Overall accuracy   : "
#         f"{overall_metrics['accuracy']:.6f}"
#     )

#     print(
#         f"Overall balanced acc: "
#         f"{overall_metrics['balanced_accuracy']:.6f}"
#     )

#     print(
#         f"Fold AUC mean      : "
#         f"{fold_auc_mean:.6f}"
#     )

#     print(
#         f"Fold AUC std       : "
#         f"{fold_auc_std:.6f}"
#     )

#     print()

#     print(
#         f"OOF predictions : "
#         f"{OOF_PREDICTIONS_FILE}"
#     )

#     print(
#         f"Fold metrics    : "
#         f"{FOLD_METRICS_FILE}"
#     )

#     print(
#         f"Summary         : "
#         f"{CV_SUMMARY_FILE}"
#     )

#     print()
#     print(
#         "FINAL STATUS: PASS"
#     )


# if __name__ == "__main__":
#     main()






"""
NeuroFusion AI
sMRI 3D CNN — Full 5x4 Nested Cross-Validation

Protocol
--------
Outer CV:
    5 folds

Inner CV:
    4 folds within each outer training partition

For every outer fold:
    1. Train 4 inner models.
    2. Record validation AUC and validation loss by epoch.
    3. Select the epoch with the highest mean inner-validation AUC.
    4. Retrain a fresh model on ALL outer-training subjects
       for the selected number of epochs.
    5. Evaluate exactly once on the untouched outer-test set.

The outer-test subjects are never used for model selection.

Existing baseline/nested results are preserved.
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

CNN_DATASET_DIR = ROOT / "cnn_dataset"

IMAGE_DIR = CNN_DATASET_DIR / "images"

NESTED_CV_DIR = (
    CNN_DATASET_DIR
    / "cross_validation"
    / "nested"
)

# IMPORTANT:
# New output directory. Previous experiment is preserved.
TRAINING_DIR = (
    CNN_DATASET_DIR
    / "cross_validation"
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
    # Partition validation.
    # --------------------------------------------------------

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
    # Confirm outer test is excluded from model selection.
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
            f"Outer {outer_fold}, Inner {inner_fold}: "
            "partition does not cover the outer fold."
        )

    # --------------------------------------------------------
    # Class validation.
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

    # --------------------------------------------------------
    # Seed unique to this inner run.
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
    # Data.
    # --------------------------------------------------------

    train_loader = create_loader(
        train_df,
        shuffle=True,
    )

    validation_loader = create_loader(
        validation_df,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model.
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
    # Train inner model.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Median of the four independently selected inner epochs.
    #
    # This gives a robust training-duration estimate without
    # using the outer test set.
    # --------------------------------------------------------

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
    # Outer train/test.
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

    # --------------------------------------------------------
    # Verify complete outer partition.
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

    # --------------------------------------------------------
    # Both classes.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Unique final training seed.
    # --------------------------------------------------------

    final_seed = (
        RANDOM_SEED
        + outer_fold * 1000
    )

    set_seed(
        final_seed
    )

    # --------------------------------------------------------
    # Data.
    # --------------------------------------------------------

    train_loader = create_loader(
        outer_train_df,
        shuffle=True,
    )

    test_loader = create_loader(
        outer_test_df,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model.
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
    # Train for selected epoch.
    #
    # No outer-test information is used.
    # No outer validation set exists.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Outer test evaluation.
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
        f"Specificity-related "
        f"threshold metrics : "
        f"see predictions"
    )

    print(
        f"F1                  : "
        f"{test_metrics['f1']:.6f}"
    )

    # --------------------------------------------------------
    # Source information.
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
    # Predictions.
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
        TRAINING_DIR
        / f"outer_fold_{outer_fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model.
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

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

    # --------------------------------------------------------
    # Save training history.
    # --------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        fold_dir
        / "final_outer_training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions.
    # --------------------------------------------------------

    prediction_df.to_csv(
        fold_dir
        / "outer_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Fold metrics.
    # --------------------------------------------------------

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

    print(
        "=" * 72
    )

    print(
        "NeuroFusion AI - FULL 5x4 NESTED CV"
    )

    print(
        "=" * 72
    )

    print()

    set_seed(
        RANDOM_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load assignments.
    # --------------------------------------------------------

    if not NESTED_ASSIGNMENTS_FILE.exists():

        raise FileNotFoundError(
            f"Nested assignments not found:\n"
            f"{NESTED_ASSIGNMENTS_FILE}"
        )

    nested_df = pd.read_csv(
        NESTED_ASSIGNMENTS_FILE
    )

    print(
        f"Assignment rows: "
        f"{len(nested_df)}"
    )

    # --------------------------------------------------------
    # Validate complete assignment structure.
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
        - set(nested_df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing nested assignment columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    unique_subjects = set(
        nested_df[
            "subject_id"
        ].astype(str)
    )

    if len(unique_subjects) != (
        EXPECTED_SUBJECTS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} "
            f"subjects; found "
            f"{len(unique_subjects)}."
        )

    # --------------------------------------------------------
    # Validate every outer/inner partition.
    # --------------------------------------------------------

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        outer_df = nested_df[
            nested_df["outer_fold"]
            == outer_fold
        ].copy()

        if outer_df.empty:

            raise RuntimeError(
                f"Outer fold {outer_fold} is empty."
            )

        outer_subjects = set(
            outer_df[
                "subject_id"
            ].astype(str)
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
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "train/validation overlap."
                )

            if train_ids.intersection(
                test_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "train/test overlap."
                )

            if validation_ids.intersection(
                test_ids
            ):

                raise RuntimeError(
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "validation/test overlap."
                )

            if (
                train_ids
                | validation_ids
                | test_ids
            ) != outer_subjects:

                raise RuntimeError(
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "partition does not cover "
                    "outer subjects exactly."
                )

            if set(
                split_df.loc[
                    split_df["role"]
                    == "INNER_TRAIN",
                    "label",
                ].astype(int)
            ) != {0, 1}:

                raise RuntimeError(
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "inner training class problem."
                )

            if set(
                split_df.loc[
                    split_df["role"]
                    == "INNER_VALIDATION",
                    "label",
                ].astype(int)
            ) != {0, 1}:

                raise RuntimeError(
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "inner validation class problem."
                )

            if set(
                split_df.loc[
                    split_df["role"]
                    == "OUTER_TEST",
                    "label",
                ].astype(int)
            ) != {0, 1}:

                raise RuntimeError(
                    f"Outer {outer_fold}, Inner "
                    f"{inner_fold}: "
                    "outer test class problem."
                )

    print(
        "Nested assignment validation: PASS"
    )

    # --------------------------------------------------------
    # Create new output directory.
    # --------------------------------------------------------

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

        outer_df = nested_df[
            nested_df["outer_fold"]
            == outer_fold
        ].copy()

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
                device,
            )

            all_inner_results.append(
                result
            )

            all_inner_histories.append(
                history_df
            )

    # --------------------------------------------------------
    # Save inner results.
    # --------------------------------------------------------

    inner_results_df = pd.DataFrame(
        all_inner_results
    )

    inner_results_df.to_csv(
        INNER_RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Select epoch separately for every outer fold.
    # --------------------------------------------------------

    epoch_selection_rows = []

    selected_epochs = {}

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        selected_epoch = select_epoch(
            all_inner_results,
            outer_fold,
        )

        selected_epochs[
            outer_fold
        ] = selected_epoch

        fold_results = [
            result
            for result in all_inner_results
            if result[
                "outer_fold"
            ] == outer_fold
        ]

        epoch_selection_rows.append(
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
                        in sorted(
                            fold_results,
                            key=lambda x:
                                x["inner_fold"],
                        )
                    ),

                "inner_best_auc":
                    ",".join(
                        f"{result['best_validation_auc']:.6f}"
                        for result
                        in sorted(
                            fold_results,
                            key=lambda x:
                                x["inner_fold"],
                        )
                    ),

                "selected_epoch":
                    selected_epoch,
            }
        )

        print()
        print(
            f"Outer fold {outer_fold}: "
            f"selected epoch = "
            f"{selected_epoch}"
        )

    epoch_selection_df = pd.DataFrame(
        epoch_selection_rows
    )

    epoch_selection_df.to_csv(
        EPOCH_SELECTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL OUTER TRAINING.
    # --------------------------------------------------------

    all_predictions = []

    all_fold_metrics = []

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):

        outer_df = nested_df[
            nested_df["outer_fold"]
            == outer_fold
        ].copy()

        (
            prediction_df,
            fold_metrics,
        ) = train_final_outer_model(
            outer_fold,
            outer_df,
            selected_epochs[
                outer_fold
            ],
            device,
        )

        all_predictions.append(
            prediction_df
        )

        all_fold_metrics.append(
            fold_metrics
        )

    # --------------------------------------------------------
    # OOF predictions.
    # --------------------------------------------------------

    oof_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    if oof_df[
        "subject_id"
    ].duplicated().any():

        duplicates = (
            oof_df.loc[
                oof_df[
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
            "Duplicate OOF subjects: "
            + ", ".join(
                sorted(
                    duplicates
                )
            )
        )

    if len(oof_df) != (
        EXPECTED_SUBJECTS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} "
            f"OOF predictions; found "
            f"{len(oof_df)}."
        )

    # --------------------------------------------------------
    # Overall OOF metrics.
    # --------------------------------------------------------

    overall_metrics = calculate_metrics(
        oof_df[
            "label"
        ].to_numpy(),

        oof_df[
            "probability"
        ].to_numpy(),
    )

    # --------------------------------------------------------
    # Fold metrics.
    # --------------------------------------------------------

    fold_metrics_df = pd.DataFrame(
        all_fold_metrics
    )

    fold_auc_mean = (
        fold_metrics_df[
            "test_auc"
        ].mean()
    )

    fold_auc_std = (
        fold_metrics_df[
            "test_auc"
        ].std(
            ddof=1
        )
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    oof_df = oof_df.sort_values(
        "subject_id"
    ).reset_index(
        drop=True
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

    summary_lines = [
        "NeuroFusion AI - FULL 5x4 NESTED CV",
        "=" * 72,
        "",
        f"Subjects evaluated       : {len(oof_df)}",
        f"Outer folds              : {OUTER_FOLDS}",
        f"Inner folds per outer    : {INNER_FOLDS}",
        f"Random seed              : {RANDOM_SEED}",
        f"Device                   : {device}",
        "",
        "Overall OOF metrics:",
        f"  AUC                    : "
        f"{overall_metrics['auc']:.6f}",
        f"  Accuracy               : "
        f"{overall_metrics['accuracy']:.6f}",
        f"  Balanced Accuracy      : "
        f"{overall_metrics['balanced_accuracy']:.6f}",
        f"  Precision              : "
        f"{overall_metrics['precision']:.6f}",
        f"  Sensitivity/Recall     : "
        f"{overall_metrics['recall']:.6f}",
        f"  F1                     : "
        f"{overall_metrics['f1']:.6f}",
        "",
        "Outer-fold AUC:",
        f"  Mean                   : "
        f"{fold_auc_mean:.6f}",
        f"  Std                    : "
        f"{fold_auc_std:.6f}",
        "",
        "Outer-fold results:",
    ]

    for _, row in (
        fold_metrics_df
        .sort_values(
            "outer_fold"
        )
        .iterrows()
    ):

        summary_lines.append(
            f"  Fold {int(row['outer_fold'])}: "
            f"AUC={row['test_auc']:.6f}, "
            f"balanced_acc="
            f"{row['test_balanced_accuracy']:.6f}, "
            f"selected_epoch="
            f"{int(row['selected_epoch'])}"
        )

    summary_lines.extend(
        [
            "",
            "Protocol:",
            "  4 inner folds used for model selection.",
            "  Selected epoch = median best epoch "
            "across the four inner folds.",
            "  Final outer model retrained from scratch "
            "on all outer-training subjects.",
            "  Outer test evaluated exactly once.",
            "  All 172 subjects evaluated exactly once "
            "as an outer-test subject.",
            "",
            "Previous experiments preserved.",
            "",
            f"Inner results: {INNER_RESULTS_FILE}",
            f"Epoch selection: {EPOCH_SELECTION_FILE}",
            f"OOF predictions: {OOF_PREDICTIONS_FILE}",
            f"Fold metrics: {FOLD_METRICS_FILE}",
            "",
            "FINAL STATUS: PASS",
        ]
    )

    CV_SUMMARY_FILE.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FULL 5x4 NESTED CV COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Subjects evaluated : "
        f"{len(oof_df)}"
    )

    print(
        f"Overall OOF AUC    : "
        f"{overall_metrics['auc']:.6f}"
    )

    print(
        f"Fold AUC mean      : "
        f"{fold_auc_mean:.6f}"
    )

    print(
        f"Fold AUC std       : "
        f"{fold_auc_std:.6f}"
    )

    print()

    print(
        f"Output directory:"
    )

    print(
        TRAINING_DIR
    )

    print()

    print(
        "FINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()