# from __future__ import annotations

# import json
# import random
# from pathlib import Path
# from statistics import median

# import joblib
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn

# from sklearn.metrics import (
#     accuracy_score,
#     balanced_accuracy_score,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     roc_auc_score,
# )
# from sklearn.preprocessing import StandardScaler

# from torch.utils.data import DataLoader, Dataset

# from smri_cnn_aal_fusion import SMRICNNAALFusion


# # ============================================================
# # CONFIGURATION
# # ============================================================

# SEED = 42

# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MANIFEST_PATH = (
#     PROJECT_ROOT
#     / "processed_data"
#     / "smri"
#     / "dataset"
#     / "smri_subject_manifest_labeled.csv"
# )

# NESTED_ASSIGNMENTS_PATH = (
#     PROJECT_ROOT
#     / "processed_data"
#     / "smri"
#     / "cnn_dataset"
#     / "cross_validation"
#     / "nested"
#     / "nested_cv_assignments.csv"
# )

# IMAGE_DIR = (
#     PROJECT_ROOT
#     / "processed_data"
#     / "smri"
#     / "cnn_dataset"
#     / "images"
# )

# OUTPUT_DIR = (
#     PROJECT_ROOT
#     / "processed_data"
#     / "smri"
#     / "cnn_dataset"
#     / "experiments"
#     / "exp05_cnn_aal_fusion"
#     / "training_full_5x4"
# )

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# BATCH_SIZE = 4
# MAX_EPOCHS = 50

# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-4

# DROPOUT = 0.30

# EARLY_STOPPING_PATIENCE = 10
# LR_PATIENCE = 4
# LR_FACTOR = 0.5

# NUM_WORKERS = 0


# # ============================================================
# # REPRODUCIBILITY
# # ============================================================

# def set_seed(seed: int) -> None:
#     """
#     Set all relevant random seeds.

#     Important:
#     pandas/NumPy fold IDs can be NumPy integer scalars.
#     Python's random.seed() expects a supported native type,
#     so we explicitly convert the seed to a Python int.
#     """

#     seed = int(seed)

#     random.seed(seed)
#     np.random.seed(seed)

#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # ============================================================
# # DATASET
# # ============================================================

# class FusionDataset(Dataset):
#     """
#     Subject-level multimodal dataset.

#     Each sample contains:

#         MRI tensor:
#             (1, 95, 117, 99)

#         AAL feature vector:
#             (116,)

#         Binary label:
#             0 = Control
#             1 = Schizophrenia

#         Subject ID
#     """

#     def __init__(
#         self,
#         records: pd.DataFrame,
#         aal_values: np.ndarray,
#     ):
#         if len(records) != len(aal_values):
#             raise ValueError(
#                 "Records and AAL array length mismatch."
#             )

#         self.records = records.reset_index(
#             drop=True
#         )

#         self.aal_values = aal_values.astype(
#             np.float32
#         )

#     def __len__(self) -> int:
#         return len(self.records)

#     def __getitem__(
#         self,
#         index: int,
#     ):
#         row = self.records.iloc[index]

#         subject_id = str(
#             row["subject_id"]
#         )

#         label = float(
#             row["label"]
#         )

#         image_path = (
#             IMAGE_DIR
#             / f"{subject_id}.npy"
#         )

#         if not image_path.exists():
#             raise FileNotFoundError(
#                 f"MRI tensor not found: {image_path}"
#             )

#         image = np.load(
#             image_path
#         ).astype(np.float32)

#         if image.shape != (
#             1,
#             95,
#             117,
#             99,
#         ):
#             raise ValueError(
#                 f"Unexpected MRI shape for "
#                 f"{subject_id}: {image.shape}"
#             )

#         if not np.isfinite(image).all():
#             raise ValueError(
#                 f"Non-finite MRI tensor: "
#                 f"{subject_id}"
#             )

#         aal = self.aal_values[index]

#         if aal.shape != (116,):
#             raise ValueError(
#                 f"Unexpected AAL shape for "
#                 f"{subject_id}: {aal.shape}"
#             )

#         if not np.isfinite(aal).all():
#             raise ValueError(
#                 f"Non-finite AAL values: "
#                 f"{subject_id}"
#             )

#         return (
#             torch.from_numpy(image),
#             torch.from_numpy(aal),
#             torch.tensor(
#                 [label],
#                 dtype=torch.float32,
#             ),
#             subject_id,
#         )


# # ============================================================
# # AAL LOADING
# # ============================================================

# def load_aal_matrix(
#     manifest: pd.DataFrame,
# ) -> tuple[np.ndarray, list[str]]:
#     """
#     Load exactly one 116-dimensional AAL vector
#     for every subject.

#     Metadata columns removed:

#         subject_id
#         acquisition_type
#         volume

#     The remaining 116 ROI columns must be identical
#     across all subjects.
#     """

#     rows = []
#     subjects = []

#     reference_columns = None

#     for _, row in manifest.iterrows():

#         subject_id = str(
#             row["subject_id"]
#         )

#         aal_path = Path(
#             row["aal_csv"]
#         )

#         if not aal_path.is_absolute():
#             aal_path = (
#                 PROJECT_ROOT
#                 / aal_path
#             )

#         if not aal_path.exists():
#             raise FileNotFoundError(
#                 f"AAL file not found: {aal_path}"
#             )

#         df = pd.read_csv(
#             aal_path
#         )

#         if len(df) != 1:
#             raise ValueError(
#                 f"{subject_id}: expected exactly "
#                 f"one AAL row, found {len(df)}"
#             )

#         feature_df = df.drop(
#             columns=[
#                 "subject_id",
#                 "acquisition_type",
#                 "volume",
#             ],
#             errors="ignore",
#         )

#         columns = list(
#             feature_df.columns
#         )

#         if reference_columns is None:
#             reference_columns = columns

#         if columns != reference_columns:
#             raise ValueError(
#                 f"AAL column mismatch for "
#                 f"{subject_id}"
#             )

#         values = (
#             feature_df.iloc[0]
#             .to_numpy(
#                 dtype=np.float32
#             )
#         )

#         if values.shape != (116,):
#             raise ValueError(
#                 f"{subject_id}: expected 116 AAL "
#                 f"features, found {values.shape}"
#             )

#         if not np.isfinite(values).all():
#             raise ValueError(
#                 f"{subject_id}: non-finite "
#                 f"AAL features"
#             )

#         rows.append(values)
#         subjects.append(subject_id)

#     matrix = np.vstack(
#         rows
#     )

#     if matrix.shape != (
#         len(manifest),
#         116,
#     ):
#         raise ValueError(
#             f"Unexpected AAL matrix shape: "
#             f"{matrix.shape}"
#         )

#     return matrix, subjects


# # ============================================================
# # MODEL TRAINING
# # ============================================================

# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     criterion: nn.Module,
# ) -> tuple[float, float]:

#     model.train()

#     total_loss = 0.0

#     all_labels = []
#     all_probs = []

#     for (
#         smri,
#         aal,
#         labels,
#         _,
#     ) in loader:

#         smri = smri.to(
#             DEVICE,
#             non_blocking=True,
#         )

#         aal = aal.to(
#             DEVICE,
#             non_blocking=True,
#         )

#         labels = labels.to(
#             DEVICE,
#             non_blocking=True,
#         )

#         optimizer.zero_grad(
#             set_to_none=True
#         )

#         logits = model(
#             smri,
#             aal,
#         )

#         loss = criterion(
#             logits,
#             labels,
#         )

#         loss.backward()

#         optimizer.step()

#         total_loss += (
#             loss.item()
#             * smri.size(0)
#         )

#         probs = torch.sigmoid(
#             logits
#         )

#         all_labels.extend(
#             labels.detach()
#             .cpu()
#             .numpy()
#             .reshape(-1)
#             .tolist()
#         )

#         all_probs.extend(
#             probs.detach()
#             .cpu()
#             .numpy()
#             .reshape(-1)
#             .tolist()
#         )

#     average_loss = (
#         total_loss
#         / len(loader.dataset)
#     )

#     if len(
#         set(all_labels)
#     ) >= 2:

#         auc = roc_auc_score(
#             all_labels,
#             all_probs,
#         )

#     else:

#         auc = float("nan")

#     return (
#         average_loss,
#         auc,
#     )


# # ============================================================
# # VALIDATION / EVALUATION
# # ============================================================

# @torch.no_grad()
# def evaluate_loader(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
# ) -> tuple[
#     float,
#     float,
#     np.ndarray,
#     np.ndarray,
# ]:

#     model.eval()

#     total_loss = 0.0

#     all_labels = []
#     all_probs = []

#     for (
#         smri,
#         aal,
#         labels,
#         _,
#     ) in loader:

#         smri = smri.to(
#             DEVICE,
#             non_blocking=True,
#         )

#         aal = aal.to(
#             DEVICE,
#             non_blocking=True,
#         )

#         labels = labels.to(
#             DEVICE,
#             non_blocking=True,
#         )

#         logits = model(
#             smri,
#             aal,
#         )

#         loss = criterion(
#             logits,
#             labels,
#         )

#         total_loss += (
#             loss.item()
#             * smri.size(0)
#         )

#         probs = torch.sigmoid(
#             logits
#         )

#         all_labels.extend(
#             labels.cpu()
#             .numpy()
#             .reshape(-1)
#             .tolist()
#         )

#         all_probs.extend(
#             probs.cpu()
#             .numpy()
#             .reshape(-1)
#             .tolist()
#         )

#     average_loss = (
#         total_loss
#         / len(loader.dataset)
#     )

#     labels_array = np.asarray(
#         all_labels,
#         dtype=np.float32,
#     )

#     probs_array = np.asarray(
#         all_probs,
#         dtype=np.float32,
#     )

#     if len(
#         np.unique(labels_array)
#     ) >= 2:

#         auc = roc_auc_score(
#             labels_array,
#             probs_array,
#         )

#     else:

#         auc = float("nan")

#     return (
#         average_loss,
#         auc,
#         labels_array,
#         probs_array,
#     )


# # ============================================================
# # DATALOADER
# # ============================================================

# def make_loader(
#     records: pd.DataFrame,
#     aal_values: np.ndarray,
#     shuffle: bool,
# ) -> DataLoader:

#     dataset = FusionDataset(
#         records,
#         aal_values,
#     )

#     return DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=shuffle,
#         num_workers=NUM_WORKERS,
#         pin_memory=torch.cuda.is_available(),
#     )


# # ============================================================
# # POSITIVE CLASS WEIGHT
# # ============================================================

# def get_pos_weight(
#     labels: np.ndarray,
# ) -> float:

#     positives = int(
#         np.sum(labels == 1)
#     )

#     negatives = int(
#         np.sum(labels == 0)
#     )

#     if positives == 0:
#         raise ValueError(
#             "No positive samples in training fold."
#         )

#     return negatives / positives


# # ============================================================
# # INNER-FOLD EPOCH SELECTION
# # ============================================================

# def select_epoch_for_outer_fold(
#     outer_fold: int,
#     assignments: pd.DataFrame,
#     manifest: pd.DataFrame,
#     aal_raw: np.ndarray,
#     subject_to_aal_index: dict[str, int],
# ) -> tuple[
#     int,
#     list[int],
#     list[dict],
# ]:

#     outer_fold = int(
#         outer_fold
#     )

#     inner_best_epochs = []
#     inner_results = []

#     inner_folds = sorted(
#         assignments.loc[
#             (
#                 assignments["outer_fold"]
#                 == outer_fold
#             )
#             & (
#                 assignments["role"]
#                 == "INNER_TRAIN"
#             ),
#             "inner_fold",
#         ].unique()
#     )

#     if len(inner_folds) != 4:
#         raise ValueError(
#             f"Outer fold {outer_fold}: expected "
#             f"4 inner folds, found {len(inner_folds)}"
#         )

#     for inner_fold_value in inner_folds:

#         inner_fold = int(
#             inner_fold_value
#         )

#         inner_assign = assignments[
#             (
#                 assignments["outer_fold"]
#                 == outer_fold
#             )
#             & (
#                 assignments["inner_fold"]
#                 == inner_fold
#             )
#         ]

#         train_ids = (
#             inner_assign.loc[
#                 inner_assign["role"]
#                 == "INNER_TRAIN",
#                 "subject_id",
#             ]
#             .astype(str)
#             .tolist()
#         )

#         val_ids = (
#             inner_assign.loc[
#                 inner_assign["role"]
#                 == "INNER_VALIDATION",
#                 "subject_id",
#             ]
#             .astype(str)
#             .tolist()
#         )

#         train_records = manifest[
#             manifest["subject_id"]
#             .astype(str)
#             .isin(train_ids)
#         ].copy()

#         val_records = manifest[
#             manifest["subject_id"]
#             .astype(str)
#             .isin(val_ids)
#         ].copy()

#         train_records = (
#             train_records
#             .set_index("subject_id")
#             .loc[train_ids]
#             .reset_index()
#         )

#         val_records = (
#             val_records
#             .set_index("subject_id")
#             .loc[val_ids]
#             .reset_index()
#         )

#         train_indices = [
#             subject_to_aal_index[sid]
#             for sid in train_records[
#                 "subject_id"
#             ].astype(str)
#         ]

#         val_indices = [
#             subject_to_aal_index[sid]
#             for sid in val_records[
#                 "subject_id"
#             ].astype(str)
#         ]

#         train_aal_raw = aal_raw[
#             train_indices
#         ]

#         val_aal_raw = aal_raw[
#             val_indices
#         ]

#         # ----------------------------------------------------
#         # CRITICAL LEAKAGE CONTROL
#         # ----------------------------------------------------
#         # Fit the scaler ONLY on inner-training data.
#         # Validation data is transformed but never used to
#         # fit the scaler.
#         # ----------------------------------------------------

#         scaler = StandardScaler()

#         train_aal = scaler.fit_transform(
#             train_aal_raw
#         )

#         val_aal = scaler.transform(
#             val_aal_raw
#         )

#         train_labels = (
#             train_records["label"]
#             .to_numpy(
#                 dtype=np.float32
#             )
#         )

#         pos_weight_value = get_pos_weight(
#             train_labels
#         )

#         criterion = nn.BCEWithLogitsLoss(
#             pos_weight=torch.tensor(
#                 [pos_weight_value],
#                 dtype=torch.float32,
#                 device=DEVICE,
#             )
#         )

#         train_loader = make_loader(
#             train_records,
#             train_aal,
#             shuffle=True,
#         )

#         val_loader = make_loader(
#             val_records,
#             val_aal,
#             shuffle=False,
#         )

#         # ----------------------------------------------------
#         # FIXED NATIVE PYTHON SEED
#         # ----------------------------------------------------

#         set_seed(
#             int(
#                 SEED
#                 + int(outer_fold) * 100
#                 + int(inner_fold)
#             )
#         )

#         model = SMRICNNAALFusion(
#             aal_features=116,
#             dropout=DROPOUT,
#         ).to(DEVICE)

#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=LEARNING_RATE,
#             weight_decay=WEIGHT_DECAY,
#         )

#         scheduler = (
#             torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer,
#                 mode="max",
#                 factor=LR_FACTOR,
#                 patience=LR_PATIENCE,
#             )
#         )

#         best_auc = -np.inf
#         best_epoch = 1
#         best_state = None

#         epochs_without_improvement = 0

#         for epoch in range(
#             1,
#             MAX_EPOCHS + 1,
#         ):

#             train_loss, train_auc = (
#                 train_one_epoch(
#                     model,
#                     train_loader,
#                     optimizer,
#                     criterion,
#                 )
#             )

#             (
#                 val_loss,
#                 val_auc,
#                 _,
#                 _,
#             ) = evaluate_loader(
#                 model,
#                 val_loader,
#                 criterion,
#             )

#             scheduler.step(
#                 val_auc
#                 if np.isfinite(val_auc)
#                 else 0.0
#             )

#             if (
#                 np.isfinite(val_auc)
#                 and val_auc > best_auc
#             ):

#                 best_auc = val_auc
#                 best_epoch = epoch

#                 best_state = {
#                     key: value.detach()
#                     .cpu()
#                     .clone()
#                     for key, value
#                     in model.state_dict().items()
#                 }

#                 epochs_without_improvement = 0

#             else:

#                 epochs_without_improvement += 1

#             if (
#                 epochs_without_improvement
#                 >= EARLY_STOPPING_PATIENCE
#             ):
#                 break

#         if best_state is not None:

#             model.load_state_dict(
#                 best_state
#             )

#         result = {
#             "outer_fold": int(
#                 outer_fold
#             ),
#             "inner_fold": int(
#                 inner_fold
#             ),
#             "best_epoch": int(
#                 best_epoch
#             ),
#             "best_validation_auc": float(
#                 best_auc
#             ),
#             "train_subjects": len(
#                 train_records
#             ),
#             "validation_subjects": len(
#                 val_records
#             ),
#             "pos_weight": float(
#                 pos_weight_value
#             ),
#         }

#         inner_results.append(
#             result
#         )

#         inner_best_epochs.append(
#             int(best_epoch)
#         )

#         print(
#             f"  Inner fold {inner_fold}: "
#             f"best epoch={best_epoch}, "
#             f"validation AUC={best_auc:.6f}"
#         )

#         del model
#         del optimizer
#         del scheduler

#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # Median is used to avoid one unusually high/low
#     # inner-fold epoch dominating the selected epoch.
#     selected_epoch = int(
#         round(
#             median(
#                 inner_best_epochs
#             )
#         )
#     )

#     return (
#         selected_epoch,
#         inner_best_epochs,
#         inner_results,
#     )


# # ============================================================
# # OUTER-FOLD TRAINING
# # ============================================================

# def train_outer_fold(
#     outer_fold: int,
#     selected_epoch: int,
#     assignments: pd.DataFrame,
#     manifest: pd.DataFrame,
#     aal_raw: np.ndarray,
#     subject_to_aal_index: dict[str, int],
# ) -> tuple[
#     pd.DataFrame,
#     dict,
# ]:

#     outer_fold = int(
#         outer_fold
#     )

#     selected_epoch = int(
#         selected_epoch
#     )

#     outer_assign = assignments[
#         assignments["outer_fold"]
#         == outer_fold
#     ]

#     train_ids = (
#         outer_assign.loc[
#             outer_assign["role"]
#             != "OUTER_TEST",
#             "subject_id",
#         ]
#         .astype(str)
#         .unique()
#         .tolist()
#     )

#     test_ids = (
#         outer_assign.loc[
#             outer_assign["role"]
#             == "OUTER_TEST",
#             "subject_id",
#         ]
#         .astype(str)
#         .tolist()
#     )

#     train_records = manifest[
#         manifest["subject_id"]
#         .astype(str)
#         .isin(train_ids)
#     ].copy()

#     test_records = manifest[
#         manifest["subject_id"]
#         .astype(str)
#         .isin(test_ids)
#     ].copy()

#     train_records = (
#         train_records
#         .set_index("subject_id")
#         .loc[train_ids]
#         .reset_index()
#     )

#     test_records = (
#         test_records
#         .set_index("subject_id")
#         .loc[test_ids]
#         .reset_index()
#     )

#     train_indices = [
#         subject_to_aal_index[sid]
#         for sid in train_records[
#             "subject_id"
#         ].astype(str)
#     ]

#     test_indices = [
#         subject_to_aal_index[sid]
#         for sid in test_records[
#             "subject_id"
#         ].astype(str)
#     ]

#     train_aal_raw = aal_raw[
#         train_indices
#     ]

#     test_aal_raw = aal_raw[
#         test_indices
#     ]

#     # --------------------------------------------------------
#     # CRITICAL LEAKAGE CONTROL
#     # --------------------------------------------------------
#     # Fit scaler ONLY on outer training data.
#     # Outer test data is never used for fitting.
#     # --------------------------------------------------------

#     scaler = StandardScaler()

#     train_aal = scaler.fit_transform(
#         train_aal_raw
#     )

#     test_aal = scaler.transform(
#         test_aal_raw
#     )

#     fold_dir = (
#         OUTPUT_DIR
#         / f"outer_fold_{outer_fold}"
#     )

#     fold_dir.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     scaler_path = (
#         fold_dir
#         / "aal_standard_scaler.joblib"
#     )

#     joblib.dump(
#         scaler,
#         scaler_path,
#     )

#     train_labels = (
#         train_records["label"]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     pos_weight_value = get_pos_weight(
#         train_labels
#     )

#     criterion = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor(
#             [pos_weight_value],
#             dtype=torch.float32,
#             device=DEVICE,
#         )
#     )

#     train_loader = make_loader(
#         train_records,
#         train_aal,
#         shuffle=True,
#     )

#     test_loader = make_loader(
#         test_records,
#         test_aal,
#         shuffle=False,
#     )

#     # --------------------------------------------------------
#     # FIXED NATIVE PYTHON SEED
#     # --------------------------------------------------------

#     outer_seed = int(
#         SEED
#         + 1000
#         + int(outer_fold)
#     )

#     set_seed(
#         outer_seed
#     )

#     model = SMRICNNAALFusion(
#         aal_features=116,
#         dropout=DROPOUT,
#     ).to(DEVICE)

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )

#     history = []

#     print(
#         f"  Outer training: {selected_epoch} epochs"
#     )

#     for epoch in range(
#         1,
#         selected_epoch + 1,
#     ):

#         train_loss, train_auc = (
#             train_one_epoch(
#                 model,
#                 train_loader,
#                 optimizer,
#                 criterion,
#             )
#         )

#         history.append(
#             {
#                 "epoch": int(epoch),
#                 "train_loss": float(
#                     train_loss
#                 ),
#                 "train_auc": float(
#                     train_auc
#                 )
#                 if np.isfinite(train_auc)
#                 else None,
#                 "learning_rate": float(
#                     optimizer.param_groups[0][
#                         "lr"
#                     ]
#                 ),
#             }
#         )

#     checkpoint_path = (
#         fold_dir
#         / "best_model.pt"
#     )

#     torch.save(
#         {
#             "model_state_dict": model.state_dict(),
#             "selected_epoch": selected_epoch,
#             "outer_fold": outer_fold,
#             "seed": outer_seed,
#             "input_shape": (
#                 1,
#                 95,
#                 117,
#                 99,
#             ),
#             "aal_features": 116,
#             "dropout": DROPOUT,
#             "batch_size": BATCH_SIZE,
#             "learning_rate": LEARNING_RATE,
#             "weight_decay": WEIGHT_DECAY,
#             "pos_weight": pos_weight_value,
#         },
#         checkpoint_path,
#     )

#     (
#         test_loss,
#         test_auc,
#         test_labels,
#         test_probs,
#     ) = evaluate_loader(
#         model,
#         test_loader,
#         criterion,
#     )

#     predictions = pd.DataFrame(
#         {
#             "subject_id": (
#                 test_records[
#                     "subject_id"
#                 ]
#                 .astype(str)
#                 .tolist()
#             ),
#             "label": test_labels.astype(
#                 int
#             ),
#             "probability": test_probs,
#             "outer_fold": outer_fold,
#         }
#     )

#     fold_report = {
#         "outer_fold": outer_fold,
#         "selected_epoch": selected_epoch,
#         "train_subjects": len(
#             train_records
#         ),
#         "test_subjects": len(
#             test_records
#         ),
#         "train_controls": int(
#             np.sum(
#                 train_labels == 0
#             )
#         ),
#         "train_schizophrenia": int(
#             np.sum(
#                 train_labels == 1
#             )
#         ),
#         "test_controls": int(
#             np.sum(
#                 test_labels == 0
#             )
#         ),
#         "test_schizophrenia": int(
#             np.sum(
#                 test_labels == 1
#             )
#         ),
#         "pos_weight": float(
#             pos_weight_value
#         ),
#         "test_loss": float(
#             test_loss
#         ),
#         "test_auc": float(
#             test_auc
#         ),
#     }

#     history_path = (
#         fold_dir
#         / "training_history.csv"
#     )

#     pd.DataFrame(
#         history
#     ).to_csv(
#         history_path,
#         index=False,
#     )

#     del model
#     del optimizer

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     return (
#         predictions,
#         fold_report,
#     )


# # ============================================================
# # FINAL METRICS
# # ============================================================

# def calculate_oof_metrics(
#     predictions: pd.DataFrame,
# ) -> dict:

#     y_true = predictions[
#         "label"
#     ].to_numpy()

#     y_prob = predictions[
#         "probability"
#     ].to_numpy()

#     y_pred = (
#         y_prob >= 0.5
#     ).astype(int)

#     auc = roc_auc_score(
#         y_true,
#         y_prob,
#     )

#     accuracy = accuracy_score(
#         y_true,
#         y_pred,
#     )

#     balanced_accuracy = (
#         balanced_accuracy_score(
#             y_true,
#             y_pred,
#         )
#     )

#     precision = precision_score(
#         y_true,
#         y_pred,
#         zero_division=0,
#     )

#     positive_count = np.sum(
#         y_true == 1
#     )

#     negative_count = np.sum(
#         y_true == 0
#     )

#     if positive_count > 0:

#         sensitivity = (
#             np.sum(
#                 (y_true == 1)
#                 & (y_pred == 1)
#             )
#             / positive_count
#         )

#     else:

#         sensitivity = 0.0

#     if negative_count > 0:

#         specificity = (
#             np.sum(
#                 (y_true == 0)
#                 & (y_pred == 0)
#             )
#             / negative_count
#         )

#     else:

#         specificity = 0.0

#     f1 = f1_score(
#         y_true,
#         y_pred,
#         zero_division=0,
#     )

#     (
#         tn,
#         fp,
#         fn,
#         tp,
#     ) = confusion_matrix(
#         y_true,
#         y_pred,
#         labels=[0, 1],
#     ).ravel()

#     return {
#         "oof_auc": float(
#             auc
#         ),
#         "oof_accuracy": float(
#             accuracy
#         ),
#         "oof_balanced_accuracy": float(
#             balanced_accuracy
#         ),
#         "oof_precision": float(
#             precision
#         ),
#         "oof_sensitivity": float(
#             sensitivity
#         ),
#         "oof_specificity": float(
#             specificity
#         ),
#         "oof_f1": float(
#             f1
#         ),
#         "tn": int(tn),
#         "fp": int(fp),
#         "fn": int(fn),
#         "tp": int(tp),
#     }


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     set_seed(
#         SEED
#     )

#     print(
#         "=" * 70
#     )

#     print(
#         "EXPERIMENT 05 — CNN + AAL FUSION"
#     )

#     print(
#         "FULL 5×4 NESTED CROSS-VALIDATION"
#     )

#     print(
#         "=" * 70
#     )

#     print(
#         "Device:",
#         DEVICE,
#     )

#     print(
#         "Manifest:",
#         MANIFEST_PATH,
#     )

#     print(
#         "Nested assignments:",
#         NESTED_ASSIGNMENTS_PATH,
#     )

#     print(
#         "Image directory:",
#         IMAGE_DIR,
#     )

#     print(
#         "Output directory:",
#         OUTPUT_DIR,
#     )

#     # ========================================================
#     # LOAD MANIFEST
#     # ========================================================

#     manifest = pd.read_csv(
#         MANIFEST_PATH
#     )

#     manifest["subject_id"] = (
#         manifest["subject_id"]
#         .astype(str)
#     )

#     if len(manifest) != 172:
#         raise ValueError(
#             f"Expected 172 labeled subjects, "
#             f"found {len(manifest)}"
#         )

#     if manifest[
#         "subject_id"
#     ].duplicated().any():

#         raise ValueError(
#             "Duplicate subject IDs in manifest."
#         )

#     # ========================================================
#     # LOAD NESTED ASSIGNMENTS
#     # ========================================================

#     assignments = pd.read_csv(
#         NESTED_ASSIGNMENTS_PATH
#     )

#     assignments["subject_id"] = (
#         assignments["subject_id"]
#         .astype(str)
#     )

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
#         - set(assignments.columns)
#     )

#     if missing:
#         raise ValueError(
#             "Nested assignments missing columns: "
#             f"{sorted(missing)}"
#         )

#     expected_roles = {
#         "INNER_TRAIN",
#         "INNER_VALIDATION",
#         "OUTER_TEST",
#     }

#     actual_roles = set(
#         assignments[
#             "role"
#         ].unique()
#     )

#     if actual_roles != expected_roles:
#         raise ValueError(
#             f"Unexpected roles: "
#             f"{actual_roles}"
#         )

#     # ========================================================
#     # VALIDATE ASSIGNMENT SUBJECTS
#     # ========================================================

#     assignment_subjects = set(
#         assignments[
#             "subject_id"
#         ]
#         .astype(str)
#         .unique()
#     )

#     manifest_subjects = set(
#         manifest[
#             "subject_id"
#         ]
#         .astype(str)
#         .unique()
#     )

#     if assignment_subjects != manifest_subjects:

#         missing_in_assignments = (
#             manifest_subjects
#             - assignment_subjects
#         )

#         extra_in_assignments = (
#             assignment_subjects
#             - manifest_subjects
#         )

#         raise ValueError(
#             "Nested-CV subjects do not exactly "
#             "match manifest subjects. "
#             f"Missing: {sorted(missing_in_assignments)[:10]}; "
#             f"Extra: {sorted(extra_in_assignments)[:10]}"
#         )

#     # ========================================================
#     # LOAD AAL MATRIX
#     # ========================================================

#     print()
#     print(
#         "Loading AAL matrix..."
#     )

#     (
#         aal_raw,
#         aal_subjects,
#     ) = load_aal_matrix(
#         manifest
#     )

#     subject_to_aal_index = {
#         subject_id: index
#         for index, subject_id
#         in enumerate(
#             aal_subjects
#         )
#     }

#     if len(
#         subject_to_aal_index
#     ) != 172:

#         raise ValueError(
#             "AAL subject mapping is not unique."
#         )

#     if not np.isfinite(
#         aal_raw
#     ).all():

#         raise ValueError(
#             "AAL matrix contains non-finite values."
#         )

#     print(
#         "AAL matrix:",
#         aal_raw.shape,
#     )

#     # ========================================================
#     # VALIDATE MRI TENSORS
#     # ========================================================

#     print(
#         "Validating MRI tensor inventory..."
#     )

#     missing_images = []

#     invalid_images = []

#     for subject_id in manifest[
#         "subject_id"
#     ]:

#         image_path = (
#             IMAGE_DIR
#             / f"{subject_id}.npy"
#         )

#         if not image_path.exists():

#             missing_images.append(
#                 subject_id
#             )

#             continue

#         # Lightweight shape/finite validation.
#         image = np.load(
#             image_path,
#             mmap_mode="r",
#         )

#         if image.shape != (
#             1,
#             95,
#             117,
#             99,
#         ):

#             invalid_images.append(
#                 (
#                     subject_id,
#                     str(image.shape),
#                 )
#             )

#             continue

#         if not np.isfinite(
#             image
#         ).all():

#             invalid_images.append(
#                 (
#                     subject_id,
#                     "non-finite",
#                 )
#             )

#     if missing_images:

#         raise FileNotFoundError(
#             "Missing MRI tensors: "
#             f"{missing_images[:10]}"
#         )

#     if invalid_images:

#         raise ValueError(
#             "Invalid MRI tensors: "
#             f"{invalid_images[:10]}"
#         )

#     print(
#         "MRI tensors validated: 172/172"
#     )

#     # ========================================================
#     # OUTER FOLD VALIDATION
#     # ========================================================

#     outer_folds = sorted(
#         int(x)
#         for x in assignments[
#             "outer_fold"
#         ].unique()
#     )

#     if outer_folds != [
#         1,
#         2,
#         3,
#         4,
#         5,
#     ]:

#         raise ValueError(
#             f"Unexpected outer folds: "
#             f"{outer_folds}"
#         )

#     # ========================================================
#     # NESTED CV
#     # ========================================================

#     all_predictions = []
#     all_fold_reports = []
#     all_inner_results = []

#     for outer_fold in outer_folds:

#         print()
#         print(
#             "=" * 70
#         )

#         print(
#             f"OUTER FOLD {outer_fold}/5"
#         )

#         print(
#             "=" * 70
#         )

#         (
#             selected_epoch,
#             inner_epochs,
#             inner_results,
#         ) = select_epoch_for_outer_fold(
#             outer_fold=outer_fold,
#             assignments=assignments,
#             manifest=manifest,
#             aal_raw=aal_raw,
#             subject_to_aal_index=(
#                 subject_to_aal_index
#             ),
#         )

#         print(
#             "Inner best epochs:",
#             inner_epochs,
#         )

#         print(
#             "Selected epoch:",
#             selected_epoch,
#         )

#         all_inner_results.extend(
#             inner_results
#         )

#         (
#             predictions,
#             fold_report,
#         ) = train_outer_fold(
#             outer_fold=outer_fold,
#             selected_epoch=selected_epoch,
#             assignments=assignments,
#             manifest=manifest,
#             aal_raw=aal_raw,
#             subject_to_aal_index=(
#                 subject_to_aal_index
#             ),
#         )

#         all_predictions.append(
#             predictions
#         )

#         all_fold_reports.append(
#             fold_report
#         )

#         print(
#             "Outer test AUC:",
#             f"{fold_report['test_auc']:.6f}",
#         )

#     # ========================================================
#     # COMBINE OOF PREDICTIONS
#     # ========================================================

#     oof = pd.concat(
#         all_predictions,
#         ignore_index=True,
#     )

#     oof_path = (
#         OUTPUT_DIR
#         / "oof_predictions.csv"
#     )

#     oof.to_csv(
#         oof_path,
#         index=False,
#     )

#     # ========================================================
#     # OOF INTEGRITY
#     # ========================================================

#     if len(oof) != 172:

#         raise RuntimeError(
#             "Expected 172 OOF predictions, "
#             f"found {len(oof)}"
#         )

#     if oof[
#         "subject_id"
#     ].duplicated().any():

#         raise RuntimeError(
#             "Duplicate subjects in OOF predictions."
#         )

#     if set(
#         oof["subject_id"]
#     ) != set(
#         manifest["subject_id"]
#     ):

#         raise RuntimeError(
#             "OOF subjects do not exactly match "
#             "manifest subjects."
#         )

#     if not np.isfinite(
#         oof["probability"]
#     ).all():

#         raise RuntimeError(
#             "OOF probabilities contain non-finite values."
#         )

#     # ========================================================
#     # FINAL METRICS
#     # ========================================================

#     metrics = calculate_oof_metrics(
#         oof
#     )

#     fold_auc = [
#         float(
#             report["test_auc"]
#         )
#         for report
#         in all_fold_reports
#     ]

#     mean_fold_auc = float(
#         np.mean(
#             fold_auc
#         )
#     )

#     fold_auc_std = float(
#         np.std(
#             fold_auc,
#             ddof=0,
#         )
#     )

#     # ========================================================
#     # SAVE FOLD RESULTS
#     # ========================================================

#     pd.DataFrame(
#         all_fold_reports
#     ).to_csv(
#         OUTPUT_DIR
#         / "outer_fold_results.csv",
#         index=False,
#     )

#     pd.DataFrame(
#         all_inner_results
#     ).to_csv(
#         OUTPUT_DIR
#         / "inner_fold_results.csv",
#         index=False,
#     )

#     # ========================================================
#     # SUMMARY
#     # ========================================================

#     selected_epochs = [
#         int(
#             report["selected_epoch"]
#         )
#         for report
#         in all_fold_reports
#     ]

#     summary = {
#         "experiment": (
#             "Experiment 05 — "
#             "3D CNN + AAL feature fusion"
#         ),
#         "seed": int(SEED),
#         "device": str(DEVICE),
#         "subjects": 172,
#         "controls": int(
#             np.sum(
#                 manifest["label"] == 0
#             )
#         ),
#         "schizophrenia": int(
#             np.sum(
#                 manifest["label"] == 1
#             )
#         ),
#         "aal_features": 116,
#         "outer_folds": 5,
#         "inner_folds": 4,
#         "batch_size": BATCH_SIZE,
#         "max_epochs": MAX_EPOCHS,
#         "learning_rate": LEARNING_RATE,
#         "weight_decay": WEIGHT_DECAY,
#         "dropout": DROPOUT,
#         "input_shape": [
#             1,
#             95,
#             117,
#             99,
#         ],
#         "subject_level": True,
#         "aal_scaler_fit_inside_training_folds": True,
#         "test_set_tuning": False,
#         **metrics,
#         "mean_outer_fold_auc": (
#             mean_fold_auc
#         ),
#         "outer_fold_auc_population_std": (
#             fold_auc_std
#         ),
#         "outer_fold_auc": fold_auc,
#         "selected_epochs": selected_epochs,
#         "final_status": "PASS",
#     }

#     with open(
#         OUTPUT_DIR
#         / "training_summary.json",
#         "w",
#         encoding="utf-8",
#     ) as file:

#         json.dump(
#             summary,
#             file,
#             indent=2,
#         )

#     # ========================================================
#     # HUMAN-READABLE REPORT
#     # ========================================================

#     report_lines = [
#         "=" * 70,
#         "EXPERIMENT 05 — CNN + AAL FEATURE FUSION",
#         "FULL 5×4 NESTED CROSS-VALIDATION",
#         "=" * 70,
#         "",
#         f"Device: {DEVICE}",
#         "Subjects: 172",
#         "Controls: 93",
#         "Schizophrenia: 79",
#         "AAL features: 116",
#         "MRI input: (1, 95, 117, 99)",
#         "",
#         "Evaluation:",
#         "  Outer folds: 5",
#         "  Inner folds: 4",
#         "  Subject-level: YES",
#         "  AAL scaler fitted inside training folds: YES",
#         "  Test-set tuning: NO",
#         "",
#         "Outer fold AUCs:",
#     ]

#     for index, auc in enumerate(
#         fold_auc,
#         start=1,
#     ):

#         report_lines.append(
#             f"  Fold {index}: {auc:.6f}"
#         )

#     report_lines.extend(
#         [
#             "",
#             (
#                 "Mean fold AUC: "
#                 f"{mean_fold_auc:.6f}"
#             ),
#             (
#                 "Fold AUC population SD: "
#                 f"{fold_auc_std:.6f}"
#             ),
#             "",
#             (
#                 "OOF ROC-AUC: "
#                 f"{metrics['oof_auc']:.6f}"
#             ),
#             (
#                 "OOF Accuracy: "
#                 f"{metrics['oof_accuracy']:.6f}"
#             ),
#             (
#                 "OOF Balanced Accuracy: "
#                 f"{metrics['oof_balanced_accuracy']:.6f}"
#             ),
#             (
#                 "OOF Precision: "
#                 f"{metrics['oof_precision']:.6f}"
#             ),
#             (
#                 "OOF Sensitivity: "
#                 f"{metrics['oof_sensitivity']:.6f}"
#             ),
#             (
#                 "OOF Specificity: "
#                 f"{metrics['oof_specificity']:.6f}"
#             ),
#             (
#                 "OOF F1: "
#                 f"{metrics['oof_f1']:.6f}"
#             ),
#             "",
#             "Confusion matrix:",
#             f"  TN: {metrics['tn']}",
#             f"  FP: {metrics['fp']}",
#             f"  FN: {metrics['fn']}",
#             f"  TP: {metrics['tp']}",
#             "",
#             "Selected epochs:",
#             str(selected_epochs),
#             "",
#             (
#                 "FINAL STATUS: "
#                 "EXPERIMENT 05 NESTED CV PASS"
#             ),
#         ]
#     )

#     with open(
#         OUTPUT_DIR
#         / "training_report.txt",
#         "w",
#         encoding="utf-8",
#     ) as file:

#         file.write(
#             "\n".join(
#                 report_lines
#             )
#         )

#     # ========================================================
#     # FINAL CONSOLE OUTPUT
#     # ========================================================

#     print()
#     print(
#         "=" * 70
#     )

#     print(
#         "EXPERIMENT 05 COMPLETE"
#     )

#     print(
#         "=" * 70
#     )

#     print(
#         "Outer fold AUCs:",
#         [
#             round(
#                 value,
#                 6,
#             )
#             for value
#             in fold_auc
#         ],
#     )

#     print(
#         "Mean fold AUC:",
#         f"{mean_fold_auc:.6f}",
#     )

#     print(
#         "Fold AUC population SD:",
#         f"{fold_auc_std:.6f}",
#     )

#     print(
#         "OOF ROC-AUC:",
#         f"{metrics['oof_auc']:.6f}",
#     )

#     print(
#         "OOF Accuracy:",
#         f"{metrics['oof_accuracy']:.6f}",
#     )

#     print(
#         "OOF Balanced Accuracy:",
#         f"{metrics['oof_balanced_accuracy']:.6f}",
#     )

#     print(
#         "OOF Precision:",
#         f"{metrics['oof_precision']:.6f}",
#     )

#     print(
#         "OOF Sensitivity:",
#         f"{metrics['oof_sensitivity']:.6f}",
#     )

#     print(
#         "OOF Specificity:",
#         f"{metrics['oof_specificity']:.6f}",
#     )

#     print(
#         "OOF F1:",
#         f"{metrics['oof_f1']:.6f}",
#     )

#     print()

#     print(
#         "Selected epochs:",
#         selected_epochs,
#     )

#     print()

#     print(
#         "Results:",
#         OUTPUT_DIR,
#     )

#     print(
#         "FINAL STATUS: "
#         "EXPERIMENT 05 NESTED CV PASS"
#     )

#     print(
#         "=" * 70
#     )


# if __name__ == "__main__":
#     main()



from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import median

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset

from smri_cnn_aal_fusion import SMRICNNAALFusion


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "dataset"
    / "smri_subject_manifest_labeled.csv"
)

NESTED_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "cross_validation"
    / "nested"
    / "nested_cv_assignments.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "cnn_dataset"
    / "experiments"
    / "exp05_cnn_aal_fusion"
    / "training_full_5x4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 4
MAX_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

EARLY_STOPPING_PATIENCE = 10

LR_PATIENCE = 4
LR_FACTOR = 0.5

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:

    # Convert NumPy/pandas integer scalars into
    # a native Python integer before calling random.seed().
    seed = int(seed)

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

class FusionDataset(Dataset):

    def __init__(
        self,
        records: pd.DataFrame,
        aal_values: np.ndarray,
    ):

        if len(records) != len(aal_values):

            raise ValueError(
                "Records and AAL array length mismatch."
            )

        self.records = (
            records
            .reset_index(drop=True)
        )

        self.aal_values = (
            aal_values
            .astype(np.float32)
        )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        row = self.records.iloc[index]

        subject_id = str(
            row["subject_id"]
        )

        label = float(
            row["label"]
        )

        image_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"MRI tensor not found: "
                f"{image_path}"
            )

        image = np.load(
            image_path
        ).astype(np.float32)

        expected_shape = (
            1,
            95,
            117,
            99,
        )

        if image.shape != expected_shape:

            raise ValueError(
                f"Unexpected MRI shape for "
                f"{subject_id}: {image.shape}"
            )

        if not np.isfinite(image).all():

            raise ValueError(
                f"Non-finite MRI tensor: "
                f"{subject_id}"
            )

        aal = self.aal_values[index]

        if aal.shape != (116,):

            raise ValueError(
                f"Unexpected AAL shape for "
                f"{subject_id}: {aal.shape}"
            )

        if not np.isfinite(aal).all():

            raise ValueError(
                f"Non-finite AAL values: "
                f"{subject_id}"
            )

        return (
            torch.from_numpy(image),
            torch.from_numpy(aal),
            torch.tensor(
                [label],
                dtype=torch.float32,
            ),
            subject_id,
        )


# ============================================================
# AAL LOADING
# ============================================================

def load_aal_matrix(
    manifest: pd.DataFrame,
):

    rows = []

    subjects = []

    reference_columns = None

    for _, row in manifest.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        aal_path = Path(
            row["aal_csv"]
        )

        if not aal_path.is_absolute():

            aal_path = (
                PROJECT_ROOT
                / aal_path
            )

        if not aal_path.exists():

            raise FileNotFoundError(
                f"AAL file not found: "
                f"{aal_path}"
            )

        df = pd.read_csv(
            aal_path
        )

        if len(df) != 1:

            raise ValueError(
                f"{subject_id}: expected exactly "
                f"one AAL row, found {len(df)}"
            )

        feature_df = df.drop(
            columns=[
                "subject_id",
                "acquisition_type",
                "volume",
            ],
            errors="ignore",
        )

        columns = list(
            feature_df.columns
        )

        if reference_columns is None:

            reference_columns = columns

        if columns != reference_columns:

            raise ValueError(
                f"AAL column mismatch for "
                f"{subject_id}"
            )

        values = (
            feature_df.iloc[0]
            .to_numpy(
                dtype=np.float32
            )
        )

        if values.shape != (116,):

            raise ValueError(
                f"{subject_id}: expected 116 AAL "
                f"features, found {values.shape}"
            )

        if not np.isfinite(values).all():

            raise ValueError(
                f"{subject_id}: non-finite "
                f"AAL features"
            )

        rows.append(values)

        subjects.append(subject_id)

    matrix = np.vstack(rows)

    if matrix.shape != (
        len(manifest),
        116,
    ):

        raise ValueError(
            f"Unexpected AAL matrix shape: "
            f"{matrix.shape}"
        )

    return matrix, subjects


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    all_labels = []

    all_probs = []

    for (
        smri,
        aal,
        labels,
        _,
    ) in loader:

        smri = smri.to(
            DEVICE,
            non_blocking=True,
        )

        aal = aal.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            smri,
            aal,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * smri.size(0)
        )

        probs = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        all_probs.extend(
            probs.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    if len(
        set(all_labels)
    ) >= 2:

        auc = roc_auc_score(
            all_labels,
            all_probs,
        )

    else:

        auc = float("nan")

    return (
        average_loss,
        auc,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_probs = []

    all_subject_ids = []

    for (
        smri,
        aal,
        labels,
        subject_ids,
    ) in loader:

        smri = smri.to(
            DEVICE,
            non_blocking=True,
        )

        aal = aal.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            smri,
            aal,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * smri.size(0)
        )

        probs = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        all_probs.extend(
            probs.cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        all_subject_ids.extend(
            list(subject_ids)
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    labels_array = np.asarray(
        all_labels,
        dtype=np.float32,
    )

    probs_array = np.asarray(
        all_probs,
        dtype=np.float32,
    )

    if len(
        np.unique(labels_array)
    ) >= 2:

        auc = roc_auc_score(
            labels_array,
            probs_array,
        )

    else:

        auc = float("nan")

    return (
        average_loss,
        auc,
        labels_array,
        probs_array,
        all_subject_ids,
    )


# ============================================================
# DATALOADER
# ============================================================

def make_loader(
    records,
    aal_values,
    shuffle,
):

    dataset = FusionDataset(
        records,
        aal_values,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

def get_pos_weight(labels):

    positives = int(
        np.sum(labels == 1)
    )

    negatives = int(
        np.sum(labels == 0)
    )

    if positives == 0:

        raise ValueError(
            "No positive samples in training fold."
        )

    return (
        negatives
        / positives
    )


# ============================================================
# INNER-FOLD EPOCH SELECTION
# ============================================================

def select_epoch_for_outer_fold(
    outer_fold,
    assignments,
    manifest,
    aal_raw,
    subject_to_aal_index,
):

    outer_fold = int(
        outer_fold
    )

    inner_best_epochs = []

    inner_results = []

    inner_folds = sorted(
        assignments.loc[
            (
                assignments["outer_fold"]
                == outer_fold
            )
            & (
                assignments["role"]
                == "INNER_TRAIN"
            ),
            "inner_fold",
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if inner_folds != [
        1,
        2,
        3,
        4,
    ]:

        raise ValueError(
            f"Outer fold {outer_fold}: "
            f"expected inner folds [1,2,3,4], "
            f"found {inner_folds}"
        )

    for inner_fold_value in inner_folds:

        inner_fold = int(
            inner_fold_value
        )

        inner_assign = assignments[
            (
                assignments["outer_fold"]
                == outer_fold
            )
            & (
                assignments["inner_fold"]
                == inner_fold
            )
        ]

        train_ids = (
            inner_assign.loc[
                inner_assign["role"]
                == "INNER_TRAIN",
                "subject_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        val_ids = (
            inner_assign.loc[
                inner_assign["role"]
                == "INNER_VALIDATION",
                "subject_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(train_ids) != len(
            set(train_ids)
        ):

            raise RuntimeError(
                f"Outer {outer_fold}, inner "
                f"{inner_fold}: duplicate "
                f"training subjects."
            )

        if len(val_ids) != len(
            set(val_ids)
        ):

            raise RuntimeError(
                f"Outer {outer_fold}, inner "
                f"{inner_fold}: duplicate "
                f"validation subjects."
            )

        train_records = manifest[
            manifest["subject_id"]
            .astype(str)
            .isin(train_ids)
        ].copy()

        val_records = manifest[
            manifest["subject_id"]
            .astype(str)
            .isin(val_ids)
        ].copy()

        train_records = (
            train_records
            .set_index("subject_id")
            .loc[train_ids]
            .reset_index()
        )

        val_records = (
            val_records
            .set_index("subject_id")
            .loc[val_ids]
            .reset_index()
        )

        train_indices = [
            subject_to_aal_index[sid]
            for sid in train_records[
                "subject_id"
            ].astype(str)
        ]

        val_indices = [
            subject_to_aal_index[sid]
            for sid in val_records[
                "subject_id"
            ].astype(str)
        ]

        train_aal_raw = aal_raw[
            train_indices
        ]

        val_aal_raw = aal_raw[
            val_indices
        ]

        # ====================================================
        # INNER TRAINING SCALER
        # ====================================================

        scaler = StandardScaler()

        train_aal = (
            scaler.fit_transform(
                train_aal_raw
            )
        )

        val_aal = (
            scaler.transform(
                val_aal_raw
            )
        )

        train_labels = (
            train_records["label"]
            .to_numpy(
                dtype=np.float32
            )
        )

        pos_weight_value = (
            get_pos_weight(
                train_labels
            )
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [pos_weight_value],
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )
        )

        # ====================================================
        # SEED
        # ====================================================

        inner_seed = int(
            SEED
            + int(outer_fold) * 100
            + int(inner_fold)
        )

        set_seed(
            inner_seed
        )

        train_loader = make_loader(
            train_records,
            train_aal,
            shuffle=True,
        )

        val_loader = make_loader(
            val_records,
            val_aal,
            shuffle=False,
        )

        model = (
            SMRICNNAALFusion(
                aal_features=116,
                dropout=DROPOUT,
            )
            .to(DEVICE)
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
                factor=LR_FACTOR,
                patience=LR_PATIENCE,
            )
        )

        best_auc = -np.inf

        best_epoch = 1

        best_state = None

        epochs_without_improvement = 0

        for epoch in range(
            1,
            MAX_EPOCHS + 1,
        ):

            train_loss, train_auc = (
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                )
            )

            (
                val_loss,
                val_auc,
                _,
                _,
                _,
            ) = evaluate_loader(
                model,
                val_loader,
                criterion,
            )

            scheduler.step(
                val_auc
                if np.isfinite(
                    val_auc
                )
                else 0.0
            )

            if (
                np.isfinite(val_auc)
                and val_auc > best_auc
            ):

                best_auc = val_auc

                best_epoch = epoch

                best_state = {
                    key: value
                    .detach()
                    .cpu()
                    .clone()
                    for key, value
                    in model.state_dict()
                    .items()
                }

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                break

        if best_state is not None:

            model.load_state_dict(
                best_state
            )

        result = {
            "outer_fold": int(
                outer_fold
            ),
            "inner_fold": int(
                inner_fold
            ),
            "best_epoch": int(
                best_epoch
            ),
            "best_validation_auc": float(
                best_auc
            ),
            "train_subjects": int(
                len(train_records)
            ),
            "validation_subjects": int(
                len(val_records)
            ),
            "pos_weight": float(
                pos_weight_value
            ),
            "seed": int(
                inner_seed
            ),
        }

        inner_results.append(
            result
        )

        inner_best_epochs.append(
            int(best_epoch)
        )

        print(
            f"  Inner fold {inner_fold}: "
            f"best epoch={best_epoch}, "
            f"validation AUC={best_auc:.6f}"
        )

        del model
        del optimizer
        del scheduler

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    selected_epoch = int(
        round(
            median(
                inner_best_epochs
            )
        )
    )

    return (
        selected_epoch,
        inner_best_epochs,
        inner_results,
    )


# ============================================================
# OUTER-FOLD TRAINING
# ============================================================

def train_outer_fold(
    outer_fold,
    selected_epoch,
    assignments,
    manifest,
    aal_raw,
    subject_to_aal_index,
):

    outer_fold = int(
        outer_fold
    )

    selected_epoch = int(
        selected_epoch
    )

    outer_assign = assignments[
        assignments["outer_fold"]
        == outer_fold
    ]

    # ========================================================
    # UNIQUE OUTER TRAINING SUBJECTS
    # ========================================================

    train_ids = (
        outer_assign.loc[
            outer_assign["role"]
            != "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    # ========================================================
    # UNIQUE OUTER TEST SUBJECTS
    #
    # IMPORTANT FIX:
    #
    # The nested assignment table repeats every OUTER_TEST
    # subject once for each inner fold.
    #
    # Therefore .unique() is mandatory here.
    # ========================================================

    test_ids = (
        outer_assign.loc[
            outer_assign["role"]
            == "OUTER_TEST",
            "subject_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    # ========================================================
    # OUTER-FOLD INTEGRITY
    # ========================================================

    if len(train_ids) != len(
        set(train_ids)
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"duplicate training subjects."
        )

    if len(test_ids) != len(
        set(test_ids)
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"duplicate test subjects."
        )

    train_set = set(
        train_ids
    )

    test_set = set(
        test_ids
    )

    overlap = (
        train_set
        & test_set
    )

    if overlap:

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"training/test overlap detected: "
            f"{sorted(overlap)[:10]}"
        )

    train_records = manifest[
        manifest["subject_id"]
        .astype(str)
        .isin(train_ids)
    ].copy()

    test_records = manifest[
        manifest["subject_id"]
        .astype(str)
        .isin(test_ids)
    ].copy()

    train_records = (
        train_records
        .set_index("subject_id")
        .loc[train_ids]
        .reset_index()
    )

    test_records = (
        test_records
        .set_index("subject_id")
        .loc[test_ids]
        .reset_index()
    )

    if len(train_records) != len(
        train_ids
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"training manifest mismatch."
        )

    if len(test_records) != len(
        test_ids
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"test manifest mismatch."
        )

    train_indices = [
        subject_to_aal_index[sid]
        for sid in train_records[
            "subject_id"
        ].astype(str)
    ]

    test_indices = [
        subject_to_aal_index[sid]
        for sid in test_records[
            "subject_id"
        ].astype(str)
    ]

    train_aal_raw = aal_raw[
        train_indices
    ]

    test_aal_raw = aal_raw[
        test_indices
    ]

    # ========================================================
    # OUTER TRAINING SCALER
    #
    # Fit ONLY on outer training subjects.
    # ========================================================

    scaler = StandardScaler()

    train_aal = (
        scaler.fit_transform(
            train_aal_raw
        )
    )

    test_aal = (
        scaler.transform(
            test_aal_raw
        )
    )

    fold_dir = (
        OUTPUT_DIR
        / f"outer_fold_{outer_fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_path = (
        fold_dir
        / "aal_standard_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    train_labels = (
        train_records["label"]
        .to_numpy(
            dtype=np.float32
        )
    )

    pos_weight_value = (
        get_pos_weight(
            train_labels
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value],
                dtype=torch.float32,
                device=DEVICE,
            )
        )
    )

    # ========================================================
    # OUTER SEED
    # ========================================================

    outer_seed = int(
        SEED
        + 1000
        + int(outer_fold)
    )

    set_seed(
        outer_seed
    )

    train_loader = make_loader(
        train_records,
        train_aal,
        shuffle=True,
    )

    test_loader = make_loader(
        test_records,
        test_aal,
        shuffle=False,
    )

    model = (
        SMRICNNAALFusion(
            aal_features=116,
            dropout=DROPOUT,
        )
        .to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    print(
        f"  Outer training: "
        f"{selected_epoch} epochs"
    )

    for epoch in range(
        1,
        selected_epoch + 1,
    ):

        train_loss, train_auc = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": float(
                    train_loss
                ),
                "train_auc": (
                    float(train_auc)
                    if np.isfinite(
                        train_auc
                    )
                    else None
                ),
                "learning_rate": float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
            }
        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = (
        fold_dir
        / "best_model.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "selected_epoch":
                selected_epoch,

            "outer_fold":
                outer_fold,

            "seed":
                outer_seed,

            "input_shape":
                (
                    1,
                    95,
                    117,
                    99,
                ),

            "aal_features":
                116,

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
        checkpoint_path,
    )

    # ========================================================
    # OUTER TEST EVALUATION
    # ========================================================

    (
        test_loss,
        test_auc,
        test_labels,
        test_probs,
        test_subject_ids,
    ) = evaluate_loader(
        model,
        test_loader,
        criterion,
    )

    # ========================================================
    # PREDICTION INTEGRITY
    # ========================================================

    if len(
        test_subject_ids
    ) != len(
        set(test_subject_ids)
    ):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"duplicate subject IDs in predictions."
        )

    if len(
        test_subject_ids
    ) != len(test_ids):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"expected {len(test_ids)} "
            f"predictions, found "
            f"{len(test_subject_ids)}."
        )

    if set(
        test_subject_ids
    ) != set(test_ids):

        raise RuntimeError(
            f"Outer fold {outer_fold}: "
            f"prediction subjects do not match "
            f"outer test subjects."
        )

    predictions = pd.DataFrame(
        {
            "subject_id":
                [
                    str(x)
                    for x
                    in test_subject_ids
                ],

            "label":
                test_labels.astype(int),

            "probability":
                test_probs,

            "outer_fold":
                outer_fold,
        }
    )

    fold_report = {
        "outer_fold":
            outer_fold,

        "selected_epoch":
            selected_epoch,

        "train_subjects":
            len(train_records),

        "test_subjects":
            len(test_records),

        "train_controls":
            int(
                np.sum(
                    train_labels == 0
                )
            ),

        "train_schizophrenia":
            int(
                np.sum(
                    train_labels == 1
                )
            ),

        "test_controls":
            int(
                np.sum(
                    test_labels == 0
                )
            ),

        "test_schizophrenia":
            int(
                np.sum(
                    test_labels == 1
                )
            ),

        "pos_weight":
            float(
                pos_weight_value
            ),

        "test_loss":
            float(
                test_loss
            ),

        "test_auc":
            float(
                test_auc
            ),

        "seed":
            outer_seed,
    }

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = (
        fold_dir
        / "training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    # ========================================================
    # SAVE FOLD PREDICTIONS
    # ========================================================

    predictions.to_csv(
        fold_dir
        / "test_predictions.csv",
        index=False,
    )

    del model
    del optimizer

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return (
        predictions,
        fold_report,
    )


# ============================================================
# FINAL OOF METRICS
# ============================================================

def calculate_oof_metrics(
    predictions,
):

    y_true = (
        predictions["label"]
        .to_numpy()
    )

    y_prob = (
        predictions["probability"]
        .to_numpy()
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    positive_count = np.sum(
        y_true == 1
    )

    negative_count = np.sum(
        y_true == 0
    )

    sensitivity = (
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
        / positive_count
        if positive_count > 0
        else 0.0
    )

    specificity = (
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
        / negative_count
        if negative_count > 0
        else 0.0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    (
        tn,
        fp,
        fn,
        tp,
    ) = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "oof_auc":
            float(auc),

        "oof_accuracy":
            float(accuracy),

        "oof_balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "oof_precision":
            float(precision),

        "oof_sensitivity":
            float(sensitivity),

        "oof_specificity":
            float(specificity),

        "oof_f1":
            float(f1),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 05 — CNN + AAL FUSION"
    )

    print(
        "FULL 5×4 NESTED CROSS-VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Manifest:",
        MANIFEST_PATH,
    )

    print(
        "Nested assignments:",
        NESTED_ASSIGNMENTS_PATH,
    )

    print(
        "Image directory:",
        IMAGE_DIR,
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
    )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest["subject_id"] = (
        manifest["subject_id"]
        .astype(str)
    )

    if len(manifest) != 172:

        raise ValueError(
            f"Expected 172 labeled subjects, "
            f"found {len(manifest)}"
        )

    if manifest[
        "subject_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate subject IDs in manifest."
        )

    # ========================================================
    # LOAD NESTED ASSIGNMENTS
    # ========================================================

    assignments = pd.read_csv(
        NESTED_ASSIGNMENTS_PATH
    )

    assignments["subject_id"] = (
        assignments["subject_id"]
        .astype(str)
    )

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
        - set(assignments.columns)
    )

    if missing:

        raise ValueError(
            "Nested assignments missing "
            f"columns: {sorted(missing)}"
        )

    expected_roles = {
        "INNER_TRAIN",
        "INNER_VALIDATION",
        "OUTER_TEST",
    }

    actual_roles = set(
        assignments[
            "role"
        ].unique()
    )

    if actual_roles != expected_roles:

        raise ValueError(
            f"Unexpected roles: "
            f"{actual_roles}"
        )

    # ========================================================
    # SUBJECT SET VALIDATION
    # ========================================================

    assignment_subjects = set(
        assignments[
            "subject_id"
        ]
        .astype(str)
        .unique()
    )

    manifest_subjects = set(
        manifest[
            "subject_id"
        ]
        .astype(str)
        .unique()
    )

    if (
        assignment_subjects
        != manifest_subjects
    ):

        raise ValueError(
            "Nested-CV subjects do not exactly "
            "match manifest subjects."
        )

    # ========================================================
    # LOAD AAL
    # ========================================================

    print()

    print(
        "Loading AAL matrix..."
    )

    (
        aal_raw,
        aal_subjects,
    ) = load_aal_matrix(
        manifest
    )

    subject_to_aal_index = {
        subject_id: index
        for index, subject_id
        in enumerate(
            aal_subjects
        )
    }

    if len(
        subject_to_aal_index
    ) != 172:

        raise ValueError(
            "AAL subject mapping is not unique."
        )

    if not np.isfinite(
        aal_raw
    ).all():

        raise ValueError(
            "AAL matrix contains "
            "non-finite values."
        )

    print(
        "AAL matrix:",
        aal_raw.shape,
    )

    # ========================================================
    # VALIDATE MRI INVENTORY
    # ========================================================

    print(
        "Validating MRI tensor inventory..."
    )

    missing_images = []

    invalid_images = []

    for subject_id in manifest[
        "subject_id"
    ]:

        image_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not image_path.exists():

            missing_images.append(
                subject_id
            )

            continue

        image = np.load(
            image_path,
            mmap_mode="r",
        )

        if image.shape != (
            1,
            95,
            117,
            99,
        ):

            invalid_images.append(
                (
                    subject_id,
                    str(
                        image.shape
                    ),
                )
            )

            continue

        if not np.isfinite(
            image
        ).all():

            invalid_images.append(
                (
                    subject_id,
                    "non-finite",
                )
            )

    if missing_images:

        raise FileNotFoundError(
            "Missing MRI tensors: "
            f"{missing_images[:10]}"
        )

    if invalid_images:

        raise ValueError(
            "Invalid MRI tensors: "
            f"{invalid_images[:10]}"
        )

    print(
        "MRI tensors validated: 172/172"
    )

    # ========================================================
    # OUTER FOLDS
    # ========================================================

    outer_folds = sorted(
        assignments[
            "outer_fold"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if outer_folds != [
        1,
        2,
        3,
        4,
        5,
    ]:

        raise ValueError(
            f"Unexpected outer folds: "
            f"{outer_folds}"
        )

    # ========================================================
    # NESTED CROSS-VALIDATION
    # ========================================================

    all_predictions = []

    all_fold_reports = []

    all_inner_results = []

    for outer_fold in outer_folds:

        outer_fold = int(
            outer_fold
        )

        print()

        print(
            "=" * 70
        )

        print(
            f"OUTER FOLD "
            f"{outer_fold}/5"
        )

        print(
            "=" * 70
        )

        (
            selected_epoch,
            inner_epochs,
            inner_results,
        ) = select_epoch_for_outer_fold(
            outer_fold=outer_fold,
            assignments=assignments,
            manifest=manifest,
            aal_raw=aal_raw,
            subject_to_aal_index=(
                subject_to_aal_index
            ),
        )

        print(
            "Inner best epochs:",
            inner_epochs,
        )

        print(
            "Selected epoch:",
            selected_epoch,
        )

        all_inner_results.extend(
            inner_results
        )

        (
            predictions,
            fold_report,
        ) = train_outer_fold(
            outer_fold=outer_fold,
            selected_epoch=selected_epoch,
            assignments=assignments,
            manifest=manifest,
            aal_raw=aal_raw,
            subject_to_aal_index=(
                subject_to_aal_index
            ),
        )

        # ----------------------------------------------------
        # Extra protection:
        # Every outer fold contributes only unique subjects.
        # ----------------------------------------------------

        if predictions[
            "subject_id"
        ].duplicated().any():

            raise RuntimeError(
                f"Outer fold {outer_fold}: "
                f"duplicate predictions."
            )

        all_predictions.append(
            predictions
        )

        all_fold_reports.append(
            fold_report
        )

        print(
            "Outer test subjects:",
            len(predictions),
        )

        print(
            "Outer test AUC:",
            f"{fold_report['test_auc']:.6f}",
        )

    # ========================================================
    # COMBINE OOF PREDICTIONS
    # ========================================================

    oof = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ========================================================
    # OOF INTEGRITY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "OOF INTEGRITY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "OOF rows before validation:",
        len(oof),
    )

    duplicate_subjects = (
        oof[
            "subject_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate OOF subjects:",
        duplicate_subjects,
    )

    # This MUST be 172.
    if len(oof) != 172:

        raise RuntimeError(
            f"Expected 172 OOF predictions, "
            f"found {len(oof)}"
        )

    # This MUST be zero.
    if duplicate_subjects != 0:

        duplicated_ids = (
            oof.loc[
                oof[
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
            "Duplicate subjects in OOF "
            f"predictions: "
            f"{duplicated_ids[:20]}"
        )

    # Exact subject-set equality.
    if set(
        oof[
            "subject_id"
        ]
        .astype(str)
    ) != manifest_subjects:

        raise RuntimeError(
            "OOF subjects do not exactly "
            "match manifest subjects."
        )

    if not np.isfinite(
        oof[
            "probability"
        ]
    ).all():

        raise RuntimeError(
            "OOF probabilities contain "
            "non-finite values."
        )

    # ========================================================
    # SAVE CLEAN OOF PREDICTIONS
    # ========================================================

    oof_path = (
        OUTPUT_DIR
        / "oof_predictions.csv"
    )

    oof.to_csv(
        oof_path,
        index=False,
    )

    print(
        "OOF rows:",
        len(oof),
    )

    print(
        "Unique OOF subjects:",
        oof[
            "subject_id"
        ].nunique(),
    )

    print(
        "OOF integrity: PASS"
    )

    # ========================================================
    # FINAL OOF METRICS
    # ========================================================

    metrics = calculate_oof_metrics(
        oof
    )

    fold_auc = [
        float(
            report[
                "test_auc"
            ]
        )
        for report
        in all_fold_reports
    ]

    mean_fold_auc = float(
        np.mean(
            fold_auc
        )
    )

    fold_auc_std = float(
        np.std(
            fold_auc,
            ddof=0,
        )
    )

    # ========================================================
    # SAVE OUTER RESULTS
    # ========================================================

    pd.DataFrame(
        all_fold_reports
    ).to_csv(
        OUTPUT_DIR
        / "outer_fold_results.csv",
        index=False,
    )

    # ========================================================
    # SAVE INNER RESULTS
    # ========================================================

    pd.DataFrame(
        all_inner_results
    ).to_csv(
        OUTPUT_DIR
        / "inner_fold_results.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    selected_epochs = [
        int(
            report[
                "selected_epoch"
            ]
        )
        for report
        in all_fold_reports
    ]

    summary = {
        "experiment":
            "Experiment 05 — "
            "3D CNN + AAL feature fusion",

        "seed":
            int(SEED),

        "device":
            str(DEVICE),

        "subjects":
            172,

        "controls":
            int(
                np.sum(
                    manifest["label"]
                    == 0
                )
            ),

        "schizophrenia":
            int(
                np.sum(
                    manifest["label"]
                    == 1
                )
            ),

        "aal_features":
            116,

        "outer_folds":
            5,

        "inner_folds":
            4,

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
            [
                1,
                95,
                117,
                99,
            ],

        "subject_level":
            True,

        "aal_scaler_fit_inside_training_folds":
            True,

        "test_set_tuning":
            False,

        "oof_rows":
            int(len(oof)),

        "oof_unique_subjects":
            int(
                oof[
                    "subject_id"
                ].nunique()
            ),

        **metrics,

        "mean_outer_fold_auc":
            mean_fold_auc,

        "outer_fold_auc_population_std":
            fold_auc_std,

        "outer_fold_auc":
            fold_auc,

        "selected_epochs":
            selected_epochs,

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

    report_lines = [
        "=" * 70,
        "EXPERIMENT 05 — CNN + AAL FEATURE FUSION",
        "FULL 5×4 NESTED CROSS-VALIDATION",
        "=" * 70,
        "",
        f"Device: {DEVICE}",
        "Subjects: 172",
        "Controls: 93",
        "Schizophrenia: 79",
        "AAL features: 116",
        "MRI input: (1, 95, 117, 99)",
        "",
        "Evaluation:",
        "  Outer folds: 5",
        "  Inner folds: 4",
        "  Subject-level: YES",
        "  AAL scaler fitted inside training folds: YES",
        "  Test-set tuning: NO",
        "",
        "OOF integrity:",
        f"  OOF rows: {len(oof)}",
        f"  Unique OOF subjects: "
        f"{oof['subject_id'].nunique()}",
        "  Duplicate OOF subjects: 0",
        "",
        "Outer fold AUCs:",
    ]

    for index, auc in enumerate(
        fold_auc,
        start=1,
    ):

        report_lines.append(
            f"  Fold {index}: "
            f"{auc:.6f}"
        )

    report_lines.extend(
        [
            "",
            (
                "Mean fold AUC: "
                f"{mean_fold_auc:.6f}"
            ),
            (
                "Fold AUC population SD: "
                f"{fold_auc_std:.6f}"
            ),
            "",
            (
                "OOF ROC-AUC: "
                f"{metrics['oof_auc']:.6f}"
            ),
            (
                "OOF Accuracy: "
                f"{metrics['oof_accuracy']:.6f}"
            ),
            (
                "OOF Balanced Accuracy: "
                f"{metrics['oof_balanced_accuracy']:.6f}"
            ),
            (
                "OOF Precision: "
                f"{metrics['oof_precision']:.6f}"
            ),
            (
                "OOF Sensitivity: "
                f"{metrics['oof_sensitivity']:.6f}"
            ),
            (
                "OOF Specificity: "
                f"{metrics['oof_specificity']:.6f}"
            ),
            (
                "OOF F1: "
                f"{metrics['oof_f1']:.6f}"
            ),
            "",
            "Confusion matrix:",
            f"  TN: {metrics['tn']}",
            f"  FP: {metrics['fp']}",
            f"  FN: {metrics['fn']}",
            f"  TP: {metrics['tp']}",
            "",
            "Selected epochs:",
            str(
                selected_epochs
            ),
            "",
            (
                "FINAL STATUS: "
                "EXPERIMENT 05 NESTED CV PASS"
            ),
        ]
    )

    with open(
        OUTPUT_DIR
        / "training_report.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 05 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Outer fold AUCs:",
        [
            round(
                value,
                6,
            )
            for value
            in fold_auc
        ],
    )

    print(
        "Mean fold AUC:",
        f"{mean_fold_auc:.6f}",
    )

    print(
        "Fold AUC population SD:",
        f"{fold_auc_std:.6f}",
    )

    print(
        "OOF ROC-AUC:",
        f"{metrics['oof_auc']:.6f}",
    )

    print(
        "OOF Accuracy:",
        f"{metrics['oof_accuracy']:.6f}",
    )

    print(
        "OOF Balanced Accuracy:",
        f"{metrics['oof_balanced_accuracy']:.6f}",
    )

    print(
        "OOF Precision:",
        f"{metrics['oof_precision']:.6f}",
    )

    print(
        "OOF Sensitivity:",
        f"{metrics['oof_sensitivity']:.6f}",
    )

    print(
        "OOF Specificity:",
        f"{metrics['oof_specificity']:.6f}",
    )

    print(
        "OOF F1:",
        f"{metrics['oof_f1']:.6f}",
    )

    print()

    print(
        "OOF rows:",
        len(oof),
    )

    print(
        "Unique OOF subjects:",
        oof[
            "subject_id"
        ].nunique(),
    )

    print(
        "Selected epochs:",
        selected_epochs,
    )

    print()

    print(
        "Results:",
        OUTPUT_DIR,
    )

    print()

    print(
        "FINAL STATUS: "
        "EXPERIMENT 05 NESTED CV PASS"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()