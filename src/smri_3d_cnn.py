"""
NeuroFusion AI
sMRI Branch - 3D CNN Diagnostic Investigation

Purpose
-------
Diagnose the poor held-out test performance of the current 3D CNN
before changing the model or retraining.

This script DOES NOT:
    - modify the model
    - modify the dataset
    - modify train/validation/test splits
    - retrain the model
    - overwrite test predictions
    - change thresholds

It analyzes:
    1. Dataset integrity
    2. Class/source distribution
    3. Tensor intensity statistics
    4. Train/validation/test distribution differences
    5. RMS vs legacy distribution differences
    6. CNN logits/probabilities
    7. Prediction variability
    8. Split-wise AUC
    9. Source-wise prediction behavior
   10. Source-only diagnostic
   11. Probability compression
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = Path(
    "processed_data/smri/cnn_dataset"
)

IMAGE_DIR = BASE_DIR / "images"

SPLIT_DIR = BASE_DIR / "splits"

TRAINING_DIR = BASE_DIR / "training"

DIAGNOSTIC_DIR = BASE_DIR / "diagnostics"

TRAIN_FILE = (
    SPLIT_DIR / "train.csv"
)

VAL_FILE = (
    SPLIT_DIR / "validation.csv"
)

TEST_FILE = (
    SPLIT_DIR / "test.csv"
)

LABELS_FILE = (
    BASE_DIR / "labels.csv"
)

METADATA_FILE = (
    BASE_DIR / "metadata.csv"
)

CHECKPOINT_FILE = (
    TRAINING_DIR / "best_model.pt"
)

EXPECTED_TENSOR_SHAPE = (
    1,
    95,
    117,
    99,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# MODEL DEFINITION
#
# EXACT COPY OF src/smri_3d_cnn.py
# ============================================================


class Conv3DBlock(nn.Module):
    """
    Basic 3D convolution block.

    Structure:
        Conv3D
        BatchNorm3D
        ReLU
        optional MaxPool3D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pool: bool = True,
    ):

        super().__init__()

        layers = [

            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        ]

        if use_pool:

            layers.append(
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=2,
                )
            )

        self.block = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


class SMRI3DCNN(nn.Module):
    """
    Exact NeuroFusion AI sMRI 3D CNN.

    Input:
        (B, 1, 95, 117, 99)

    Output:
        (B, 1) classification logits
    """

    def __init__(
        self,
        dropout: float = 0.30,
    ):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            Conv3DBlock(
                in_channels=1,
                out_channels=16,
                use_pool=True,
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            Conv3DBlock(
                in_channels=16,
                out_channels=32,
                use_pool=True,
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            Conv3DBlock(
                in_channels=32,
                out_channels=64,
                use_pool=True,
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            Conv3DBlock(
                in_channels=64,
                out_channels=128,
                use_pool=False,
            ),
        )

        self.global_pool = (
            nn.AdaptiveAvgPool3d(
                output_size=1
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                in_features=128,
                out_features=64,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                in_features=64,
                out_features=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.global_pool(x)

        logits = self.classifier(x)

        return logits


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def print_section(
    title: str,
):
    """
    Print a formatted section heading.
    """

    print()

    print(
        "=" * 72
    )

    print(title)

    print(
        "=" * 72
    )


def find_source_column(
    df: pd.DataFrame,
):
    """
    Identify the acquisition/source column.
    """

    candidates = [

        "acquisition_source",

        "acquisition_type",

        "source",

        "selected_source",

        "representation",

        "lineage",

        "preprocessing_source",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    return None


def safe_auc(
    y_true,
    y_score,
):
    """
    Calculate AUC only when both classes
    are present.
    """

    y_true = np.asarray(
        y_true
    )

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    return roc_auc_score(
        y_true,
        y_score
    )


def load_split(
    path: Path,
    split_name: str,
):
    """
    Load one subject-level split.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Split file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = [
        "subject_id",
        "label",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"{split_name} split does not "
                f"contain required column: {column}"
            )

    df = df.copy()

    df["split"] = split_name

    return df


def tensor_statistics(
    path: Path,
):
    """
    Calculate intensity statistics
    for one CNN tensor.
    """

    array = np.load(
        path
    )

    if array.shape != EXPECTED_TENSOR_SHAPE:

        raise ValueError(
            f"Invalid shape {array.shape}; "
            f"expected {EXPECTED_TENSOR_SHAPE}"
        )

    if array.dtype != np.float32:

        raise ValueError(
            f"Invalid dtype {array.dtype}; "
            f"expected float32"
        )

    if not np.isfinite(
        array
    ).all():

        raise ValueError(
            "Tensor contains NaN or Inf"
        )

    flat = array.reshape(
        -1
    )

    nonzero = flat[
        flat > 0
    ]

    if len(nonzero) == 0:

        brain_mean = 0.0

        brain_std = 0.0

        brain_p01 = 0.0

        brain_p50 = 0.0

        brain_p99 = 0.0

    else:

        brain_mean = float(
            nonzero.mean()
        )

        brain_std = float(
            nonzero.std()
        )

        brain_p01 = float(
            np.percentile(
                nonzero,
                1
            )
        )

        brain_p50 = float(
            np.percentile(
                nonzero,
                50
            )
        )

        brain_p99 = float(
            np.percentile(
                nonzero,
                99
            )
        )

    return {

        "tensor_min": float(
            flat.min()
        ),

        "tensor_max": float(
            flat.max()
        ),

        "tensor_mean": float(
            flat.mean()
        ),

        "tensor_std": float(
            flat.std()
        ),

        "nonzero_fraction": float(
            np.count_nonzero(flat)
            / flat.size
        ),

        "brain_mean": brain_mean,

        "brain_std": brain_std,

        "brain_p01": brain_p01,

        "brain_p50": brain_p50,

        "brain_p99": brain_p99,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print_section(
        "NeuroFusion AI - sMRI 3D CNN Diagnostic Investigation"
    )

    print(
        f"Seed             : {SEED}"
    )

    print(
        f"Dataset directory: {BASE_DIR}"
    )

    print(
        f"Expected tensor  : "
        f"{EXPECTED_TENSOR_SHAPE}"
    )

    # --------------------------------------------------------
    # CREATE DIAGNOSTIC DIRECTORY
    # --------------------------------------------------------

    DIAGNOSTIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. LOAD DATA TABLES
    # --------------------------------------------------------

    print_section(
        "1. Loading Dataset Tables"
    )

    train_df = load_split(
        TRAIN_FILE,
        "train"
    )

    val_df = load_split(
        VAL_FILE,
        "validation"
    )

    test_df = load_split(
        TEST_FILE,
        "test"
    )

    if not LABELS_FILE.exists():

        raise FileNotFoundError(
            f"Labels file not found:\n"
            f"{LABELS_FILE}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    labels_df = pd.read_csv(
        LABELS_FILE
    )

    metadata_df = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Train subjects      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation subjects : "
        f"{len(val_df)}"
    )

    print(
        f"Test subjects       : "
        f"{len(test_df)}"
    )

    print(
        f"Total split rows    : "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    print()

    print(
        "Metadata columns:"
    )

    print(
        list(metadata_df.columns)
    )

    # Prevent unused-variable warnings
    _ = labels_df

    # --------------------------------------------------------
    # 2. IDENTIFY ACQUISITION SOURCE
    # --------------------------------------------------------

    print_section(
        "2. Identifying Acquisition Source"
    )

    source_col = find_source_column(
        metadata_df
    )

    if source_col is None:

        print(
            "WARNING: No recognized source "
            "column found."
        )

        combined = pd.concat(
            [
                train_df,
                val_df,
                test_df,
            ],
            ignore_index=True
        )

    else:

        print(
            f"Detected source column: "
            f"{source_col}"
        )

        metadata_keep = (
            metadata_df.copy()
        )

        duplicate_columns = [

            column

            for column in metadata_keep.columns

            if (
                column != "subject_id"
                and column in train_df.columns
            )
        ]

        if duplicate_columns:

            metadata_keep = (
                metadata_keep.drop(
                    columns=duplicate_columns
                )
            )

        combined = pd.concat(
            [
                train_df,
                val_df,
                test_df,
            ],
            ignore_index=True
        )

        combined = combined.merge(
            metadata_keep,
            on="subject_id",
            how="left",
            validate="one_to_one"
        )

    # --------------------------------------------------------
    # 3. SPLIT INTEGRITY
    # --------------------------------------------------------

    print_section(
        "3. Split Integrity"
    )

    all_ids = (
        combined[
            "subject_id"
        ]
        .astype(str)
    )

    duplicate_ids = (
        all_ids[
            all_ids.duplicated()
        ]
        .unique()
    )

    train_ids = set(
        train_df[
            "subject_id"
        ].astype(str)
    )

    val_ids = set(
        val_df[
            "subject_id"
        ].astype(str)
    )

    test_ids = set(
        test_df[
            "subject_id"
        ].astype(str)
    )

    print(
        f"Total subjects : "
        f"{len(all_ids)}"
    )

    print(
        f"Duplicate IDs  : "
        f"{len(duplicate_ids)}"
    )

    print(
        f"Train ∩ Val    : "
        f"{len(train_ids & val_ids)}"
    )

    print(
        f"Train ∩ Test   : "
        f"{len(train_ids & test_ids)}"
    )

    print(
        f"Val ∩ Test     : "
        f"{len(val_ids & test_ids)}"
    )

    expected_ids = (
        train_ids
        | val_ids
        | test_ids
    )

    print(
        f"Unique split IDs: "
        f"{len(expected_ids)}"
    )

    if len(duplicate_ids) > 0:

        print()

        print(
            "DUPLICATE SUBJECTS:"
        )

        for subject_id in duplicate_ids:

            print(
                f"  {subject_id}"
            )

    # --------------------------------------------------------
    # 4. CLASS DISTRIBUTION
    # --------------------------------------------------------

    print_section(
        "4. Class Distribution"
    )

    class_summary = (
        combined
        .groupby(
            [
                "split",
                "label"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        class_summary
    )

    # --------------------------------------------------------
    # 5. ACQUISITION SOURCE DISTRIBUTION
    # --------------------------------------------------------

    source_summary = None

    source_split_summary = None

    if source_col is not None:

        print_section(
            "5. Acquisition Source Distribution"
        )

        source_summary = (
            combined
            .groupby(
                [
                    source_col,
                    "label"
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        print(
            source_summary
        )

        print()

        print(
            "Source distribution by split:"
        )

        source_split_summary = (
            combined
            .groupby(
                [
                    "split",
                    source_col,
                    "label"
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        print(
            source_split_summary.to_string(
                index=False
            )
        )

        source_split_summary.to_csv(
            DIAGNOSTIC_DIR
            / "source_split_distribution.csv",
            index=False
        )

    # --------------------------------------------------------
    # 6. LOAD ALL TENSOR STATISTICS
    # --------------------------------------------------------

    print_section(
        "6. Tensor Intensity Diagnostics"
    )

    statistics_rows = []

    failed_tensors = []

    for index, row in combined.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        tensor_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not tensor_path.exists():

            failed_tensors.append(
                f"{subject_id}: tensor missing"
            )

            continue

        try:

            stats = tensor_statistics(
                tensor_path
            )

            result = {

                "subject_id":
                    subject_id,

                "split":
                    row["split"],

                "label":
                    int(row["label"]),
            }

            if source_col is not None:

                result["source"] = str(
                    row[source_col]
                )

            result.update(
                stats
            )

            statistics_rows.append(
                result
            )

        except Exception as exc:

            failed_tensors.append(
                f"{subject_id}: {exc}"
            )

        processed = index + 1

        if processed % 25 == 0:

            print(
                f"Processed tensor statistics: "
                f"{processed}/{len(combined)}"
            )

    stats_df = pd.DataFrame(
        statistics_rows
    )

    stats_path = (
        DIAGNOSTIC_DIR
        / "tensor_intensity_statistics.csv"
    )

    stats_df.to_csv(
        stats_path,
        index=False
    )

    print()

    print(
        f"Valid tensors : "
        f"{len(stats_df)}"
    )

    print(
        f"Failed tensors: "
        f"{len(failed_tensors)}"
    )

    if failed_tensors:

        print()

        print(
            "Tensor failures:"
        )

        for failure in failed_tensors:

            print(
                f"  {failure}"
            )

    # --------------------------------------------------------
    # 7. INTENSITY SUMMARY BY SPLIT
    # --------------------------------------------------------

    print_section(
        "7. Intensity Summary by Split"
    )

    numeric_columns = [

        "tensor_min",

        "tensor_max",

        "tensor_mean",

        "tensor_std",

        "nonzero_fraction",

        "brain_mean",

        "brain_std",

        "brain_p01",

        "brain_p50",

        "brain_p99",
    ]

    split_summary = (
        stats_df
        .groupby("split")[
            numeric_columns
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        split_summary.to_string()
    )

    split_summary.to_csv(
        DIAGNOSTIC_DIR
        / "intensity_summary_by_split.csv"
    )

    # --------------------------------------------------------
    # 8. INTENSITY SUMMARY BY CLASS
    # --------------------------------------------------------

    print_section(
        "8. Intensity Summary by Class"
    )

    class_intensity_summary = (
        stats_df
        .groupby("label")[
            numeric_columns
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        class_intensity_summary.to_string()
    )

    class_intensity_summary.to_csv(
        DIAGNOSTIC_DIR
        / "intensity_summary_by_class.csv"
    )

    # --------------------------------------------------------
    # 9. INTENSITY SUMMARY BY SOURCE
    # --------------------------------------------------------

    if source_col is not None:

        print_section(
            "9. Intensity Summary by Acquisition Source"
        )

        source_intensity_summary = (
            stats_df
            .groupby("source")[
                numeric_columns
            ]
            .agg(
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            )
        )

        print(
            source_intensity_summary.to_string()
        )

        source_intensity_summary.to_csv(
            DIAGNOSTIC_DIR
            / "intensity_summary_by_source.csv"
        )

    # --------------------------------------------------------
    # 10. LOAD EXACT TRAINED CNN CHECKPOINT
    # --------------------------------------------------------

    print_section(
        "10. Loading Trained CNN Checkpoint"
    )

    if not CHECKPOINT_FILE.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
        weights_only=False,
    )

    saved_dropout = checkpoint.get(
        "dropout",
        0.30
    )

    model = SMRI3DCNN(
        dropout=saved_dropout
    )

    # --------------------------------------------------------
    # EXACT CHECKPOINT LOADING
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Checkpoint epoch       : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Validation AUC         : "
        f"{checkpoint.get('validation_auc')}"
    )

    print(
        f"Random seed            : "
        f"{checkpoint.get('random_seed')}"
    )

    print(
        f"Saved input shape      : "
        f"{checkpoint.get('input_shape')}"
    )

    print(
        f"Dropout                : "
        f"{checkpoint.get('dropout')}"
    )

    print(
        f"Batch size             : "
        f"{checkpoint.get('batch_size')}"
    )

    print(
        f"Learning rate          : "
        f"{checkpoint.get('learning_rate')}"
    )

    print(
        f"Weight decay           : "
        f"{checkpoint.get('weight_decay')}"
    )

    print(
        f"Positive class weight  : "
        f"{checkpoint.get('pos_weight')}"
    )

    print(
        f"Model parameters       : "
        f"{parameter_count:,}"
    )

    print(
        "Checkpoint loading     : PASS"
    )

    # --------------------------------------------------------
    # 11. CNN INFERENCE ON ALL SUBJECTS
    # --------------------------------------------------------

    print_section(
        "11. CNN Logit / Probability Diagnostics"
    )

    prediction_rows = []

    with torch.no_grad():

        for index, row in combined.iterrows():

            subject_id = str(
                row["subject_id"]
            )

            tensor_path = (
                IMAGE_DIR
                / f"{subject_id}.npy"
            )

            if not tensor_path.exists():

                continue

            try:

                array = np.load(
                    tensor_path
                )

                if (
                    array.shape
                    != EXPECTED_TENSOR_SHAPE
                ):

                    continue

                tensor = torch.from_numpy(
                    array
                ).unsqueeze(0)

                tensor = tensor.to(
                    device=device,
                    dtype=torch.float32
                )

                logits = model(
                    tensor
                )

                probabilities = (
                    torch.sigmoid(
                        logits
                    )
                )

                logit_value = float(
                    logits
                    .squeeze()
                    .cpu()
                    .item()
                )

                probability_value = float(
                    probabilities
                    .squeeze()
                    .cpu()
                    .item()
                )

                result = {

                    "subject_id":
                        subject_id,

                    "split":
                        row["split"],

                    "label":
                        int(row["label"]),

                    "logit":
                        logit_value,

                    "probability":
                        probability_value,
                }

                if source_col is not None:

                    result["source"] = str(
                        row[source_col]
                    )

                prediction_rows.append(
                    result
                )

            except Exception as exc:

                print(
                    f"WARNING: inference failed "
                    f"for {subject_id}: {exc}"
                )

            processed = index + 1

            if processed % 25 == 0:

                print(
                    f"Inference processed: "
                    f"{processed}/{len(combined)}"
                )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_path = (
        DIAGNOSTIC_DIR
        / "cnn_diagnostic_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print()

    print(
        f"Inference subjects: "
        f"{len(predictions_df)}"
    )

    # --------------------------------------------------------
    # 12. PREDICTION SUMMARY BY SPLIT
    # --------------------------------------------------------

    print_section(
        "12. Prediction Summary by Split"
    )

    prediction_summary = (
        predictions_df
        .groupby("split")
        .agg(

            subjects=(
                "subject_id",
                "count"
            ),

            positive_cases=(
                "label",
                "sum"
            ),

            logit_mean=(
                "logit",
                "mean"
            ),

            logit_std=(
                "logit",
                "std"
            ),

            logit_min=(
                "logit",
                "min"
            ),

            logit_max=(
                "logit",
                "max"
            ),

            probability_mean=(
                "probability",
                "mean"
            ),

            probability_std=(
                "probability",
                "std"
            ),

            probability_min=(
                "probability",
                "min"
            ),

            probability_max=(
                "probability",
                "max"
            ),
        )
        .reset_index()
    )

    print(
        prediction_summary.to_string(
            index=False
        )
    )

    prediction_summary.to_csv(
        DIAGNOSTIC_DIR
        / "prediction_summary_by_split.csv",
        index=False
    )

    # --------------------------------------------------------
    # 13. AUC BY SPLIT
    # --------------------------------------------------------

    print_section(
        "13. AUC by Split"
    )

    auc_rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = (
            predictions_df[
                predictions_df["split"]
                == split_name
            ]
        )

        auc = safe_auc(
            subset["label"].values,
            subset["probability"].values
        )

        auc_rows.append({

            "split":
                split_name,

            "subjects":
                len(subset),

            "controls":
                int(
                    (
                        subset["label"]
                        == 0
                    ).sum()
                ),

            "schizophrenia":
                int(
                    (
                        subset["label"]
                        == 1
                    ).sum()
                ),

            "auc":
                auc,
        })

        if np.isnan(auc):

            print(
                f"{split_name:12s} "
                f"subjects={len(subset):3d} "
                f"AUC=NA"
            )

        else:

            print(
                f"{split_name:12s} "
                f"subjects={len(subset):3d} "
                f"AUC={auc:.6f}"
            )

    auc_df = pd.DataFrame(
        auc_rows
    )

    auc_df.to_csv(
        DIAGNOSTIC_DIR
        / "auc_by_split.csv",
        index=False
    )

    # --------------------------------------------------------
    # 14. PROBABILITY DISTRIBUTION BY CLASS
    # --------------------------------------------------------

    print_section(
        "14. Probability Distribution by Class"
    )

    class_prediction_summary = (
        predictions_df
        .groupby(
            [
                "split",
                "label"
            ]
        )
        .agg(

            subjects=(
                "subject_id",
                "count"
            ),

            logit_mean=(
                "logit",
                "mean"
            ),

            logit_std=(
                "logit",
                "std"
            ),

            logit_min=(
                "logit",
                "min"
            ),

            logit_max=(
                "logit",
                "max"
            ),

            probability_mean=(
                "probability",
                "mean"
            ),

            probability_std=(
                "probability",
                "std"
            ),

            probability_min=(
                "probability",
                "min"
            ),

            probability_max=(
                "probability",
                "max"
            ),
        )
        .reset_index()
    )

    print(
        class_prediction_summary.to_string(
            index=False
        )
    )

    class_prediction_summary.to_csv(
        DIAGNOSTIC_DIR
        / "prediction_summary_by_class.csv",
        index=False
    )

    # --------------------------------------------------------
    # 15. SOURCE-WISE CNN PREDICTIONS
    # --------------------------------------------------------

    if source_col is not None:

        print_section(
            "15. CNN Predictions by Acquisition Source"
        )

        source_prediction_summary = (
            predictions_df
            .groupby(
                [
                    "split",
                    "source"
                ]
            )
            .agg(

                subjects=(
                    "subject_id",
                    "count"
                ),

                controls=(
                    "label",
                    lambda x:
                    int(
                        (x == 0).sum()
                    )
                ),

                schizophrenia=(
                    "label",
                    lambda x:
                    int(
                        (x == 1).sum()
                    )
                ),

                logit_mean=(
                    "logit",
                    "mean"
                ),

                logit_std=(
                    "logit",
                    "std"
                ),

                probability_mean=(
                    "probability",
                    "mean"
                ),

                probability_std=(
                    "probability",
                    "std"
                ),

                probability_min=(
                    "probability",
                    "min"
                ),

                probability_max=(
                    "probability",
                    "max"
                ),
            )
            .reset_index()
        )

        print(
            source_prediction_summary.to_string(
                index=False
            )
        )

        source_prediction_summary.to_csv(
            DIAGNOSTIC_DIR
            / "prediction_summary_by_source.csv",
            index=False
        )

    # --------------------------------------------------------
    # 16. SOURCE-ONLY DIAGNOSTIC
    # --------------------------------------------------------

    source_auc = np.nan

    if source_col is not None:

        print_section(
            "16. Source-Only Diagnostic"
        )

        source_values = (
            predictions_df[
                "source"
            ]
            .astype(str)
            .str.upper()
        )

        unique_sources = sorted(
            source_values.unique()
        )

        print(
            "Unique sources:"
        )

        print(
            unique_sources
        )

        # RMS = 1
        # LEGACY = 0
        source_score = np.where(
            source_values
            .str.contains("RMS"),
            1.0,
            0.0
        )

        source_auc = safe_auc(
            predictions_df[
                "label"
            ].values,
            source_score
        )

        source_prediction_df = (
            predictions_df[
                [
                    "subject_id",
                    "split",
                    "label",
                    "source",
                ]
            ].copy()
        )

        source_prediction_df[
            "source_score"
        ] = source_score

        source_prediction_df.to_csv(
            DIAGNOSTIC_DIR
            / "source_only_diagnostic.csv",
            index=False
        )

        if np.isnan(source_auc):

            print(
                "Source-only AUC: NA"
            )

        else:

            print(
                f"Source-only AUC "
                f"(RMS=1, LEGACY=0): "
                f"{source_auc:.6f}"
            )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "This is a diagnostic only."
        )

        print(
            "A high source-only AUC indicates "
            "strong acquisition/source-label "
            "confounding."
        )

    # --------------------------------------------------------
    # 17. PROBABILITY COMPRESSION CHECK
    # --------------------------------------------------------

    print_section(
        "17. Probability Compression Check"
    )

    global_probability_min = (
        predictions_df[
            "probability"
        ].min()
    )

    global_probability_max = (
        predictions_df[
            "probability"
        ].max()
    )

    global_probability_std = (
        predictions_df[
            "probability"
        ].std()
    )

    global_logit_min = (
        predictions_df[
            "logit"
        ].min()
    )

    global_logit_max = (
        predictions_df[
            "logit"
        ].max()
    )

    global_logit_std = (
        predictions_df[
            "logit"
        ].std()
    )

    print(
        f"Probability range : "
        f"{global_probability_min:.6f} .. "
        f"{global_probability_max:.6f}"
    )

    print(
        f"Probability std   : "
        f"{global_probability_std:.6f}"
    )

    print(
        f"Logit range       : "
        f"{global_logit_min:.6f} .. "
        f"{global_logit_max:.6f}"
    )

    print(
        f"Logit std         : "
        f"{global_logit_std:.6f}"
    )

    probability_compressed = (
        global_probability_std < 0.01
    )

    if probability_compressed:

        print()

        print(
            "WARNING: CNN probabilities "
            "are highly compressed."
        )

    else:

        print()

        print(
            "CNN probabilities show "
            "measurable variation."
        )

    # --------------------------------------------------------
    # 18. DISTRIBUTION SHIFT CHECK
    # --------------------------------------------------------

    print_section(
        "18. Distribution Shift Check"
    )

    distribution_rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = stats_df[
            stats_df["split"]
            == split_name
        ]

        for column in [

            "tensor_mean",

            "tensor_std",

            "nonzero_fraction",

            "brain_mean",

            "brain_std",

            "brain_p50",

            "brain_p99",
        ]:

            distribution_rows.append({

                "split":
                    split_name,

                "feature":
                    column,

                "mean":
                    subset[column].mean(),

                "std":
                    subset[column].std(),

                "min":
                    subset[column].min(),

                "max":
                    subset[column].max(),
            })

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    print(
        distribution_df.to_string(
            index=False
        )
    )

    distribution_df.to_csv(
        DIAGNOSTIC_DIR
        / "distribution_shift_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # 19. MODEL OUTPUT BY CLASS AND SOURCE
    # --------------------------------------------------------

    print_section(
        "19. Detailed Model Output Diagnostics"
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        print()

        print(
            f"--- {split_name.upper()} ---"
        )

        subset = predictions_df[
            predictions_df["split"]
            == split_name
        ]

        for label_value, label_name in [
            (0, "CONTROL"),
            (1, "SCHIZOPHRENIA"),
        ]:

            class_subset = subset[
                subset["label"]
                == label_value
            ]

            if len(class_subset) == 0:

                continue

            print(
                f"{label_name:15s} "
                f"N={len(class_subset):3d} "
                f"logit mean="
                f"{class_subset['logit'].mean():.6f} "
                f"prob mean="
                f"{class_subset['probability'].mean():.6f} "
                f"prob std="
                f"{class_subset['probability'].std():.6f}"
            )

    # --------------------------------------------------------
    # 20. FINAL DIAGNOSTIC REPORT
    # --------------------------------------------------------

    print_section(
        "20. Creating Diagnostic Report"
    )

    report_lines = []

    report_lines.append(
        "NeuroFusion AI - "
        "sMRI 3D CNN Diagnostic Report"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        f"Seed: {SEED}"
    )

    report_lines.append(
        f"Expected tensor shape: "
        f"{EXPECTED_TENSOR_SHAPE}"
    )

    report_lines.append(
        f"Device: {device}"
    )

    report_lines.append("")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    report_lines.append(
        "DATASET"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Train subjects: "
        f"{len(train_df)}"
    )

    report_lines.append(
        f"Validation subjects: "
        f"{len(val_df)}"
    )

    report_lines.append(
        f"Test subjects: "
        f"{len(test_df)}"
    )

    report_lines.append(
        f"Total subjects: "
        f"{len(combined)}"
    )

    report_lines.append(
        f"Duplicate subjects: "
        f"{len(duplicate_ids)}"
    )

    report_lines.append(
        f"Tensor loading failures: "
        f"{len(failed_tensors)}"
    )

    report_lines.append("")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    report_lines.append(
        "CLASS DISTRIBUTION"
    )

    report_lines.append(
        "-" * 40
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = combined[
            combined["split"]
            == split_name
        ]

        controls = int(
            (
                subset["label"]
                == 0
            ).sum()
        )

        schizophrenia = int(
            (
                subset["label"]
                == 1
            ).sum()
        )

        report_lines.append(
            f"{split_name}: "
            f"controls={controls}, "
            f"schizophrenia={schizophrenia}"
        )

    report_lines.append("")

    # --------------------------------------------------------
    # SOURCE INFORMATION
    # --------------------------------------------------------

    if source_col is not None:

        report_lines.append(
            "ACQUISITION SOURCE"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            f"Source column: "
            f"{source_col}"
        )

        source_counts = (
            predictions_df[
                "source"
            ]
            .value_counts()
        )

        for source, count in (
            source_counts.items()
        ):

            report_lines.append(
                f"{source}: {count}"
            )

        report_lines.append("")

        if np.isnan(source_auc):

            report_lines.append(
                "Source-only AUC: NA"
            )

        else:

            report_lines.append(
                f"Source-only AUC: "
                f"{source_auc:.6f}"
            )

        report_lines.append("")

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    report_lines.append(
        "CNN CHECKPOINT"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    report_lines.append(
        f"Validation AUC at checkpoint: "
        f"{checkpoint.get('validation_auc')}"
    )

    report_lines.append(
        f"Input shape: "
        f"{checkpoint.get('input_shape')}"
    )

    report_lines.append(
        f"Dropout: "
        f"{checkpoint.get('dropout')}"
    )

    report_lines.append(
        f"Batch size: "
        f"{checkpoint.get('batch_size')}"
    )

    report_lines.append(
        f"Learning rate: "
        f"{checkpoint.get('learning_rate')}"
    )

    report_lines.append(
        f"Weight decay: "
        f"{checkpoint.get('weight_decay')}"
    )

    report_lines.append(
        f"Positive class weight: "
        f"{checkpoint.get('pos_weight')}"
    )

    report_lines.append("")

    # --------------------------------------------------------
    # MODEL OUTPUT
    # --------------------------------------------------------

    report_lines.append(
        "CNN PREDICTIONS"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Probability min: "
        f"{global_probability_min:.6f}"
    )

    report_lines.append(
        f"Probability max: "
        f"{global_probability_max:.6f}"
    )

    report_lines.append(
        f"Probability std: "
        f"{global_probability_std:.6f}"
    )

    report_lines.append(
        f"Logit min: "
        f"{global_logit_min:.6f}"
    )

    report_lines.append(
        f"Logit max: "
        f"{global_logit_max:.6f}"
    )

    report_lines.append(
        f"Logit std: "
        f"{global_logit_std:.6f}"
    )

    report_lines.append("")

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    report_lines.append(
        "AUC BY SPLIT"
    )

    report_lines.append(
        "-" * 40
    )

    for _, row in auc_df.iterrows():

        auc_value = row["auc"]

        if pd.isna(auc_value):

            auc_text = "NA"

        else:

            auc_text = (
                f"{auc_value:.6f}"
            )

        report_lines.append(
            f"{row['split']}: "
            f"AUC={auc_text}"
        )

    report_lines.append("")

    # --------------------------------------------------------
    # INTERPRETATION FLAGS
    # --------------------------------------------------------

    report_lines.append(
        "INTERPRETATION FLAGS"
    )

    report_lines.append(
        "-" * 40
    )

    if probability_compressed:

        report_lines.append(
            "WARNING: CNN probabilities "
            "are highly compressed."
        )

    else:

        report_lines.append(
            "CNN probabilities show "
            "measurable variation."
        )

    if len(duplicate_ids) == 0:

        report_lines.append(
            "PASS: No subject overlap detected."
        )

    else:

        report_lines.append(
            "WARNING: Subject overlap detected."
        )

    if len(failed_tensors) == 0:

        report_lines.append(
            "PASS: All expected tensors "
            "were readable."
        )

    else:

        report_lines.append(
            "WARNING: Tensor loading "
            "failures detected."
        )

    if (
        len(predictions_df)
        == len(combined)
    ):

        report_lines.append(
            "PASS: CNN inference completed "
            "for all subjects."
        )

    else:

        report_lines.append(
            "WARNING: CNN inference did "
            "not complete for all subjects."
        )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        "This diagnostic does not retrain "
        "or modify the CNN."
    )

    report_lines.append(
        "The existing test evaluation "
        "files are preserved."
    )

    report_lines.append(
        "Diagnostic findings should not be "
        "used to tune the existing test set."
    )

    report_lines.append(
        "Future model selection should use "
        "training/development data or a "
        "proper cross-validation protocol."
    )

    report_path = (
        DIAGNOSTIC_DIR
        / "smri_3d_cnn_diagnostic_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    # --------------------------------------------------------
    # 21. FINAL STATUS
    # --------------------------------------------------------

    print_section(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        f"Subjects analyzed      : "
        f"{len(combined)}"
    )

    print(
        f"Tensor statistics      : "
        f"{len(stats_df)}"
    )

    print(
        f"CNN predictions        : "
        f"{len(predictions_df)}"
    )

    print(
        f"Duplicate subjects     : "
        f"{len(duplicate_ids)}"
    )

    print(
        f"Tensor failures        : "
        f"{len(failed_tensors)}"
    )

    print(
        f"Probability range      : "
        f"{global_probability_min:.6f} .. "
        f"{global_probability_max:.6f}"
    )

    print(
        f"Probability std        : "
        f"{global_probability_std:.6f}"
    )

    print()

    print(
        f"Report: "
        f"{report_path}"
    )

    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print(
        f"Tensor statistics: "
        f"{stats_path}"
    )

    print()

    print(
        "FINAL STATUS: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()