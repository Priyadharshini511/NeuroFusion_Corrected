"""
NeuroFusion AI
sMRI Branch - RMS-Only 3D CNN Diagnostic

Purpose
-------
Evaluate the EXISTING trained sMRI 3D CNN only on RMS-acquisition
subjects.

This is a diagnostic experiment.

It DOES NOT:
    - retrain the model
    - modify the dataset
    - modify train/validation/test splits
    - modify the checkpoint
    - modify existing evaluation files

The purpose is to determine whether the poor held-out performance
is related to the mixed LEGACY/RMS acquisition composition.

Important:
The existing checkpoint was trained on the complete 172-subject
dataset. This script only filters its predictions to RMS subjects.
It does NOT retrain an RMS-only model.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from smri_3d_cnn import SMRI3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = Path(
    "processed_data/smri/cnn_dataset"
)

IMAGE_DIR = (
    BASE_DIR / "images"
)

SPLIT_DIR = (
    BASE_DIR / "splits"
)

TRAINING_DIR = (
    BASE_DIR / "training"
)

DIAGNOSTIC_DIR = (
    BASE_DIR / "diagnostics"
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

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
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


def load_split(
    path,
    split_name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{split_name} split not found:\n"
            f"{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required_columns = [
        "subject_id",
        "label",
    ]

    for column in required_columns:

        if column not in dataframe.columns:

            raise ValueError(
                f"{split_name} split is missing "
                f"required column: {column}"
            )

    dataframe = dataframe.copy()

    dataframe["split"] = (
        split_name
    )

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .astype(str)
    )

    return dataframe


def safe_auc(
    labels,
    probabilities,
):

    labels = np.asarray(
        labels
    )

    probabilities = np.asarray(
        probabilities
    )

    if len(
        np.unique(labels)
    ) < 2:

        return np.nan

    return roc_auc_score(
        labels,
        probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "NeuroFusion AI - RMS-Only 3D CNN Diagnostic"
    )

    print(
        f"Seed            : {SEED}"
    )

    print(
        f"Expected tensor : "
        f"{EXPECTED_TENSOR_SHAPE}"
    )


    # ========================================================
    # CREATE DIAGNOSTIC DIRECTORY
    # ========================================================

    DIAGNOSTIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 1. LOAD SPLITS
    # ========================================================

    print_section(
        "1. Loading Dataset Splits"
    )

    train_df = load_split(
        TRAIN_FILE,
        "train"
    )

    validation_df = load_split(
        VALIDATION_FILE,
        "validation"
    )

    test_df = load_split(
        TEST_FILE,
        "test"
    )

    all_df = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True
    )

    print(
        f"Total subjects: "
        f"{len(all_df)}"
    )


    # ========================================================
    # 2. LOAD METADATA
    # ========================================================

    print_section(
        "2. Loading Acquisition Metadata"
    )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    metadata_df = pd.read_csv(
        METADATA_FILE
    )

    required_metadata = [
        "subject_id",
        "acquisition_source",
    ]

    for column in required_metadata:

        if column not in metadata_df.columns:

            raise ValueError(
                f"metadata.csv is missing "
                f"column: {column}"
            )

    metadata_df["subject_id"] = (
        metadata_df["subject_id"]
        .astype(str)
    )

    metadata_df[
        "acquisition_source"
    ] = (
        metadata_df[
            "acquisition_source"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    all_df = all_df.merge(
        metadata_df[
            [
                "subject_id",
                "acquisition_source",
            ]
        ],
        on="subject_id",
        how="left",
        validate="one_to_one"
    )

    missing_sources = (
        all_df[
            "acquisition_source"
        ]
        .isna()
        .sum()
    )

    print(
        f"Subjects without source metadata: "
        f"{missing_sources}"
    )


    # ========================================================
    # 3. FILTER RMS SUBJECTS
    # ========================================================

    print_section(
        "3. RMS-Only Cohort"
    )

    rms_df = all_df[
        all_df[
            "acquisition_source"
        ] == "RMS"
    ].copy()

    rms_df = rms_df.sort_values(
        [
            "split",
            "subject_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"RMS subjects: "
        f"{len(rms_df)}"
    )

    print()

    rms_class_distribution = (
        rms_df
        .groupby(
            [
                "split",
                "label",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        rms_class_distribution
    )

    rms_df.to_csv(
        DIAGNOSTIC_DIR
        / "rms_only_subjects.csv",
        index=False
    )


    # ========================================================
    # 4. CHECK RMS TENSORS
    # ========================================================

    print_section(
        "4. RMS Tensor Validation"
    )

    tensor_failures = []

    valid_subjects = []

    for _, row in rms_df.iterrows():

        subject_id = (
            row["subject_id"]
        )

        tensor_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not tensor_path.exists():

            tensor_failures.append(
                f"{subject_id}: "
                f"tensor missing"
            )

            continue

        try:

            array = np.load(
                tensor_path
            )

            if (
                array.shape
                != EXPECTED_TENSOR_SHAPE
            ):

                tensor_failures.append(
                    f"{subject_id}: "
                    f"invalid shape "
                    f"{array.shape}"
                )

                continue

            if array.dtype != np.float32:

                tensor_failures.append(
                    f"{subject_id}: "
                    f"invalid dtype "
                    f"{array.dtype}"
                )

                continue

            if not np.isfinite(
                array
            ).all():

                tensor_failures.append(
                    f"{subject_id}: "
                    f"non-finite values"
                )

                continue

            valid_subjects.append(
                subject_id
            )

        except Exception as exc:

            tensor_failures.append(
                f"{subject_id}: "
                f"{exc}"
            )

    print(
        f"Valid RMS tensors : "
        f"{len(valid_subjects)}"
    )

    print(
        f"Tensor failures   : "
        f"{len(tensor_failures)}"
    )


    # ========================================================
    # 5. LOAD EXISTING CHECKPOINT
    # ========================================================

    print_section(
        "5. Loading Existing CNN Checkpoint"
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
        weights_only=False
    )

    dropout = checkpoint.get(
        "dropout",
        0.30
    )

    model = SMRI3DCNN(
        dropout=dropout
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Validation AUC: "
        f"{checkpoint.get('validation_auc')}"
    )

    print(
        "Checkpoint loading: PASS"
    )


    # ========================================================
    # 6. RMS-ONLY INFERENCE
    # ========================================================

    print_section(
        "6. RMS-Only CNN Inference"
    )

    prediction_rows = []

    inference_failures = []

    for index, row in rms_df.iterrows():

        subject_id = (
            row["subject_id"]
        )

        tensor_path = (
            IMAGE_DIR
            / f"{subject_id}.npy"
        )

        if not tensor_path.exists():

            inference_failures.append(
                f"{subject_id}: "
                f"tensor missing"
            )

            continue

        try:

            array = np.load(
                tensor_path
            )

            tensor = torch.from_numpy(
                array
            )

            tensor = tensor.unsqueeze(
                0
            )

            tensor = tensor.to(
                device=device,
                dtype=torch.float32
            )

            with torch.no_grad():

                logits = model(
                    tensor
                )

                probabilities = (
                    torch.sigmoid(
                        logits
                    )
                )

            logit = float(
                logits
                .squeeze()
                .cpu()
                .item()
            )

            probability = float(
                probabilities
                .squeeze()
                .cpu()
                .item()
            )

            prediction_rows.append({

                "subject_id":
                    subject_id,

                "split":
                    row["split"],

                "label":
                    int(
                        row["label"]
                    ),

                "source":
                    row[
                        "acquisition_source"
                    ],

                "logit":
                    logit,

                "probability":
                    probability,
            })

        except Exception as exc:

            inference_failures.append(
                f"{subject_id}: "
                f"{exc}"
            )

        processed = (
            index + 1
        )

        if (
            processed % 25
            == 0
        ):

            print(
                f"Processed: "
                f"{processed}/"
                f"{len(rms_df)}"
            )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_path = (
        DIAGNOSTIC_DIR
        / "rms_only_cnn_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print()

    print(
        f"Predictions generated: "
        f"{len(predictions_df)}"
    )

    print(
        f"Inference failures: "
        f"{len(inference_failures)}"
    )


    # ========================================================
    # 7. AUC BY SPLIT
    # ========================================================

    print_section(
        "7. RMS-Only AUC by Split"
    )

    auc_rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = predictions_df[
            predictions_df[
                "split"
            ] == split_name
        ]

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
                f"{split_name:12s}"
                f" AUC=NA"
            )

        else:

            print(
                f"{split_name:12s}"
                f" AUC={auc:.6f}"
            )

    auc_df = pd.DataFrame(
        auc_rows
    )

    auc_df.to_csv(
        DIAGNOSTIC_DIR
        / "rms_only_auc_by_split.csv",
        index=False
    )


    # ========================================================
    # 8. RMS PROBABILITY SUMMARY
    # ========================================================

    print_section(
        "8. RMS-Only Probability Summary"
    )

    probability_summary = (
        predictions_df
        .groupby(
            [
                "split",
                "label",
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
        probability_summary.to_string(
            index=False
        )
    )

    probability_summary.to_csv(
        DIAGNOSTIC_DIR
        / "rms_only_probability_summary.csv",
        index=False
    )


    # ========================================================
    # 9. OVERALL RMS PROBABILITY RANGE
    # ========================================================

    print_section(
        "9. RMS-Only Probability Compression"
    )

    if len(predictions_df) > 0:

        probability_min = (
            predictions_df[
                "probability"
            ].min()
        )

        probability_max = (
            predictions_df[
                "probability"
            ].max()
        )

        probability_mean = (
            predictions_df[
                "probability"
            ].mean()
        )

        probability_std = (
            predictions_df[
                "probability"
            ].std()
        )

        logit_min = (
            predictions_df[
                "logit"
            ].min()
        )

        logit_max = (
            predictions_df[
                "logit"
            ].max()
        )

        logit_mean = (
            predictions_df[
                "logit"
            ].mean()
        )

        logit_std = (
            predictions_df[
                "logit"
            ].std()
        )

        print(
            f"Probability min : "
            f"{probability_min:.6f}"
        )

        print(
            f"Probability max : "
            f"{probability_max:.6f}"
        )

        print(
            f"Probability mean: "
            f"{probability_mean:.6f}"
        )

        print(
            f"Probability std : "
            f"{probability_std:.6f}"
        )

        print()

        print(
            f"Logit min       : "
            f"{logit_min:.6f}"
        )

        print(
            f"Logit max       : "
            f"{logit_max:.6f}"
        )

        print(
            f"Logit mean      : "
            f"{logit_mean:.6f}"
        )

        print(
            f"Logit std       : "
            f"{logit_std:.6f}"
        )


    # ========================================================
    # 10. SAVE REPORT
    # ========================================================

    print_section(
        "10. Creating RMS-Only Diagnostic Report"
    )

    report_lines = []

    report_lines.append(
        "NeuroFusion AI"
    )

    report_lines.append(
        "sMRI 3D CNN RMS-Only Diagnostic"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT: This experiment uses the "
        "existing trained checkpoint only."
    )

    report_lines.append(
        "No retraining was performed."
    )

    report_lines.append("")

    report_lines.append(
        f"Total RMS subjects: "
        f"{len(rms_df)}"
    )

    report_lines.append(
        f"Valid RMS tensors: "
        f"{len(valid_subjects)}"
    )

    report_lines.append(
        f"Tensor failures: "
        f"{len(tensor_failures)}"
    )

    report_lines.append(
        f"Inference failures: "
        f"{len(inference_failures)}"
    )

    report_lines.append("")

    report_lines.append(
        "RMS CLASS DISTRIBUTION"
    )

    report_lines.append(
        "-" * 40
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = predictions_df[
            predictions_df[
                "split"
            ] == split_name
        ]

        report_lines.append(
            f"{split_name}: "
            f"{len(subset)} subjects, "
            f"{int((subset['label'] == 0).sum())} controls, "
            f"{int((subset['label'] == 1).sum())} schizophrenia"
        )

    report_lines.append("")

    report_lines.append(
        "RMS-ONLY AUC"
    )

    report_lines.append(
        "-" * 40
    )

    for _, row in auc_df.iterrows():

        if pd.isna(
            row["auc"]
        ):

            auc_text = "NA"

        else:

            auc_text = (
                f"{row['auc']:.6f}"
            )

        report_lines.append(
            f"{row['split']}: "
            f"AUC={auc_text}"
        )

    report_lines.append("")

    if len(predictions_df) > 0:

        report_lines.append(
            "RMS-ONLY CNN OUTPUT"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            f"Probability min: "
            f"{probability_min:.6f}"
        )

        report_lines.append(
            f"Probability max: "
            f"{probability_max:.6f}"
        )

        report_lines.append(
            f"Probability mean: "
            f"{probability_mean:.6f}"
        )

        report_lines.append(
            f"Probability std: "
            f"{probability_std:.6f}"
        )

        report_lines.append(
            f"Logit min: "
            f"{logit_min:.6f}"
        )

        report_lines.append(
            f"Logit max: "
            f"{logit_max:.6f}"
        )

        report_lines.append(
            f"Logit mean: "
            f"{logit_mean:.6f}"
        )

        report_lines.append(
            f"Logit std: "
            f"{logit_std:.6f}"
        )

    report_lines.append("")

    report_lines.append(
        "CHECKPOINT"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Epoch: "
        f"{checkpoint.get('epoch')}"
    )

    report_lines.append(
        f"Original validation AUC: "
        f"{checkpoint.get('validation_auc')}"
    )

    report_lines.append(
        f"Input shape: "
        f"{checkpoint.get('input_shape')}"
    )

    report_lines.append(
        f"Random seed: "
        f"{checkpoint.get('random_seed')}"
    )

    report_path = (
        DIAGNOSTIC_DIR
        / "rms_only_diagnostic_report.txt"
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
    # FINAL STATUS
    # ========================================================

    print_section(
        "RMS-ONLY DIAGNOSTIC SUMMARY"
    )

    print(
        f"RMS subjects      : "
        f"{len(rms_df)}"
    )

    print(
        f"Valid tensors     : "
        f"{len(valid_subjects)}"
    )

    print(
        f"Predictions       : "
        f"{len(predictions_df)}"
    )

    print(
        f"Tensor failures   : "
        f"{len(tensor_failures)}"
    )

    print(
        f"Inference failures: "
        f"{len(inference_failures)}"
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

    print()

    if (
        len(tensor_failures) == 0
        and len(inference_failures) == 0
        and len(predictions_df)
        == len(rms_df)
    ):

        print(
            "FINAL STATUS: PASS"
        )

    else:

        print(
            "FINAL STATUS: REVIEW"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()