"""
NeuroFusion AI
sMRI Branch - Spatial Signal Analysis

Purpose
-------
Investigate whether the current MNI-space sMRI volumes contain
measurable spatial differences between diagnostic groups.

This script is DIAGNOSTIC ONLY.

It DOES NOT:
    - retrain the CNN
    - modify the CNN checkpoint
    - modify train/validation/test splits
    - modify the .npy tensors
    - delete or overwrite source data

It analyzes:
    1. MNI tensor integrity
    2. Global spatial dimensions
    3. Class-wise mean brain volumes
    4. Class-wise standard-deviation volumes
    5. RMS control vs RMS schizophrenia spatial differences
    6. Whole-cohort control vs schizophrenia spatial differences
    7. Voxel-wise standardized effect sizes
    8. Number of informative voxels at several effect-size thresholds
    9. Spatial concentration of the strongest differences

Important
---------
The analysis is descriptive. It does NOT perform formal voxel-wise
multiple-comparison correction and should NOT be interpreted as a
statistical neuroimaging significance map.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd


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

METADATA_FILE = (
    BASE_DIR / "metadata.csv"
)

OUTPUT_DIR = (
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

EXPECTED_TENSOR_SHAPE = (
    1,
    95,
    117,
    99,
)

# Effect-size thresholds for descriptive analysis.
EFFECT_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.50,
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    SEED
)

np.random.seed(
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

    dataframe["subject_id"] = (
        dataframe[
            "subject_id"
        ]
        .astype(str)
    )

    dataframe["split"] = (
        split_name
    )

    return dataframe


def cohens_d(
    group0,
    group1,
):

    """
    Calculate voxel-wise Cohen's d.

    group0 and group1:
        shape = (subjects, voxels)

    Returns:
        shape = (voxels,)
    """

    n0 = group0.shape[0]

    n1 = group1.shape[0]

    mean0 = (
        group0.mean(
            axis=0
        )
    )

    mean1 = (
        group1.mean(
            axis=0
        )
    )

    variance0 = (
        group0.var(
            axis=0,
            ddof=1
        )
    )

    variance1 = (
        group1.var(
            axis=0,
            ddof=1
        )
    )

    pooled_variance = (
        (
            (n0 - 1)
            * variance0
        )
        +
        (
            (n1 - 1)
            * variance1
        )
    ) / (
        n0 + n1 - 2
    )

    pooled_std = np.sqrt(
        np.maximum(
            pooled_variance,
            1e-12
        )
    )

    effect = (
        mean1 - mean0
    ) / pooled_std

    effect[
        ~np.isfinite(effect)
    ] = 0.0

    return effect


def summarize_effect(
    effect,
    group0,
    group1,
    comparison_name,
):

    absolute_effect = np.abs(
        effect
    )

    result = {

        "comparison":
            comparison_name,

        "group0_n":
            int(
                group0.shape[0]
            ),

        "group1_n":
            int(
                group1.shape[0]
            ),

        "mean_effect":
            float(
                np.mean(effect)
            ),

        "mean_absolute_effect":
            float(
                np.mean(
                    absolute_effect
                )
            ),

        "median_absolute_effect":
            float(
                np.median(
                    absolute_effect
                )
            ),

        "max_absolute_effect":
            float(
                np.max(
                    absolute_effect
                )
            ),

        "p95_absolute_effect":
            float(
                np.percentile(
                    absolute_effect,
                    95
                )
            ),

        "p99_absolute_effect":
            float(
                np.percentile(
                    absolute_effect,
                    99
                )
            ),
    }

    for threshold in EFFECT_THRESHOLDS:

        count = int(
            np.sum(
                absolute_effect
                >= threshold
            )
        )

        percentage = (
            100.0
            * count
            / len(
                absolute_effect
            )
        )

        threshold_name = (
            f"voxels_abs_d_ge_"
            f"{str(threshold).replace('.', '_')}"
        )

        percentage_name = (
            f"percent_abs_d_ge_"
            f"{str(threshold).replace('.', '_')}"
        )

        result[
            threshold_name
        ] = count

        result[
            percentage_name
        ] = percentage

    return result


def load_tensor(
    subject_id,
):

    path = (
        IMAGE_DIR
        / f"{subject_id}.npy"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Tensor not found:\n{path}"
        )

    array = np.load(
        path
    )

    if (
        array.shape
        != EXPECTED_TENSOR_SHAPE
    ):

        raise ValueError(
            f"{subject_id}: invalid shape "
            f"{array.shape}; expected "
            f"{EXPECTED_TENSOR_SHAPE}"
        )

    if array.dtype != np.float32:

        raise ValueError(
            f"{subject_id}: invalid dtype "
            f"{array.dtype}; expected float32"
        )

    if not np.isfinite(
        array
    ).all():

        raise ValueError(
            f"{subject_id}: NaN/Inf detected"
        )

    return array


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "NeuroFusion AI - sMRI Spatial Signal Analysis"
    )

    print(
        f"Seed             : {SEED}"
    )

    print(
        f"Expected tensor  : "
        f"{EXPECTED_TENSOR_SHAPE}"
    )

    print(
        "Analysis type    : Diagnostic only"
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
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
        f"Train subjects      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation subjects : "
        f"{len(validation_df)}"
    )

    print(
        f"Test subjects       : "
        f"{len(test_df)}"
    )

    print(
        f"Total subjects      : "
        f"{len(all_df)}"
    )


    # ========================================================
    # 2. LOAD ACQUISITION SOURCE
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
        metadata_df[
            "subject_id"
        ]
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

    print(
        "Acquisition metadata merged."
    )

    print()

    print(
        all_df[
            "acquisition_source"
        ]
        .value_counts(
            dropna=False
        )
    )


    # ========================================================
    # 3. LOAD ALL TENSORS
    # ========================================================

    print_section(
        "3. Loading All MNI Tensors"
    )

    tensor_arrays = []

    tensor_records = []

    failures = []

    for index, row in all_df.iterrows():

        subject_id = (
            row["subject_id"]
        )

        try:

            array = load_tensor(
                subject_id
            )

            # Remove channel dimension.
            volume = array[0]

            tensor_arrays.append(
                volume
            )

            tensor_records.append({

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
            })

        except Exception as exc:

            failures.append(
                f"{subject_id}: {exc}"
            )

        processed = (
            index + 1
        )

        if (
            processed % 25
            == 0
        ):

            print(
                f"Processed tensors: "
                f"{processed}/"
                f"{len(all_df)}"
            )

    if len(failures) > 0:

        print()

        print(
            "Tensor failures:"
        )

        for failure in failures:

            print(
                f"  {failure}"
            )

        raise RuntimeError(
            "Tensor validation failed."
        )

    tensor_array = np.stack(
        tensor_arrays,
        axis=0
    )

    tensor_metadata = pd.DataFrame(
        tensor_records
    )

    print()

    print(
        f"Loaded tensor subjects: "
        f"{tensor_array.shape[0]}"
    )

    print(
        f"Volume shape: "
        f"{tensor_array.shape[1:]}"
    )

    print(
        f"Tensor failures: "
        f"{len(failures)}"
    )


    # ========================================================
    # 4. BASIC SPATIAL STATISTICS
    # ========================================================

    print_section(
        "4. Global Spatial Statistics"
    )

    # Flatten only for statistical computation.
    flat = tensor_array.reshape(
        tensor_array.shape[0],
        -1
    )

    global_statistics = {

        "subjects":
            tensor_array.shape[0],

        "voxels_per_subject":
            tensor_array.shape[1]
            * tensor_array.shape[2]
            * tensor_array.shape[3],

        "global_mean":
            float(
                flat.mean()
            ),

        "global_std":
            float(
                flat.std()
            ),

        "global_min":
            float(
                flat.min()
            ),

        "global_max":
            float(
                flat.max()
            ),

        "nonzero_fraction":
            float(
                np.mean(
                    flat > 0
                )
            ),
    }

    for key, value in global_statistics.items():

        print(
            f"{key:22s}: "
            f"{value}"
        )


    # ========================================================
    # 5. DEFINE GROUPS
    # ========================================================

    print_section(
        "5. Defining Diagnostic Groups"
    )

    labels = (
        tensor_metadata[
            "label"
        ].values
    )

    sources = (
        tensor_metadata[
            "source"
        ]
        .astype(str)
        .values
    )

    group_control = (
        labels == 0
    )

    group_schizophrenia = (
        labels == 1
    )

    group_rms = (
        sources == "RMS"
    )

    group_legacy = (
        sources == "LEGACY"
    )

    group_rms_control = (
        group_rms
        & group_control
    )

    group_rms_schizophrenia = (
        group_rms
        & group_schizophrenia
    )

    print(
        f"All controls              : "
        f"{group_control.sum()}"
    )

    print(
        f"All schizophrenia         : "
        f"{group_schizophrenia.sum()}"
    )

    print(
        f"RMS controls              : "
        f"{group_rms_control.sum()}"
    )

    print(
        f"RMS schizophrenia         : "
        f"{group_rms_schizophrenia.sum()}"
    )

    print(
        f"RMS subjects              : "
        f"{group_rms.sum()}"
    )

    print(
        f"LEGACY subjects           : "
        f"{group_legacy.sum()}"
    )


    # ========================================================
    # 6. CLASS-WISE MEAN VOLUMES
    # ========================================================

    print_section(
        "6. Class-Wise Mean Spatial Volumes"
    )

    mean_control = (
        tensor_array[
            group_control
        ]
        .mean(
            axis=0
        )
    )

    mean_schizophrenia = (
        tensor_array[
            group_schizophrenia
        ]
        .mean(
            axis=0
        )
    )

    mean_difference = (
        mean_schizophrenia
        - mean_control
    )

    np.save(
        OUTPUT_DIR
        / "mean_control_volume.npy",
        mean_control.astype(
            np.float32
        )
    )

    np.save(
        OUTPUT_DIR
        / "mean_schizophrenia_volume.npy",
        mean_schizophrenia.astype(
            np.float32
        )
    )

    np.save(
        OUTPUT_DIR
        / "mean_class_difference_volume.npy",
        mean_difference.astype(
            np.float32
        )
    )

    print(
        f"Control mean range: "
        f"{mean_control.min():.6f} .. "
        f"{mean_control.max():.6f}"
    )

    print(
        f"Schizophrenia mean range: "
        f"{mean_schizophrenia.min():.6f} .. "
        f"{mean_schizophrenia.max():.6f}"
    )

    print(
        f"Mean difference range: "
        f"{mean_difference.min():.6f} .. "
        f"{mean_difference.max():.6f}"
    )


    # ========================================================
    # 7. WHOLE COHORT VOXEL-WISE EFFECT SIZE
    # ========================================================

    print_section(
        "7. Whole-Cohort Voxel-Wise Effect Size"
    )

    control_flat = flat[
        group_control
    ]

    schizophrenia_flat = flat[
        group_schizophrenia
    ]

    whole_cohort_effect = cohens_d(
        control_flat,
        schizophrenia_flat
    )

    whole_cohort_effect_volume = (
        whole_cohort_effect
        .reshape(
            EXPECTED_TENSOR_SHAPE[
                1:
            ]
        )
    )

    np.save(
        OUTPUT_DIR
        / "whole_cohort_cohens_d.npy",
        whole_cohort_effect_volume.astype(
            np.float32
        )
    )

    whole_cohort_summary = summarize_effect(
        whole_cohort_effect,
        control_flat,
        schizophrenia_flat,
        "whole_cohort_control_vs_schizophrenia"
    )

    print(
        f"Mean absolute Cohen's d: "
        f"{whole_cohort_summary['mean_absolute_effect']:.6f}"
    )

    print(
        f"Median absolute Cohen's d: "
        f"{whole_cohort_summary['median_absolute_effect']:.6f}"
    )

    print(
        f"95th percentile |d|: "
        f"{whole_cohort_summary['p95_absolute_effect']:.6f}"
    )

    print(
        f"99th percentile |d|: "
        f"{whole_cohort_summary['p99_absolute_effect']:.6f}"
    )

    print(
        f"Maximum |d|: "
        f"{whole_cohort_summary['max_absolute_effect']:.6f}"
    )


    # ========================================================
    # 8. RMS-ONLY VOXEL-WISE EFFECT SIZE
    # ========================================================

    print_section(
        "8. RMS-Only Voxel-Wise Effect Size"
    )

    rms_control_flat = flat[
        group_rms_control
    ]

    rms_schizophrenia_flat = flat[
        group_rms_schizophrenia
    ]

    rms_effect = cohens_d(
        rms_control_flat,
        rms_schizophrenia_flat
    )

    rms_effect_volume = (
        rms_effect
        .reshape(
            EXPECTED_TENSOR_SHAPE[
                1:
            ]
        )
    )

    np.save(
        OUTPUT_DIR
        / "rms_only_cohens_d.npy",
        rms_effect_volume.astype(
            np.float32
        )
    )

    rms_summary = summarize_effect(
        rms_effect,
        rms_control_flat,
        rms_schizophrenia_flat,
        "rms_only_control_vs_schizophrenia"
    )

    print(
        f"Mean absolute Cohen's d: "
        f"{rms_summary['mean_absolute_effect']:.6f}"
    )

    print(
        f"Median absolute Cohen's d: "
        f"{rms_summary['median_absolute_effect']:.6f}"
    )

    print(
        f"95th percentile |d|: "
        f"{rms_summary['p95_absolute_effect']:.6f}"
    )

    print(
        f"99th percentile |d|: "
        f"{rms_summary['p99_absolute_effect']:.6f}"
    )

    print(
        f"Maximum |d|: "
        f"{rms_summary['max_absolute_effect']:.6f}"
    )


    # ========================================================
    # 9. EFFECT SIZE SUMMARY
    # ========================================================

    print_section(
        "9. Effect-Size Threshold Summary"
    )

    effect_summary_df = pd.DataFrame(
        [
            whole_cohort_summary,
            rms_summary,
        ]
    )

    print(
        effect_summary_df.to_string(
            index=False
        )
    )

    effect_summary_df.to_csv(
        OUTPUT_DIR
        / "spatial_effect_size_summary.csv",
        index=False
    )


    # ========================================================
    # 10. TRAIN / VALIDATION / TEST SPATIAL SIGNAL
    # ========================================================

    print_section(
        "10. Spatial Signal by Split"
    )

    split_effect_rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        split_mask = (
            tensor_metadata[
                "split"
            ].values
            == split_name
        )

        split_labels = labels[
            split_mask
        ]

        split_flat = flat[
            split_mask
        ]

        split_control = (
            split_labels == 0
        )

        split_schizophrenia = (
            split_labels == 1
        )

        control_count = int(
            split_control.sum()
        )

        schizophrenia_count = int(
            split_schizophrenia.sum()
        )

        print()

        print(
            f"{split_name.upper()}"
        )

        print(
            f"Controls: "
            f"{control_count}"
        )

        print(
            f"Schizophrenia: "
            f"{schizophrenia_count}"
        )

        if (
            control_count < 2
            or schizophrenia_count < 2
        ):

            print(
                "Insufficient subjects for "
                "effect-size calculation."
            )

            continue

        split_control_flat = (
            split_flat[
                split_control
            ]
        )

        split_schizophrenia_flat = (
            split_flat[
                split_schizophrenia
            ]
        )

        split_effect = cohens_d(
            split_control_flat,
            split_schizophrenia_flat
        )

        split_summary = summarize_effect(
            split_effect,
            split_control_flat,
            split_schizophrenia_flat,
            split_name
        )

        split_effect_rows.append(
            split_summary
        )

        print(
            f"Mean |d|: "
            f"{split_summary['mean_absolute_effect']:.6f}"
        )

        print(
            f"Median |d|: "
            f"{split_summary['median_absolute_effect']:.6f}"
        )

        print(
            f"95th percentile |d|: "
            f"{split_summary['p95_absolute_effect']:.6f}"
        )

        print(
            f"Maximum |d|: "
            f"{split_summary['max_absolute_effect']:.6f}"
        )

    split_effect_df = pd.DataFrame(
        split_effect_rows
    )

    split_effect_df.to_csv(
        OUTPUT_DIR
        / "spatial_effect_size_by_split.csv",
        index=False
    )


    # ========================================================
    # 11. RMS CONTROL VS LEGACY CONTROL
    # ========================================================

    print_section(
        "11. Acquisition-Lineage Spatial Difference"
    )

    legacy_control_flat = flat[
        group_legacy
    ]

    rms_control_only_flat = flat[
        group_rms_control
    ]

    lineage_effect = cohens_d(
        legacy_control_flat,
        rms_control_only_flat
    )

    lineage_effect_volume = (
        lineage_effect
        .reshape(
            EXPECTED_TENSOR_SHAPE[
                1:
            ]
        )
    )

    np.save(
        OUTPUT_DIR
        / "legacy_vs_rms_control_cohens_d.npy",
        lineage_effect_volume.astype(
            np.float32
        )
    )

    lineage_summary = summarize_effect(
        lineage_effect,
        legacy_control_flat,
        rms_control_only_flat,
        "legacy_control_vs_rms_control"
    )

    print(
        f"LEGACY controls: "
        f"{legacy_control_flat.shape[0]}"
    )

    print(
        f"RMS controls: "
        f"{rms_control_only_flat.shape[0]}"
    )

    print(
        f"Mean |d|: "
        f"{lineage_summary['mean_absolute_effect']:.6f}"
    )

    print(
        f"Median |d|: "
        f"{lineage_summary['median_absolute_effect']:.6f}"
    )

    print(
        f"95th percentile |d|: "
        f"{lineage_summary['p95_absolute_effect']:.6f}"
    )

    print(
        f"Maximum |d|: "
        f"{lineage_summary['max_absolute_effect']:.6f}"
    )


    # ========================================================
    # 12. SAVE ALL COMPARISON SUMMARIES
    # ========================================================

    print_section(
        "12. Saving Comparison Summary"
    )

    comparison_rows = [
        whole_cohort_summary,
        rms_summary,
        lineage_summary,
    ]

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        OUTPUT_DIR
        / "spatial_signal_comparison_summary.csv",
        index=False
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )


    # ========================================================
    # 13. TOP EFFECT-SIZE VOXELS
    # ========================================================

    print_section(
        "13. Strongest Spatial Differences"
    )

    # --------------------------------------------------------
    # Whole cohort
    # --------------------------------------------------------

    absolute_whole = np.abs(
        whole_cohort_effect
    )

    top_n = min(
        1000,
        len(
            absolute_whole
        )
    )

    top_indices = np.argsort(
        absolute_whole
    )[
        -top_n:
    ][::-1]

    top_coordinates = np.unravel_index(
        top_indices,
        EXPECTED_TENSOR_SHAPE[
            1:
        ]
    )

    top_whole_df = pd.DataFrame({

        "z_index":
            top_coordinates[0],

        "y_index":
            top_coordinates[1],

        "x_index":
            top_coordinates[2],

        "cohens_d":
            whole_cohort_effect[
                top_indices
            ],

        "absolute_cohens_d":
            absolute_whole[
                top_indices
            ],
    })

    top_whole_df.to_csv(
        OUTPUT_DIR
        / "top_1000_whole_cohort_effect_voxels.csv",
        index=False
    )

    print(
        "Top whole-cohort effect voxels saved."
    )

    # --------------------------------------------------------
    # RMS only
    # --------------------------------------------------------

    absolute_rms = np.abs(
        rms_effect
    )

    top_n_rms = min(
        1000,
        len(
            absolute_rms
        )
    )

    top_indices_rms = np.argsort(
        absolute_rms
    )[
        -top_n_rms:
    ][::-1]

    rms_coordinates = np.unravel_index(
        top_indices_rms,
        EXPECTED_TENSOR_SHAPE[
            1:
        ]
    )

    top_rms_df = pd.DataFrame({

        "z_index":
            rms_coordinates[0],

        "y_index":
            rms_coordinates[1],

        "x_index":
            rms_coordinates[2],

        "cohens_d":
            rms_effect[
                top_indices_rms
            ],

        "absolute_cohens_d":
            absolute_rms[
                top_indices_rms
            ],
    })

    top_rms_df.to_csv(
        OUTPUT_DIR
        / "top_1000_rms_effect_voxels.csv",
        index=False
    )

    print(
        "Top RMS-only effect voxels saved."
    )


    # ========================================================
    # 14. SPATIAL SUMMARY
    # ========================================================

    print_section(
        "14. Spatial Signal Summary"
    )

    summary = {

        "total_subjects":
            int(
                tensor_array.shape[0]
            ),

        "controls":
            int(
                group_control.sum()
            ),

        "schizophrenia":
            int(
                group_schizophrenia.sum()
            ),

        "rms_subjects":
            int(
                group_rms.sum()
            ),

        "legacy_subjects":
            int(
                group_legacy.sum()
            ),

        "rms_controls":
            int(
                group_rms_control.sum()
            ),

        "rms_schizophrenia":
            int(
                group_rms_schizophrenia.sum()
            ),

        "whole_cohort_mean_abs_d":
            whole_cohort_summary[
                "mean_absolute_effect"
            ],

        "whole_cohort_median_abs_d":
            whole_cohort_summary[
                "median_absolute_effect"
            ],

        "whole_cohort_p95_abs_d":
            whole_cohort_summary[
                "p95_absolute_effect"
            ],

        "whole_cohort_max_abs_d":
            whole_cohort_summary[
                "max_absolute_effect"
            ],

        "rms_mean_abs_d":
            rms_summary[
                "mean_absolute_effect"
            ],

        "rms_median_abs_d":
            rms_summary[
                "median_absolute_effect"
            ],

        "rms_p95_abs_d":
            rms_summary[
                "p95_absolute_effect"
            ],

        "rms_max_abs_d":
            rms_summary[
                "max_absolute_effect"
            ],

        "lineage_mean_abs_d":
            lineage_summary[
                "mean_absolute_effect"
            ],

        "lineage_median_abs_d":
            lineage_summary[
                "median_absolute_effect"
            ],

        "lineage_p95_abs_d":
            lineage_summary[
                "p95_absolute_effect"
            ],

        "lineage_max_abs_d":
            lineage_summary[
                "max_absolute_effect"
            ],
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "spatial_signal_summary.csv",
        index=False
    )

    for key, value in summary.items():

        print(
            f"{key:32s}: "
            f"{value}"
        )


    # ========================================================
    # 15. CREATE TEXT REPORT
    # ========================================================

    print_section(
        "15. Creating Spatial Signal Report"
    )

    report_lines = []

    report_lines.append(
        "NeuroFusion AI"
    )

    report_lines.append(
        "sMRI Spatial Signal Analysis"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        "This is a descriptive diagnostic analysis."
    )

    report_lines.append(
        "It does not perform formal voxel-wise "
        "multiple-comparison correction."
    )

    report_lines.append(
        "Effect sizes should not be interpreted "
        "as statistically significant neuroimaging findings."
    )

    report_lines.append("")

    report_lines.append(
        "COHORT"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Total subjects: "
        f"{tensor_array.shape[0]}"
    )

    report_lines.append(
        f"Controls: "
        f"{group_control.sum()}"
    )

    report_lines.append(
        f"Schizophrenia: "
        f"{group_schizophrenia.sum()}"
    )

    report_lines.append(
        f"RMS subjects: "
        f"{group_rms.sum()}"
    )

    report_lines.append(
        f"LEGACY subjects: "
        f"{group_legacy.sum()}"
    )

    report_lines.append(
        f"RMS controls: "
        f"{group_rms_control.sum()}"
    )

    report_lines.append(
        f"RMS schizophrenia: "
        f"{group_rms_schizophrenia.sum()}"
    )

    report_lines.append("")

    report_lines.append(
        "WHOLE-COHORT CONTROL VS SCHIZOPHRENIA"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Mean |d|: "
        f"{whole_cohort_summary['mean_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"Median |d|: "
        f"{whole_cohort_summary['median_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"95th percentile |d|: "
        f"{whole_cohort_summary['p95_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"99th percentile |d|: "
        f"{whole_cohort_summary['p99_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"Maximum |d|: "
        f"{whole_cohort_summary['max_absolute_effect']:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "RMS-ONLY CONTROL VS SCHIZOPHRENIA"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Mean |d|: "
        f"{rms_summary['mean_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"Median |d|: "
        f"{rms_summary['median_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"95th percentile |d|: "
        f"{rms_summary['p95_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"99th percentile |d|: "
        f"{rms_summary['p99_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"Maximum |d|: "
        f"{rms_summary['max_absolute_effect']:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "LEGACY CONTROL VS RMS CONTROL"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        f"Mean |d|: "
        f"{lineage_summary['mean_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"Median |d|: "
        f"{lineage_summary['median_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"95th percentile |d|: "
        f"{lineage_summary['p95_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"99th percentile |d|: "
        f"{lineage_summary['p99_absolute_effect']:.6f}"
    )

    report_lines.append(
        f"Maximum |d|: "
        f"{lineage_summary['max_absolute_effect']:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT INTERPRETATION"
    )

    report_lines.append(
        "-" * 40
    )

    report_lines.append(
        "A large spatial effect does not by itself "
        "prove disease-specific anatomy."
    )

    report_lines.append(
        "Acquisition-lineage differences are "
        "especially important because LEGACY "
        "and RMS are strongly associated with "
        "diagnostic class in the current cohort."
    )

    report_lines.append(
        "RMS-only results are therefore particularly "
        "important for assessing disease-related "
        "spatial signal."
    )

    report_lines.append(
        "The analysis is descriptive and should "
        "not be used as a substitute for a properly "
        "validated predictive model."
    )

    report_path = (
        OUTPUT_DIR
        / "smri_spatial_signal_analysis_report.txt"
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
    # FINAL SUMMARY
    # ========================================================

    print_section(
        "SPATIAL SIGNAL ANALYSIS SUMMARY"
    )

    print(
        f"Subjects analyzed : "
        f"{tensor_array.shape[0]}"
    )

    print(
        f"Tensor shape      : "
        f"{tensor_array.shape[1:]}"
    )

    print(
        f"Tensor failures   : "
        f"{len(failures)}"
    )

    print()

    print(
        f"Whole-cohort mean |d| : "
        f"{whole_cohort_summary['mean_absolute_effect']:.6f}"
    )

    print(
        f"RMS-only mean |d|     : "
        f"{rms_summary['mean_absolute_effect']:.6f}"
    )

    print(
        f"Lineage mean |d|      : "
        f"{lineage_summary['mean_absolute_effect']:.6f}"
    )

    print()

    print(
        f"Report: "
        f"{report_path}"
    )

    print()

    print(
        "No model or dataset files were modified."
    )

    print(
        "FINAL STATUS: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()