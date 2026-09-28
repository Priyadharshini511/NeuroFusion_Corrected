from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd


# ================================================================
# NeuroFusion AI
# Exp20 - NEW PATIENT INFERENCE
# ================================================================
#
# This script performs inference using the already-trained
# exp20_final_deployment_model.joblib.
#
# It DOES NOT retrain the model.
#
# Required input:
#   subject_id
#   age
#   gender_code
#   116 GMVOL_AAL_* features
#
# gender_code MUST use the same coding as the training
# demographic table:
#       1
#       2
#
# Do NOT pass "male"/"female" directly.
#
# ================================================================


PROJECT_ROOT = Path(".")

MODEL_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "smri"
    / "final_model"
    / "exp20_final_deployment_model.joblib"
)


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Deployment model not found:\n"
            f"{MODEL_FILE.resolve()}"
        )

    package = joblib.load(
        MODEL_FILE
    )

    required = [
        "model",
        "calibrator",
        "feature_order",
        "nuisance_columns",
    ]

    for key in required:

        if key not in package:
            raise ValueError(
                f"Saved model package missing: {key}"
            )

    return package


# ================================================================
# BUILD NUISANCE MATRIX
# ================================================================

def build_nuisance(
    df,
    nuisance_columns
):

    if "age" not in df.columns:
        raise ValueError(
            "Input must contain an 'age' column."
        )

    if "gender_code" not in df.columns:
        raise ValueError(
            "Input must contain a 'gender_code' column."
        )

    age = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    if age.isna().any():
        raise ValueError(
            "Input contains missing or invalid age."
        )

    gender = (
        df["gender_code"]
        .astype(str)
        .str.strip()
    )

    allowed = {
        "1",
        "2"
    }

    invalid_gender = sorted(
        set(gender) - allowed
    )

    if invalid_gender:
        raise ValueError(
            f"Invalid gender codes: {invalid_gender}. "
            f"Expected only 1 or 2."
        )

    nuisance = pd.DataFrame(
        {
            "age": age.astype(float)
        }
    )

    if "gender_2" in nuisance_columns:

        nuisance["gender_2"] = (
            gender == "2"
        ).astype(float)

    nuisance = nuisance.reindex(
        columns=nuisance_columns,
        fill_value=0.0
    )

    return nuisance.to_numpy(
        dtype=float
    )


# ================================================================
# LOAD FEATURES
# ================================================================

def load_input(
    input_file,
    feature_order
):

    input_file = Path(
        input_file
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file.resolve()}"
        )

    df = pd.read_csv(
        input_file
    )

    if "subject_id" not in df.columns:
        raise ValueError(
            "Input must contain 'subject_id'."
        )

    missing = [
        f
        for f in feature_order
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required "
            f"GM-volume features."
        )

    X = df[
        feature_order
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "GM-volume features contain "
            "NaN or infinite values."
        )

    return df, X


# ================================================================
# INFERENCE
# ================================================================

def run_inference(
    package,
    df,
    X
):

    model = package["model"]

    calibrator = package[
        "calibrator"
    ]

    nuisance_columns = package[
        "nuisance_columns"
    ]

    nuisance = build_nuisance(
        df,
        nuisance_columns
    )

    # ------------------------------------------------------------
    # Same transformation sequence as final Exp20 model
    # ------------------------------------------------------------

    residualized = (
        model["residualizer"]
        .transform(
            X,
            nuisance
        )
    )

    selected = (
        model["selector"]
        .transform(
            residualized
        )
    )

    scaled = (
        model["scaler"]
        .transform(
            selected
        )
    )

    decision_score = (
        model["svm"]
        .decision_function(
            scaled
        )
    )

    probability = (
        calibrator
        .predict_proba(
            decision_score.reshape(
                -1,
                1
            )
        )[:, 1]
    )

    predicted_class = (
        probability >= 0.5
    ).astype(int)

    return (
        decision_score,
        probability,
        predicted_class
    )


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NeuroFusion AI Exp20 "
            "new-patient sMRI inference."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV containing subject_id, age, "
            "gender_code and 116 GMVOL_AAL features."
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path."
        )
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "NeuroFusion AI - Exp20 NEW PATIENT INFERENCE"
    )
    print("=" * 80)

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    print()
    print("[1/4] Loading deployment model...")

    package = load_model()

    print("  Model: PASS")

    print(
        f"  Features: "
        f"{len(package['feature_order'])}"
    )

    print(
        f"  Selected k: "
        f"{package['selected_k']}"
    )

    print(
        f"  Selected C: "
        f"{package['selected_C']}"
    )

    # ------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------

    print()
    print("[2/4] Loading patient features...")

    df, X = load_input(
        args.input,
        package["feature_order"]
    )

    print(
        f"  Subjects: {len(df)}"
    )

    print(
        "  Feature validation: PASS"
    )

    # ------------------------------------------------------------
    # NUISANCE
    # ------------------------------------------------------------

    print()
    print("[3/4] Validating nuisance variables...")

    nuisance = build_nuisance(
        df,
        package["nuisance_columns"]
    )

    print(
        f"  Nuisance schema: "
        f"{package['nuisance_columns']}"
    )

    print(
        "  Age validation: PASS"
    )

    print(
        "  Gender-code validation: PASS"
    )

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

    print()
    print("[4/4] Running sMRI prediction...")

    (
        decision_score,
        probability,
        predicted_class
    ) = run_inference(
        package,
        df,
        X
    )

    results = pd.DataFrame(
        {
            "subject_id":
                df["subject_id"].astype(str),

            "smri_decision_score":
                decision_score,

            "smri_probability":
                probability,

            "predicted_class":
                predicted_class
        }
    )

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------

    if args.output:

        output_file = Path(
            args.output
        )

    else:

        output_file = (
            PROJECT_ROOT
            / "processed_data"
            / "smri"
            / "final_model"
            / "new_patient_predictions.csv"
        )

    results.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)

    print()
    print(results.to_string(
        index=False
    ))

    print()
    print("Output:")
    print(
        output_file.resolve()
    )

    print()
    print("Output columns:")
    print(
        "  smri_decision_score"
    )
    print(
        "  smri_probability"
    )
    print(
        "  predicted_class"
    )

    print()
    print("IMPORTANT:")
    print(
        "This is an sMRI-only prediction."
    )

    print(
        "It is not the final multimodal "
        "NeuroFusion prediction."
    )

    print(
        "The input MRI must have undergone "
        "the same preprocessing/VBM/AAL "
        "pipeline used for training."
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()