# from sklearn.model_selection import train_test_split

# from src.data_loader import load_dataset

# from src.profiler import (
#     profile_dataset,
#     detect_feature_types
# )

# from src.problem_detector import (
#     detect_problem_type
# )

# from src.preprocessor import (
#     build_preprocessor
# )

# from src.model_registry import (
#     get_models
# )

# from src.trainer import train_models

# from src.evaluator import (
#     evaluate_classification,
#     evaluate_regression
# )


# def run_autods(
#     file_path,
#     target
# ):

#     print("\n==========================")
#     print(" AUTONOMOUS DATA SCIENTIST")
#     print("==========================\n")

#     # 1. Load dataset

#     df = load_dataset(
#         file_path
#     )

#     print(
#         f"Dataset loaded: "
#         f"{df.shape[0]} rows × "
#         f"{df.shape[1]} columns"
#     )

#     # 2. Profile

#     profile = profile_dataset(
#         df
#     )

#     feature_types = detect_feature_types(
#         df
#     )

#     print(
#         "\nNumeric:",
#         feature_types["numeric"]
#     )

#     print(
#         "Categorical:",
#         feature_types["categorical"]
#     )

#     # 3. Detect problem

#     problem_type = (
#         detect_problem_type(
#             df,
#             target
#         )
#     )

#     print(
#         "\nDetected problem:",
#         problem_type.upper()
#     )

#     # 4. Separate features and target

#     X = df.drop(
#         columns=[target]
#     )

#     y = df[target]

#     # Remove missing targets

#     valid_rows = (
#         y.notna()
#     )

#     X = X.loc[
#         valid_rows
#     ]

#     y = y.loc[
#         valid_rows
#     ]

#     # 5. Split

#     stratify = None

#     if problem_type == "classification":
#         stratify = y

#     X_train, X_test, y_train, y_test = (
#         train_test_split(
#             X,
#             y,
#             test_size=0.20,
#             random_state=42,
#             stratify=stratify
#         )
#     )

#     print(
#         "\nTraining rows:",
#         len(X_train)
#     )

#     print(
#         "Testing rows:",
#         len(X_test)
#     )

#     # 6. Preprocessor

#     preprocessor = (
#         build_preprocessor(
#             X_train
#         )
#     )

#     # 7. Models

#     models = get_models(
#         problem_type
#     )

#     # 8. Evaluator

#     if problem_type == "classification":
#         evaluator = (
#             evaluate_classification
#         )

#     else:
#         evaluator = (
#             evaluate_regression
#         )

#     # 9. Train

#     results, trained_models = (
#         train_models(
#             X_train,
#             X_test,
#             y_train,
#             y_test,
#             preprocessor,
#             models,
#             problem_type,
#             evaluator
#         )
#     )

#     return {
#         "profile": profile,
#         "problem_type": problem_type,
#         "results": results,
#         "models": trained_models
#     }
    
    
# ============================================================
# AutoDS - Main Machine Learning Pipeline
# ============================================================

import json
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Import our AutoDS modules
# ------------------------------------------------------------

from src.data_loader import load_dataset

from src.profiler import (
    profile_dataset,
    detect_feature_types
)

from src.problem_detector import (
    detect_problem_type
)

from src.preprocessor import (
    build_preprocessor
)

from src.model_registry import (
    get_models
)

from src.trainer import (
    train_models
)

from src.evaluator import (
    evaluate_classification,
    evaluate_regression
)


# ============================================================
# MAIN AUTODS PIPELINE
# ============================================================

def run_autods(
    file_path,
    target
):

    print("\n===================================")
    print("      AUTONOMOUS DATA SCIENTIST")
    print("===================================\n")

    # ========================================================
    # STEP 1
    # Load Dataset
    # ========================================================

    print("Loading dataset...")

    df = load_dataset(
        file_path
    )

    print(
        f"\nDataset loaded successfully."
    )

    print(
        f"Rows    : {df.shape[0]}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )


    # ========================================================
    # STEP 2
    # Dataset Profiling
    # ========================================================

    print(
        "\nProfiling dataset..."
    )

    profile = profile_dataset(
        df
    )


    # ========================================================
    # STEP 3
    # Feature Type Detection
    # ========================================================

    feature_types = detect_feature_types(
        df
    )

    print(
        "\nNumeric:",
        feature_types["numeric"]
    )

    print(
        "\nCategorical:",
        feature_types["categorical"]
    )

    print(
        "\nDatetime:",
        feature_types["datetime"]
    )


    # ========================================================
    # STEP 4
    # Problem Type Detection
    # ========================================================

    problem_type = detect_problem_type(
        df,
        target
    )

    print(
        "\nDetected problem:",
        problem_type.upper()
    )


    # ========================================================
    # STEP 5
    # Separate Features and Target
    # ========================================================

    X = df.drop(
        columns=[target]
    )

    y = df[target]


    # ========================================================
    # STEP 6
    # Remove rows where target is missing
    # ========================================================

    valid_rows = y.notna()

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]


    # ========================================================
    # STEP 7
    # Train Test Split
    # ========================================================

    stratify = None

    if problem_type == "classification":

        stratify = y


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify
        )
    )


    print(
        "\nTraining rows:",
        len(X_train)
    )

    print(
        "Testing rows:",
        len(X_test)
    )


    # ========================================================
    # STEP 8
    # Build Automatic Preprocessor
    # ========================================================

    print(
        "\nCreating preprocessing pipeline..."
    )

    preprocessor = build_preprocessor(
        X_train
    )


    # ========================================================
    # STEP 9
    # Get Machine Learning Models
    # ========================================================

    models = get_models(
        problem_type
    )


    print(
        "\nModels selected:"
    )

    for model_name in models.keys():

        print(
            "-",
            model_name
        )


    # ========================================================
    # STEP 10
    # Select Evaluation Function
    # ========================================================

    if problem_type == "classification":

        evaluator = (
            evaluate_classification
        )

    else:

        evaluator = (
            evaluate_regression
        )


    # ========================================================
    # STEP 11
    # Train All Models
    # ========================================================

    print(
        "\nStarting model training..."
    )


    results, trained_models = train_models(

        X_train,
        X_test,

        y_train,
        y_test,

        preprocessor,

        models,

        problem_type,

        evaluator
    )


    # ========================================================
    # STEP 18
    # SELECT THE BEST MODEL
    # ========================================================

    print(
        "\n==================================="
    )

    print(
        "Selecting Best Model"
    )

    print(
        "==================================="
    )


    # For classification:
    # Choose model with highest F1 score

    if problem_type == "classification":

        best_model_name = max(

            results,

            key=lambda model_name:
            results[model_name]["f1"]
        )


    # For regression:
    # Choose model with highest R2 score

    else:

        best_model_name = max(

            results,

            key=lambda model_name:
            results[model_name]["r2"]
        )


    print(
        "\nBest Model:",
        best_model_name
    )


    # Get actual trained model pipeline

    best_model = trained_models[
        best_model_name
    ]


    # ========================================================
    # STEP 19
    # SAVE THE BEST MODEL
    # ========================================================

    print(
        "\nSaving best model..."
    )


    # Create folder automatically
    # if it does not already exist

    model_directory = Path(
        "artifacts/models"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # Model filename

    model_path = (
        model_directory
        / "best_model.joblib"
    )


    # Save complete pipeline
    # preprocessing + ML model

    joblib.dump(
        best_model,
        model_path
    )


    print(
        "Best model saved at:"
    )

    print(
        model_path
    )


    # ========================================================
    # STEP 20
    # SAVE MODEL METRICS
    # ========================================================

    print(
        "\nSaving model results..."
    )


    metrics_directory = Path(
        "artifacts/metrics"
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    metrics_path = (
        metrics_directory
        / "model_results.json"
    )


    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    print(
        "Model results saved at:"
    )

    print(
        metrics_path
    )


    # ========================================================
    # RETURN RESULTS TO MAIN.PY
    # ========================================================

    return {

        "profile":
            profile,

        "problem_type":
            problem_type,

        "results":
            results,

        "models":
            trained_models,

        "best_model_name":
            best_model_name,

        "best_model":
            best_model,

        "model_path":
            str(model_path),

        "metrics_path":
            str(metrics_path)
    }    