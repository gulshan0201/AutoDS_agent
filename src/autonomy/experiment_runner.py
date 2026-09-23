# ============================================================
# AutoDS Phase 4
# Safe Autonomous Experiment Runner
# ============================================================

import time

from pathlib import Path

import joblib
import pandas as pd


from sklearn.model_selection import (
    train_test_split
)

from sklearn.pipeline import Pipeline


from sklearn.linear_model import (
    LogisticRegression,
    Ridge
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


from src.preprocessor import (
    build_preprocessor
)

from src.evaluator import (
    evaluate_classification,
    evaluate_regression
)


# ============================================================
# EXPERIMENT REGISTRY
# ============================================================

def get_experiment_registry(
    problem_type
):

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if problem_type == "classification":

        return {

            "clf_logreg_balanced": {

                "description":
                    "Logistic Regression with balanced class weights.",

                "factory":
                    lambda:
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=1.0
                    )
            },


            "clf_logreg_regularized": {

                "description":
                    "More strongly regularized Logistic Regression.",

                "factory":
                    lambda:
                    LogisticRegression(
                        max_iter=3000,
                        C=0.5
                    )
            },


            "clf_rf_balanced": {

                "description":
                    "Random Forest with balanced class weights.",

                "factory":
                    lambda:
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
            },


            "clf_rf_regularized": {

                "description":
                    "Regularized Random Forest with controlled depth.",

                "factory":
                    lambda:
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1
                    )
            },


            "clf_gb_low_lr": {

                "description":
                    "Gradient Boosting with lower learning rate "
                    "and additional estimators.",

                "factory":
                    lambda:
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42
                    )
            },


            "clf_gb_more_trees": {

                "description":
                    "Gradient Boosting using more estimators.",

                "factory":
                    lambda:
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    )
            }
        }


    # ========================================================
    # REGRESSION
    # ========================================================

    if problem_type == "regression":

        return {

            "reg_ridge_1": {

                "description":
                    "Ridge Regression with alpha 1.",

                "factory":
                    lambda:
                    Ridge(
                        alpha=1.0
                    )
            },


            "reg_ridge_10": {

                "description":
                    "Stronger Ridge regularization.",

                "factory":
                    lambda:
                    Ridge(
                        alpha=10.0
                    )
            },


            "reg_rf_regularized": {

                "description":
                    "Regularized Random Forest Regressor.",

                "factory":
                    lambda:
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
            },


            "reg_rf_more_trees": {

                "description":
                    "Random Forest Regressor with more trees.",

                "factory":
                    lambda:
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=42,
                        n_jobs=-1
                    )
            },


            "reg_gb_low_lr": {

                "description":
                    "Gradient Boosting Regressor using low "
                    "learning rate.",

                "factory":
                    lambda:
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42
                    )
            },


            "reg_gb_more_trees": {

                "description":
                    "Gradient Boosting Regressor using additional "
                    "estimators.",

                "factory":
                    lambda:
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    )
            }
        }


    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )


# ============================================================
# LIST EXPERIMENTS
# ============================================================

def list_available_experiments(
    problem_type
):

    registry = get_experiment_registry(
        problem_type
    )

    return [

        {
            "experiment_id":
                experiment_id,

            "description":
                config[
                    "description"
                ]
        }

        for experiment_id, config
        in registry.items()
    ]


# ============================================================
# RUN ONE VERIFIED EXPERIMENT
# ============================================================

def run_experiment(
    dataset_path,
    target,
    problem_type,
    schema_report,
    experiment_id
):

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # LOAD NORMALIZED DATASET
    # --------------------------------------------------------

    df = pd.read_csv(
        dataset_path
    )


    # --------------------------------------------------------
    # REMOVE TARGET + EXCLUDED FEATURES
    # --------------------------------------------------------

    excluded_columns = (
        schema_report.get(
            "excluded_columns",
            []
        )
    )


    columns_to_remove = (
        [target]
        +
        excluded_columns
    )


    X = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


    y = df[
        target
    ]


    # --------------------------------------------------------
    # REMOVE MISSING TARGET
    # --------------------------------------------------------

    valid_rows = (
        y.notna()
    )

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    stratify = None


    if problem_type == "classification":

        value_counts = (
            y.value_counts()
        )

        if (
            len(value_counts) > 1
            and value_counts.min() >= 2
        ):

            stratify = y


    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=stratify
    )


    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = (
        build_preprocessor(
            X_train
        )
    )


    # --------------------------------------------------------
    # GET SAFE EXPERIMENT
    # --------------------------------------------------------

    registry = get_experiment_registry(
        problem_type
    )


    if experiment_id not in registry:

        raise ValueError(
            f"Unknown experiment: {experiment_id}"
        )


    estimator = (
        registry[
            experiment_id
        ][
            "factory"
        ]()
    )


    # --------------------------------------------------------
    # CREATE PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline(

        steps=[

            (
                "preprocessing",
                preprocessor
            ),

            (
                "model",
                estimator
            )
        ]
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_predictions = (
        pipeline.predict(
            X_train
        )
    )


    test_predictions = (
        pipeline.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # VERIFIED METRICS
    # --------------------------------------------------------

    if problem_type == "classification":

        train_metrics = (
            evaluate_classification(
                y_train,
                train_predictions
            )
        )

        test_metrics = (
            evaluate_classification(
                y_test,
                test_predictions
            )
        )

        primary_metric = "f1"

    else:

        train_metrics = (
            evaluate_regression(
                y_train,
                train_predictions
            )
        )

        test_metrics = (
            evaluate_regression(
                y_test,
                test_predictions
            )
        )

        primary_metric = "r2"


    train_score = float(
        train_metrics[
            primary_metric
        ]
    )


    validation_score = float(
        test_metrics[
            primary_metric
        ]
    )


    generalization_gap = round(

        train_score
        -
        validation_score,

        4
    )


    # --------------------------------------------------------
    # SAVE CANDIDATE MODEL
    # --------------------------------------------------------

    output_directory = Path(
        "artifacts/autonomy/models"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    model_path = (

        output_directory

        /

        f"{experiment_id}.joblib"
    )


    joblib.dump(
        pipeline,
        model_path
    )


    runtime_seconds = round(

        time.perf_counter()
        -
        start_time,

        3
    )


    return {

        "experiment_id":
            experiment_id,

        "status":
            "success",

        "primary_metric":
            primary_metric,

        "train_score":
            train_score,

        "validation_score":
            validation_score,

        "generalization_gap":
            generalization_gap,

        "train_metrics":
            train_metrics,

        "test_metrics":
            test_metrics,

        "runtime_seconds":
            runtime_seconds,

        "artifact_path":
            str(
                model_path
            )
    }