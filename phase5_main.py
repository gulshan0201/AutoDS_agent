# ============================================================
# AutoDS Phase 5 Runner
# ============================================================

import json

from pathlib import Path


from src.pipeline import (
    run_autods
)


from src.advanced_ml.advanced_engine import (
    run_advanced_ml
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = (
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


TARGET = (
    "Churn"
)


# ============================================================
# DEVELOPMENT SETTINGS
# ============================================================

CV_FOLDS = 3

OPTUNA_TRIALS = 8

MIN_IMPROVEMENT = 0.002


# ============================================================
# RUN PHASE 1 + 2
# This recreates verified schema and processed dataset
# ============================================================

print(
    "\n======================================"
)

print(
    "PREPARING VERIFIED AUTODS DATA"
)

print(
    "======================================"
)


autods_result = run_autods(

    DATASET,

    TARGET
)


# ============================================================
# FIND THE CURRENT CHAMPION
# ============================================================

possible_models = [

    Path(
        "artifacts/autonomy/"
        "final_champion_model.joblib"
    ),

    Path(
        "artifacts/autonomy/"
        "autonomous_best_model.joblib"
    ),

    Path(
        "artifacts/models/"
        "best_model.joblib"
    )
]


baseline_model_path = None


for candidate in possible_models:

    if candidate.exists():

        baseline_model_path = str(
            candidate
        )

        break


if baseline_model_path is None:

    raise FileNotFoundError(
        "No existing champion model was found."
    )


print(
    "\nCurrent champion:"
)

print(
    baseline_model_path
)


# ============================================================
# PROCESSED DATA
# ============================================================

processed_dataset = (
    autods_result[
        "processed_dataset"
    ]
)


# ============================================================
# RUN PHASE 5
# ============================================================

phase5_result = run_advanced_ml(

    dataset_path=
        processed_dataset,

    target=
        TARGET,

    problem_type=
        autods_result[
            "problem_type"
        ],

    schema_report=
        autods_result[
            "schema_report"
        ],

    quality_report=
        autods_result[
            "quality_report"
        ],

    baseline_model_path=
        baseline_model_path,

    n_trials=
        OPTUNA_TRIALS,

    cv_folds=
        CV_FOLDS,

    min_improvement=
        MIN_IMPROVEMENT
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\n======================================"
)

print(
    "PHASE 5 FINAL SUMMARY"
)

print(
    "======================================"
)


print(
    "\nMetric:",
    phase5_result[
        "primary_metric"
    ]
)


print(
    "Baseline:",
    phase5_result[
        "baseline_test_score"
    ]
)


print(
    "Advanced Winner:",
    phase5_result[
        "selected_advanced_model"
    ]
)


print(
    "Advanced Score:",
    phase5_result[
        "advanced_test_score"
    ]
)


print(
    "Improvement:",
    phase5_result[
        "improvement"
    ]
)


print(
    "Promoted:",
    phase5_result[
        "promoted"
    ]
)


print(
    "\nFinal Champion:"
)

print(
    phase5_result[
        "final_champion_path"
    ]
)


print(
    "\n======================================"
)

print(
    "PHASE 5 COMPLETED"
)

print(
    "======================================\n"
)