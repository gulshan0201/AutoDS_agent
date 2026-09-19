# from src.pipeline import run_autods


# DATASET = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# TARGET = "Churn"


# result = run_autods(
#     DATASET,
#     TARGET
# )


# print(
#     "\n======================"
# )

# print(
#     "MODEL LEADERBOARD"
# )

# print(
#     "======================"
# )


# for model_name, metrics in (
#     result["results"].items()
# ):

#     print(
#         f"\n{model_name}"
#     )

#     for metric, value in metrics.items():

#         print(
#             f"  {metric}: {value}"
#         )

# ============================================================
# AutoDS - Application Entry Point
# ============================================================

from src.pipeline import run_autods


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET = (
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


TARGET = (
    "Churn"
)


# ============================================================
# RUN AUTODS
# ============================================================

result = run_autods(
    DATASET,
    TARGET
)


# ============================================================
# DISPLAY MODEL LEADERBOARD
# ============================================================

print(
    "\n\n==================================="
)

print(
    "       MODEL LEADERBOARD"
)

print(
    "==================================="
)


for model_name, metrics in (
    result["results"].items()
):

    print(
        f"\n{model_name}"
    )

    print(
        "-" * len(model_name)
    )

    for metric, value in metrics.items():

        print(
            f"{metric:12}: {value}"
        )


# ============================================================
# DISPLAY WINNING MODEL
# ============================================================

print(
    "\n==================================="
)

print(
    "       CHAMPION MODEL"
)

print(
    "==================================="
)


print(
    "\nProblem Type :",
    result["problem_type"]
)


print(
    "Best Model   :",
    result["best_model_name"]
)


print(
    "\nSaved Model:"
)

print(
    result["model_path"]
)


print(
    "\nSaved Metrics:"
)

print(
    result["metrics_path"]
)


print(
    "\n==================================="
)

print(
    "AutoDS Phase 1 completed successfully!"
)

print(
    "===================================\n"
)