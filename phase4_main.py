# ============================================================
# AutoDS Phase 4
# Autonomous Self-Correction Runner
# ============================================================

import json

from pathlib import Path


from src.pipeline import (
    run_autods
)


from src.agent_context import (
    build_agent_context
)


from src.graphs.autods_graph import (
    build_autods_graph
)


from src.graphs.autonomous_loop_graph import (
    build_autonomous_loop_graph
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


GOAL = (
    "Predict which customers are likely to churn "
    "and improve predictive performance autonomously."
)


# ------------------------------------------------------------
# AUTONOMOUS SEARCH SETTINGS
# ------------------------------------------------------------

MAX_ROUNDS = 3

PATIENCE = 2

MIN_IMPROVEMENT = 0.002

TARGET_SCORE = None


# ============================================================
# PHASE 1 + PHASE 2
# ============================================================

print(
    "\n======================================="
)

print(
    "RUNNING PHASE 1 + PHASE 2"
)

print(
    "======================================="
)


autods_result = run_autods(
    DATASET,
    TARGET
)


# ============================================================
# BUILD VERIFIED CONTEXT
# ============================================================

initial_state = build_agent_context(

    dataset_path=DATASET,

    target=TARGET,

    goal=GOAL,

    autods_result=autods_result
)


# ============================================================
# REUSE EXISTING PHASE 3 RESULTS IF AVAILABLE
# ============================================================

phase3_report_path = Path(
    "artifacts/agents/"
    "phase3_agent_report.json"
)


if phase3_report_path.exists():

    print(
        "\nExisting Phase 3 agent report found."
    )

    print(
        "Reusing Phase 3 reasoning to save API calls."
    )


    with open(
        phase3_report_path,
        "r",
        encoding="utf-8"
    ) as file:

        phase3_report = json.load(
            file
        )


    initial_state.update({

        "problem_analysis":
            phase3_report.get(
                "problem_agent",
                {}
            ),

        "dataset_analysis":
            phase3_report.get(
                "dataset_agent",
                {}
            ),

        "quality_analysis":
            phase3_report.get(
                "quality_agent",
                {}
            ),

        "eda_analysis":
            phase3_report.get(
                "eda_agent",
                {}
            ),

        "ml_plan":
            phase3_report.get(
                "ml_planning_agent",
                {}
            ),

        "final_analysis":
            phase3_report.get(
                "lead_data_scientist",
                {}
            )
    })


    phase3_state = (
        initial_state
    )


else:

    print(
        "\nPhase 3 report not found."
    )

    print(
        "Running Phase 3 agents..."
    )


    phase3_graph = (
        build_autods_graph()
    )


    phase3_state = (
        phase3_graph.invoke(
            initial_state
        )
    )


# ============================================================
# PHASE 4 CONFIGURATION
# ============================================================

phase3_state.update({

    "max_rounds":
        MAX_ROUNDS,

    "patience":
        PATIENCE,

    "min_improvement":
        MIN_IMPROVEMENT,

    "target_score":
        TARGET_SCORE
})


# ============================================================
# BUILD PHASE 4 GRAPH
# ============================================================

phase4_graph = (
    build_autonomous_loop_graph()
)


# ============================================================
# RUN AUTONOMOUS LOOP
# ============================================================

print(
    "\n======================================="
)

print(
    "STARTING PHASE 4 AUTONOMOUS LOOP"
)

print(
    "======================================="
)


final_state = phase4_graph.invoke(

    phase3_state,

    config={
        "recursion_limit": 50
    }
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n======================================="
)

print(
    "PHASE 4 AUTONOMOUS RESULTS"
)

print(
    "======================================="
)


print(
    "\nPrimary Metric:",
    final_state[
        "primary_metric"
    ]
)


print(
    "Baseline Score:",
    final_state[
        "baseline_score"
    ]
)


print(
    "Final Best Score:",
    final_state[
        "best_score"
    ]
)


print(
    "Best Experiment:",
    final_state[
        "best_experiment_id"
    ]
)


print(
    "Rounds Executed:",
    final_state[
        "round_number"
    ]
)


print(
    "Stop Reason:",
    final_state[
        "loop_status"
    ]
)


print(
    "\nChampion Model:"
)

print(
    final_state[
        "autonomy_best_model_path"
    ]
)


# ============================================================
# EXPERIMENT HISTORY
# ============================================================

print(
    "\n======================================="
)

print(
    "EXPERIMENT HISTORY"
)

print(
    "======================================="
)


for experiment in final_state[
    "experiment_history"
]:

    print(
        "\nExperiment:",
        experiment.get(
            "experiment_id"
        )
    )

    print(
        "Status:",
        experiment.get(
            "status"
        )
    )

    if experiment.get(
        "status"
    ) == "success":

        print(
            "Validation Score:",
            experiment.get(
                "validation_score"
            )
        )

        print(
            "Generalization Gap:",
            experiment.get(
                "generalization_gap"
            )
        )

        print(
            "Improved:",
            experiment.get(
                "improved"
            )
        )

    else:

        print(
            "Error:",
            experiment.get(
                "error"
            )
        )


# ============================================================
# SAVE PHASE 4 REPORT
# ============================================================

report_directory = Path(
    "artifacts/autonomy"
)

report_directory.mkdir(
    parents=True,
    exist_ok=True
)


report_path = (

    report_directory

    /

    "phase4_autonomy_report.json"
)


phase4_report = {

    "goal":
        GOAL,

    "primary_metric":
        final_state[
            "primary_metric"
        ],

    "baseline_score":
        final_state[
            "baseline_score"
        ],

    "final_best_score":
        final_state[
            "best_score"
        ],

    "best_experiment":
        final_state[
            "best_experiment_id"
        ],

    "best_model_path":
        final_state[
            "autonomy_best_model_path"
        ],

    "rounds":
        final_state[
            "round_number"
        ],

    "stop_reason":
        final_state[
            "loop_status"
        ],

    "experiment_history":
        final_state[
            "experiment_history"
        ],

    "autonomy_summary":
        final_state[
            "autonomy_summary"
        ]
}


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        phase4_report,
        file,
        indent=4,
        default=str
    )


print(
    "\nPhase 4 report saved:"
)

print(
    report_path
)


print(
    "\n======================================="
)

print(
    "PHASE 4 COMPLETED SUCCESSFULLY"
)

print(
    "=======================================\n"
)