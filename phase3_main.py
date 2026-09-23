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


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

DATASET = (
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


TARGET = (
    "Churn"
)


GOAL = (
    "Predict which customers are likely to churn "
    "and identify the important modelling considerations."
)


# ============================================================
# RUN PHASE 1 + PHASE 2
# ============================================================

print(
    "\n======================================="
)

print(
    "RUNNING AUTODS PHASE 1 + PHASE 2"
)

print(
    "======================================="
)


autods_result = run_autods(
    DATASET,
    TARGET
)


# ============================================================
# BUILD VERIFIED AGENT CONTEXT
# ============================================================

print(
    "\n======================================="
)

print(
    "BUILDING PHASE 3 AGENT CONTEXT"
)

print(
    "======================================="
)


initial_state = build_agent_context(

    dataset_path=DATASET,

    target=TARGET,

    goal=GOAL,

    autods_result=autods_result
)


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = build_autods_graph()


# ============================================================
# RUN AGENT GRAPH
# ============================================================

print(
    "\n======================================="
)

print(
    "STARTING AUTODS AGENTIC BRAIN"
)

print(
    "======================================="
)


final_state = graph.invoke(
    initial_state
)


# ============================================================
# DISPLAY FINAL AGENT RESULT
# ============================================================

print(
    "\n======================================="
)

print(
    "AUTODS AGENTIC ANALYSIS"
)

print(
    "=======================================\n"
)


final_analysis = final_state[
    "final_analysis"
]


print(
    "Executive Summary:"
)

print(
    final_analysis[
        "executive_summary"
    ]
)


print(
    "\nCurrent Status:"
)

print(
    final_analysis[
        "current_project_status"
    ]
)


print(
    "\nStrongest Findings:"
)

for item in final_analysis[
    "strongest_findings"
]:

    print(
        "-",
        item
    )


print(
    "\nWarnings:"
)

for item in final_analysis[
    "important_warnings"
]:

    print(
        "-",
        item
    )


print(
    "\nRecommended Next Actions:"
)

for item in final_analysis[
    "recommended_next_actions"
]:

    print(
        "-",
        item
    )


# ============================================================
# SAVE COMPLETE AGENT REPORT
# ============================================================

output_directory = Path(
    "artifacts/agents"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True
)


output_path = (
    output_directory
    /
    "phase3_agent_report.json"
)


report = {

    "goal":
        final_state[
            "goal"
        ],

    "problem_agent":
        final_state[
            "problem_analysis"
        ],

    "dataset_agent":
        final_state[
            "dataset_analysis"
        ],

    "quality_agent":
        final_state[
            "quality_analysis"
        ],

    "eda_agent":
        final_state[
            "eda_analysis"
        ],

    "ml_planning_agent":
        final_state[
            "ml_plan"
        ],

    "lead_data_scientist":
        final_state[
            "final_analysis"
        ]
}


with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        default=str
    )


print(
    "\nAgent report saved:"
)

print(
    output_path
)


print(
    "\n======================================="
)

print(
    "PHASE 3 COMPLETED SUCCESSFULLY"
)

print(
    "=======================================\n"
)