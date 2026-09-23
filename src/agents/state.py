# from typing import Any, TypedDict


# class AutoDSAgentState(TypedDict, total=False):
#     """
#     Shared state used by all AutoDS LangGraph agents.
#     """

#     # User input
#     goal: str
#     dataset_path: str
#     target: str

#     # Phase 1 + Phase 2 facts
#     problem_type: str
#     profile: dict[str, Any]
#     schema_report: dict[str, Any]
#     quality_report: dict[str, Any]
#     model_results: dict[str, Any]
#     best_model_name: str
#     eda_summary: dict[str, Any]

#     # Agent outputs
#     problem_analysis: dict[str, Any]
#     dataset_analysis: dict[str, Any]
#     quality_analysis: dict[str, Any]
#     eda_analysis: dict[str, Any]
#     ml_plan: dict[str, Any]
#     final_analysis: dict[str, Any]

#     # Execution information
#     errors: list[str]

from typing import Any, TypedDict


class AutoDSAgentState(TypedDict, total=False):
    """
    Shared state for AutoDS Phases 3 and 4.
    """

    # ========================================================
    # USER INPUT
    # ========================================================

    goal: str
    dataset_path: str
    target: str


    # ========================================================
    # PHASE 1 + PHASE 2 FACTS
    # ========================================================

    problem_type: str

    profile: dict[str, Any]

    schema_report: dict[str, Any]

    quality_report: dict[str, Any]

    model_results: dict[str, Any]

    best_model_name: str

    eda_summary: dict[str, Any]

    processed_dataset: str

    model_path: str


    # ========================================================
    # PHASE 3 AGENT OUTPUTS
    # ========================================================

    problem_analysis: dict[str, Any]

    dataset_analysis: dict[str, Any]

    quality_analysis: dict[str, Any]

    eda_analysis: dict[str, Any]

    ml_plan: dict[str, Any]

    final_analysis: dict[str, Any]


    # ========================================================
    # PHASE 4 AUTONOMOUS LOOP
    # ========================================================

    primary_metric: str

    baseline_score: float

    best_score: float

    best_experiment_id: str

    autonomy_best_model_path: str

    experiment_history: list[dict[str, Any]]

    tried_experiments: list[str]

    available_experiments: list[dict[str, Any]]

    correction_decision: dict[str, Any]

    last_experiment: dict[str, Any]

    round_number: int

    max_rounds: int

    patience: int

    no_improvement_rounds: int

    min_improvement: float

    target_score: float | None

    loop_status: str

    autonomy_summary: dict[str, Any]


    # ========================================================
    # ERRORS
    # ========================================================

    errors: list[str]