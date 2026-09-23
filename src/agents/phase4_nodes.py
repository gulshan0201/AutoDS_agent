# ============================================================
# AutoDS Phase 4
# Autonomous Critic / Experiment / Judge Nodes
# ============================================================

import shutil
import traceback

from pathlib import Path


from src.agents.llm_client import (
    ask_structured
)

from src.agents.schemas import (
    CorrectionDecision,
    AutonomySummary
)

from src.autonomy.experiment_runner import (
    list_available_experiments,
    run_experiment
)


# ============================================================
# INITIALIZE AUTONOMOUS LOOP
# ============================================================

def initialize_autonomy_node(
    state
):

    print(
        "\n[AUTONOMY] Initializing correction loop..."
    )


    problem_type = (
        state[
            "problem_type"
        ]
    )


    primary_metric = (

        "f1"

        if problem_type == "classification"

        else "r2"
    )


    baseline_model = (
        state[
            "best_model_name"
        ]
    )


    baseline_score = float(

        state[
            "model_results"
        ][
            baseline_model
        ][
            primary_metric
        ]
    )


    experiments = (
        list_available_experiments(
            problem_type
        )
    )


    print(
        "Baseline model:",
        baseline_model
    )

    print(
        "Baseline",
        primary_metric,
        ":",
        baseline_score
    )


    return {

        "primary_metric":
            primary_metric,

        "baseline_score":
            baseline_score,

        "best_score":
            baseline_score,

        "best_experiment_id":
            "phase1_baseline",

        "autonomy_best_model_path":
            state.get(
                "model_path",
                ""
            ),

        "experiment_history":
            [],

        "tried_experiments":
            [],

        "available_experiments":
            experiments,

        "round_number":
            0,

        "no_improvement_rounds":
            0,

        "loop_status":
            "continue"
    }


# ============================================================
# CRITIC AGENT
# ============================================================

def critic_agent_node(
    state
):

    print(
        "\n[AUTONOMY] Critic Agent"
    )


    tried = set(
        state.get(
            "tried_experiments",
            []
        )
    )


    remaining = [

        experiment

        for experiment
        in state[
            "available_experiments"
        ]

        if experiment[
            "experiment_id"
        ] not in tried
    ]


    # --------------------------------------------------------
    # DETERMINISTIC STOP CONDITIONS
    # --------------------------------------------------------

    if not remaining:

        return {

            "correction_decision": {

                "continue_search":
                    False,

                "experiment_id":
                    None,

                "diagnosis": [
                    "No untested experiments remain."
                ],

                "rationale":
                    "Experiment search space exhausted.",

                "expected_effect":
                    "Stop autonomous search."
            }
        }


    if (
        state[
            "round_number"
        ]
        >=
        state[
            "max_rounds"
        ]
    ):

        return {

            "correction_decision": {

                "continue_search":
                    False,

                "experiment_id":
                    None,

                "diagnosis": [
                    "Maximum autonomous rounds reached."
                ],

                "rationale":
                    "Execution budget exhausted.",

                "expected_effect":
                    "Stop autonomous search."
            }
        }


    if (
        state[
            "no_improvement_rounds"
        ]
        >=
        state[
            "patience"
        ]
    ):

        return {

            "correction_decision": {

                "continue_search":
                    False,

                "experiment_id":
                    None,

                "diagnosis": [
                    "Performance has plateaued."
                ],

                "rationale":
                    "Patience threshold reached.",

                "expected_effect":
                    "Stop unnecessary experiments."
            }
        }


    target_score = state.get(
        "target_score"
    )


    if (
        target_score is not None
        and
        state[
            "best_score"
        ]
        >=
        target_score
    ):

        return {

            "correction_decision": {

                "continue_search":
                    False,

                "experiment_id":
                    None,

                "diagnosis": [
                    "Requested target score achieved."
                ],

                "rationale":
                    "No further search is required.",

                "expected_effect":
                    "Stop autonomous search."
            }
        }


    # --------------------------------------------------------
    # LLM CRITIC PAYLOAD
    # --------------------------------------------------------

    payload = {

        "problem_type":
            state[
                "problem_type"
            ],

        "primary_metric":
            state[
                "primary_metric"
            ],

        "baseline_score":
            state[
                "baseline_score"
            ],

        "current_best_score":
            state[
                "best_score"
            ],

        "current_best_experiment":
            state[
                "best_experiment_id"
            ],

        "round_number":
            state[
                "round_number"
            ],

        "quality_analysis":
            state.get(
                "quality_analysis",
                {}
            ),

        "eda_analysis":
            state.get(
                "eda_analysis",
                {}
            ),

        "ml_plan":
            state.get(
                "ml_plan",
                {}
            ),

        "previous_experiments":
            state.get(
                "experiment_history",
                []
            ),

        "remaining_safe_experiments":
            remaining
    }


    try:

        result = ask_structured(

            CorrectionDecision,

            """
            You are the AutoDS Critic Agent.

            Examine the verified experiment history,
            data-quality findings, EDA findings and
            existing champion.

            Select exactly ONE experiment from the supplied
            remaining_safe_experiments.

            Never invent an experiment ID.

            If continued experimentation is no longer useful,
            set continue_search to false.

            If an experiment failed, analyze the supplied
            error information and select another safe strategy.

            Do not claim an experiment will improve performance.
            State only why it is reasonable to test.
            """,

            payload
        )

    except Exception as error:

        # ----------------------------------------------------
        # SAFE FALLBACK IF FREE API IS TEMPORARILY UNAVAILABLE
        # ----------------------------------------------------

        result = {

            "continue_search":
                True,

            "experiment_id":
                remaining[0][
                    "experiment_id"
                ],

            "diagnosis": [
                "LLM critic temporarily unavailable."
            ],

            "rationale":
                "Using deterministic safe fallback experiment.",

            "expected_effect":
                "Continue the bounded experiment search."
        }


    # --------------------------------------------------------
    # VERIFY LLM CHOICE
    # --------------------------------------------------------

    allowed_ids = {

        item[
            "experiment_id"
        ]

        for item in remaining
    }


    if result[
        "continue_search"
    ]:

        selected_id = result.get(
            "experiment_id"
        )

        if selected_id not in allowed_ids:

            result = {

                "continue_search":
                    False,

                "experiment_id":
                    None,

                "diagnosis": [
                    "Critic returned an unsupported experiment."
                ],

                "rationale":
                    "Unsafe or unknown actions are not executed.",

                "expected_effect":
                    "Stop safely."
            }


    if result[
        "continue_search"
    ]:

        print(
            "Selected experiment:",
            result[
                "experiment_id"
            ]
        )

    else:

        print(
            "Critic decided to stop."
        )


    return {

        "correction_decision":
            result
    }


# ============================================================
# EXECUTE EXPERIMENT
# ============================================================

def experiment_node(
    state
):

    decision = (
        state[
            "correction_decision"
        ]
    )


    experiment_id = (
        decision[
            "experiment_id"
        ]
    )


    print(
        "\n[AUTONOMY] Executing:",
        experiment_id
    )


    history = list(
        state.get(
            "experiment_history",
            []
        )
    )


    tried = list(
        state.get(
            "tried_experiments",
            []
        )
    )


    try:

        result = run_experiment(

            dataset_path=
                state[
                    "processed_dataset"
                ],

            target=
                state[
                    "target"
                ],

            problem_type=
                state[
                    "problem_type"
                ],

            schema_report=
                state[
                    "schema_report"
                ],

            experiment_id=
                experiment_id
        )


    except Exception as error:

        result = {

            "experiment_id":
                experiment_id,

            "status":
                "failed",

            "error":
                str(error),

            "traceback":
                traceback.format_exc(
                    limit=8
                )
        }


    history.append(
        result
    )


    tried.append(
        experiment_id
    )


    return {

        "last_experiment":
            result,

        "experiment_history":
            history,

        "tried_experiments":
            tried,

        "round_number":
            state[
                "round_number"
            ]
            +
            1
    }


# ============================================================
# JUDGE EXPERIMENT
# ============================================================

def judge_experiment_node(
    state
):

    print(
        "\n[AUTONOMY] Evaluating experiment..."
    )


    result = dict(
        state[
            "last_experiment"
        ]
    )


    current_best = float(
        state[
            "best_score"
        ]
    )


    best_experiment = (
        state[
            "best_experiment_id"
        ]
    )


    best_model_path = (
        state[
            "autonomy_best_model_path"
        ]
    )


    no_improvement = (
        state[
            "no_improvement_rounds"
        ]
    )


    # --------------------------------------------------------
    # FAILED EXPERIMENT
    # --------------------------------------------------------

    if result[
        "status"
    ] != "success":

        print(
            "Experiment failed."
        )

        no_improvement += 1

        result[
            "improved"
        ] = False


    # --------------------------------------------------------
    # SUCCESSFUL EXPERIMENT
    # --------------------------------------------------------

    else:

        candidate_score = float(
            result[
                "validation_score"
            ]
        )


        improvement = round(

            candidate_score
            -
            current_best,

            6
        )


        result[
            "improvement"
        ] = improvement


        if (
            improvement
            >=
            state[
                "min_improvement"
            ]
        ):

            print(
                "New champion found!"
            )

            print(
                "Previous:",
                current_best
            )

            print(
                "New:",
                candidate_score
            )


            current_best = (
                candidate_score
            )


            best_experiment = (
                result[
                    "experiment_id"
                ]
            )


            # ------------------------------------------------
            # PROMOTE CHAMPION MODEL
            # ------------------------------------------------

            champion_directory = Path(
                "artifacts/autonomy"
            )

            champion_directory.mkdir(
                parents=True,
                exist_ok=True
            )


            champion_path = (

                champion_directory

                /

                "autonomous_best_model.joblib"
            )


            shutil.copy2(

                result[
                    "artifact_path"
                ],

                champion_path
            )


            best_model_path = str(
                champion_path
            )


            no_improvement = 0


            result[
                "improved"
            ] = True


        else:

            print(
                "No meaningful improvement."
            )

            no_improvement += 1

            result[
                "improved"
            ] = False


    # --------------------------------------------------------
    # UPDATE HISTORY WITH JUDGMENT
    # --------------------------------------------------------

    history = list(
        state[
            "experiment_history"
        ]
    )


    history[-1] = result


    # --------------------------------------------------------
    # DETERMINE LOOP STATUS
    # --------------------------------------------------------

    loop_status = "continue"


    target_score = (
        state.get(
            "target_score"
        )
    )


    if (
        target_score is not None
        and
        current_best >= target_score
    ):

        loop_status = (
            "target_reached"
        )


    elif (
        state[
            "round_number"
        ]
        >=
        state[
            "max_rounds"
        ]
    ):

        loop_status = (
            "max_rounds_reached"
        )


    elif (
        no_improvement
        >=
        state[
            "patience"
        ]
    ):

        loop_status = (
            "performance_plateau"
        )


    return {

        "best_score":
            current_best,

        "best_experiment_id":
            best_experiment,

        "autonomy_best_model_path":
            best_model_path,

        "no_improvement_rounds":
            no_improvement,

        "loop_status":
            loop_status,

        "experiment_history":
            history,

        "last_experiment":
            result
    }


# ============================================================
# FINAL AUTONOMY SUMMARY
# ============================================================

def finalizer_node(
    state
):

    print(
        "\n[AUTONOMY] Finalizing autonomous search..."
    )


    successful = [

        item

        for item in state[
            "experiment_history"
        ]

        if item.get(
            "status"
        ) == "success"
    ]


    failed = [

        item

        for item in state[
            "experiment_history"
        ]

        if item.get(
            "status"
        ) == "failed"
    ]


    payload = {

        "problem_type":
            state[
                "problem_type"
            ],

        "primary_metric":
            state[
                "primary_metric"
            ],

        "baseline_score":
            state[
                "baseline_score"
            ],

        "final_best_score":
            state[
                "best_score"
            ],

        "best_experiment":
            state[
                "best_experiment_id"
            ],

        "stop_reason":
            state[
                "loop_status"
            ],

        "experiment_history":
            state[
                "experiment_history"
            ]
    }


    try:

        summary = ask_structured(

            AutonomySummary,

            """
            Act as the final AutoDS autonomous-search reviewer.

            Summarize what the bounded experiment loop actually
            tested and why it stopped.

            Never invent scores.

            Do not state that an untested method is better.

            Suggestions for future experiments must clearly
            be recommendations.
            """,

            payload
        )


    except Exception:

        summary = {

            "stop_reason":
                state[
                    "loop_status"
                ],

            "what_changed": [
                "AutoDS completed a bounded autonomous experiment search."
            ],

            "successful_actions": [

                item[
                    "experiment_id"
                ]

                for item in successful
            ],

            "failed_actions": [

                item[
                    "experiment_id"
                ]

                for item in failed
            ],

            "next_recommendations": [
                "Proceed to advanced model optimization."
            ]
        }


    return {

        "autonomy_summary":
            summary
    }


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_critic(
    state
):

    decision = (
        state[
            "correction_decision"
        ]
    )


    if decision[
        "continue_search"
    ]:

        return "experiment"


    return "finish"


def route_after_judge(
    state
):

    if (
        state[
            "loop_status"
        ]
        ==
        "continue"
    ):

        return "retry"


    return "finish"