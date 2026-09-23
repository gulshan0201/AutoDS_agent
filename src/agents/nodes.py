from src.agents.llm_client import (
    ask_structured
)

from src.agents.schemas import (
    ProblemAnalysis,
    DatasetAnalysis,
    QualityAnalysis,
    EDAAnalysis,
    MLPlan,
    FinalAnalysis
)


# ============================================================
# PROBLEM AGENT
# ============================================================

def problem_agent_node(
    state
):

    print(
        "\n[AGENT] Problem Understanding Agent"
    )

    payload = {

        "user_goal":
            state["goal"],

        "target":
            state["target"],

        "detected_problem_type":
            state["problem_type"],

        "target_distribution":
            state[
                "eda_summary"
            ].get(
                "target_summary",
                {}
            ),

        "data_quality_warnings":
            state[
                "quality_report"
            ].get(
                "warnings",
                []
            )
    }


    result = ask_structured(

        ProblemAnalysis,

        """
        Act as the Problem Understanding Agent.

        Translate the user's goal into a precise machine-learning
        objective.

        Recommend an appropriate primary evaluation metric and
        useful secondary metrics.

        Respect the already detected problem type unless the supplied
        facts clearly indicate a contradiction.
        """,

        payload
    )


    return {

        "problem_analysis":
            result
    }


# ============================================================
# DATASET AGENT
# ============================================================

def dataset_agent_node(
    state
):

    print(
        "\n[AGENT] Dataset Understanding Agent"
    )

    payload = {

        "profile":
            state["profile"],

        "schema_report":
            state[
                "schema_report"
            ],

        "target":
            state["target"],

        "problem_analysis":
            state[
                "problem_analysis"
            ]
    }


    result = ask_structured(

        DatasetAnalysis,

        """
        Act as a senior data profiling specialist.

        Explain the supplied dataset structure.

        Pay special attention to:
        - converted numeric columns
        - identifier columns
        - constant columns
        - high-cardinality features
        - features that may create ML problems

        Do not invent information that is absent.
        """,

        payload
    )


    return {

        "dataset_analysis":
            result
    }


# ============================================================
# QUALITY AGENT
# ============================================================

def quality_agent_node(
    state
):

    print(
        "\n[AGENT] Data Quality Agent"
    )

    payload = {

        "quality_report":
            state[
                "quality_report"
            ],

        "schema_report":
            state[
                "schema_report"
            ],

        "dataset_analysis":
            state[
                "dataset_analysis"
            ]
    }


    result = ask_structured(

        QualityAnalysis,

        """
        Act as a Data Quality and ML Risk Agent.

        Review missing values, duplicates, imbalance,
        suspected leakage, outliers, identifiers,
        constant features, and high-cardinality fields.

        Separate confirmed problems from warnings.

        Recommend actions before advanced modelling.
        """,

        payload
    )


    return {

        "quality_analysis":
            result
    }


# ============================================================
# EDA AGENT
# ============================================================

def eda_agent_node(
    state
):

    print(
        "\n[AGENT] EDA Interpretation Agent"
    )

    payload = {

        "goal":
            state["goal"],

        "problem_type":
            state[
                "problem_type"
            ],

        "target":
            state["target"],

        "eda_summary":
            state[
                "eda_summary"
            ],

        "quality_analysis":
            state[
                "quality_analysis"
            ]
    }


    result = ask_structured(

        EDAAnalysis,

        """
        Act as an Exploratory Data Analysis Agent.

        You are NOT viewing chart pixels.

        Analyze only the deterministic numeric,
        categorical and target summaries supplied.

        Identify useful patterns, suspicious patterns,
        potential feature opportunities, and additional
        analyses that should be performed.
        """,

        payload
    )


    return {

        "eda_analysis":
            result
    }


# ============================================================
# ML PLANNING AGENT
# ============================================================

def ml_planning_agent_node(
    state
):

    print(
        "\n[AGENT] Machine Learning Planning Agent"
    )

    payload = {

        "goal":
            state["goal"],

        "problem_analysis":
            state[
                "problem_analysis"
            ],

        "dataset_analysis":
            state[
                "dataset_analysis"
            ],

        "quality_analysis":
            state[
                "quality_analysis"
            ],

        "eda_analysis":
            state[
                "eda_analysis"
            ],

        "existing_model_results":
            state[
                "model_results"
            ],

        "current_best_model":
            state[
                "best_model_name"
            ]
    }


    result = ask_structured(

        MLPlan,

        """
        Act as a senior Machine Learning Planning Agent.

        Design the NEXT modelling strategy.

        Existing model results are verified historical
        experiment results.

        Recommend:
        - metric strategy
        - validation strategy
        - model families
        - preprocessing improvements
        - feature engineering
        - next experiments
        - stopping rule

        Do not claim a recommended model will outperform
        the existing champion until it has actually been tested.
        """,

        payload
    )


    return {

        "ml_plan":
            result
    }


# ============================================================
# SYNTHESIS AGENT
# ============================================================

def synthesis_agent_node(
    state
):

    print(
        "\n[AGENT] Lead Data Scientist Agent"
    )

    payload = {

        "goal":
            state["goal"],

        "problem_analysis":
            state[
                "problem_analysis"
            ],

        "dataset_analysis":
            state[
                "dataset_analysis"
            ],

        "quality_analysis":
            state[
                "quality_analysis"
            ],

        "eda_analysis":
            state[
                "eda_analysis"
            ],

        "ml_plan":
            state[
                "ml_plan"
            ],

        "verified_model_results":
            state[
                "model_results"
            ],

        "verified_best_model":
            state[
                "best_model_name"
            ]
    }


    result = ask_structured(

        FinalAnalysis,

        """
        You are the Lead Autonomous Data Scientist.

        Synthesize the specialist-agent findings into
        one concise project-level analysis.

        Treat the supplied model results as verified facts.

        Recommendations are future actions, not completed results.
        """,

        payload
    )


    return {

        "final_analysis":
            result
    }