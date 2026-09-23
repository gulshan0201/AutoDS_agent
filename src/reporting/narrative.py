from src.agents.llm_client import (
    ask_structured
)

from src.agents.schemas import (
    ReportNarrative
)


def generate_report_narrative(
    phase5_report,
    quality_report,
    diagnostics,
    explainability
):

    payload = {

        "phase5_verified_results":
            phase5_report,

        "verified_data_quality":
            quality_report,

        "verified_final_diagnostics":
            diagnostics,

        "verified_explainability":
            explainability
    }


    try:

        return ask_structured(

            ReportNarrative,

            """
            Act as the final reporting Data Scientist.

            Produce a stakeholder-friendly interpretation
            using ONLY the verified facts supplied.

            Important:

            - Never invent performance values.
            - Never invent feature importance.
            - Never claim causality.
            - SHAP explains model contribution, not causation.
            - Clearly mention limitations.
            - Recommendations must be described as future actions.
            """,

            payload
        )


    except Exception:

        return {

            "executive_summary":
                "AutoDS completed automated model development, "
                "advanced optimization and final model evaluation.",

            "model_performance_interpretation":
                "The final verified metrics are available in "
                "the model evaluation section.",

            "data_quality_interpretation":
                "Data-quality findings should be reviewed together "
                "with the stored quality report.",

            "explainability_interpretation":
                "Model explanation artifacts were generated using "
                "the available feature-importance methods.",

            "limitations": [
                "Results depend on the supplied dataset.",
                "Model relationships should not be interpreted "
                "as proof of causation."
            ],

            "recommendations": [
                "Validate the final model on new unseen data.",
                "Monitor performance after deployment."
            ]
        }