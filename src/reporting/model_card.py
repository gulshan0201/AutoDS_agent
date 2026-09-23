import json

from pathlib import Path


def generate_model_card(

    dataset_path,

    target,

    phase5_report,

    quality_report,

    diagnostics,

    explainability,

    output_directory=
        "artifacts/phase6/model_card"
):

    output_directory = Path(
        output_directory
    )


    output_directory.mkdir(

        parents=True,

        exist_ok=True
    )


    card = {

        "model_name":
            "AutoDS Final Champion",

        "model_path":
            phase5_report.get(
                "final_champion_path"
            ),

        "problem_type":
            phase5_report.get(
                "problem_type"
            ),

        "target":
            target,

        "dataset":
            dataset_path,

        "dataset_rows":
            quality_report.get(
                "rows"
            ),

        "dataset_columns":
            quality_report.get(
                "columns"
            ),

        "primary_metric":
            phase5_report.get(
                "primary_metric"
            ),

        "verified_final_metrics":
            diagnostics.get(
                "metrics",
                {}
            ),

        "selected_advanced_model":
            phase5_report.get(
                "selected_advanced_model"
            ),

        "advanced_model_promoted":
            phase5_report.get(
                "promoted"
            ),

        "data_quality_score":
            quality_report.get(
                "quality_score"
            ),

        "data_quality_warnings":
            quality_report.get(
                "warnings",
                []
            ),

        "top_explainability_features":
            explainability.get(
                "top_features",
                []
            ),

        "known_limitations": [

            "Performance is dependent on the training dataset.",

            "The model should be validated on future unseen data.",

            "Feature importance and SHAP values indicate model "
            "influence, not causal relationships.",

            "Data drift may reduce future model performance."
        ]
    }


    # ========================================================
    # JSON
    # ========================================================

    json_path = (

        output_directory

        /

        "model_card.json"
    )


    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            card,

            file,

            indent=4,

            default=str
        )


    # ========================================================
    # MARKDOWN
    # ========================================================

    markdown_path = (

        output_directory

        /

        "model_card.md"
    )


    lines = [

        "# AutoDS Final Model Card",

        "",

        f"**Problem Type:** "
        f"{card['problem_type']}",

        "",

        f"**Target:** "
        f"{card['target']}",

        "",

        f"**Primary Metric:** "
        f"{card['primary_metric']}",

        "",

        "## Verified Final Metrics",

        ""
    ]


    for metric, value in (
        card[
            "verified_final_metrics"
        ].items()
    ):

        lines.append(
            f"- {metric}: {value}"
        )


    lines.extend([

        "",

        "## Data Quality",

        "",

        f"Quality score: "
        f"{card['data_quality_score']}",

        "",

        "## Known Limitations",

        ""
    ])


    for limitation in (
        card[
            "known_limitations"
        ]
    ):

        lines.append(
            f"- {limitation}"
        )


    with open(

        markdown_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


    return {

        "card":
            card,

        "json_path":
            str(
                json_path
            ),

        "markdown_path":
            str(
                markdown_path
            )
    }