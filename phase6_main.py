import json

from pathlib import Path


from src.reporting.diagnostics import (
    generate_diagnostics
)

from src.explainability.explainability_engine import (
    generate_explainability
)

from src.reporting.narrative import (
    generate_report_narrative
)

from src.reporting.model_card import (
    generate_model_card
)

from src.reporting.notebook_generator import (
    generate_notebook
)

from src.reporting.pdf_report import (
    generate_pdf_report
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Churn"


PROCESSED_DATASET = (
    "data/processed/"
    "phase2_normalized.csv"
)


PHASE5_REPORT_PATH = (
    "artifacts/advanced_ml/"
    "phase5_advanced_ml_report.json"
)


QUALITY_REPORT_PATH = (
    "artifacts/data_quality/"
    "data_quality_report.json"
)


FINAL_MODEL_PATH = (
    "artifacts/advanced_ml/"
    "phase5_final_champion.joblib"
)


LABEL_ENCODER_PATH = (
    "artifacts/advanced_ml/"
    "target_label_encoder.joblib"
)


OUTPUT_DIRECTORY = Path(
    "artifacts/phase6"
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD VERIFIED REPORTS
# ============================================================

print(
    "\n======================================"
)

print(
    "PHASE 6 - EXPLAINABILITY & REPORTING"
)

print(
    "======================================"
)


with open(
    PHASE5_REPORT_PATH,
    "r",
    encoding="utf-8"
) as file:

    phase5_report = json.load(
        file
    )


with open(
    QUALITY_REPORT_PATH,
    "r",
    encoding="utf-8"
) as file:

    quality_report = json.load(
        file
    )


problem_type = (
    phase5_report[
        "problem_type"
    ]
)


schema_report = (
    quality_report.get(
        "schema_findings",
        {}
    )
)


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

print(
    "\nGenerating final model diagnostics..."
)


diagnostics = generate_diagnostics(

    dataset_path=
        PROCESSED_DATASET,

    target=
        TARGET,

    problem_type=
        problem_type,

    schema_report=
        schema_report,

    model_path=
        FINAL_MODEL_PATH,

    label_encoder_path=
        LABEL_ENCODER_PATH
)


print(
    "Final metrics:"
)

print(
    diagnostics[
        "metrics"
    ]
)


# ============================================================
# EXPLAINABILITY
# ============================================================

print(
    "\nGenerating SHAP explainability..."
)


explainability = (
    generate_explainability(

        dataset_path=
            PROCESSED_DATASET,

        target=
            TARGET,

        schema_report=
            schema_report,

        model_path=
            FINAL_MODEL_PATH,

        problem_type=
            problem_type
    )
)


print(
    "SHAP status:",
    explainability[
        "shap_status"
    ]
)


if explainability[
    "shap_error"
]:

    print(
        "SHAP message:",
        explainability[
            "shap_error"
        ]
    )


# ============================================================
# VERIFIED AI NARRATIVE
# ============================================================

print(
    "\nGenerating verified AI interpretation..."
)


narrative = (
    generate_report_narrative(

        phase5_report,

        quality_report,

        diagnostics,

        explainability
    )
)


# ============================================================
# MODEL CARD
# ============================================================

print(
    "\nGenerating model card..."
)


model_card = (
    generate_model_card(

        dataset_path=
            PROCESSED_DATASET,

        target=
            TARGET,

        phase5_report=
            phase5_report,

        quality_report=
            quality_report,

        diagnostics=
            diagnostics,

        explainability=
            explainability
    )
)


# ============================================================
# NOTEBOOK
# ============================================================

print(
    "\nGenerating reproducible notebook..."
)


notebook_path = (
    generate_notebook(

        dataset_path=
            PROCESSED_DATASET,

        target=
            TARGET,

        model_path=
            FINAL_MODEL_PATH,

        schema_report=
            schema_report,

        phase5_report_path=
            PHASE5_REPORT_PATH,

        explainability_path=
            (
                explainability.get(
                    "shap_importance_path"
                )
                or
                explainability.get(
                    "native_importance_path"
                )
            )
    )
)


# ============================================================
# BUILD FINAL REPORT DATA
# ============================================================

report_data = {

    "phase5":
        phase5_report,

    "quality":
        quality_report,

    "diagnostics":
        diagnostics,

    "explainability":
        explainability,

    "narrative":
        narrative,

    "model_card":
        model_card[
            "card"
        ]
}


# ============================================================
# PDF
# ============================================================

print(
    "\nGenerating professional PDF report..."
)


pdf_path = (
    generate_pdf_report(

        report_data
    )
)


# ============================================================
# SAVE PHASE 6 SUMMARY
# ============================================================

summary = {

    "problem_type":
        problem_type,

    "final_model":
        FINAL_MODEL_PATH,

    "final_metrics":
        diagnostics[
            "metrics"
        ],

    "shap_status":
        explainability[
            "shap_status"
        ],

    "top_features":
        explainability[
            "top_features"
        ],

    "model_card_json":
        model_card[
            "json_path"
        ],

    "model_card_markdown":
        model_card[
            "markdown_path"
        ],

    "notebook":
        notebook_path,

    "pdf_report":
        pdf_path
}


summary_path = (

    OUTPUT_DIRECTORY

    /

    "phase6_summary.json"
)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        summary,

        file,

        indent=4,

        default=str
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n======================================"
)

print(
    "PHASE 6 RESULTS"
)

print(
    "======================================"
)


print(
    "\nFinal Model:"
)

print(
    FINAL_MODEL_PATH
)


print(
    "\nSHAP Status:"
)

print(
    explainability[
        "shap_status"
    ]
)


print(
    "\nModel Card:"
)

print(
    model_card[
        "markdown_path"
    ]
)


print(
    "\nNotebook:"
)

print(
    notebook_path
)


print(
    "\nPDF Report:"
)

print(
    pdf_path
)


print(
    "\nSummary:"
)

print(
    summary_path
)


print(
    "\n======================================"
)

print(
    "PHASE 6 COMPLETED"
)

print(
    "======================================\n"
)