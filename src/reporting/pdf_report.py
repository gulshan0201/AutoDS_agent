from pathlib import Path


from reportlab.lib import colors

from reportlab.lib.enums import (
    TA_CENTER
)

from reportlab.lib.pagesizes import (
    A4
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.units import (
    inch
)

from reportlab.lib.utils import (
    ImageReader
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


# ============================================================
# IMAGE HELPER
# ============================================================

def scaled_image(
    path,
    max_width=6.3 * inch,
    max_height=4.5 * inch
):

    path = Path(
        path
    )


    if not path.exists():

        return None


    reader = ImageReader(
        str(
            path
        )
    )


    width, height = (
        reader.getSize()
    )


    scale = min(

        max_width / width,

        max_height / height
    )


    return Image(

        str(
            path
        ),

        width=
            width * scale,

        height=
            height * scale
    )


# ============================================================
# PAGE NUMBER
# ============================================================

def add_page_number(
    canvas,
    doc
):

    canvas.saveState()


    canvas.setFont(
        "Helvetica",
        8
    )


    canvas.drawRightString(

        A4[0] - 35,

        20,

        f"Page {doc.page}"
    )


    canvas.restoreState()


# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf_report(

    report_data,

    output_path=
        "artifacts/phase6/AutoDS_Final_Report.pdf"
):

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(

        parents=True,

        exist_ok=True
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "AutoDSTitle",

        parent=styles[
            "Title"
        ],

        alignment=TA_CENTER,

        fontSize=22,

        leading=27,

        spaceAfter=20
    )


    section_style = ParagraphStyle(

        "AutoDSSection",

        parent=styles[
            "Heading2"
        ],

        spaceBefore=12,

        spaceAfter=8
    )


    body_style = ParagraphStyle(

        "AutoDSBody",

        parent=styles[
            "BodyText"
        ],

        fontSize=10,

        leading=15
    )


    document = SimpleDocTemplate(

        str(
            output_path
        ),

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=45,

        bottomMargin=40
    )


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(

        Paragraph(

            "Autonomous Data Scientist",
            title_style
        )
    )


    story.append(

        Paragraph(

            "Automated Machine Learning, Explainability "
            "and Model Evaluation Report",

            styles[
                "Heading3"
            ]
        )
    )


    story.append(
        Spacer(
            1,
            20
        )
    )


    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(

        Paragraph(
            "Executive Summary",
            section_style
        )
    )


    narrative = report_data[
        "narrative"
    ]


    story.append(

        Paragraph(

            narrative.get(
                "executive_summary",
                ""
            ),

            body_style
        )
    )


    # ========================================================
    # DATASET
    # ========================================================

    story.append(

        Paragraph(
            "Dataset Overview",
            section_style
        )
    )


    quality = report_data[
        "quality"
    ]


    dataset_table = [

        [
            "Attribute",
            "Value"
        ],

        [
            "Rows",
            str(
                quality.get(
                    "rows"
                )
            )
        ],

        [
            "Columns",
            str(
                quality.get(
                    "columns"
                )
            )
        ],

        [
            "Data Quality Score",
            str(
                quality.get(
                    "quality_score"
                )
            )
        ],

        [
            "Problem Type",
            str(
                report_data[
                    "phase5"
                ].get(
                    "problem_type"
                )
            )
        ]
    ]


    table = Table(

        dataset_table,

        colWidths=[
            2.2 * inch,
            3.8 * inch
        ]
    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            )
        ])
    )


    story.append(
        table
    )


    # ========================================================
    # FINAL MODEL METRICS
    # ========================================================

    story.append(

        Paragraph(
            "Final Model Performance",
            section_style
        )
    )


    diagnostics = report_data[
        "diagnostics"
    ]


    metric_table = [
        [
            "Metric",
            "Value"
        ]
    ]


    for metric, value in (
        diagnostics.get(
            "metrics",
            {}
        ).items()
    ):

        metric_table.append([
            str(metric),
            str(value)
        ])


    metric_table_object = Table(

        metric_table,

        colWidths=[
            3 * inch,
            3 * inch
        ]
    )


    metric_table_object.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            )
        ])
    )


    story.append(
        metric_table_object
    )


    story.append(
        Spacer(
            1,
            10
        )
    )


    story.append(

        Paragraph(

            narrative.get(
                "model_performance_interpretation",
                ""
            ),

            body_style
        )
    )


    # ========================================================
    # DIAGNOSTIC IMAGES
    # ========================================================

    for image_name, image_path in (
        diagnostics.get(
            "images",
            {}
        ).items()
    ):

        image_object = (
            scaled_image(
                image_path
            )
        )


        if image_object:

            story.append(
                Spacer(
                    1,
                    10
                )
            )

            story.append(
                image_object
            )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    story.append(
        PageBreak()
    )


    story.append(

        Paragraph(
            "Model Explainability",
            section_style
        )
    )


    explainability = report_data[
        "explainability"
    ]


    story.append(

        Paragraph(

            narrative.get(
                "explainability_interpretation",
                ""
            ),

            body_style
        )
    )


    shap_plot = (
        explainability.get(
            "shap_plot_path"
        )
    )


    if shap_plot:

        image_object = (
            scaled_image(
                shap_plot
            )
        )


        if image_object:

            story.append(
                Spacer(
                    1,
                    12
                )
            )

            story.append(
                image_object
            )


    # ========================================================
    # TOP FEATURES
    # ========================================================

    top_features = (
        explainability.get(
            "top_features",
            []
        )
    )


    if top_features:

        story.append(

            Paragraph(
                "Top Model Features",
                section_style
            )
        )


        feature_table = [

            [
                "Feature",
                "Importance"
            ]
        ]


        for item in top_features[
            :15
        ]:

            value = (

                item.get(
                    "mean_abs_shap"
                )

                if
                "mean_abs_shap"
                in item

                else
                item.get(
                    "importance"
                )
            )


            feature_table.append([

                str(
                    item.get(
                        "feature"
                    )
                ),

                str(
                    value
                )
            ])


        feature_table_object = Table(

            feature_table,

            colWidths=[
                4.5 * inch,
                1.5 * inch
            ]
        )


        feature_table_object.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                )
            ])
        )


        story.append(
            feature_table_object
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    story.append(

        Paragraph(
            "Data Quality Interpretation",
            section_style
        )
    )


    story.append(

        Paragraph(

            narrative.get(
                "data_quality_interpretation",
                ""
            ),

            body_style
        )
    )


    warnings = quality.get(
        "warnings",
        []
    )


    for warning in warnings:

        story.append(

            Paragraph(

                f"- {warning}",

                body_style
            )
        )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    story.append(

        Paragraph(
            "Limitations",
            section_style
        )
    )


    for limitation in narrative.get(
        "limitations",
        []
    ):

        story.append(

            Paragraph(

                f"- {limitation}",

                body_style
            )
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    story.append(

        Paragraph(
            "Recommendations",
            section_style
        )
    )


    for recommendation in narrative.get(
        "recommendations",
        []
    ):

        story.append(

            Paragraph(

                f"- {recommendation}",

                body_style
            )
        )


    # ========================================================
    # BUILD
    # ========================================================

    document.build(

        story,

        onFirstPage=
            add_page_number,

        onLaterPages=
            add_page_number
    )


    return str(
        output_path
    )