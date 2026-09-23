from pydantic import BaseModel


# ============================================================
# PROBLEM AGENT
# ============================================================

class ProblemAnalysis(BaseModel):

    objective: str

    task_type: str

    target: str

    primary_metric: str

    secondary_metrics: list[str]

    rationale: str

    assumptions: list[str]


# ============================================================
# DATASET AGENT
# ============================================================

class DatasetAnalysis(BaseModel):

    dataset_summary: str

    important_schema_findings: list[str]

    feature_concerns: list[str]

    preprocessing_recommendations: list[str]


# ============================================================
# DATA QUALITY AGENT
# ============================================================

class QualityAnalysis(BaseModel):

    overall_assessment: str

    highest_priority_issues: list[str]

    model_risks: list[str]

    recommended_actions: list[str]

    human_review_required: bool


# ============================================================
# EDA AGENT
# ============================================================

class EDAAnalysis(BaseModel):

    important_patterns: list[str]

    potentially_useful_features: list[str]

    suspicious_patterns: list[str]

    additional_checks: list[str]

    interpretation: str


# ============================================================
# ML PLANNING AGENT
# ============================================================

class MLPlan(BaseModel):

    primary_metric: str

    secondary_metrics: list[str]

    validation_strategy: str

    recommended_model_families: list[str]

    preprocessing_actions: list[str]

    feature_engineering_ideas: list[str]

    next_experiments: list[str]

    stopping_rule: str


# ============================================================
# SYNTHESIS AGENT
# ============================================================

class FinalAnalysis(BaseModel):

    executive_summary: str

    current_project_status: str

    strongest_findings: list[str]

    important_warnings: list[str]

    recommended_next_actions: list[str]

# ============================================================
# PHASE 4
# AUTONOMOUS CORRECTION DECISION
# ============================================================

class CorrectionDecision(BaseModel):

    continue_search: bool

    experiment_id: str | None = None

    diagnosis: list[str]

    rationale: str

    expected_effect: str


# ============================================================
# PHASE 4
# FINAL AUTONOMY SUMMARY
# ============================================================

class AutonomySummary(BaseModel):

    stop_reason: str

    what_changed: list[str]

    successful_actions: list[str]

    failed_actions: list[str]

    next_recommendations: list[str]