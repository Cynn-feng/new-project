from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    GENERAL_QA = "general_qa"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    REASONING = "reasoning"
    RAG = "rag"


class TaskClassification(BaseModel):
    task_type: TaskType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class TaskFeatures(BaseModel):
    input_length: int = Field(..., ge=0)
    line_count: int = Field(default=1, ge=1)
    question_count: int = Field(default=0, ge=0)
    code_marker_count: int = Field(default=0, ge=0)
    retrieval_marker_count: int = Field(default=0, ge=0)
    reasoning_marker_count: int = Field(default=0, ge=0)
    summarization_marker_count: int = Field(default=0, ge=0)
    contains_code: bool = False
    needs_long_context: bool = False
    needs_tools: bool = False
    needs_json: bool = False
    high_reasoning: bool = False
    code_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    summarization_signal: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelSelection(BaseModel):
    model_alias: str
    provider: str
    model: str
    reason: str
    fallback_model_alias: str | None = None


class DecisionTrace(BaseModel):
    source: str
    applied_route_to: str
    applied_reason: str
    rule_route_to: str
    ai_route_to: str | None = None
    ai_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ai_model_version: str | None = None
    notes: list[str] = Field(default_factory=list)


class LLMUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)


class LLMResponse(BaseModel):
    content: str = ""
    provider: str
    model: str
    latency_ms: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    finish_reason: str | None = None
    success: bool = True
    error_message: str | None = None
    simulated: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    valid: bool = True
    issues: list[str] = Field(default_factory=list)
    json_valid: bool | None = None
    contains_code_block: bool | None = None
    passed_basic_checks: bool = True


class ChatResponse(BaseModel):
    task_id: int | None = None
    execution_id: int | None = None
    task_type: TaskType
    selected_model: str
    resolved_model: str
    provider: str
    reason: str
    answer: str
    latency_ms: int = Field(..., ge=0)
    fallback_triggered: bool = False
    classification: TaskClassification
    features: TaskFeatures
    decision_trace: DecisionTrace | None = None
    validation: ValidationResult
    usage: LLMUsage


class RouterInspectResponse(BaseModel):
    task_type: TaskType
    features: TaskFeatures
    selected_model_key: str
    selected_model_name: str
    reason: str
    decision_trace: DecisionTrace | None = None


class AdminTopModel(BaseModel):
    selected_model_key: str
    selected_model_name: str
    actual_model_used: str
    count: int = Field(..., ge=0)


class AdminErrorBreakdown(BaseModel):
    error_type: str
    count: int = Field(..., ge=0)


class AdminStatsResponse(BaseModel):
    total_requests: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    top_models: list[AdminTopModel] = Field(default_factory=list)
    error_breakdown: list[AdminErrorBreakdown] = Field(default_factory=list)


class AIAnalysisResponse(BaseModel):
    summary: str
    observations: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class RouterRecommendationItem(BaseModel):
    priority: int = Field(..., ge=1)
    condition: str
    route_to: Literal[
        "fast_general",
        "strong_reasoning",
        "code_specialist",
        "structured_output_model",
    ]
    reason: str


class RouterRecommendationResponse(BaseModel):
    summary: str
    recommended_changes: list[RouterRecommendationItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RecommendationReviewItem(BaseModel):
    condition: str
    route_to: str
    status: Literal[
        "already_covered",
        "compatible",
        "unsupported_route",
        "needs_manual_review",
    ]
    comment: str


class RecommendationReviewResponse(BaseModel):
    summary: str
    reviews: list[RecommendationReviewItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PolicyRecommendationResponse(BaseModel):
    id: int
    summary: str
    condition: str
    route_to: str
    reason: str
    priority: int = Field(..., ge=1)
    status: str
    review_status: str | None = None
    review_comment: str | None = None
    source: str


class GenerateRecommendationsResponse(BaseModel):
    summary: str
    created_count: int = Field(default=0, ge=0)
    recommendations: list[PolicyRecommendationResponse] = Field(default_factory=list)


class PolicyDecisionResponse(BaseModel):
    recommendation_id: int
    status: str
    message: str


class ActiveRoutingRule(BaseModel):
    rule_name: str
    condition_description: str
    route_to: str
    reason: str
    source: str


class AvailableModelSnapshot(BaseModel):
    model_key: str
    model_name: str
    capabilities: list[str] = Field(default_factory=list)
    cost_level: str
    speed_level: str


class ApprovedPendingRecommendation(BaseModel):
    id: int
    condition: str
    route_to: str
    reason: str
    priority: int = Field(..., ge=1)
    status: str
    review_status: str | None = None


class PolicySnapshotResponse(BaseModel):
    active_policy_version: str
    active_routing_rules: list[ActiveRoutingRule] = Field(default_factory=list)
    available_models: list[AvailableModelSnapshot] = Field(default_factory=list)
    approved_but_not_applied: list[ApprovedPendingRecommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PolicySimulationSampleResult(BaseModel):
    input_message: str
    current_task_type: str
    current_selected_model_key: str
    simulated_selected_model_key: str
    changed: bool
    explanation: str


class PolicySimulationResponse(BaseModel):
    recommendation_id: int
    recommendation_condition: str
    recommendation_route_to: str
    total_samples: int = Field(default=0, ge=0)
    changed_samples: int = Field(default=0, ge=0)
    unchanged_samples: int = Field(default=0, ge=0)
    sample_results: list[PolicySimulationSampleResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PolicySimulationSummary(BaseModel):
    total_samples: int = Field(default=0, ge=0)
    changed_samples: int = Field(default=0, ge=0)
    unchanged_samples: int = Field(default=0, ge=0)
    change_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_condition: bool = False


class EstimatedImpact(BaseModel):
    level: str
    explanation: str


class RolloutPhase(BaseModel):
    phase_name: str
    traffic_percentage: int = Field(..., ge=0, le=100)
    success_criteria: list[str] = Field(default_factory=list)
    rollback_triggers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RollbackPlan(BaseModel):
    trigger_conditions: list[str] = Field(default_factory=list)
    immediate_action: list[str] = Field(default_factory=list)
    restore_target: str


class PolicyRolloutPlanResponse(BaseModel):
    recommendation_id: int
    recommendation_condition: str
    recommendation_route_to: str
    simulation_summary: PolicySimulationSummary
    estimated_impact: EstimatedImpact
    rollout_strategy: str
    phases: list[RolloutPhase] = Field(default_factory=list)
    monitoring_metrics: list[str] = Field(default_factory=list)
    rollback_plan: RollbackPlan
    notes: list[str] = Field(default_factory=list)


class ApplyGuardrailCheck(BaseModel):
    check_name: str
    passed: bool
    severity: str
    message: str


class PolicyApplyReadinessResponse(BaseModel):
    recommendation_id: int
    recommendation_condition: str
    recommendation_route_to: str
    current_status: str
    readiness: str
    guardrail_checks: list[ApplyGuardrailCheck] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    non_blocking_warnings: list[str] = Field(default_factory=list)
    next_step: str
    notes: list[str] = Field(default_factory=list)


class GovernanceCounts(BaseModel):
    total_recommendations: int = Field(default=0, ge=0)
    pending_count: int = Field(default=0, ge=0)
    approved_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    approved_but_not_applied_count: int = Field(default=0, ge=0)
    ready_for_future_apply_count: int = Field(default=0, ge=0)
    not_ready_count: int = Field(default=0, ge=0)
    unnecessary_count: int = Field(default=0, ge=0)
    manual_review_required_count: int = Field(default=0, ge=0)


class GovernanceRecommendationSummary(BaseModel):
    id: int
    condition: str
    route_to: str
    priority: int = Field(..., ge=1)
    status: str
    review_status: str | None = None
    readiness: str
    has_simulation: bool = False
    has_rollout_plan: bool = False
    source: str


class UnifiedGovernanceDashboardResponse(BaseModel):
    active_policy_version: str
    current_router_summary: list[str] = Field(default_factory=list)
    governance_counts: GovernanceCounts
    recent_recommendations: list[GovernanceRecommendationSummary] = Field(default_factory=list)
    approved_but_not_applied: list[GovernanceRecommendationSummary] = Field(default_factory=list)
    readiness_overview: dict[str, list[GovernanceRecommendationSummary]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PolicyAuditEventResponse(BaseModel):
    id: int
    recommendation_id: int
    event_type: str
    event_status: str | None = None
    event_summary: str
    event_details_json: str | None = None
    created_at: datetime


class PolicyAuditTimelineResponse(BaseModel):
    recommendation_id: int
    timeline: list[PolicyAuditEventResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GovernanceAuditOverviewResponse(BaseModel):
    total_audit_events: int = Field(default=0, ge=0)
    recent_events: list[PolicyAuditEventResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RecommendationCoreSummary(BaseModel):
    id: int
    summary: str
    condition: str
    route_to: str
    reason: str
    priority: int = Field(..., ge=1)
    status: str
    review_status: str | None = None
    review_comment: str | None = None
    source: str


class RecommendationSimulationSummary(BaseModel):
    total_samples: int = Field(default=0, ge=0)
    changed_samples: int = Field(default=0, ge=0)
    unchanged_samples: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class RecommendationRolloutSummary(BaseModel):
    rollout_strategy: str
    monitoring_metrics: list[str] = Field(default_factory=list)
    rollback_plan: RollbackPlan
    notes: list[str] = Field(default_factory=list)


class RecommendationReadinessSummary(BaseModel):
    readiness: str
    blocking_issues: list[str] = Field(default_factory=list)
    non_blocking_warnings: list[str] = Field(default_factory=list)
    next_step: str
    notes: list[str] = Field(default_factory=list)


class RecommendationAuditSummary(BaseModel):
    total_events: int = Field(default=0, ge=0)
    latest_event_type: str | None = None
    latest_event_status: str | None = None
    timeline_preview: list[str] = Field(default_factory=list)


class GovernanceReviewPackResponse(BaseModel):
    generated_at: datetime
    active_policy_version: str
    recommendation: RecommendationCoreSummary
    simulation: RecommendationSimulationSummary
    rollout_plan: RecommendationRolloutSummary
    apply_readiness: RecommendationReadinessSummary
    audit_summary: RecommendationAuditSummary
    executive_summary: str
    notes: list[str] = Field(default_factory=list)


class PortfolioRecommendationItem(BaseModel):
    id: int
    summary: str
    condition: str
    route_to: str
    priority: int = Field(..., ge=1)
    status: str
    review_status: str | None = None
    readiness: str
    source: str
    latest_event_type: str | None = None
    latest_event_status: str | None = None
    risk_label: str
    recommended_admin_action: str


class PortfolioCounts(BaseModel):
    total: int = Field(default=0, ge=0)
    pending: int = Field(default=0, ge=0)
    approved: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    ready_for_future_apply: int = Field(default=0, ge=0)
    not_ready: int = Field(default=0, ge=0)
    unnecessary: int = Field(default=0, ge=0)
    manual_review_required: int = Field(default=0, ge=0)
    high_priority_count: int = Field(default=0, ge=0)
    low_impact_count: int = Field(default=0, ge=0)
    high_impact_count: int = Field(default=0, ge=0)


class GovernancePortfolioReportResponse(BaseModel):
    generated_at: datetime
    active_policy_version: str
    counts: PortfolioCounts
    top_priority_recommendations: list[PortfolioRecommendationItem] = Field(default_factory=list)
    blocked_recommendations: list[PortfolioRecommendationItem] = Field(default_factory=list)
    unnecessary_recommendations: list[PortfolioRecommendationItem] = Field(default_factory=list)
    long_pending_recommendations: list[PortfolioRecommendationItem] = Field(default_factory=list)
    recommendation_portfolio: list[PortfolioRecommendationItem] = Field(default_factory=list)
    executive_summary: str
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    database_url: str


class ModelProfileResponse(BaseModel):
    alias: str
    provider: str
    model: str
    capabilities: list[str]
    max_context: int = Field(..., ge=1)
    cost_level: str
    speed_level: str
    fallback_model_alias: str | None = None
