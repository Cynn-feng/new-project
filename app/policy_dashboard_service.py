from collections import Counter

from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.policy_apply_gate_service import evaluate_apply_readiness
from app.policy_snapshot_service import get_policy_snapshot
from app.policy_workflow_service import list_recommendations
from app.schemas import (
    GovernanceCounts,
    GovernanceRecommendationSummary,
    PolicyApplyReadinessResponse,
    PolicyRecommendationResponse,
    UnifiedGovernanceDashboardResponse,
)

READINESS_BUCKETS = (
    "ready_for_future_apply",
    "not_ready",
    "unnecessary",
    "manual_review_required",
)


def get_unified_governance_dashboard(
    limit: int = 20,
    sample_limit: int = 20,
    db: Session | None = None,
) -> UnifiedGovernanceDashboardResponse:
    init_db()
    bounded_limit = max(1, min(limit, 100))
    owns_session = db is None
    session = db or SessionLocal()
    try:
        snapshot = get_policy_snapshot(db=session)
        recommendations = list_recommendations(status=None, db=session)

        readiness_map: dict[int, PolicyApplyReadinessResponse] = {}
        readiness_failures: list[str] = []
        for recommendation in recommendations:
            readiness = _safe_readiness_evaluation(
                recommendation=recommendation,
                sample_limit=sample_limit,
                db=session,
            )
            readiness_map[recommendation.id] = readiness
            if readiness.notes and any("safe fallback" in note.casefold() for note in readiness.notes):
                readiness_failures.append(f"Recommendation {recommendation.id} readiness defaulted to not_ready.")

        summaries = [
            _build_recommendation_summary(recommendation, readiness_map[recommendation.id])
            for recommendation in recommendations
        ]
        recent_recommendations = summaries[:bounded_limit]
        approved_but_not_applied = [summary for summary in summaries if summary.status == "approved"]

        readiness_overview: dict[str, list[GovernanceRecommendationSummary]] = {
            bucket: [] for bucket in READINESS_BUCKETS
        }
        for summary in summaries:
            readiness_overview.setdefault(summary.readiness, []).append(summary)

        counts = _build_governance_counts(
            summaries=summaries,
            approved_but_not_applied=approved_but_not_applied,
        )

        notes = list(snapshot.notes)
        notes.extend(
            [
                "This dashboard is read-only and does not change recommendation status.",
                "No recommendation was auto-applied and no rollout was executed.",
                "Use readiness and rollout plan outputs as governance signals only.",
            ]
        )
        notes.extend(readiness_failures)

        return UnifiedGovernanceDashboardResponse(
            active_policy_version=snapshot.active_policy_version,
            current_router_summary=_build_current_router_summary(snapshot.active_routing_rules),
            governance_counts=counts,
            recent_recommendations=recent_recommendations,
            approved_but_not_applied=approved_but_not_applied,
            readiness_overview=readiness_overview,
            notes=notes,
        )
    finally:
        if owns_session:
            session.close()


def _safe_readiness_evaluation(
    *,
    recommendation: PolicyRecommendationResponse,
    sample_limit: int,
    db: Session,
) -> PolicyApplyReadinessResponse:
    try:
        return evaluate_apply_readiness(
            recommendation_id=recommendation.id,
            sample_limit=sample_limit,
            record_audit=False,
            db=db,
        )
    except Exception as exc:
        return PolicyApplyReadinessResponse(
            recommendation_id=recommendation.id,
            recommendation_condition=recommendation.condition,
            recommendation_route_to=recommendation.route_to,
            current_status=recommendation.status,
            readiness="not_ready",
            guardrail_checks=[],
            blocking_issues=[f"Readiness evaluation failed: {exc}"],
            non_blocking_warnings=[],
            next_step="Do not apply; fix blocking issues first.",
            notes=[
                "Safe fallback readiness was generated for dashboard aggregation.",
                "This fallback does not apply any recommendation or modify router.py.",
            ],
        )


def _build_recommendation_summary(
    recommendation: PolicyRecommendationResponse,
    readiness: PolicyApplyReadinessResponse,
) -> GovernanceRecommendationSummary:
    return GovernanceRecommendationSummary(
        id=recommendation.id,
        condition=recommendation.condition,
        route_to=recommendation.route_to,
        priority=recommendation.priority,
        status=recommendation.status,
        review_status=recommendation.review_status,
        readiness=readiness.readiness,
        has_simulation=_guardrail_passed(readiness, "dry_run_simulation_completed"),
        has_rollout_plan=_guardrail_passed(readiness, "rollout_plan_available"),
        source=recommendation.source,
    )


def _guardrail_passed(readiness: PolicyApplyReadinessResponse, check_name: str) -> bool:
    for check in readiness.guardrail_checks:
        if check.check_name == check_name:
            return check.passed
    return False


def _build_governance_counts(
    *,
    summaries: list[GovernanceRecommendationSummary],
    approved_but_not_applied: list[GovernanceRecommendationSummary],
) -> GovernanceCounts:
    status_counts = Counter(summary.status for summary in summaries)
    readiness_counts = Counter(summary.readiness for summary in summaries)
    return GovernanceCounts(
        total_recommendations=len(summaries),
        pending_count=status_counts.get("pending", 0),
        approved_count=status_counts.get("approved", 0),
        rejected_count=status_counts.get("rejected", 0),
        approved_but_not_applied_count=len(approved_but_not_applied),
        ready_for_future_apply_count=readiness_counts.get("ready_for_future_apply", 0),
        not_ready_count=readiness_counts.get("not_ready", 0),
        unnecessary_count=readiness_counts.get("unnecessary", 0),
        manual_review_required_count=readiness_counts.get("manual_review_required", 0),
    )


def _build_current_router_summary(active_routing_rules) -> list[str]:
    summary_lines: list[str] = []
    for rule in active_routing_rules:
        if rule.rule_name == "code_signal_priority":
            summary_lines.append("Code-containing or coding-classified inputs route to code_specialist.")
        elif rule.rule_name == "retrieval_long_context_priority":
            summary_lines.append("Retrieval-heavy and long-context inputs route to long_context_rag.")
        elif rule.rule_name == "reasoning_priority":
            summary_lines.append("Reasoning-heavy inputs route to strong_reasoning.")
        elif rule.rule_name == "summarization_fast_path":
            summary_lines.append("Light summarization requests usually remain on fast_general.")
        elif rule.rule_name == "general_default":
            summary_lines.append("The default live route remains fast_general when specialist signals stay low.")
        else:
            summary_lines.append(f"{rule.condition_description} Routes to {rule.route_to}.")
    return summary_lines
