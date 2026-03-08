from datetime import datetime

from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.policy_apply_gate_service import evaluate_apply_readiness
from app.policy_audit_service import get_recommendation_timeline
from app.policy_rollout_plan_service import RESTORE_TARGET, create_rollout_plan
from app.policy_simulation_service import simulate_recommendation_application
from app.policy_snapshot_service import get_policy_snapshot
from app.schemas import (
    GovernanceReviewPackResponse,
    PolicyApplyReadinessResponse,
    PolicyRecommendationResponse,
    PolicyRolloutPlanResponse,
    PolicySimulationResponse,
    RecommendationAuditSummary,
    RecommendationCoreSummary,
    RecommendationReadinessSummary,
    RecommendationRolloutSummary,
    RecommendationSimulationSummary,
    RollbackPlan,
)
from app.storage.entities import PolicyRecommendation


def generate_governance_review_pack(
    recommendation_id: int,
    sample_limit: int = 20,
    db: Session | None = None,
) -> GovernanceReviewPackResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        recommendation = session.get(PolicyRecommendation, recommendation_id)
        if recommendation is None:
            raise KeyError(f"Policy recommendation {recommendation_id} was not found.")

        snapshot = get_policy_snapshot(db=session)
        core = _build_core_summary(recommendation)
        simulation = _build_simulation_summary(recommendation_id=recommendation_id, sample_limit=sample_limit, db=session)
        rollout = _build_rollout_summary(recommendation_id=recommendation_id, sample_limit=sample_limit, db=session)
        readiness = _build_readiness_summary(recommendation_id=recommendation_id, sample_limit=sample_limit, db=session)
        timeline = get_recommendation_timeline(recommendation_id=recommendation_id, db=session)
        audit_summary = _build_audit_summary(timeline.timeline)

        return GovernanceReviewPackResponse(
            generated_at=datetime.now(),
            active_policy_version=snapshot.active_policy_version,
            recommendation=core,
            simulation=simulation,
            rollout_plan=rollout,
            apply_readiness=readiness,
            audit_summary=audit_summary,
            executive_summary=_build_executive_summary(
                recommendation=core,
                simulation=simulation,
                readiness=readiness,
                rollout=rollout,
            ),
            notes=[
                "This governance review pack is read-only and export-friendly JSON.",
                "No recommendation was applied and no rollout was executed while generating this report.",
                "app/router/rule_router.py still defines the active live routing behavior.",
            ],
        )
    finally:
        if owns_session:
            session.close()


def _build_core_summary(recommendation: PolicyRecommendation) -> RecommendationCoreSummary:
    return RecommendationCoreSummary(
        id=recommendation.id,
        summary=recommendation.summary,
        condition=recommendation.condition,
        route_to=recommendation.route_to,
        reason=recommendation.reason,
        priority=recommendation.priority,
        status=recommendation.status,
        review_status=recommendation.review_status,
        review_comment=recommendation.review_comment,
        source=recommendation.source,
    )


def _build_simulation_summary(
    *,
    recommendation_id: int,
    sample_limit: int,
    db: Session,
) -> RecommendationSimulationSummary:
    try:
        simulation = simulate_recommendation_application(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            record_audit=False,
            db=db,
        )
        return RecommendationSimulationSummary(
            total_samples=simulation.total_samples,
            changed_samples=simulation.changed_samples,
            unchanged_samples=simulation.unchanged_samples,
            notes=list(simulation.notes),
        )
    except Exception as exc:
        return RecommendationSimulationSummary(
            total_samples=0,
            changed_samples=0,
            unchanged_samples=0,
            notes=[
                f"Simulation data unavailable: {exc}",
                "This fallback does not change live routing behavior.",
            ],
        )


def _build_rollout_summary(
    *,
    recommendation_id: int,
    sample_limit: int,
    db: Session,
) -> RecommendationRolloutSummary:
    try:
        rollout_plan = create_rollout_plan(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            record_audit=False,
            db=db,
        )
        return RecommendationRolloutSummary(
            rollout_strategy=rollout_plan.rollout_strategy,
            monitoring_metrics=list(rollout_plan.monitoring_metrics),
            rollback_plan=rollout_plan.rollback_plan,
            notes=list(rollout_plan.notes),
        )
    except Exception as exc:
        return RecommendationRolloutSummary(
            rollout_strategy="unavailable",
            monitoring_metrics=[],
            rollback_plan=RollbackPlan(
                trigger_conditions=[],
                immediate_action=["Continue using current router.py behavior only."],
                restore_target=RESTORE_TARGET,
            ),
            notes=[
                f"Rollout plan unavailable: {exc}",
                "No staged rollout should proceed from this fallback section.",
            ],
        )


def _build_readiness_summary(
    *,
    recommendation_id: int,
    sample_limit: int,
    db: Session,
) -> RecommendationReadinessSummary:
    try:
        readiness = evaluate_apply_readiness(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            record_audit=False,
            db=db,
        )
        return RecommendationReadinessSummary(
            readiness=readiness.readiness,
            blocking_issues=list(readiness.blocking_issues),
            non_blocking_warnings=list(readiness.non_blocking_warnings),
            next_step=readiness.next_step,
            notes=list(readiness.notes),
        )
    except Exception as exc:
        return RecommendationReadinessSummary(
            readiness="not_ready",
            blocking_issues=[f"Readiness evaluation unavailable: {exc}"],
            non_blocking_warnings=[],
            next_step="Do not apply; fix blocking issues first.",
            notes=[
                "A safe fallback readiness section was generated for reporting.",
                "This fallback does not apply any recommendation.",
            ],
        )


def _build_audit_summary(timeline) -> RecommendationAuditSummary:
    latest = timeline[-1] if timeline else None
    preview = [
        f"{event.event_type}: {event.event_summary}"
        for event in timeline[-5:]
    ]
    return RecommendationAuditSummary(
        total_events=len(timeline),
        latest_event_type=latest.event_type if latest else None,
        latest_event_status=latest.event_status if latest else None,
        timeline_preview=preview,
    )


def _build_executive_summary(
    *,
    recommendation: RecommendationCoreSummary,
    simulation: RecommendationSimulationSummary,
    readiness: RecommendationReadinessSummary,
    rollout: RecommendationRolloutSummary,
) -> str:
    if recommendation.status != "approved":
        return (
            f"This recommendation is currently {recommendation.status} and remains in governance review. "
            "It has not been applied to the live router."
        )
    if readiness.readiness == "not_ready":
        return (
            "This recommendation is approved, but it is not ready for future apply because blocking guardrail checks remain."
        )
    if readiness.readiness == "manual_review_required":
        return (
            "This recommendation is approved, but manual governance review is required before any future rollout can be considered."
        )
    if readiness.readiness == "unnecessary":
        return (
            "This recommendation appears unnecessary because the current router already covers the behavior "
            "or the dry-run showed no simulated routing change."
        )

    change_ratio = simulation.changed_samples / simulation.total_samples if simulation.total_samples else 0.0
    impact_label = "low-impact" if change_ratio < 0.2 else "medium-impact" if change_ratio < 0.5 else "high-impact"
    return (
        f"This recommendation is approved, {impact_label} in dry-run simulation, and eligible for a future controlled "
        f"rollout strategy of '{rollout.rollout_strategy}', but it has not been applied."
    )
