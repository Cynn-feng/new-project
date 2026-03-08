from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.policy_apply_gate_service import evaluate_apply_readiness
from app.policy_rollout_plan_service import create_rollout_plan
from app.policy_snapshot_service import get_policy_snapshot
from app.schemas import (
    GovernancePortfolioReportResponse,
    PolicyApplyReadinessResponse,
    PolicyRolloutPlanResponse,
    PortfolioCounts,
    PortfolioRecommendationItem,
)
from app.storage.entities import PolicyAuditEvent, PolicyRecommendation

LONG_PENDING_DAYS = 7


def generate_governance_portfolio_report(
    limit: int = 50,
    sample_limit: int = 20,
    db: Session | None = None,
) -> GovernancePortfolioReportResponse:
    init_db()
    bounded_limit = max(1, min(limit, 100))
    owns_session = db is None
    session = db or SessionLocal()
    try:
        snapshot = get_policy_snapshot(db=session)
        recommendations = list(
            session.scalars(
                select(PolicyRecommendation)
                .order_by(
                    PolicyRecommendation.created_at.desc(),
                    PolicyRecommendation.priority.asc(),
                    PolicyRecommendation.id.desc(),
                )
                .limit(bounded_limit)
            )
        )
        recommendation_ids = [item.id for item in recommendations]
        latest_events = _load_latest_audit_events(recommendation_ids, session)

        readiness_map: dict[int, PolicyApplyReadinessResponse] = {}
        rollout_map: dict[int, PolicyRolloutPlanResponse | None] = {}
        impact_map: dict[int, str] = {}

        for recommendation in recommendations:
            readiness_map[recommendation.id] = _safe_readiness(
                recommendation_id=recommendation.id,
                sample_limit=sample_limit,
                db=session,
            )
            rollout = _safe_rollout_plan(
                recommendation=recommendation,
                sample_limit=sample_limit,
                db=session,
            )
            rollout_map[recommendation.id] = rollout
            if rollout is not None:
                impact_map[recommendation.id] = rollout.estimated_impact.level

        portfolio = [
            _build_portfolio_item(
                recommendation=recommendation,
                readiness=readiness_map[recommendation.id],
                latest_event=latest_events.get(recommendation.id),
                impact_level=impact_map.get(recommendation.id),
            )
            for recommendation in recommendations
        ]

        counts = _build_portfolio_counts(portfolio=portfolio, impact_map=impact_map)
        top_priority = [
            item for item in sorted(portfolio, key=lambda current: (current.priority, current.id))
            if item.priority <= 2 and item.readiness in {"ready_for_future_apply", "manual_review_required"}
        ]
        blocked = [item for item in portfolio if item.readiness == "not_ready"]
        unnecessary = [item for item in portfolio if item.readiness == "unnecessary"]
        long_pending = [
            portfolio_item
            for recommendation, portfolio_item in zip(recommendations, portfolio, strict=False)
            if recommendation.status == "pending" and _is_long_pending(recommendation.created_at)
        ]

        return GovernancePortfolioReportResponse(
            generated_at=datetime.now(),
            active_policy_version=snapshot.active_policy_version,
            counts=counts,
            top_priority_recommendations=top_priority,
            blocked_recommendations=blocked,
            unnecessary_recommendations=unnecessary,
            long_pending_recommendations=long_pending,
            recommendation_portfolio=portfolio,
            executive_summary=_build_executive_summary(counts),
            notes=[
                "This portfolio report is read-only and does not change recommendation lifecycle state.",
                "No recommendation was applied and no rollout was executed while generating this portfolio view.",
                "app/router/rule_router.py still controls live routing behavior.",
            ],
        )
    finally:
        if owns_session:
            session.close()


def _safe_readiness(
    *,
    recommendation_id: int,
    sample_limit: int,
    db: Session,
) -> PolicyApplyReadinessResponse:
    try:
        return evaluate_apply_readiness(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            record_audit=False,
            db=db,
        )
    except Exception as exc:
        return PolicyApplyReadinessResponse(
            recommendation_id=recommendation_id,
            recommendation_condition="",
            recommendation_route_to="",
            current_status="unknown",
            readiness="not_ready",
            guardrail_checks=[],
            blocking_issues=[f"Readiness evaluation failed: {exc}"],
            non_blocking_warnings=[],
            next_step="Do not apply; fix blocking issues first.",
            notes=["Safe fallback readiness was used for portfolio reporting."],
        )


def _safe_rollout_plan(
    *,
    recommendation: PolicyRecommendation,
    sample_limit: int,
    db: Session,
) -> PolicyRolloutPlanResponse | None:
    if recommendation.status != "approved":
        return None
    try:
        return create_rollout_plan(
            recommendation_id=recommendation.id,
            sample_limit=sample_limit,
            record_audit=False,
            db=db,
        )
    except Exception:
        return None


def _load_latest_audit_events(
    recommendation_ids: list[int],
    session: Session,
) -> dict[int, PolicyAuditEvent]:
    if not recommendation_ids:
        return {}
    events = list(
        session.scalars(
            select(PolicyAuditEvent)
            .where(PolicyAuditEvent.recommendation_id.in_(recommendation_ids))
            .order_by(
                PolicyAuditEvent.recommendation_id.asc(),
                PolicyAuditEvent.created_at.desc(),
                PolicyAuditEvent.id.desc(),
            )
        )
    )
    latest: dict[int, PolicyAuditEvent] = {}
    for event in events:
        latest.setdefault(event.recommendation_id, event)
    return latest


def _build_portfolio_item(
    *,
    recommendation: PolicyRecommendation,
    readiness: PolicyApplyReadinessResponse,
    latest_event: PolicyAuditEvent | None,
    impact_level: str | None,
) -> PortfolioRecommendationItem:
    risk_label = _risk_label(
        status=recommendation.status,
        readiness=readiness.readiness,
        blocking_issues=readiness.blocking_issues,
        impact_level=impact_level,
    )
    return PortfolioRecommendationItem(
        id=recommendation.id,
        summary=recommendation.summary,
        condition=recommendation.condition,
        route_to=recommendation.route_to,
        priority=recommendation.priority,
        status=recommendation.status,
        review_status=recommendation.review_status,
        readiness=readiness.readiness,
        source=recommendation.source,
        latest_event_type=latest_event.event_type if latest_event else None,
        latest_event_status=latest_event.event_status if latest_event else None,
        risk_label=risk_label,
        recommended_admin_action=_recommended_action(
            status=recommendation.status,
            readiness=readiness.readiness,
        ),
    )


def _risk_label(
    *,
    status: str,
    readiness: str,
    blocking_issues: list[str],
    impact_level: str | None,
) -> str:
    if readiness == "unnecessary" or status == "rejected":
        return "none"
    if readiness == "not_ready" and blocking_issues:
        return "high"
    if readiness == "manual_review_required":
        return "medium"
    if readiness == "ready_for_future_apply" and impact_level == "low":
        return "low"
    if readiness == "ready_for_future_apply" and impact_level == "high":
        return "medium"
    if status == "pending":
        return "medium"
    return "medium"


def _recommended_action(*, status: str, readiness: str) -> str:
    if readiness == "ready_for_future_apply":
        return "consider future controlled rollout planning"
    if readiness == "not_ready":
        return "review pending recommendation" if status == "pending" else "fix blocking issues before further review"
    if readiness == "manual_review_required":
        return "manual governance review required"
    if readiness == "unnecessary" or status == "rejected":
        return "no action needed"
    return "review pending recommendation"


def _build_portfolio_counts(
    *,
    portfolio: list[PortfolioRecommendationItem],
    impact_map: dict[int, str],
) -> PortfolioCounts:
    status_counts = Counter(item.status for item in portfolio)
    readiness_counts = Counter(item.readiness for item in portfolio)
    impact_counts = Counter(impact_map.values())
    return PortfolioCounts(
        total=len(portfolio),
        pending=status_counts.get("pending", 0),
        approved=status_counts.get("approved", 0),
        rejected=status_counts.get("rejected", 0),
        ready_for_future_apply=readiness_counts.get("ready_for_future_apply", 0),
        not_ready=readiness_counts.get("not_ready", 0),
        unnecessary=readiness_counts.get("unnecessary", 0),
        manual_review_required=readiness_counts.get("manual_review_required", 0),
        high_priority_count=sum(1 for item in portfolio if item.priority <= 2),
        low_impact_count=impact_counts.get("low", 0),
        high_impact_count=impact_counts.get("high", 0),
    )


def _build_executive_summary(counts: PortfolioCounts) -> str:
    if counts.total == 0:
        return "No policy recommendations are currently present in the governance portfolio."

    if counts.ready_for_future_apply == 0:
        summary = (
            "Most recommendations remain pending or not ready, with no recommendation currently eligible "
            "for future controlled rollout."
        )
    elif counts.ready_for_future_apply <= max(1, counts.total // 4):
        summary = (
            "Most recommendations remain pending or not ready, with only a small subset eligible for future "
            "controlled rollout."
        )
    else:
        summary = (
            f"{counts.ready_for_future_apply} recommendations are currently eligible for future controlled rollout, "
            "while others remain in governance review."
        )

    if counts.unnecessary > 0:
        summary += (
            f" {counts.unnecessary} recommendation(s) appear unnecessary because the current router already covers "
            "the intended behavior or the dry-run showed no change."
        )
    if counts.high_priority_count > 0:
        summary += f" {counts.high_priority_count} high-priority recommendation(s) deserve focused governance review."
    summary += " No live policy changes have been applied."
    return summary


def _is_long_pending(created_at: datetime | None) -> bool:
    if created_at is None:
        return False
    created_at_naive = created_at.replace(tzinfo=None) if created_at.tzinfo else created_at
    return created_at_naive <= datetime.now(UTC).replace(tzinfo=None) - timedelta(days=LONG_PENDING_DAYS)
