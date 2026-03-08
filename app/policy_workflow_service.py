from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.policy_audit_service import record_policy_audit_event
from app.recommendation_review_service import review_recommendations
from app.router_recommendation_service import recommend_router_updates
from app.schemas import (
    GenerateRecommendationsResponse,
    PolicyDecisionResponse,
    PolicyRecommendationResponse,
)
from app.storage.entities import PolicyRecommendation

VALID_POLICY_STATUSES = {"pending", "approved", "rejected"}


def generate_and_store_recommendations(
    limit: int = 50,
    db: Session | None = None,
) -> GenerateRecommendationsResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        recommendation_response = recommend_router_updates(limit=limit, db=session)
        review_response = review_recommendations(limit=limit)
        review_map = {
            _review_key(item.condition, item.route_to): item
            for item in review_response.reviews
        }

        created_records: list[PolicyRecommendation] = []
        for item in recommendation_response.recommended_changes:
            review_item = review_map.get(_review_key(item.condition, item.route_to))
            record = PolicyRecommendation(
                summary=recommendation_response.summary,
                condition=item.condition,
                route_to=item.route_to,
                reason=item.reason,
                priority=item.priority,
                status="pending",
                review_status=review_item.status if review_item else "needs_manual_review",
                review_comment=(
                    review_item.comment
                    if review_item
                    else "No matching review result was found for this recommendation."
                ),
                source="ai",
            )
            session.add(record)
            created_records.append(record)

        session.commit()
        for record in created_records:
            session.refresh(record)
            _record_audit_event_safe(
                recommendation_id=record.id,
                event_type="generated",
                event_status=record.status,
                event_summary="Policy recommendation was generated and stored.",
                event_details_json={
                    "condition": record.condition,
                    "route_to": record.route_to,
                    "priority": record.priority,
                    "source": record.source,
                },
            )
            _record_audit_event_safe(
                recommendation_id=record.id,
                event_type="reviewed",
                event_status=record.review_status,
                event_summary="Policy recommendation review result was merged into stored governance state.",
                event_details_json={
                    "review_status": record.review_status,
                    "review_comment": record.review_comment,
                },
            )

        return GenerateRecommendationsResponse(
            summary=(
                f"Stored {len(created_records)} pending policy recommendation(s). "
                "This workflow is governance-only and does not auto-apply router policy changes."
            ),
            created_count=len(created_records),
            recommendations=[_to_policy_response(record) for record in created_records],
        )
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def list_recommendations(
    status: str | None = None,
    db: Session | None = None,
) -> list[PolicyRecommendationResponse]:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        statement = select(PolicyRecommendation)
        if status:
            statement = statement.where(PolicyRecommendation.status == status)
        statement = statement.order_by(
            PolicyRecommendation.created_at.desc(),
            PolicyRecommendation.priority.asc(),
            PolicyRecommendation.id.desc(),
        )
        records = list(session.scalars(statement))
        return [_to_policy_response(record) for record in records]
    finally:
        if owns_session:
            session.close()


def update_recommendation_status(
    recommendation_id: int,
    new_status: str,
    db: Session | None = None,
) -> PolicyDecisionResponse:
    if new_status not in VALID_POLICY_STATUSES:
        raise ValueError(f"Unsupported policy status: {new_status}")

    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        record = session.get(PolicyRecommendation, recommendation_id)
        if record is None:
            raise KeyError(f"Policy recommendation {recommendation_id} was not found.")

        record.status = new_status
        session.add(record)
        session.commit()
        session.refresh(record)
        _record_audit_event_safe(
            recommendation_id=record.id,
            event_type=record.status,
            event_status=record.status,
            event_summary=f"Policy recommendation status changed to {record.status}.",
            event_details_json={
                "recommendation_id": record.id,
                "route_to": record.route_to,
                "review_status": record.review_status,
            },
        )

        return PolicyDecisionResponse(
            recommendation_id=record.id,
            status=record.status,
            message=(
                f"Recommendation {record.id} marked as {record.status}. "
                "No router policy was auto-applied."
            ),
        )
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def _review_key(condition: str, route_to: str) -> tuple[str, str]:
    return (condition.strip().casefold(), route_to.strip().casefold())


def _to_policy_response(record: PolicyRecommendation) -> PolicyRecommendationResponse:
    return PolicyRecommendationResponse(
        id=record.id,
        summary=record.summary,
        condition=record.condition,
        route_to=record.route_to,
        reason=record.reason,
        priority=record.priority,
        status=record.status,
        review_status=record.review_status,
        review_comment=record.review_comment,
        source=record.source,
    )


def _record_audit_event_safe(**kwargs) -> None:
    try:
        record_policy_audit_event(**kwargs)
    except Exception:
        pass
