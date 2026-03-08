import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.schemas import (
    GovernanceAuditOverviewResponse,
    PolicyAuditEventResponse,
    PolicyAuditTimelineResponse,
)
from app.storage.entities import PolicyAuditEvent, PolicyRecommendation


def record_policy_audit_event(
    *,
    recommendation_id: int,
    event_type: str,
    event_status: str | None = None,
    event_summary: str,
    event_details_json: str | dict[str, Any] | list[Any] | None = None,
    db: Session | None = None,
) -> PolicyAuditEventResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        record = PolicyAuditEvent(
            recommendation_id=recommendation_id,
            event_type=event_type,
            event_status=event_status,
            event_summary=event_summary,
            event_details_json=_serialize_event_details(event_details_json),
        )
        session.add(record)
        if owns_session:
            session.commit()
        else:
            session.flush()
        session.refresh(record)
        return _to_event_response(record)
    except Exception:
        if owns_session:
            session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def get_recommendation_timeline(
    recommendation_id: int,
    db: Session | None = None,
) -> PolicyAuditTimelineResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        recommendation = session.get(PolicyRecommendation, recommendation_id)
        if recommendation is None:
            raise KeyError(f"Policy recommendation {recommendation_id} was not found.")

        events = list(
            session.scalars(
                select(PolicyAuditEvent)
                .where(PolicyAuditEvent.recommendation_id == recommendation_id)
                .order_by(PolicyAuditEvent.created_at.asc(), PolicyAuditEvent.id.asc())
            )
        )
        return PolicyAuditTimelineResponse(
            recommendation_id=recommendation_id,
            timeline=[_to_event_response(event) for event in events],
            notes=[
                "This audit timeline is append-only and read-only from the API perspective.",
                "Timeline events describe governance activity only and do not imply live router changes.",
            ],
        )
    finally:
        if owns_session:
            session.close()


def get_recent_audit_events(
    limit: int = 50,
    db: Session | None = None,
) -> list[PolicyAuditEventResponse]:
    init_db()
    bounded_limit = max(1, min(limit, 200))
    owns_session = db is None
    session = db or SessionLocal()
    try:
        events = list(
            session.scalars(
                select(PolicyAuditEvent)
                .order_by(PolicyAuditEvent.created_at.desc(), PolicyAuditEvent.id.desc())
                .limit(bounded_limit)
            )
        )
        return [_to_event_response(event) for event in events]
    finally:
        if owns_session:
            session.close()


def get_audit_overview(
    limit: int = 50,
    db: Session | None = None,
) -> GovernanceAuditOverviewResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        total_events = session.scalar(select(func.count(PolicyAuditEvent.id))) or 0
        recent_events = get_recent_audit_events(limit=limit, db=session)
        return GovernanceAuditOverviewResponse(
            total_audit_events=int(total_events),
            recent_events=recent_events,
            notes=[
                "Audit events are append-only and generated server-side.",
                "This overview is read-only and does not modify recommendation state.",
            ],
        )
    finally:
        if owns_session:
            session.close()


def _serialize_event_details(value: str | dict[str, Any] | list[Any] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _to_event_response(record: PolicyAuditEvent) -> PolicyAuditEventResponse:
    return PolicyAuditEventResponse(
        id=record.id,
        recommendation_id=record.recommendation_id,
        event_type=record.event_type,
        event_status=record.event_status,
        event_summary=record.event_summary,
        event_details_json=record.event_details_json,
        created_at=record.created_at,
    )
