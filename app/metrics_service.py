from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.schemas import AdminErrorBreakdown, AdminStatsResponse, AdminTopModel
from app.storage.entities import ExecutionRecord, TaskRecord


def get_admin_stats(db: Session | None = None) -> AdminStatsResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        final_executions = _final_execution_subquery()
        total_requests = session.scalar(select(func.count(TaskRecord.id))) or 0
        successful_requests = session.scalar(
            select(func.count()).select_from(final_executions).where(final_executions.c.success.is_(True))
        ) or 0
        fallback_requests = session.scalar(
            select(func.count(TaskRecord.id)).where(TaskRecord.fallback_triggered.is_(True))
        ) or 0
        avg_latency_ms = session.scalar(select(func.avg(final_executions.c.latency_ms))) or 0.0

        top_model_rows = session.execute(
            select(
                func.coalesce(ExecutionRecord.selected_model_key, ExecutionRecord.model_alias).label("selected_model_key"),
                func.coalesce(ExecutionRecord.selected_model_name, ExecutionRecord.model).label("selected_model_name"),
                func.coalesce(ExecutionRecord.actual_model_used, ExecutionRecord.model).label("actual_model_used"),
                func.count(ExecutionRecord.id).label("count"),
            )
            .group_by(
                func.coalesce(ExecutionRecord.selected_model_key, ExecutionRecord.model_alias),
                func.coalesce(ExecutionRecord.selected_model_name, ExecutionRecord.model),
                func.coalesce(ExecutionRecord.actual_model_used, ExecutionRecord.model),
            )
            .order_by(desc("count"), "selected_model_key", "actual_model_used")
            .limit(5)
        ).all()

        error_rows = session.execute(
            select(
                ExecutionRecord.error_type,
                func.count(ExecutionRecord.id).label("count"),
            )
            .where(ExecutionRecord.error_type.is_not(None))
            .group_by(ExecutionRecord.error_type)
            .order_by(desc("count"), ExecutionRecord.error_type)
        ).all()

        success_rate = (successful_requests / total_requests) if total_requests else 0.0
        fallback_rate = (fallback_requests / total_requests) if total_requests else 0.0
        return AdminStatsResponse(
            total_requests=total_requests,
            success_rate=round(success_rate, 4),
            fallback_rate=round(fallback_rate, 4),
            avg_latency_ms=round(float(avg_latency_ms), 2) if avg_latency_ms else 0.0,
            top_models=[
                AdminTopModel(
                    selected_model_key=row.selected_model_key,
                    selected_model_name=row.selected_model_name,
                    actual_model_used=row.actual_model_used,
                    count=row.count,
                )
                for row in top_model_rows
            ],
            error_breakdown=[
                AdminErrorBreakdown(
                    error_type=row.error_type,
                    count=row.count,
                )
                for row in error_rows
            ],
        )
    finally:
        if owns_session:
            session.close()


def _final_execution_subquery():
    latest_attempts = (
        select(
            ExecutionRecord.task_id.label("task_id"),
            func.max(ExecutionRecord.attempt_number).label("max_attempt_number"),
        )
        .group_by(ExecutionRecord.task_id)
        .subquery()
    )
    return (
        select(
            ExecutionRecord.task_id.label("task_id"),
            ExecutionRecord.id.label("execution_id"),
            ExecutionRecord.latency_ms.label("latency_ms"),
            ExecutionRecord.success.label("success"),
            ExecutionRecord.fallback_triggered.label("fallback_triggered"),
            ExecutionRecord.actual_model_used.label("actual_model_used"),
        )
        .join(
            latest_attempts,
            (ExecutionRecord.task_id == latest_attempts.c.task_id)
            & (ExecutionRecord.attempt_number == latest_attempts.c.max_attempt_number),
        )
        .subquery()
    )
