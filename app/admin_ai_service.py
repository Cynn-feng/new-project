import json
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.gateway import call_model
from app.metrics_service import get_admin_stats
from app.schemas import AIAnalysisResponse, AdminStatsResponse
from app.storage.entities import ExecutionRecord, TaskRecord


def get_recent_logs(limit: int = 50, db: Session | None = None) -> list[dict[str, Any]]:
    init_db()
    bounded_limit = max(1, min(limit, 200))
    owns_session = db is None
    session = db or SessionLocal()
    try:
        rows = session.execute(
            select(ExecutionRecord, TaskRecord.task_type, TaskRecord.user_message)
            .join(TaskRecord, TaskRecord.id == ExecutionRecord.task_id)
            .order_by(desc(ExecutionRecord.created_at), desc(ExecutionRecord.id))
            .limit(bounded_limit)
        ).all()
        return [
            {
                "execution_id": execution.id,
                "task_id": execution.task_id,
                "attempt_number": execution.attempt_number,
                "task_type": task_type,
                "selected_model_key": execution.selected_model_key or execution.model_alias,
                "selected_model_name": execution.selected_model_name or execution.model,
                "actual_model_used": execution.actual_model_used or execution.model,
                "fallback_triggered": execution.fallback_triggered,
                "success": execution.success,
                "error_type": execution.error_type,
                "latency_ms": execution.latency_ms,
                "created_at": execution.created_at.isoformat() if execution.created_at else None,
                "message_preview": user_message[:160],
            }
            for execution, task_type, user_message in rows
        ]
    finally:
        if owns_session:
            session.close()


def build_analysis_prompt(stats: AdminStatsResponse, recent_logs: list[dict[str, Any]]) -> str:
    return (
        "Analyze the following AI-managed platform telemetry snapshot. "
        "You must stay read-only: do not claim that any routing policy was changed or should be auto-applied. "
        "Respond as strict JSON with keys summary, observations, anomalies, recommendations. "
        "summary must be a short string. The other keys must be arrays of short strings.\n\n"
        f"Admin stats:\n{json.dumps(stats.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Recent execution logs:\n{json.dumps(recent_logs, ensure_ascii=False, indent=2)}"
    )


def analyze_platform_with_ai(
    model_name: str = "gpt-4o-mini",
    limit: int = 50,
    db: Session | None = None,
) -> AIAnalysisResponse:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        stats = get_admin_stats(session)
        recent_logs = get_recent_logs(limit=limit, db=session)
        prompt = build_analysis_prompt(stats, recent_logs)
        result = call_model(
            prompt,
            model_name=model_name,
            system_prompt=(
                "You are an operations analyst for an AI routing platform. "
                "Analyze traffic, latency, errors, and fallback behavior. "
                "Return JSON only. Stay read-only and do not modify or claim to modify routing policy."
            ),
            max_tokens=1400,
            temperature=0.1,
            metadata={
                "needs_json": True,
                "admin_analysis": True,
                "analysis_mode": "platform_admin",
            },
        )
        if not result.success:
            return _safe_fallback_analysis(
                stats=stats,
                recent_logs=recent_logs,
                reason=result.error_type or "generation_error",
            )

        parsed = _parse_analysis_response(result.content)
        if parsed is None:
            return _safe_fallback_analysis(
                stats=stats,
                recent_logs=recent_logs,
                reason="invalid_json",
            )
        return parsed
    finally:
        if owns_session:
            session.close()


def _parse_analysis_response(content: str) -> AIAnalysisResponse | None:
    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    return AIAnalysisResponse(
        summary=_as_text(payload.get("summary"), default="AI analysis completed."),
        observations=_as_text_list(payload.get("observations")),
        anomalies=_as_text_list(payload.get("anomalies")),
        recommendations=_as_text_list(payload.get("recommendations")),
    )


def _safe_fallback_analysis(
    *,
    stats: AdminStatsResponse,
    recent_logs: list[dict[str, Any]],
    reason: str,
) -> AIAnalysisResponse:
    top_model = stats.top_models[0].actual_model_used if stats.top_models else "unknown"
    observations = [
        f"Observed {stats.total_requests} total requests with a {stats.success_rate:.2%} success rate.",
        f"Average latency is {stats.avg_latency_ms:.2f} ms and the current top model is {top_model}.",
    ]
    anomalies = []
    if stats.fallback_rate > 0.2:
        anomalies.append(f"Fallback usage is elevated at {stats.fallback_rate:.2%}.")
    if stats.error_breakdown:
        top_error = stats.error_breakdown[0]
        anomalies.append(f"Most common logged error type is {top_error.error_type} ({top_error.count} events).")
    anomalies.append(f"AI analysis returned a safe fallback because of {reason}.")

    recommendations = [
        "Review the recent error breakdown before adjusting model routing thresholds.",
        "Inspect recent fallback-triggered executions to confirm whether provider health or prompt quality caused them.",
        "Keep this admin AI endpoint advisory only and apply routing changes through normal configuration review.",
    ]
    if recent_logs:
        recommendations.append("Sample additional recent executions if anomaly investigation needs more context.")

    return AIAnalysisResponse(
        summary=(
            "Safe fallback analysis generated from aggregated stats and recent execution logs because the AI response "
            "was unavailable or invalid."
        ),
        observations=observations,
        anomalies=anomalies,
        recommendations=recommendations,
    )


def _as_text(value: Any, *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return items
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []
