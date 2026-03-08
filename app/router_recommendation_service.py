import json
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.admin_ai_service import analyze_platform_with_ai, get_recent_logs
from app.core.database import SessionLocal
from app.gateway import call_model
from app.metrics_service import get_admin_stats
from app.schemas import (
    AIAnalysisResponse,
    AdminStatsResponse,
    RouterRecommendationItem,
    RouterRecommendationResponse,
)


def get_recent_logs_for_recommendation(
    limit: int = 50,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    return get_recent_logs(limit=limit, db=db)


def build_router_recommendation_prompt(
    stats: AdminStatsResponse,
    analysis: AIAnalysisResponse,
    recent_logs: list[dict[str, Any]],
) -> str:
    allowed_routes = [
        "fast_general",
        "strong_reasoning",
        "code_specialist",
        "structured_output_model",
    ]
    return (
        "Review the platform telemetry and provide read-only router policy recommendations. "
        "You may recommend changes, but you must not claim that any file was edited, any policy was applied, "
        "or any router behavior was automatically changed. "
        "Respond as strict JSON with keys summary, recommended_changes, notes. "
        "summary must be a short string. recommended_changes must be an array of objects with keys "
        "priority, condition, route_to, reason. route_to must be one of "
        f"{allowed_routes}. notes must be an array of short strings.\n\n"
        f"Admin stats:\n{json.dumps(stats.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"AI analysis:\n{json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Recent execution logs:\n{json.dumps(recent_logs, ensure_ascii=False, indent=2)}"
    )


def recommend_router_updates(
    model_name: str = "gpt-4o-mini",
    limit: int = 50,
    db: Session | None = None,
) -> RouterRecommendationResponse:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        stats = get_admin_stats(session)
        analysis = analyze_platform_with_ai(model_name=model_name, limit=limit, db=session)
        recent_logs = get_recent_logs_for_recommendation(limit=limit, db=session)
        prompt = build_router_recommendation_prompt(stats, analysis, recent_logs)
        result = call_model(
            prompt,
            model_name=model_name,
            system_prompt=(
                "You are a senior routing policy advisor for an AI platform. "
                "Recommend router policy changes in a read-only manner. "
                "Do not modify code, do not auto-apply policies, and do not claim any changes were executed. "
                "Return JSON only."
            ),
            max_tokens=1600,
            temperature=0.1,
            metadata={
                "needs_json": True,
                "router_recommendation": True,
                "analysis_mode": "router_recommendation",
            },
        )
        if not result.success:
            return _safe_fallback_recommendation(
                stats=stats,
                analysis=analysis,
                reason=result.error_type or "generation_error",
            )

        parsed = _parse_router_recommendation_response(result.content)
        if parsed is None:
            return _safe_fallback_recommendation(
                stats=stats,
                analysis=analysis,
                reason="invalid_json",
            )
        return parsed
    finally:
        if owns_session:
            session.close()


def _parse_router_recommendation_response(content: str) -> RouterRecommendationResponse | None:
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

    try:
        items = [
            RouterRecommendationItem.model_validate(item)
            for item in _as_list(payload.get("recommended_changes"))
        ]
        return RouterRecommendationResponse(
            summary=_as_text(payload.get("summary"), default="Router recommendation completed."),
            recommended_changes=items,
            notes=_as_text_list(payload.get("notes")),
        )
    except ValidationError:
        return None


def _safe_fallback_recommendation(
    *,
    stats: AdminStatsResponse,
    analysis: AIAnalysisResponse,
    reason: str,
) -> RouterRecommendationResponse:
    notes = [
        f"Read-only fallback response generated because recommendation parsing failed due to {reason}.",
        f"Current success rate is {stats.success_rate:.2%} and fallback rate is {stats.fallback_rate:.2%}.",
        "No router file was modified and no policy was auto-applied.",
    ]
    if analysis.anomalies:
        notes.append(f"Most recent analysis anomaly: {analysis.anomalies[0]}")

    return RouterRecommendationResponse(
        summary=(
            "Safe fallback router recommendation generated from existing admin telemetry. "
            "No recommended changes were auto-applied."
        ),
        recommended_changes=[],
        notes=notes,
    )


def _as_text(value: Any, *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return items
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []
