import re

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.models import get_model_registry
from app.policy_audit_service import record_policy_audit_event
from app.router_inspect_service import inspect_route
from app.schemas import PolicySimulationResponse, PolicySimulationSampleResult
from app.storage.entities import ExecutionRecord, PolicyRecommendation, TaskRecord

SUPPORTED_BOOLEAN_FIELDS = {
    "contains_code",
    "high_reasoning",
    "needs_json",
    "needs_tools",
}


def simulate_recommendation_application(
    recommendation_id: int,
    sample_limit: int = 20,
    *,
    record_audit: bool = True,
    db: Session | None = None,
) -> PolicySimulationResponse:
    init_db()
    bounded_limit = max(1, min(sample_limit, 100))
    owns_session = db is None
    session = db or SessionLocal()
    try:
        recommendation = session.get(PolicyRecommendation, recommendation_id)
        if recommendation is None:
            raise KeyError(f"Policy recommendation {recommendation_id} was not found.")
        if recommendation.status != "approved":
            raise ValueError("Only approved recommendations can be simulated.")

        sample_messages = _get_recent_sample_messages(session, limit=bounded_limit)
        sample_results: list[PolicySimulationSampleResult] = []

        for message in sample_messages:
            inspection = inspect_route(message)
            task_type = _stringify_task_type(inspection["task_type"])
            current_model = str(inspection["selected_model_key"])
            features = inspection["features"]
            matched, unsupported = recommendation_matches(
                features=features,
                task_type=task_type,
                condition=recommendation.condition,
            )

            simulated_model = recommendation.route_to if matched else current_model
            changed = simulated_model != current_model
            explanation = _build_sample_explanation(
                unsupported=unsupported,
                matched=matched,
                changed=changed,
                current_model=current_model,
                simulated_model=simulated_model,
            )

            sample_results.append(
                PolicySimulationSampleResult(
                    input_message=message,
                    current_task_type=task_type,
                    current_selected_model_key=current_model,
                    simulated_selected_model_key=simulated_model,
                    changed=changed,
                    explanation=explanation,
                )
            )

        changed_samples = sum(1 for item in sample_results if item.changed)
        total_samples = len(sample_results)
        registry = get_model_registry()
        notes = [
            "This was a simulation only; no live routing policy was changed.",
            "Approved recommendations are not auto-applied by this dry-run workflow.",
            "app/router/rule_router.py still controls real production routing.",
        ]
        if not sample_results:
            notes.append("No recent execution samples were available for dry-run analysis.")
        if not registry.has(recommendation.route_to):
            notes.append(
                f"Recommended route '{recommendation.route_to}' is not present in MODEL_REGISTRY; "
                "the dry-run reports a hypothetical override only."
            )

        response = PolicySimulationResponse(
            recommendation_id=recommendation.id,
            recommendation_condition=recommendation.condition,
            recommendation_route_to=recommendation.route_to,
            total_samples=total_samples,
            changed_samples=changed_samples,
            unchanged_samples=total_samples - changed_samples,
            sample_results=sample_results,
            notes=notes,
        )
        if record_audit:
            _record_audit_event_safe(
                recommendation_id=recommendation.id,
                event_type="simulated",
                event_status="completed",
                event_summary="Dry-run policy simulation was completed.",
                event_details_json={
                    "sample_limit": bounded_limit,
                    "total_samples": response.total_samples,
                    "changed_samples": response.changed_samples,
                    "unchanged_samples": response.unchanged_samples,
                },
            )
        return response
    finally:
        if owns_session:
            session.close()


def recommendation_matches(
    *,
    features: dict,
    task_type: str,
    condition: str,
) -> tuple[bool, bool]:
    supported_clauses = _parse_supported_clauses(condition)
    if supported_clauses is None:
        return False, True

    matched_all = True
    for clause in supported_clauses:
        supported, matched = _match_clause(features=features, task_type=task_type, clause=clause)
        if not supported:
            return False, True
        if not matched:
            matched_all = False
    return matched_all, False


def condition_supported_for_simulation(condition: str) -> bool:
    return _parse_supported_clauses(condition) is not None


def _get_recent_sample_messages(session: Session, *, limit: int) -> list[str]:
    rows = session.execute(
        select(TaskRecord.user_message)
        .join(ExecutionRecord, ExecutionRecord.task_id == TaskRecord.id)
        .order_by(desc(ExecutionRecord.created_at), desc(ExecutionRecord.id))
        .limit(limit)
    ).all()
    return [message for (message,) in rows if isinstance(message, str) and message.strip()]


def _match_clause(*, features: dict, task_type: str, clause: str) -> tuple[bool, bool]:
    bool_match = re.fullmatch(r"(contains_code|high_reasoning|needs_json|needs_tools)\s*==\s*(true|false)", clause)
    if bool_match:
        feature_name, expected = bool_match.groups()
        if feature_name not in SUPPORTED_BOOLEAN_FIELDS:
            return False, False
        actual = bool(features.get(feature_name, False))
        return True, actual is (expected == "true")

    task_match = re.fullmatch(r"task_type\s*==\s*'?([a-z_]+)'?", clause)
    if task_match:
        expected_task_type = task_match.group(1)
        return True, task_type == expected_task_type

    return False, False


def _normalize_condition(condition: str) -> str:
    normalized = condition.strip().casefold().replace('"', "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _parse_supported_clauses(condition: str) -> list[str] | None:
    normalized = _normalize_condition(condition)
    if not normalized or " or " in normalized or "||" in normalized:
        return None
    clauses = [clause.strip() for clause in re.split(r"\band\b|&&", normalized) if clause.strip()]
    if not clauses:
        return None
    return clauses


def _stringify_task_type(task_type: object) -> str:
    if hasattr(task_type, "value"):
        return str(task_type.value)
    return str(task_type)


def _build_sample_explanation(
    *,
    unsupported: bool,
    matched: bool,
    changed: bool,
    current_model: str,
    simulated_model: str,
) -> str:
    if unsupported:
        return "Condition format unsupported for simulation; no change applied."
    if matched and changed:
        return f"Recommendation condition matched; simulated route changed to {simulated_model}."
    if matched:
        return (
            f"Recommendation condition matched, but the simulated route stayed on {current_model} "
            "because it already matched the current router outcome."
        )
    return "Recommendation condition did not match; no routing change."


def _record_audit_event_safe(**kwargs) -> None:
    try:
        record_policy_audit_event(**kwargs)
    except Exception:
        pass
