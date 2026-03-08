from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.models import get_model_registry
from app.policy_audit_service import record_policy_audit_event
from app.policy_rollout_plan_service import create_rollout_plan
from app.policy_simulation_service import (
    condition_supported_for_simulation,
    simulate_recommendation_application,
)
from app.schemas import ApplyGuardrailCheck, PolicyApplyReadinessResponse
from app.storage.entities import PolicyRecommendation


def evaluate_apply_readiness(
    recommendation_id: int,
    sample_limit: int = 20,
    *,
    record_audit: bool = True,
    db: Session | None = None,
) -> PolicyApplyReadinessResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        recommendation = session.get(PolicyRecommendation, recommendation_id)
        if recommendation is None:
            raise KeyError(f"Policy recommendation {recommendation_id} was not found.")

        checks: list[ApplyGuardrailCheck] = []
        warnings: list[str] = []

        registry = get_model_registry()
        route_exists = registry.has(recommendation.route_to)
        condition_supported = condition_supported_for_simulation(recommendation.condition)

        checks.append(
            ApplyGuardrailCheck(
                check_name="recommendation_exists",
                passed=True,
                severity="info",
                message=f"Recommendation {recommendation.id} exists and can be evaluated.",
            )
        )
        checks.append(
            ApplyGuardrailCheck(
                check_name="status_is_approved",
                passed=recommendation.status == "approved",
                severity="blocking",
                message=(
                    "Recommendation status is approved."
                    if recommendation.status == "approved"
                    else f"Recommendation status is {recommendation.status}; only approved recommendations are eligible."
                ),
            )
        )
        checks.append(
            ApplyGuardrailCheck(
                check_name="review_status_supported",
                passed=recommendation.review_status != "unsupported_route",
                severity="blocking",
                message=(
                    "Review status does not indicate an unsupported route."
                    if recommendation.review_status != "unsupported_route"
                    else "Review status is unsupported_route, so this recommendation is not eligible for application."
                ),
            )
        )
        checks.append(
            ApplyGuardrailCheck(
                check_name="simulation_condition_supported",
                passed=condition_supported,
                severity="blocking",
                message=(
                    "Recommendation condition is supported by the dry-run matcher."
                    if condition_supported
                    else "Recommendation condition is unsupported or ambiguous for safe dry-run evaluation."
                ),
            )
        )
        checks.append(
            ApplyGuardrailCheck(
                check_name="route_exists_in_registry",
                passed=route_exists,
                severity="blocking",
                message=(
                    f"Route '{recommendation.route_to}' exists in MODEL_REGISTRY."
                    if route_exists
                    else f"Route '{recommendation.route_to}' does not exist in MODEL_REGISTRY."
                ),
            )
        )

        simulation = None
        simulation_error: str | None = None
        try:
            simulation = simulate_recommendation_application(
                recommendation_id=recommendation_id,
                sample_limit=sample_limit,
                record_audit=record_audit,
                db=session,
            )
            checks.append(
                ApplyGuardrailCheck(
                    check_name="dry_run_simulation_completed",
                    passed=True,
                    severity="info",
                    message=(
                        f"Dry-run simulation completed across {simulation.total_samples} sample(s) "
                        f"with {simulation.changed_samples} simulated route changes."
                    ),
                )
            )
        except ValueError as exc:
            simulation_error = str(exc)
            checks.append(
                ApplyGuardrailCheck(
                    check_name="dry_run_simulation_completed",
                    passed=False,
                    severity="blocking",
                    message=f"Dry-run simulation could not run: {simulation_error}",
                )
            )

        rollout_plan = None
        rollout_error: str | None = None
        try:
            rollout_plan = create_rollout_plan(
                recommendation_id=recommendation_id,
                sample_limit=sample_limit,
                record_audit=record_audit,
                db=session,
            )
            checks.append(
                ApplyGuardrailCheck(
                    check_name="rollout_plan_available",
                    passed=True,
                    severity="info",
                    message=f"Rollout plan is available with strategy '{rollout_plan.rollout_strategy}'.",
                )
            )
        except ValueError as exc:
            rollout_error = str(exc)
            checks.append(
                ApplyGuardrailCheck(
                    check_name="rollout_plan_available",
                    passed=False,
                    severity="blocking",
                    message=f"Rollout plan could not be created: {rollout_error}",
                )
            )

        already_covered = recommendation.review_status == "already_covered"
        zero_change = simulation is not None and simulation.changed_samples == 0
        checks.append(
            ApplyGuardrailCheck(
                check_name="not_already_effectively_covered",
                passed=not already_covered,
                severity="info",
                message=(
                    "Recommendation is not marked as already covered by the current router."
                    if not already_covered
                    else "Recommendation appears already covered by the current router and may be unnecessary."
                ),
            )
        )
        checks.append(
            ApplyGuardrailCheck(
                check_name="dry_run_shows_non_zero_change",
                passed=not zero_change,
                severity="info",
                message=(
                    "Dry-run indicates at least one sample would change under this recommendation."
                    if not zero_change
                    else "Dry-run shows zero changed samples, so a rollout appears unnecessary."
                ),
            )
        )

        extreme_change = False
        if simulation is not None and simulation.total_samples > 0:
            change_ratio = simulation.changed_samples / simulation.total_samples
            extreme_change = change_ratio > 0.8
            checks.append(
                ApplyGuardrailCheck(
                    check_name="change_ratio_not_extreme",
                    passed=not extreme_change,
                    severity="blocking",
                    message=(
                        f"Dry-run change ratio is {change_ratio:.0%}, which remains within the conservative guardrail."
                        if not extreme_change
                        else f"Dry-run change ratio is {change_ratio:.0%}, which is too extreme for a future apply gate."
                    ),
                )
            )

        rollout_manual_review = rollout_plan is not None and rollout_plan.rollout_strategy == "manual review before rollout"
        checks.append(
            ApplyGuardrailCheck(
                check_name="rollout_strategy_safe_for_future_apply",
                passed=not rollout_manual_review,
                severity="warning",
                message=(
                    "Rollout strategy does not require manual review."
                    if not rollout_manual_review
                    else "Rollout plan requires manual review before any future rollout can be considered."
                ),
            )
        )

        blocking_issues = [check.message for check in checks if check.severity == "blocking" and not check.passed]
        warnings.extend(check.message for check in checks if check.severity == "warning" and not check.passed)

        readiness = _determine_readiness(
            blocking_issues=blocking_issues,
            already_covered=already_covered,
            zero_change=zero_change,
            rollout_manual_review=rollout_manual_review,
        )

        if simulation is None and simulation_error:
            warnings.append(f"Simulation details unavailable: {simulation_error}")
        if rollout_plan is None and rollout_error:
            warnings.append(f"Rollout planning details unavailable: {rollout_error}")

        next_step = _determine_next_step(readiness)

        response = PolicyApplyReadinessResponse(
            recommendation_id=recommendation.id,
            recommendation_condition=recommendation.condition,
            recommendation_route_to=recommendation.route_to,
            current_status=recommendation.status,
            readiness=readiness,
            guardrail_checks=checks,
            blocking_issues=blocking_issues,
            non_blocking_warnings=warnings,
            next_step=next_step,
            notes=[
                "This apply readiness gate is read-only; it does not apply any recommendation.",
                "No rollout was executed and no recommendation status was changed.",
                "app/router/rule_router.py still defines the live routing behavior.",
            ],
        )
        if record_audit:
            _record_audit_event_safe(
                recommendation_id=recommendation.id,
                event_type="apply_readiness_evaluated",
                event_status=response.readiness,
                event_summary="Apply readiness was evaluated for this recommendation.",
                event_details_json={
                    "readiness": response.readiness,
                    "blocking_issues": response.blocking_issues,
                    "warning_count": len(response.non_blocking_warnings),
                },
            )
        return response
    finally:
        if owns_session:
            session.close()


def _determine_readiness(
    *,
    blocking_issues: list[str],
    already_covered: bool,
    zero_change: bool,
    rollout_manual_review: bool,
) -> str:
    if blocking_issues:
        return "not_ready"
    if already_covered or zero_change:
        return "unnecessary"
    if rollout_manual_review:
        return "manual_review_required"
    return "ready_for_future_apply"


def _determine_next_step(readiness: str) -> str:
    if readiness == "not_ready":
        return "Do not apply; fix blocking issues first."
    if readiness == "unnecessary":
        return "No rollout needed; recommendation appears unnecessary."
    if readiness == "manual_review_required":
        return "Manual governance review required before any rollout."
    return "Eligible for a future controlled rollout, but not applied yet."


def _record_audit_event_safe(**kwargs) -> None:
    try:
        record_policy_audit_event(**kwargs)
    except Exception:
        pass
