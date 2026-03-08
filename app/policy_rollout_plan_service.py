from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.models import get_model_registry
from app.policy_audit_service import record_policy_audit_event
from app.policy_simulation_service import (
    condition_supported_for_simulation,
    simulate_recommendation_application,
)
from app.schemas import (
    EstimatedImpact,
    PolicyRolloutPlanResponse,
    PolicySimulationSummary,
    RollbackPlan,
    RolloutPhase,
)
from app.storage.entities import PolicyRecommendation

RESTORE_TARGET = "router_v1_feature_scored"


def create_rollout_plan(
    recommendation_id: int,
    sample_limit: int = 20,
    *,
    record_audit: bool = True,
    db: Session | None = None,
) -> PolicyRolloutPlanResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        recommendation = session.get(PolicyRecommendation, recommendation_id)
        if recommendation is None:
            raise KeyError(f"Policy recommendation {recommendation_id} was not found.")
        if recommendation.status != "approved":
            raise ValueError("Rollout planning is only available for approved recommendations.")

        simulation = simulate_recommendation_application(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            record_audit=record_audit,
            db=session,
        )

        total_samples = simulation.total_samples
        changed_samples = simulation.changed_samples
        unchanged_samples = simulation.unchanged_samples
        change_ratio = changed_samples / total_samples if total_samples > 0 else 0.0
        unsupported_condition = not condition_supported_for_simulation(recommendation.condition)
        target_route_exists = get_model_registry().has(recommendation.route_to)

        simulation_summary = PolicySimulationSummary(
            total_samples=total_samples,
            changed_samples=changed_samples,
            unchanged_samples=unchanged_samples,
            change_ratio=change_ratio,
            unsupported_condition=unsupported_condition,
        )
        estimated_impact = _build_estimated_impact(
            change_ratio=change_ratio,
            changed_samples=changed_samples,
            total_samples=total_samples,
        )
        rollout_strategy = _determine_rollout_strategy(
            unsupported_condition=unsupported_condition,
            target_route_exists=target_route_exists,
            changed_samples=changed_samples,
            impact_level=estimated_impact.level,
        )
        phases = _build_rollout_phases(
            strategy=rollout_strategy,
            impact_level=estimated_impact.level,
            unsupported_condition=unsupported_condition,
            target_route_exists=target_route_exists,
        )

        response = PolicyRolloutPlanResponse(
            recommendation_id=recommendation.id,
            recommendation_condition=recommendation.condition,
            recommendation_route_to=recommendation.route_to,
            simulation_summary=simulation_summary,
            estimated_impact=estimated_impact,
            rollout_strategy=rollout_strategy,
            phases=phases,
            monitoring_metrics=[
                "success_rate",
                "fallback_rate",
                "avg_latency_ms",
                "error_breakdown",
                "route_distribution_change",
            ],
            rollback_plan=RollbackPlan(
                trigger_conditions=[
                    "success_rate drops materially from the pre-rollout baseline",
                    "fallback_rate increases materially from the pre-rollout baseline",
                    "avg_latency_ms increases materially from the pre-rollout baseline",
                    "error_breakdown shows a sustained spike in recommendation-related failures",
                ],
                immediate_action=[
                    "stop the staged rollout immediately",
                    "continue using current router.py behavior only",
                    "review simulation evidence, logs, and recommendation assumptions before any retry",
                ],
                restore_target=RESTORE_TARGET,
            ),
            notes=_build_rollout_notes(
                strategy=rollout_strategy,
                unsupported_condition=unsupported_condition,
                target_route_exists=target_route_exists,
            ),
        )
        if record_audit:
            _record_audit_event_safe(
                recommendation_id=recommendation.id,
                event_type="rollout_planned",
                event_status=rollout_strategy,
                event_summary="A staged rollout plan was created for this recommendation.",
                event_details_json={
                    "rollout_strategy": rollout_strategy,
                    "impact_level": estimated_impact.level,
                    "changed_samples": simulation.changed_samples,
                    "total_samples": simulation.total_samples,
                },
            )
        return response
    finally:
        if owns_session:
            session.close()


def _build_estimated_impact(
    *,
    change_ratio: float,
    changed_samples: int,
    total_samples: int,
) -> EstimatedImpact:
    if change_ratio < 0.2:
        level = "low"
    elif change_ratio < 0.5:
        level = "medium"
    else:
        level = "high"

    return EstimatedImpact(
        level=level,
        explanation=(
            f"The dry-run changed {changed_samples} of {total_samples} sampled requests "
            f"({change_ratio:.0%} simulated route change rate)."
        ),
    )


def _determine_rollout_strategy(
    *,
    unsupported_condition: bool,
    target_route_exists: bool,
    changed_samples: int,
    impact_level: str,
) -> str:
    if unsupported_condition or not target_route_exists:
        return "manual review before rollout"
    if changed_samples == 0:
        return "no rollout needed"
    if impact_level == "low":
        return "very limited rollout"
    return "staged rollout"


def _build_rollout_phases(
    *,
    strategy: str,
    impact_level: str,
    unsupported_condition: bool,
    target_route_exists: bool,
) -> list[RolloutPhase]:
    base_success_criteria = [
        "no significant drop in success_rate compared with the pre-rollout baseline",
        "no major increase in fallback_rate compared with the pre-rollout baseline",
        "no major increase in avg_latency_ms compared with the pre-rollout baseline",
    ]
    base_rollback_triggers = [
        "fallback_rate spikes above the expected control band",
        "avg_latency_ms spikes above the expected control band",
        "success_rate drops below the agreed rollout threshold",
    ]

    if strategy == "manual review before rollout":
        notes = ["Do not execute traffic changes until the condition and target route are manually validated."]
        if unsupported_condition:
            notes.append("The recommendation condition is unsupported or ambiguous for safe dry-run simulation.")
        if not target_route_exists:
            notes.append("The recommended target route does not exist in MODEL_REGISTRY.")
        return [
            RolloutPhase(
                phase_name="Phase 1: Manual policy review",
                traffic_percentage=0,
                success_criteria=["confirm the recommendation condition is precise and auditable"],
                rollback_triggers=["review finds ambiguous condition semantics or unsafe target routing"],
                notes=notes,
            ),
            RolloutPhase(
                phase_name="Phase 2: Governance sign-off",
                traffic_percentage=0,
                success_criteria=["obtain explicit approval for any future staged rollout plan"],
                rollback_triggers=["sign-off is withheld because the recommendation remains unclear"],
                notes=["No traffic should move in this phase; this is still planning-only."],
            ),
        ]

    if strategy == "no rollout needed":
        return [
            RolloutPhase(
                phase_name="Phase 1: Confirm no-op result",
                traffic_percentage=0,
                success_criteria=["verify that the dry-run shows no simulated route change on current samples"],
                rollback_triggers=["fresh evidence contradicts the no-op simulation result"],
                notes=["Keep live routing unchanged because the recommendation does not affect sampled traffic."],
            ),
            RolloutPhase(
                phase_name="Phase 2: Monitor only",
                traffic_percentage=0,
                success_criteria=["continue normal monitoring without enabling any staged rollout"],
                rollback_triggers=["new traffic patterns suggest the recommendation may affect future requests"],
                notes=["No rollout should proceed unless new evidence shows non-zero impact."],
            ),
        ]

    if strategy == "very limited rollout":
        return [
            RolloutPhase(
                phase_name="Phase 1: 5% canary",
                traffic_percentage=5,
                success_criteria=base_success_criteria,
                rollback_triggers=base_rollback_triggers,
                notes=["Use a narrow canary because the simulated impact is low but non-zero."],
            ),
            RolloutPhase(
                phase_name="Phase 2: 10% expansion",
                traffic_percentage=10,
                success_criteria=base_success_criteria,
                rollback_triggers=base_rollback_triggers,
                notes=["Expand only if Phase 1 remains stable over the agreed observation window."],
            ),
            RolloutPhase(
                phase_name="Phase 3: 25% checkpoint",
                traffic_percentage=25,
                success_criteria=base_success_criteria,
                rollback_triggers=base_rollback_triggers,
                notes=["Pause after 25% and reassess before any broader rollout."],
            ),
        ]

    final_note = "Proceed to 100% only after each checkpoint passes its stability review."
    if impact_level == "high":
        final_note = "Because simulated impact is high, require explicit approval at every phase before continuing."
    return [
        RolloutPhase(
            phase_name="Phase 1: 10% canary",
            traffic_percentage=10,
            success_criteria=base_success_criteria,
            rollback_triggers=base_rollback_triggers,
            notes=["Start with a bounded canary to validate the simulated behavior against production telemetry."],
        ),
        RolloutPhase(
            phase_name="Phase 2: 25% expansion",
            traffic_percentage=25,
            success_criteria=base_success_criteria,
            rollback_triggers=base_rollback_triggers,
            notes=["Advance only if route distribution and error rates remain within expected bands."],
        ),
        RolloutPhase(
            phase_name="Phase 3: 50% rollout",
            traffic_percentage=50,
            success_criteria=base_success_criteria,
            rollback_triggers=base_rollback_triggers,
            notes=["Compare treatment and control cohorts before expanding further."],
        ),
        RolloutPhase(
            phase_name="Phase 4: 100% rollout",
            traffic_percentage=100,
            success_criteria=base_success_criteria,
            rollback_triggers=base_rollback_triggers,
            notes=[final_note],
        ),
    ]


def _build_rollout_notes(
    *,
    strategy: str,
    unsupported_condition: bool,
    target_route_exists: bool,
) -> list[str]:
    notes = [
        "This rollout plan is planning-only; no live routing policy was changed.",
        "No recommendation was auto-applied and no staged rollout was executed.",
        "app/router/rule_router.py still defines the active production routing behavior.",
    ]
    if strategy == "manual review before rollout":
        notes.append("The recommendation should remain in governance review until the simulation assumptions are made explicit.")
    if unsupported_condition:
        notes.append("The recommendation condition is not safely supported by the current dry-run matcher.")
    if not target_route_exists:
        notes.append("The recommended route target is not available in MODEL_REGISTRY and needs manual correction.")
    return notes


def _record_audit_event_safe(**kwargs) -> None:
    try:
        record_policy_audit_event(**kwargs)
    except Exception:
        pass
