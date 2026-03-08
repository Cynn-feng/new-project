from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.models import get_model_registry
from app.schemas import (
    ActiveRoutingRule,
    ApprovedPendingRecommendation,
    AvailableModelSnapshot,
    PolicySnapshotResponse,
)
from app.storage.entities import PolicyRecommendation

ACTIVE_POLICY_VERSION = "router_v1_feature_scored"


def get_current_routing_rules() -> list[ActiveRoutingRule]:
    return [
        ActiveRoutingRule(
            rule_name="code_signal_priority",
            condition_description=(
                "When the request has strong code features, explicit code blocks, or the task is classified as coding."
            ),
            route_to="code_specialist",
            reason=(
                "The live router gives code_signal the strongest specialist boost, and coding classification adds an extra score bonus."
            ),
            source="app/router/rule_router.py",
        ),
        ActiveRoutingRule(
            rule_name="retrieval_long_context_priority",
            condition_description=(
                "When retrieval intent, tool usage needs, or long-context pressure is high, especially for RAG-classified requests."
            ),
            route_to="long_context_rag",
            reason=(
                "The live router boosts retrieval_signal, needs_tools, and needs_long_context toward the RAG route."
            ),
            source="app/router/rule_router.py",
        ),
        ActiveRoutingRule(
            rule_name="reasoning_priority",
            condition_description=(
                "When the request needs multi-step reasoning, analysis, or structured output pressure raises reasoning difficulty."
            ),
            route_to="strong_reasoning",
            reason=(
                "The live router boosts reasoning_signal and also adds extra score when needs_json is true. "
                "There is not yet a separate live structured_output_model route."
            ),
            source="app/router/rule_router.py",
        ),
        ActiveRoutingRule(
            rule_name="summarization_fast_path",
            condition_description=(
                "When the request is mainly summarization and specialist signals remain low."
            ),
            route_to="fast_general",
            reason=(
                "The live router gives fast_general a stable base score and a summarization bonus when specialization is not strongly indicated."
            ),
            source="app/router/rule_router.py",
        ),
        ActiveRoutingRule(
            rule_name="general_default",
            condition_description="When no specialist route out-scores the general baseline.",
            route_to="fast_general",
            reason="fast_general remains the default live route when code, retrieval, and reasoning signals stay below the general threshold.",
            source="app/router/rule_router.py",
        ),
    ]


def get_policy_snapshot(db: Session | None = None) -> PolicySnapshotResponse:
    init_db()
    owns_session = db is None
    session = db or SessionLocal()
    try:
        registry = get_model_registry()
        approved_rows = list(
            session.scalars(
                select(PolicyRecommendation)
                .where(PolicyRecommendation.status == "approved")
                .order_by(PolicyRecommendation.priority.asc(), PolicyRecommendation.id.desc())
            )
        )

        return PolicySnapshotResponse(
            active_policy_version=ACTIVE_POLICY_VERSION,
            active_routing_rules=get_current_routing_rules(),
            available_models=[
                AvailableModelSnapshot(
                    model_key=profile.alias,
                    model_name=profile.model,
                    capabilities=list(profile.capabilities),
                    cost_level=profile.cost_level,
                    speed_level=profile.speed_level,
                )
                for profile in registry.list_profiles()
            ],
            approved_but_not_applied=[
                ApprovedPendingRecommendation(
                    id=row.id,
                    condition=row.condition,
                    route_to=row.route_to,
                    reason=row.reason,
                    priority=row.priority,
                    status=row.status,
                    review_status=row.review_status,
                )
                for row in approved_rows
            ],
            notes=[
                "Approved recommendations are tracked in governance state.",
                "Approved recommendations are not yet active and are not auto-applied.",
                "The live routing behavior is still defined by app/router/rule_router.py.",
            ],
        )
    finally:
        if owns_session:
            session.close()
