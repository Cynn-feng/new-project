from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.admin_ai_service import analyze_platform_with_ai
from app.agent.orchestrator import AgentOrchestrator, get_orchestrator
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.metrics_service import get_admin_stats
from app.models import ModelRegistry, get_model_registry
from app.policy_apply_gate_service import evaluate_apply_readiness
from app.policy_audit_service import get_audit_overview, get_recommendation_timeline
from app.policy_dashboard_service import get_unified_governance_dashboard
from app.policy_portfolio_service import generate_governance_portfolio_report
from app.policy_report_service import generate_governance_review_pack
from app.policy_rollout_plan_service import create_rollout_plan
from app.policy_simulation_service import simulate_recommendation_application
from app.policy_snapshot_service import get_policy_snapshot
from app.policy_workflow_service import (
    generate_and_store_recommendations,
    list_recommendations,
    update_recommendation_status,
)
from app.recommendation_review_service import review_recommendations
from app.router_inspect_service import inspect_route
from app.router_recommendation_service import recommend_router_updates
from app.schemas import (
    AIAnalysisResponse,
    AdminStatsResponse,
    ChatRequest,
    ChatResponse,
    GenerateRecommendationsResponse,
    GovernanceAuditOverviewResponse,
    GovernancePortfolioReportResponse,
    GovernanceReviewPackResponse,
    HealthResponse,
    ModelProfileResponse,
    PolicyApplyReadinessResponse,
    PolicyAuditTimelineResponse,
    PolicyDecisionResponse,
    PolicyRecommendationResponse,
    PolicyRolloutPlanResponse,
    PolicySimulationResponse,
    PolicySnapshotResponse,
    RecommendationReviewResponse,
    RouterInspectRequest,
    RouterInspectResponse,
    RouterRecommendationResponse,
    UnifiedGovernanceDashboardResponse,
)

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        app_name=settings.app_name,
        database_url=settings.database_url,
    )


@api_router.get("/models", response_model=list[ModelProfileResponse], tags=["registry"])
def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> list[ModelProfileResponse]:
    # 暴露模型注册表，便于前端或运维查看当前有哪些路由目标。
    return [
        ModelProfileResponse(
            alias=profile.alias,
            provider=profile.provider,
            model=profile.model,
            capabilities=list(profile.capabilities),
            max_context=profile.max_context,
            cost_level=profile.cost_level,
            speed_level=profile.speed_level,
            fallback_model_alias=profile.fallback_model_alias,
        )
        for profile in registry.list_profiles()
    ]


@api_router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    # API 层只负责接收请求和注入依赖，编排逻辑全部下沉到 orchestrator。
    return orchestrator.handle_chat(payload, db)


@api_router.post("/router/inspect", response_model=RouterInspectResponse, tags=["router"])
def router_inspect(payload: RouterInspectRequest) -> RouterInspectResponse:
    return RouterInspectResponse.model_validate(inspect_route(payload.message))


@api_router.get("/admin/stats", response_model=AdminStatsResponse, tags=["admin"])
def admin_stats(db: Session = Depends(get_db)) -> AdminStatsResponse:
    return get_admin_stats(db)


@api_router.get("/admin/ai/analyze", response_model=AIAnalysisResponse, tags=["admin"])
def admin_ai_analyze(
    model_name: str = Query(default="gpt-4o-mini"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> AIAnalysisResponse:
    return analyze_platform_with_ai(model_name=model_name, limit=limit, db=db)


@api_router.get("/admin/ai/recommend-router-update", response_model=RouterRecommendationResponse, tags=["admin"])
def admin_router_recommendation(
    model_name: str = Query(default="gpt-4o-mini"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> RouterRecommendationResponse:
    return recommend_router_updates(model_name=model_name, limit=limit, db=db)


@api_router.get("/admin/ai/review-recommendations", response_model=RecommendationReviewResponse, tags=["admin"])
def admin_review_recommendations(limit: int = Query(default=50, ge=1, le=200)) -> RecommendationReviewResponse:
    return review_recommendations(limit=limit)


@api_router.post(
    "/admin/policy/recommendations/generate",
    response_model=GenerateRecommendationsResponse,
    tags=["admin"],
)
def admin_generate_policy_recommendations(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> GenerateRecommendationsResponse:
    return generate_and_store_recommendations(limit=limit, db=db)


@api_router.get(
    "/admin/policy/recommendations",
    response_model=list[PolicyRecommendationResponse],
    tags=["admin"],
)
def admin_list_policy_recommendations(
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[PolicyRecommendationResponse]:
    return list_recommendations(status=status, db=db)


@api_router.post(
    "/admin/policy/recommendations/{recommendation_id}/approve",
    response_model=PolicyDecisionResponse,
    tags=["admin"],
)
def admin_approve_policy_recommendation(
    recommendation_id: int,
    db: Session = Depends(get_db),
) -> PolicyDecisionResponse:
    try:
        return update_recommendation_status(recommendation_id, "approved", db=db)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.post(
    "/admin/policy/recommendations/{recommendation_id}/reject",
    response_model=PolicyDecisionResponse,
    tags=["admin"],
)
def admin_reject_policy_recommendation(
    recommendation_id: int,
    db: Session = Depends(get_db),
) -> PolicyDecisionResponse:
    try:
        return update_recommendation_status(recommendation_id, "rejected", db=db)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/snapshot",
    response_model=PolicySnapshotResponse,
    tags=["admin"],
)
def admin_policy_snapshot(db: Session = Depends(get_db)) -> PolicySnapshotResponse:
    return get_policy_snapshot(db=db)


@api_router.get(
    "/admin/policy/recommendations/{recommendation_id}/simulate",
    response_model=PolicySimulationResponse,
    tags=["admin"],
)
def admin_simulate_policy_recommendation(
    recommendation_id: int,
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> PolicySimulationResponse:
    try:
        return simulate_recommendation_application(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/recommendations/{recommendation_id}/rollout-plan",
    response_model=PolicyRolloutPlanResponse,
    tags=["admin"],
)
def admin_policy_rollout_plan(
    recommendation_id: int,
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> PolicyRolloutPlanResponse:
    try:
        return create_rollout_plan(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/recommendations/{recommendation_id}/apply-readiness",
    response_model=PolicyApplyReadinessResponse,
    tags=["admin"],
)
def admin_policy_apply_readiness(
    recommendation_id: int,
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> PolicyApplyReadinessResponse:
    try:
        return evaluate_apply_readiness(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/dashboard",
    response_model=UnifiedGovernanceDashboardResponse,
    tags=["admin"],
)
def admin_policy_dashboard(
    limit: int = Query(default=20, ge=1, le=100),
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> UnifiedGovernanceDashboardResponse:
    return get_unified_governance_dashboard(limit=limit, sample_limit=sample_limit, db=db)


@api_router.get(
    "/admin/policy/recommendations/{recommendation_id}/timeline",
    response_model=PolicyAuditTimelineResponse,
    tags=["admin"],
)
def admin_policy_recommendation_timeline(
    recommendation_id: int,
    db: Session = Depends(get_db),
) -> PolicyAuditTimelineResponse:
    try:
        return get_recommendation_timeline(recommendation_id=recommendation_id, db=db)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/audit/overview",
    response_model=GovernanceAuditOverviewResponse,
    tags=["admin"],
)
def admin_policy_audit_overview(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> GovernanceAuditOverviewResponse:
    return get_audit_overview(limit=limit, db=db)


@api_router.get(
    "/admin/policy/recommendations/{recommendation_id}/report",
    response_model=GovernanceReviewPackResponse,
    tags=["admin"],
)
def admin_policy_recommendation_report(
    recommendation_id: int,
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> GovernanceReviewPackResponse:
    try:
        return generate_governance_review_pack(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/recommendations/{recommendation_id}/report/export",
    response_model=GovernanceReviewPackResponse,
    tags=["admin"],
)
def admin_policy_recommendation_report_export(
    recommendation_id: int,
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> GovernanceReviewPackResponse:
    try:
        return generate_governance_review_pack(
            recommendation_id=recommendation_id,
            sample_limit=sample_limit,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.get(
    "/admin/policy/portfolio-report",
    response_model=GovernancePortfolioReportResponse,
    tags=["admin"],
)
def admin_policy_portfolio_report(
    limit: int = Query(default=50, ge=1, le=100),
    sample_limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> GovernancePortfolioReportResponse:
    return generate_governance_portfolio_report(limit=limit, sample_limit=sample_limit, db=db)
