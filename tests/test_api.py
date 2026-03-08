import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agent.orchestrator import get_orchestrator
from app.core.config import _load_env_file, get_settings
from app.core.database import SessionLocal, init_db
from app.gateway import ModelCallResult, call_model
from app.main import app
from app.models import LiteLLMGateway, get_model_registry
from app.schemas import (
    AIAnalysisResponse,
    ApplyGuardrailCheck,
    EstimatedImpact,
    PolicyApplyReadinessResponse,
    PolicySimulationResponse,
    PolicySimulationSampleResult,
    PolicyRolloutPlanResponse,
    PolicySimulationSummary,
    RollbackPlan,
    RolloutPhase,
    LLMResponse,
    RecommendationReviewItem,
    RecommendationReviewResponse,
    RouterRecommendationItem,
    RouterRecommendationResponse,
)
from app.storage.entities import ExecutionRecord, PolicyAuditEvent, PolicyRecommendation, TaskRecord


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def _generate_policy_recommendation(self, token: str) -> dict:
        with (
            patch(
                "app.policy_workflow_service.recommend_router_updates",
                return_value=RouterRecommendationResponse(
                    summary=f"Generated policy recommendation {token}",
                    recommended_changes=[
                        RouterRecommendationItem(
                            priority=1,
                            condition=f"If the request contains code snippets or traceback details {token}.",
                            route_to="code_specialist",
                            reason="Code-heavy requests should use the coding route.",
                        )
                    ],
                    notes=["Read-only recommendation generation."],
                ),
            ),
            patch(
                "app.policy_workflow_service.review_recommendations",
                return_value=RecommendationReviewResponse(
                    summary=f"Reviewed policy recommendation {token}",
                    reviews=[
                        RecommendationReviewItem(
                            condition=f"If the request contains code snippets or traceback details {token}.",
                            route_to="code_specialist",
                            status="already_covered",
                            comment="Current routing already covers this code-focused recommendation.",
                        )
                    ],
                    notes=["Read-only review."],
                ),
            ),
        ):
            response = self.client.post("/api/v1/admin/policy/recommendations/generate")

        self.assertEqual(response.status_code, 200)
        return response.json()

    def _insert_execution_sample(
        self,
        *,
        message: str,
        task_type: str = "general_qa",
        model_alias: str = "fast_general",
    ) -> int:
        init_db()
        profile = get_model_registry().get(model_alias)
        db = SessionLocal()
        try:
            task = TaskRecord(
                user_message=message,
                task_type=task_type,
                classification_reason="test fixture",
                classification_confidence=0.9,
                features_json={"fixture": True},
                selected_model_alias=model_alias,
                final_model_alias=model_alias,
                selected_provider=profile.provider,
                final_provider=profile.provider,
                routing_reason="test fixture",
                fallback_triggered=False,
                request_metadata={"fixture": True},
            )
            db.add(task)
            db.flush()

            execution = ExecutionRecord(
                task_id=task.id,
                attempt_number=1,
                selected_model_key=model_alias,
                selected_model_name=profile.model,
                actual_model_used=profile.model,
                provider=profile.provider,
                model_alias=model_alias,
                model=profile.model,
                latency_ms=5,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                estimated_cost=0.0,
                success=True,
                error_type=None,
                error_message=None,
                finish_reason="stop",
                fallback_triggered=False,
                used_fallback=False,
                raw_response_json={"fixture": True},
            )
            db.add(execution)
            db.commit()
            db.refresh(execution)
            return execution.id
        finally:
            db.close()

    def _insert_policy_recommendation(
        self,
        *,
        condition: str,
        route_to: str,
        status: str = "approved",
        priority: int = 1,
        review_status: str = "compatible",
    ) -> int:
        init_db()
        db = SessionLocal()
        try:
            recommendation = PolicyRecommendation(
                summary=f"Simulation fixture {uuid4().hex}",
                condition=condition,
                route_to=route_to,
                reason="Simulation fixture recommendation.",
                priority=priority,
                status=status,
                review_status=review_status,
                review_comment="Inserted by test fixture.",
                source="ai",
            )
            db.add(recommendation)
            db.commit()
            db.refresh(recommendation)
            return recommendation.id
        finally:
            db.close()

    def _build_simulation_response(
        self,
        *,
        recommendation_id: int,
        condition: str,
        route_to: str,
        total_samples: int,
        changed_samples: int,
        unchanged_samples: int,
        sample_results: list[PolicySimulationSampleResult] | None = None,
    ) -> PolicySimulationResponse:
        return PolicySimulationResponse(
            recommendation_id=recommendation_id,
            recommendation_condition=condition,
            recommendation_route_to=route_to,
            total_samples=total_samples,
            changed_samples=changed_samples,
            unchanged_samples=unchanged_samples,
            sample_results=sample_results or [],
            notes=["Dry-run fixture response."],
        )

    def _build_rollout_plan_response(
        self,
        *,
        recommendation_id: int,
        condition: str,
        route_to: str,
        total_samples: int,
        changed_samples: int,
        unchanged_samples: int,
        change_ratio: float,
        rollout_strategy: str,
        impact_level: str,
    ) -> PolicyRolloutPlanResponse:
        return PolicyRolloutPlanResponse(
            recommendation_id=recommendation_id,
            recommendation_condition=condition,
            recommendation_route_to=route_to,
            simulation_summary=PolicySimulationSummary(
                total_samples=total_samples,
                changed_samples=changed_samples,
                unchanged_samples=unchanged_samples,
                change_ratio=change_ratio,
                unsupported_condition=False,
            ),
            estimated_impact=EstimatedImpact(
                level=impact_level,
                explanation="Fixture rollout impact.",
            ),
            rollout_strategy=rollout_strategy,
            phases=[
                RolloutPhase(
                    phase_name="Phase 1",
                    traffic_percentage=10 if rollout_strategy != "no rollout needed" else 0,
                    success_criteria=["keep success rate stable"],
                    rollback_triggers=["fallback spike"],
                    notes=["Fixture phase."],
                )
            ],
            monitoring_metrics=["success_rate", "fallback_rate"],
            rollback_plan=RollbackPlan(
                trigger_conditions=["success_rate drops materially"],
                immediate_action=["stop rollout"],
                restore_target="router_v1_feature_scored",
            ),
            notes=["Fixture rollout plan."],
        )

    def _build_apply_readiness_response(
        self,
        *,
        recommendation_id: int,
        condition: str,
        route_to: str,
        current_status: str,
        readiness: str,
        has_simulation: bool,
        has_rollout_plan: bool,
        blocking_issues: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> PolicyApplyReadinessResponse:
        guardrail_checks = [
            ApplyGuardrailCheck(
                check_name="dry_run_simulation_completed",
                passed=has_simulation,
                severity="info" if has_simulation else "blocking",
                message="Simulation fixture available." if has_simulation else "Simulation fixture unavailable.",
            ),
            ApplyGuardrailCheck(
                check_name="rollout_plan_available",
                passed=has_rollout_plan,
                severity="info" if has_rollout_plan else "blocking",
                message="Rollout plan fixture available." if has_rollout_plan else "Rollout plan fixture unavailable.",
            ),
        ]
        return PolicyApplyReadinessResponse(
            recommendation_id=recommendation_id,
            recommendation_condition=condition,
            recommendation_route_to=route_to,
            current_status=current_status,
            readiness=readiness,
            guardrail_checks=guardrail_checks,
            blocking_issues=blocking_issues or [],
            non_blocking_warnings=warnings or [],
            next_step="Fixture next step.",
            notes=["Fixture readiness response."],
        )

    def _build_dashboard_readiness_side_effect(self, readiness_map: dict[int, PolicyApplyReadinessResponse]):
        def _side_effect(recommendation_id: int, sample_limit: int, db, record_audit: bool = True):
            if recommendation_id in readiness_map:
                return readiness_map[recommendation_id]

            record = db.get(PolicyRecommendation, recommendation_id)
            self.assertIsNotNone(record)
            return self._build_apply_readiness_response(
                recommendation_id=record.id,
                condition=record.condition,
                route_to=record.route_to,
                current_status=record.status,
                readiness="unnecessary" if record.status == "approved" else "not_ready",
                has_simulation=False,
                has_rollout_plan=False,
                blocking_issues=[] if record.status == "approved" else [f"Recommendation status is {record.status}."],
            )

        return _side_effect

    def _build_portfolio_rollout_side_effect(self, rollout_map: dict[int, PolicyRolloutPlanResponse]):
        def _side_effect(recommendation_id: int, sample_limit: int, db, record_audit: bool = True):
            if recommendation_id in rollout_map:
                return rollout_map[recommendation_id]
            raise ValueError("Rollout fixture unavailable.")

        return _side_effect

    def _get_audit_events_for_recommendation(self, recommendation_id: int) -> list[PolicyAuditEvent]:
        db = SessionLocal()
        try:
            return list(
                db.scalars(
                    select(PolicyAuditEvent)
                    .where(PolicyAuditEvent.recommendation_id == recommendation_id)
                    .order_by(PolicyAuditEvent.created_at.asc(), PolicyAuditEvent.id.asc())
                )
            )
        finally:
            db.close()

    def test_health(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")

    def test_models(self) -> None:
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        aliases = {item["alias"] for item in payload}
        self.assertIn("fast_general", aliases)
        self.assertIn("code_specialist", aliases)

    def test_home_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("智能路由演示台", response.text)
        self.assertIn("/workspace", response.text)
        self.assertIn("/classification", response.text)
        self.assertIn("/governance", response.text)

    def test_workspace_page(self) -> None:
        response = self.client.get("/workspace")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("对话演示台", response.text)
        self.assertIn("决策模型", response.text)
        self.assertIn("/api/v1/chat", response.text)

    def test_classification_page(self) -> None:
        response = self.client.get("/classification")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("决策展示台", response.text)
        self.assertIn("规则回退", response.text)
        self.assertIn("/api/v1/models", response.text)

    def test_governance_page(self) -> None:
        response = self.client.get("/governance")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("治理中心", response.text)
        self.assertIn("/api/v1/admin/policy/dashboard", response.text)

    def test_system_info(self) -> None:
        response = self.client.get("/system/info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["workspace"], "/workspace")
        self.assertEqual(payload["classification"], "/classification")
        self.assertEqual(payload["governance"], "/governance")

    def test_chat_rejects_blank_message(self) -> None:
        response = self.client.post("/api/v1/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 422)

    def test_chat_route_for_coding(self) -> None:
        with patch.dict("os.environ", {"MOCK_LLM_RESPONSES": "true"}, clear=False):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            get_orchestrator.cache_clear()

            response = self.client.post(
                "/api/v1/chat",
                json={"message": "Help me debug this Python function"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["task_type"], "coding")
            self.assertEqual(payload["selected_model"], "code_specialist")
            self.assertGreaterEqual(payload["features"]["code_signal"], 0.5)
            self.assertIn("decision_trace", payload)
            self.assertIn(payload["decision_trace"]["source"], {"ai_model", "rule_router"})
            self.assertIn("```python", payload["answer"])

        get_settings.cache_clear()
        get_model_registry.cache_clear()
        get_orchestrator.cache_clear()

    def test_chat_persists_extended_execution_fields(self) -> None:
        with patch.dict("os.environ", {"MOCK_LLM_RESPONSES": "true"}, clear=False):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            get_orchestrator.cache_clear()

            response = self.client.post(
                "/api/v1/chat",
                json={"message": "Help me debug this Python function"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()

            db = SessionLocal()
            try:
                execution = db.get(ExecutionRecord, payload["execution_id"])
            finally:
                db.close()

            self.assertIsNotNone(execution)
            self.assertEqual(execution.selected_model_key, "code_specialist")
            self.assertEqual(execution.selected_model_name, payload["resolved_model"])
            self.assertEqual(execution.actual_model_used, payload["resolved_model"])
            self.assertFalse(execution.fallback_triggered)
            self.assertTrue(execution.success)
            self.assertIsNone(execution.error_type)

        get_settings.cache_clear()
        get_model_registry.cache_clear()
        get_orchestrator.cache_clear()

    def test_chat_falls_back_to_general_when_classifier_errors(self) -> None:
        with (
            patch.dict("os.environ", {"MOCK_LLM_RESPONSES": "true"}, clear=False),
            patch("app.agent.orchestrator.TaskClassifier.classify", side_effect=RuntimeError("classifier down")),
        ):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            get_orchestrator.cache_clear()

            response = self.client.post(
                "/api/v1/chat",
                json={"message": "Explain how agent routing works"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["task_type"], "general_qa")
            self.assertEqual(payload["selected_model"], "fast_general")
            self.assertIn("fell back", payload["classification"]["reason"])

        get_settings.cache_clear()
        get_model_registry.cache_clear()
        get_orchestrator.cache_clear()

    def test_admin_stats_endpoint_returns_aggregates(self) -> None:
        baseline_response = self.client.get("/api/v1/admin/stats")
        self.assertEqual(baseline_response.status_code, 200)
        baseline = baseline_response.json()

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSES": "true"}, clear=False):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            get_orchestrator.cache_clear()

            normal_response = self.client.post(
                "/api/v1/chat",
                json={"message": "Summarize this release note in three bullet points."},
            )
            self.assertEqual(normal_response.status_code, 200)

        def flaky_generate(_gateway_self, request):
            if request.model_alias == "fast_general":
                return LLMResponse(
                    content="",
                    provider=request.provider,
                    model=request.model,
                    success=False,
                    error_message="forced provider failure",
                    raw={"error": "forced provider failure"},
                )
            return LLMResponse(
                content="Recovered answer from fallback path.",
                provider=request.provider,
                model=request.model,
                latency_ms=12,
                success=True,
                raw={"mode": "test-fallback"},
            )

        with (
            patch.dict("os.environ", {"MOCK_LLM_RESPONSES": "false"}, clear=False),
            patch("app.models.litellm_gateway.LiteLLMGateway.generate", new=flaky_generate),
        ):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            get_orchestrator.cache_clear()

            fallback_response = self.client.post(
                "/api/v1/chat",
                json={"message": "Tell me what an agent router does."},
            )
            self.assertEqual(fallback_response.status_code, 200)
            self.assertTrue(fallback_response.json()["fallback_triggered"])

        get_settings.cache_clear()
        get_model_registry.cache_clear()
        get_orchestrator.cache_clear()

        response = self.client.get("/api/v1/admin/stats")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertGreaterEqual(payload["total_requests"], baseline["total_requests"] + 2)
        self.assertGreaterEqual(payload["success_rate"], 0.0)
        self.assertLessEqual(payload["success_rate"], 1.0)
        self.assertGreaterEqual(payload["fallback_rate"], 0.0)
        self.assertLessEqual(payload["fallback_rate"], 1.0)
        self.assertGreaterEqual(payload["avg_latency_ms"], 0.0)
        self.assertTrue(payload["top_models"])
        self.assertIn("generation_error", {item["error_type"] for item in payload["error_breakdown"]})

    def test_call_model_supports_custom_system_prompt_and_fallback(self) -> None:
        seen_system_prompts: list[str] = []
        call_count = {"value": 0}

        def fake_generate(_gateway_self, request):
            seen_system_prompts.append(request.messages[0].content)
            call_count["value"] += 1
            if call_count["value"] == 1:
                return LLMResponse(
                    content="",
                    provider=request.provider,
                    model=request.model,
                    success=False,
                    error_message="forced timeout",
                    raw={"error": "forced timeout"},
                )
            return LLMResponse(
                content='{"summary":"ok","observations":[],"anomalies":[],"recommendations":[]}',
                provider=request.provider,
                model=request.model,
                latency_ms=9,
                success=True,
                raw={"mode": "test"},
            )

        with patch("app.models.litellm_gateway.LiteLLMGateway.generate", new=fake_generate):
            result = call_model(
                "Analyze these stats.",
                model_name="gpt-4o-mini",
                system_prompt="CUSTOM SYSTEM",
                metadata={"needs_json": True, "admin_analysis": True},
            )

        self.assertEqual(seen_system_prompts[0], "CUSTOM SYSTEM")
        self.assertTrue(result.success)
        self.assertTrue(result.fallback_triggered)
        self.assertEqual(result.error_type, None)

    def test_admin_ai_analyze_returns_model_json(self) -> None:
        with patch(
            "app.admin_ai_service.call_model",
            return_value=ModelCallResult(
                content=json.dumps(
                    {
                        "summary": "Platform is stable.",
                        "observations": ["Success rate remains healthy."],
                        "anomalies": ["Fallback usage increased slightly."],
                        "recommendations": ["Review recent fallback-triggered requests."],
                    }
                ),
                latency_ms=11,
                model_used="openai/gpt-4o-mini",
                fallback_triggered=False,
                success=True,
                error_type=None,
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/analyze")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"], "Platform is stable.")
        self.assertEqual(payload["observations"], ["Success rate remains healthy."])
        self.assertEqual(payload["anomalies"], ["Fallback usage increased slightly."])
        self.assertEqual(payload["recommendations"], ["Review recent fallback-triggered requests."])

    def test_admin_ai_analyze_returns_safe_fallback_when_json_invalid(self) -> None:
        with patch(
            "app.admin_ai_service.call_model",
            return_value=ModelCallResult(
                content="not-json",
                latency_ms=7,
                model_used="openai/gpt-4o-mini",
                fallback_triggered=False,
                success=True,
                error_type=None,
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/analyze")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Safe fallback analysis", payload["summary"])
        self.assertTrue(payload["observations"])
        self.assertTrue(payload["anomalies"])
        self.assertTrue(payload["recommendations"])

    def test_admin_router_recommendation_returns_model_json(self) -> None:
        with (
            patch(
                "app.router_recommendation_service.analyze_platform_with_ai",
                return_value=AIAnalysisResponse(
                    summary="Analysis summary.",
                    observations=["Observation."],
                    anomalies=["Anomaly."],
                    recommendations=["Recommendation."],
                ),
            ),
            patch(
                "app.router_recommendation_service.call_model",
                return_value=ModelCallResult(
                    content=json.dumps(
                        {
                            "summary": "Recommend a targeted structured-output path.",
                            "recommended_changes": [
                                {
                                    "priority": 1,
                                    "condition": "If requests require strict JSON output and frequently fail validation.",
                                    "route_to": "structured_output_model",
                                    "reason": "A specialized structured route can reduce invalid_json failures.",
                                }
                            ],
                            "notes": [
                                "Read-only recommendation only.",
                                "No policy was auto-applied.",
                            ],
                        }
                    ),
                    latency_ms=10,
                    model_used="openai/gpt-4o-mini",
                    fallback_triggered=False,
                    success=True,
                    error_type=None,
                ),
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/recommend-router-update")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"], "Recommend a targeted structured-output path.")
        self.assertEqual(len(payload["recommended_changes"]), 1)
        self.assertEqual(payload["recommended_changes"][0]["route_to"], "structured_output_model")
        self.assertIn("No policy was auto-applied.", payload["notes"])

    def test_admin_router_recommendation_returns_safe_fallback_when_json_invalid(self) -> None:
        with (
            patch(
                "app.router_recommendation_service.analyze_platform_with_ai",
                return_value=AIAnalysisResponse(
                    summary="Analysis summary.",
                    observations=["Observation."],
                    anomalies=["Anomaly."],
                    recommendations=["Recommendation."],
                ),
            ),
            patch(
                "app.router_recommendation_service.call_model",
                return_value=ModelCallResult(
                    content="not-json",
                    latency_ms=8,
                    model_used="openai/gpt-4o-mini",
                    fallback_triggered=False,
                    success=True,
                    error_type=None,
                ),
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/recommend-router-update")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Safe fallback router recommendation", payload["summary"])
        self.assertEqual(payload["recommended_changes"], [])
        self.assertTrue(payload["notes"])

    def test_admin_router_recommendation_rejects_invalid_route_targets(self) -> None:
        with (
            patch(
                "app.router_recommendation_service.analyze_platform_with_ai",
                return_value=AIAnalysisResponse(
                    summary="Analysis summary.",
                    observations=["Observation."],
                    anomalies=["Anomaly."],
                    recommendations=["Recommendation."],
                ),
            ),
            patch(
                "app.router_recommendation_service.call_model",
                return_value=ModelCallResult(
                    content=json.dumps(
                        {
                            "summary": "Unsafe recommendation.",
                            "recommended_changes": [
                                {
                                    "priority": 1,
                                    "condition": "Whenever a request contains code.",
                                    "route_to": "unknown_model",
                                    "reason": "This should be rejected by validation.",
                                }
                            ],
                            "notes": ["This output should not pass validation."],
                        }
                    ),
                    latency_ms=8,
                    model_used="openai/gpt-4o-mini",
                    fallback_triggered=False,
                    success=True,
                    error_type=None,
                ),
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/recommend-router-update")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommended_changes"], [])
        self.assertTrue(any("No router file was modified" in note for note in payload["notes"]))

    def test_router_inspect_routes_code_input_to_code_specialist(self) -> None:
        response = self.client.post(
            "/api/v1/router/inspect",
            json={"message": "def normalize_items(items):\n    return [item.strip() for item in items]"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_type"], "coding")
        self.assertEqual(payload["selected_model_key"], "code_specialist")
        self.assertIn("decision_trace", payload)
        self.assertIn(payload["decision_trace"]["source"], {"ai_model", "rule_router"})
        self.assertTrue(
            "Route scores favored" in payload["reason"]
            or "DeepSeek decision model" in payload["reason"]
            or "Mock DeepSeek routing" in payload["reason"]
        )

    def test_router_inspect_routes_reasoning_input_to_strong_reasoning(self) -> None:
        response = self.client.post(
            "/api/v1/router/inspect",
            json={
                "message": "Compare Redis and Postgres for a task queue and explain the durability trade-offs.",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_type"], "reasoning")
        self.assertEqual(payload["selected_model_key"], "strong_reasoning")
        self.assertTrue(payload["features"]["high_reasoning"] or payload["features"]["reasoning_signal"] >= 0.5)

    def test_router_inspect_routes_general_summary_to_fast_general(self) -> None:
        response = self.client.post(
            "/api/v1/router/inspect",
            json={
                "message": (
                    "Please summarize this product update into three short bullet points. "
                    "This week we launched a new permissions page, improved audit logging coverage, "
                    "reduced support response times, fixed a mobile form submission bug that affected checkout, "
                    "and tightened dashboard performance for large enterprise accounts."
                ),
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_type"], "summarization")
        self.assertEqual(payload["selected_model_key"], "fast_general")
        self.assertIn("summarization_signal", payload["features"])

    def test_review_recommendations_marks_already_covered(self) -> None:
        with patch(
            "app.recommendation_review_service.recommend_router_updates",
            return_value=RouterRecommendationResponse(
                summary="Recommendation summary.",
                recommended_changes=[
                    RouterRecommendationItem(
                        priority=1,
                        condition="If the request contains code snippets or debug traces.",
                        route_to="code_specialist",
                        reason="Code-heavy requests should use the coding route.",
                    )
                ],
                notes=["Read-only recommendation."],
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/review-recommendations")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reviews"][0]["status"], "already_covered")
        self.assertIn("already", payload["reviews"][0]["comment"].casefold())

    def test_review_recommendations_marks_unsupported_route(self) -> None:
        with patch(
            "app.recommendation_review_service.recommend_router_updates",
            return_value=RouterRecommendationResponse(
                summary="Recommendation summary.",
                recommended_changes=[
                    RouterRecommendationItem(
                        priority=1,
                        condition="If strict JSON output keeps failing validation.",
                        route_to="structured_output_model",
                        reason="A structured route might help JSON-heavy traffic.",
                    )
                ],
                notes=["Read-only recommendation."],
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/review-recommendations")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reviews"][0]["status"], "unsupported_route")
        self.assertIn("does not exist", payload["reviews"][0]["comment"])

    def test_review_recommendations_endpoint_response_shape(self) -> None:
        with patch(
            "app.recommendation_review_service.recommend_router_updates",
            return_value=RouterRecommendationResponse(
                summary="Recommendation summary.",
                recommended_changes=[
                    RouterRecommendationItem(
                        priority=2,
                        condition="If a request needs deeper multi-step reasoning.",
                        route_to="strong_reasoning",
                        reason="Reasoning-heavy prompts benefit from stronger analysis.",
                    )
                ],
                notes=["Read-only recommendation."],
            ),
        ):
            response = self.client.get("/api/v1/admin/ai/review-recommendations")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("summary", payload)
        self.assertIn("reviews", payload)
        self.assertIn("notes", payload)
        self.assertTrue(payload["reviews"])
        self.assertEqual(
            set(payload["reviews"][0].keys()),
            {"condition", "route_to", "status", "comment"},
        )

    def test_generate_policy_recommendations_creates_pending_records(self) -> None:
        token = f"policy-{uuid4().hex}"
        payload = self._generate_policy_recommendation(token)

        self.assertEqual(payload["created_count"], 1)
        self.assertEqual(len(payload["recommendations"]), 1)
        self.assertEqual(payload["recommendations"][0]["status"], "pending")
        self.assertEqual(payload["recommendations"][0]["review_status"], "already_covered")
        self.assertEqual(payload["recommendations"][0]["source"], "ai")
        self.assertIn("does not auto-apply", payload["summary"])

    def test_list_policy_recommendations_returns_records(self) -> None:
        token = f"policy-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        created_id = created["recommendations"][0]["id"]

        response = self.client.get("/api/v1/admin/policy/recommendations", params={"status": "pending"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        matching = [item for item in payload if item["id"] == created_id]
        self.assertEqual(len(matching), 1)
        self.assertIn(token, matching[0]["summary"])
        self.assertEqual(matching[0]["status"], "pending")

    def test_approve_policy_recommendation_changes_status_to_approved(self) -> None:
        token = f"policy-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation_id"], recommendation_id)
        self.assertEqual(payload["status"], "approved")
        self.assertIn("No router policy was auto-applied.", payload["message"])

    def test_reject_policy_recommendation_changes_status_to_rejected(self) -> None:
        token = f"policy-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/reject")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation_id"], recommendation_id)
        self.assertEqual(payload["status"], "rejected")
        self.assertIn("No router policy was auto-applied.", payload["message"])

    def test_policy_snapshot_returns_200_and_required_sections(self) -> None:
        response = self.client.get("/api/v1/admin/policy/snapshot")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("active_policy_version", payload)
        self.assertIn("active_routing_rules", payload)
        self.assertIn("available_models", payload)
        self.assertIn("approved_but_not_applied", payload)
        self.assertIn("notes", payload)
        self.assertTrue(payload["active_routing_rules"])
        self.assertTrue(payload["available_models"])
        self.assertEqual(
            set(payload["available_models"][0].keys()),
            {"model_key", "model_name", "capabilities", "cost_level", "speed_level"},
        )
        self.assertTrue(any("tracked" in note.casefold() for note in payload["notes"]))
        self.assertTrue(any("not yet active" in note.casefold() for note in payload["notes"]))
        self.assertTrue(any("rule_router.py" in note for note in payload["notes"]))

    def test_policy_snapshot_includes_approved_recommendations(self) -> None:
        token = f"policy-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        approve_response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")
        self.assertEqual(approve_response.status_code, 200)

        response = self.client.get("/api/v1/admin/policy/snapshot")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        matching = [item for item in payload["approved_but_not_applied"] if item["id"] == recommendation_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["status"], "approved")
        self.assertEqual(matching[0]["review_status"], "already_covered")
        self.assertIn(token, matching[0]["condition"])

    def test_policy_snapshot_is_read_only_and_preserves_recommendation_status(self) -> None:
        token = f"policy-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        approve_response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")
        self.assertEqual(approve_response.status_code, 200)

        with patch(
            "app.models.litellm_gateway.LiteLLMGateway.generate",
            side_effect=AssertionError("policy snapshot must not call the model"),
        ):
            snapshot_response = self.client.get("/api/v1/admin/policy/snapshot")

        self.assertEqual(snapshot_response.status_code, 200)

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_policy_simulation_endpoint_returns_200_for_approved_recommendation(self) -> None:
        token = uuid4().hex
        self._insert_execution_sample(
            message=f"def normalize_{token}(items):\n    return [item.strip() for item in items]",
            task_type="coding",
            model_alias="code_specialist",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate",
            params={"sample_limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation_id"], recommendation_id)
        self.assertEqual(payload["recommendation_condition"], "contains_code == true")
        self.assertEqual(payload["recommendation_route_to"], "fast_general")
        self.assertEqual(payload["total_samples"], 1)
        self.assertEqual(len(payload["sample_results"]), 1)
        self.assertTrue(any("simulation only" in note.casefold() for note in payload["notes"]))

    def test_policy_simulation_rejects_non_approved_recommendations(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
            status="pending",
        )

        response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only approved recommendations can be simulated.", response.json()["detail"])

    def test_policy_simulation_changed_samples_computed_correctly(self) -> None:
        token = uuid4().hex
        self._insert_execution_sample(
            message=f"Explain the product roadmap status for customer {token}.",
            task_type="general_qa",
            model_alias="fast_general",
        )
        self._insert_execution_sample(
            message=f"def transform_{token}(value):\n    return value.strip().lower()",
            task_type="coding",
            model_alias="code_specialist",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate",
            params={"sample_limit": 2},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_samples"], 2)
        self.assertEqual(payload["changed_samples"], 1)
        self.assertEqual(payload["unchanged_samples"], 1)
        self.assertEqual(sum(1 for item in payload["sample_results"] if item["changed"]), 1)

    def test_policy_simulation_unsupported_conditions_do_not_crash(self) -> None:
        token = uuid4().hex
        self._insert_execution_sample(
            message=f"Summarize this launch note for customer {token}.",
            task_type="summarization",
            model_alias="fast_general",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="traffic_risk_score > 0.8",
            route_to="strong_reasoning",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate",
            params={"sample_limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["changed_samples"], 0)
        self.assertEqual(payload["unchanged_samples"], 1)
        self.assertIn("unsupported", payload["sample_results"][0]["explanation"].casefold())

    def test_policy_simulation_is_read_only_and_preserves_status_and_router_behavior(self) -> None:
        token = uuid4().hex
        message = f"def cleanup_{token}(rows):\n    return [row.strip() for row in rows if row]"
        self._insert_execution_sample(
            message=message,
            task_type="coding",
            model_alias="code_specialist",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
        )

        before_inspect = self.client.post("/api/v1/router/inspect", json={"message": message})
        self.assertEqual(before_inspect.status_code, 200)
        before_payload = before_inspect.json()

        with patch(
            "app.models.litellm_gateway.LiteLLMGateway.generate",
            side_effect=AssertionError("policy simulation must not call the model"),
        ):
            simulate_response = self.client.get(
                f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate",
                params={"sample_limit": 1},
            )

        self.assertEqual(simulate_response.status_code, 200)

        after_inspect = self.client.post("/api/v1/router/inspect", json={"message": message})
        self.assertEqual(after_inspect.status_code, 200)
        after_payload = after_inspect.json()

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")
        self.assertEqual(before_payload["selected_model_key"], after_payload["selected_model_key"])

    def test_rollout_plan_endpoint_returns_200_for_approved_recommendation(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=1,
            unchanged_samples=9,
        )

        with patch(
            "app.policy_rollout_plan_service.simulate_recommendation_application",
            return_value=simulated,
        ):
            response = self.client.get(
                f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan",
                params={"sample_limit": 10},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation_id"], recommendation_id)
        self.assertEqual(payload["rollout_strategy"], "very limited rollout")

    def test_rollout_plan_endpoint_rejects_non_approved_recommendation(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="pending",
        )

        response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan")

        self.assertEqual(response.status_code, 400)
        self.assertIn("only available for approved recommendations", response.json()["detail"].casefold())

    def test_rollout_plan_response_contains_required_sections(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="task_type == 'reasoning'",
            route_to="strong_reasoning",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="task_type == 'reasoning'",
            route_to="strong_reasoning",
            total_samples=10,
            changed_samples=3,
            unchanged_samples=7,
            sample_results=[
                PolicySimulationSampleResult(
                    input_message="Compare two storage backends.",
                    current_task_type="reasoning",
                    current_selected_model_key="fast_general",
                    simulated_selected_model_key="strong_reasoning",
                    changed=True,
                    explanation="Recommendation condition matched; simulated route changed to strong_reasoning.",
                )
            ],
        )

        with patch(
            "app.policy_rollout_plan_service.simulate_recommendation_application",
            return_value=simulated,
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("rollout_strategy", payload)
        self.assertIn("phases", payload)
        self.assertIn("monitoring_metrics", payload)
        self.assertIn("rollback_plan", payload)
        self.assertTrue(payload["phases"])
        self.assertTrue(payload["monitoring_metrics"])
        self.assertIn("restore_target", payload["rollback_plan"])

    def test_high_impact_rollout_plan_produces_staged_rollout(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="fast_general",
            total_samples=10,
            changed_samples=6,
            unchanged_samples=4,
        )

        with patch(
            "app.policy_rollout_plan_service.simulate_recommendation_application",
            return_value=simulated,
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["estimated_impact"]["level"], "high")
        self.assertEqual(payload["rollout_strategy"], "staged rollout")
        self.assertEqual([phase["traffic_percentage"] for phase in payload["phases"]], [10, 25, 50, 100])

    def test_zero_change_rollout_plan_produces_no_rollout_needed(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="task_type == 'summarization'",
            route_to="fast_general",
            total_samples=8,
            changed_samples=0,
            unchanged_samples=8,
        )

        with patch(
            "app.policy_rollout_plan_service.simulate_recommendation_application",
            return_value=simulated,
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rollout_strategy"], "no rollout needed")
        self.assertEqual([phase["traffic_percentage"] for phase in payload["phases"]], [0, 0])

    def test_rollout_plan_endpoint_is_read_only_and_does_not_mutate_state(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=12,
            changed_samples=2,
            unchanged_samples=10,
        )

        with (
            patch(
                "app.policy_rollout_plan_service.simulate_recommendation_application",
                return_value=simulated,
            ),
            patch(
                "app.models.litellm_gateway.LiteLLMGateway.generate",
                side_effect=AssertionError("rollout planning must not call the model"),
            ),
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan")

        self.assertEqual(response.status_code, 200)

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_apply_readiness_returns_response_for_approved_recommendation(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            review_status="compatible",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=2,
            unchanged_samples=8,
        )
        rollout_plan = self._build_rollout_plan_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=2,
            unchanged_samples=8,
            change_ratio=0.2,
            rollout_strategy="very limited rollout",
            impact_level="medium",
        )

        with (
            patch(
                "app.policy_apply_gate_service.simulate_recommendation_application",
                return_value=simulated,
            ),
            patch(
                "app.policy_apply_gate_service.create_rollout_plan",
                return_value=rollout_plan,
            ),
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation_id"], recommendation_id)
        self.assertEqual(payload["current_status"], "approved")
        self.assertEqual(payload["readiness"], "ready_for_future_apply")
        self.assertTrue(payload["guardrail_checks"])
        self.assertEqual(payload["blocking_issues"], [])
        self.assertEqual(payload["next_step"], "Eligible for a future controlled rollout, but not applied yet.")

    def test_apply_readiness_non_approved_recommendation_is_not_ready(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="pending",
            review_status="compatible",
        )

        response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["current_status"], "pending")
        self.assertEqual(payload["readiness"], "not_ready")
        self.assertTrue(payload["blocking_issues"])

    def test_apply_readiness_unsupported_route_leads_to_not_ready(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            review_status="unsupported_route",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=1,
            unchanged_samples=9,
        )
        rollout_plan = self._build_rollout_plan_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=1,
            unchanged_samples=9,
            change_ratio=0.1,
            rollout_strategy="very limited rollout",
            impact_level="low",
        )

        with (
            patch(
                "app.policy_apply_gate_service.simulate_recommendation_application",
                return_value=simulated,
            ),
            patch(
                "app.policy_apply_gate_service.create_rollout_plan",
                return_value=rollout_plan,
            ),
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["readiness"], "not_ready")
        self.assertTrue(any("unsupported_route" in issue for issue in payload["blocking_issues"]))

    def test_apply_readiness_zero_change_recommendation_becomes_unnecessary(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
            review_status="compatible",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="task_type == 'summarization'",
            route_to="fast_general",
            total_samples=8,
            changed_samples=0,
            unchanged_samples=8,
        )
        rollout_plan = self._build_rollout_plan_response(
            recommendation_id=recommendation_id,
            condition="task_type == 'summarization'",
            route_to="fast_general",
            total_samples=8,
            changed_samples=0,
            unchanged_samples=8,
            change_ratio=0.0,
            rollout_strategy="no rollout needed",
            impact_level="low",
        )

        with (
            patch(
                "app.policy_apply_gate_service.simulate_recommendation_application",
                return_value=simulated,
            ),
            patch(
                "app.policy_apply_gate_service.create_rollout_plan",
                return_value=rollout_plan,
            ),
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["readiness"], "unnecessary")
        self.assertEqual(payload["next_step"], "No rollout needed; recommendation appears unnecessary.")

    def test_apply_readiness_manual_review_rollout_strategy_becomes_manual_review_required(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            review_status="compatible",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=2,
            unchanged_samples=8,
        )
        rollout_plan = self._build_rollout_plan_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=10,
            changed_samples=2,
            unchanged_samples=8,
            change_ratio=0.2,
            rollout_strategy="manual review before rollout",
            impact_level="medium",
        )

        with (
            patch(
                "app.policy_apply_gate_service.simulate_recommendation_application",
                return_value=simulated,
            ),
            patch(
                "app.policy_apply_gate_service.create_rollout_plan",
                return_value=rollout_plan,
            ),
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["readiness"], "manual_review_required")
        self.assertIn("Manual governance review required before any rollout.", payload["next_step"])

    def test_apply_readiness_endpoint_is_read_only_and_does_not_mutate_state(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            review_status="compatible",
        )
        simulated = self._build_simulation_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=12,
            changed_samples=2,
            unchanged_samples=10,
        )
        rollout_plan = self._build_rollout_plan_response(
            recommendation_id=recommendation_id,
            condition="contains_code == true",
            route_to="code_specialist",
            total_samples=12,
            changed_samples=2,
            unchanged_samples=10,
            change_ratio=0.17,
            rollout_strategy="very limited rollout",
            impact_level="low",
        )

        with (
            patch(
                "app.policy_apply_gate_service.simulate_recommendation_application",
                return_value=simulated,
            ),
            patch(
                "app.policy_apply_gate_service.create_rollout_plan",
                return_value=rollout_plan,
            ),
            patch(
                "app.models.litellm_gateway.LiteLLMGateway.generate",
                side_effect=AssertionError("apply readiness must not call the model"),
            ),
        ):
            response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness")

        self.assertEqual(response.status_code, 200)

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_policy_dashboard_returns_200_and_required_sections(self) -> None:
        approved_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
        )
        pending_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
            status="pending",
            review_status="compatible",
        )
        readiness_map = {
            approved_id: self._build_apply_readiness_response(
                recommendation_id=approved_id,
                condition="contains_code == true",
                route_to="code_specialist",
                current_status="approved",
                readiness="ready_for_future_apply",
                has_simulation=True,
                has_rollout_plan=True,
            ),
            pending_id: self._build_apply_readiness_response(
                recommendation_id=pending_id,
                condition="task_type == 'summarization'",
                route_to="fast_general",
                current_status="pending",
                readiness="not_ready",
                has_simulation=False,
                has_rollout_plan=False,
                blocking_issues=["Recommendation status is pending."],
            ),
        }

        with patch(
            "app.policy_dashboard_service.evaluate_apply_readiness",
            side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
        ):
            response = self.client.get("/api/v1/admin/policy/dashboard", params={"limit": 10, "sample_limit": 5})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("active_policy_version", payload)
        self.assertIn("governance_counts", payload)
        self.assertIn("recent_recommendations", payload)
        self.assertIn("approved_but_not_applied", payload)
        self.assertIn("readiness_overview", payload)
        self.assertTrue(payload["current_router_summary"])
        self.assertTrue(payload["recent_recommendations"])

    def test_policy_dashboard_counts_are_internally_consistent(self) -> None:
        pending_id = self._insert_policy_recommendation(
            condition="needs_json == true",
            route_to="strong_reasoning",
            status="pending",
            review_status="compatible",
        )
        approved_ready_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
        )
        approved_manual_id = self._insert_policy_recommendation(
            condition="high_reasoning == true",
            route_to="strong_reasoning",
            status="approved",
            review_status="needs_manual_review",
        )
        rejected_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
            status="rejected",
            review_status="compatible",
        )
        readiness_map = {
            pending_id: self._build_apply_readiness_response(
                recommendation_id=pending_id,
                condition="needs_json == true",
                route_to="strong_reasoning",
                current_status="pending",
                readiness="not_ready",
                has_simulation=False,
                has_rollout_plan=False,
                blocking_issues=["Pending recommendations are not eligible."],
            ),
            approved_ready_id: self._build_apply_readiness_response(
                recommendation_id=approved_ready_id,
                condition="contains_code == true",
                route_to="code_specialist",
                current_status="approved",
                readiness="ready_for_future_apply",
                has_simulation=True,
                has_rollout_plan=True,
            ),
            approved_manual_id: self._build_apply_readiness_response(
                recommendation_id=approved_manual_id,
                condition="high_reasoning == true",
                route_to="strong_reasoning",
                current_status="approved",
                readiness="manual_review_required",
                has_simulation=True,
                has_rollout_plan=True,
                warnings=["Manual review required."],
            ),
            rejected_id: self._build_apply_readiness_response(
                recommendation_id=rejected_id,
                condition="task_type == 'summarization'",
                route_to="fast_general",
                current_status="rejected",
                readiness="unnecessary",
                has_simulation=False,
                has_rollout_plan=False,
            ),
        }

        with patch(
            "app.policy_dashboard_service.evaluate_apply_readiness",
            side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
        ):
            response = self.client.get("/api/v1/admin/policy/dashboard", params={"limit": 20, "sample_limit": 5})

        self.assertEqual(response.status_code, 200)
        counts = response.json()["governance_counts"]
        self.assertEqual(
            counts["total_recommendations"],
            counts["pending_count"] + counts["approved_count"] + counts["rejected_count"],
        )
        self.assertEqual(counts["approved_but_not_applied_count"], counts["approved_count"])
        self.assertEqual(
            counts["total_recommendations"],
            counts["ready_for_future_apply_count"]
            + counts["not_ready_count"]
            + counts["unnecessary_count"]
            + counts["manual_review_required_count"],
        )

    def test_policy_dashboard_approved_recommendations_appear_in_approved_but_not_applied(self) -> None:
        approved_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
        )
        rejected_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
            status="rejected",
            review_status="compatible",
        )
        readiness_map = {
            approved_id: self._build_apply_readiness_response(
                recommendation_id=approved_id,
                condition="contains_code == true",
                route_to="code_specialist",
                current_status="approved",
                readiness="ready_for_future_apply",
                has_simulation=True,
                has_rollout_plan=True,
            ),
            rejected_id: self._build_apply_readiness_response(
                recommendation_id=rejected_id,
                condition="task_type == 'summarization'",
                route_to="fast_general",
                current_status="rejected",
                readiness="not_ready",
                has_simulation=False,
                has_rollout_plan=False,
                blocking_issues=["Rejected recommendation should not be applied."],
            ),
        }

        with patch(
            "app.policy_dashboard_service.evaluate_apply_readiness",
            side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
        ):
            response = self.client.get("/api/v1/admin/policy/dashboard")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        approved_items = payload["approved_but_not_applied"]
        self.assertEqual(len([item for item in approved_items if item["id"] == approved_id]), 1)
        self.assertEqual(len([item for item in approved_items if item["id"] == rejected_id]), 0)
        self.assertIn("ready_for_future_apply", payload["readiness_overview"])

    def test_policy_dashboard_is_read_only_and_does_not_mutate_state(self) -> None:
        approved_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
        )
        readiness = self._build_apply_readiness_response(
            recommendation_id=approved_id,
            condition="contains_code == true",
            route_to="code_specialist",
            current_status="approved",
            readiness="ready_for_future_apply",
            has_simulation=True,
            has_rollout_plan=True,
        )

        with (
            patch(
                "app.policy_dashboard_service.evaluate_apply_readiness",
                return_value=readiness,
            ),
            patch(
                "app.models.litellm_gateway.LiteLLMGateway.generate",
                side_effect=AssertionError("policy dashboard must not call the model"),
            ),
        ):
            response = self.client.get("/api/v1/admin/policy/dashboard")

        self.assertEqual(response.status_code, 200)

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, approved_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_generating_recommendations_creates_generated_audit_events(self) -> None:
        token = f"audit-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        events = self._get_audit_events_for_recommendation(recommendation_id)
        event_types = [event.event_type for event in events]

        self.assertIn("generated", event_types)
        self.assertIn("reviewed", event_types)

    def test_approving_recommendation_creates_approved_audit_event(self) -> None:
        token = f"audit-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")

        self.assertEqual(response.status_code, 200)
        event_types = [event.event_type for event in self._get_audit_events_for_recommendation(recommendation_id)]
        self.assertIn("approved", event_types)

    def test_rejecting_recommendation_creates_rejected_audit_event(self) -> None:
        token = f"audit-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/reject")

        self.assertEqual(response.status_code, 200)
        event_types = [event.event_type for event in self._get_audit_events_for_recommendation(recommendation_id)]
        self.assertIn("rejected", event_types)

    def test_simulation_creates_simulated_audit_event(self) -> None:
        token = uuid4().hex
        self._insert_execution_sample(
            message=f"def audit_{token}(value):\n    return value.strip()",
            task_type="coding",
            model_alias="code_specialist",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
            status="approved",
            review_status="compatible",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate",
            params={"sample_limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        event_types = [event.event_type for event in self._get_audit_events_for_recommendation(recommendation_id)]
        self.assertIn("simulated", event_types)

    def test_rollout_plan_creates_rollout_planned_audit_event(self) -> None:
        token = uuid4().hex
        self._insert_execution_sample(
            message=f"def rollout_{token}(value):\n    return value.strip().lower()",
            task_type="coding",
            model_alias="code_specialist",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
            status="approved",
            review_status="compatible",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/rollout-plan",
            params={"sample_limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        event_types = [event.event_type for event in self._get_audit_events_for_recommendation(recommendation_id)]
        self.assertIn("rollout_planned", event_types)

    def test_apply_readiness_creates_apply_readiness_evaluated_audit_event(self) -> None:
        token = uuid4().hex
        self._insert_execution_sample(
            message=f"def readiness_{token}(value):\n    return value.strip().lower()",
            task_type="coding",
            model_alias="code_specialist",
        )
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="fast_general",
            status="approved",
            review_status="compatible",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/apply-readiness",
            params={"sample_limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        event_types = [event.event_type for event in self._get_audit_events_for_recommendation(recommendation_id)]
        self.assertIn("apply_readiness_evaluated", event_types)

    def test_policy_timeline_returns_events_in_chronological_order(self) -> None:
        token = f"timeline-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]
        self._insert_execution_sample(
            message=f"def timeline_{token}(value):\n    return value.strip()",
            task_type="coding",
            model_alias="code_specialist",
        )

        approve_response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")
        self.assertEqual(approve_response.status_code, 200)
        simulate_response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/simulate",
            params={"sample_limit": 1},
        )
        self.assertEqual(simulate_response.status_code, 200)

        response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/timeline")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation_id"], recommendation_id)
        self.assertTrue(payload["timeline"])
        ids = [item["id"] for item in payload["timeline"]]
        self.assertEqual(ids, sorted(ids))
        event_types = [item["event_type"] for item in payload["timeline"]]
        self.assertLess(event_types.index("generated"), event_types.index("approved"))
        self.assertLess(event_types.index("approved"), event_types.index("simulated"))

    def test_policy_audit_overview_returns_total_and_recent_events(self) -> None:
        token = f"overview-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        response = self.client.get("/api/v1/admin/policy/audit/overview", params={"limit": 200})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("total_audit_events", payload)
        self.assertIn("recent_events", payload)
        self.assertGreaterEqual(payload["total_audit_events"], len(payload["recent_events"]))
        self.assertTrue(any(item["recommendation_id"] == recommendation_id for item in payload["recent_events"]))

    def test_audit_endpoints_are_read_only_and_do_not_mutate_recommendation_status(self) -> None:
        token = f"audit-ro-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]

        approve_response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")
        self.assertEqual(approve_response.status_code, 200)

        before_events = self._get_audit_events_for_recommendation(recommendation_id)

        with patch(
            "app.models.litellm_gateway.LiteLLMGateway.generate",
            side_effect=AssertionError("audit endpoints must not call the model"),
        ):
            timeline_response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/timeline")
            overview_response = self.client.get("/api/v1/admin/policy/audit/overview")

        self.assertEqual(timeline_response.status_code, 200)
        self.assertEqual(overview_response.status_code, 200)

        after_events = self._get_audit_events_for_recommendation(recommendation_id)
        self.assertEqual(len(before_events), len(after_events))

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_policy_report_returns_200_and_required_sections(self) -> None:
        token = f"report-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]
        approve_response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")
        self.assertEqual(approve_response.status_code, 200)
        self._insert_execution_sample(
            message=f"def report_{token}(value):\n    return value.strip().lower()",
            task_type="coding",
            model_alias="code_specialist",
        )

        response = self.client.get(
            f"/api/v1/admin/policy/recommendations/{recommendation_id}/report",
            params={"sample_limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("recommendation", payload)
        self.assertIn("simulation", payload)
        self.assertIn("rollout_plan", payload)
        self.assertIn("apply_readiness", payload)
        self.assertIn("audit_summary", payload)
        self.assertIn("executive_summary", payload)
        self.assertTrue(payload["executive_summary"].strip())

    def test_policy_report_works_when_some_subsections_are_unavailable(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="pending",
            review_status="compatible",
        )

        response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/report")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["recommendation"]["status"], "pending")
        self.assertTrue(payload["simulation"]["notes"])
        self.assertTrue(payload["rollout_plan"]["notes"])
        self.assertEqual(payload["apply_readiness"]["readiness"], "not_ready")

    def test_policy_report_executive_summary_is_present_and_non_empty(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
            status="approved",
            review_status="already_covered",
        )

        response = self.client.get(f"/api/v1/admin/policy/recommendations/{recommendation_id}/report")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsInstance(payload["executive_summary"], str)
        self.assertTrue(payload["executive_summary"].strip())

    def test_policy_report_endpoint_is_read_only_and_does_not_mutate_state(self) -> None:
        token = f"report-ro-{uuid4().hex}"
        created = self._generate_policy_recommendation(token)
        recommendation_id = created["recommendations"][0]["id"]
        approve_response = self.client.post(f"/api/v1/admin/policy/recommendations/{recommendation_id}/approve")
        self.assertEqual(approve_response.status_code, 200)
        self._insert_execution_sample(
            message=f"def report_readonly_{token}(value):\n    return value.strip()",
            task_type="coding",
            model_alias="code_specialist",
        )

        before_events = self._get_audit_events_for_recommendation(recommendation_id)

        with patch(
            "app.models.litellm_gateway.LiteLLMGateway.generate",
            side_effect=AssertionError("policy report must not call the model"),
        ):
            response = self.client.get(
                f"/api/v1/admin/policy/recommendations/{recommendation_id}/report/export",
                params={"sample_limit": 1},
            )

        self.assertEqual(response.status_code, 200)

        after_events = self._get_audit_events_for_recommendation(recommendation_id)
        self.assertEqual(len(before_events), len(after_events))

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_policy_portfolio_report_returns_200_and_required_sections(self) -> None:
        approved_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
            priority=1,
        )
        readiness_map = {
            approved_id: self._build_apply_readiness_response(
                recommendation_id=approved_id,
                condition="contains_code == true",
                route_to="code_specialist",
                current_status="approved",
                readiness="ready_for_future_apply",
                has_simulation=True,
                has_rollout_plan=True,
            )
        }
        rollout_map = {
            approved_id: self._build_rollout_plan_response(
                recommendation_id=approved_id,
                condition="contains_code == true",
                route_to="code_specialist",
                total_samples=10,
                changed_samples=1,
                unchanged_samples=9,
                change_ratio=0.1,
                rollout_strategy="very limited rollout",
                impact_level="low",
            )
        }

        with (
            patch(
                "app.policy_portfolio_service.evaluate_apply_readiness",
                side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
            ),
            patch(
                "app.policy_portfolio_service.create_rollout_plan",
                side_effect=self._build_portfolio_rollout_side_effect(rollout_map),
            ),
        ):
            response = self.client.get("/api/v1/admin/policy/portfolio-report", params={"limit": 50, "sample_limit": 5})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("counts", payload)
        self.assertIn("recommendation_portfolio", payload)
        self.assertIn("executive_summary", payload)
        self.assertTrue(payload["executive_summary"].strip())
        self.assertEqual(payload["counts"]["total"], len(payload["recommendation_portfolio"]))

    def test_policy_portfolio_blocked_and_unnecessary_sections_are_populated(self) -> None:
        blocked_id = self._insert_policy_recommendation(
            condition="needs_json == true",
            route_to="strong_reasoning",
            status="approved",
            review_status="compatible",
            priority=2,
        )
        unnecessary_id = self._insert_policy_recommendation(
            condition="task_type == 'summarization'",
            route_to="fast_general",
            status="approved",
            review_status="already_covered",
            priority=3,
        )
        readiness_map = {
            blocked_id: self._build_apply_readiness_response(
                recommendation_id=blocked_id,
                condition="needs_json == true",
                route_to="strong_reasoning",
                current_status="approved",
                readiness="not_ready",
                has_simulation=True,
                has_rollout_plan=False,
                blocking_issues=["Blocking issue fixture."],
            ),
            unnecessary_id: self._build_apply_readiness_response(
                recommendation_id=unnecessary_id,
                condition="task_type == 'summarization'",
                route_to="fast_general",
                current_status="approved",
                readiness="unnecessary",
                has_simulation=True,
                has_rollout_plan=True,
            ),
        }
        rollout_map = {
            blocked_id: self._build_rollout_plan_response(
                recommendation_id=blocked_id,
                condition="needs_json == true",
                route_to="strong_reasoning",
                total_samples=10,
                changed_samples=6,
                unchanged_samples=4,
                change_ratio=0.6,
                rollout_strategy="staged rollout",
                impact_level="high",
            ),
            unnecessary_id: self._build_rollout_plan_response(
                recommendation_id=unnecessary_id,
                condition="task_type == 'summarization'",
                route_to="fast_general",
                total_samples=8,
                changed_samples=0,
                unchanged_samples=8,
                change_ratio=0.0,
                rollout_strategy="no rollout needed",
                impact_level="low",
            ),
        }

        with (
            patch(
                "app.policy_portfolio_service.evaluate_apply_readiness",
                side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
            ),
            patch(
                "app.policy_portfolio_service.create_rollout_plan",
                side_effect=self._build_portfolio_rollout_side_effect(rollout_map),
            ),
        ):
            response = self.client.get("/api/v1/admin/policy/portfolio-report")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(blocked_id, {item["id"] for item in payload["blocked_recommendations"]})
        self.assertIn(unnecessary_id, {item["id"] for item in payload["unnecessary_recommendations"]})

    def test_policy_portfolio_top_priority_includes_approved_ready_items(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
            priority=1,
        )
        readiness_map = {
            recommendation_id: self._build_apply_readiness_response(
                recommendation_id=recommendation_id,
                condition="contains_code == true",
                route_to="code_specialist",
                current_status="approved",
                readiness="ready_for_future_apply",
                has_simulation=True,
                has_rollout_plan=True,
            )
        }
        rollout_map = {
            recommendation_id: self._build_rollout_plan_response(
                recommendation_id=recommendation_id,
                condition="contains_code == true",
                route_to="code_specialist",
                total_samples=12,
                changed_samples=1,
                unchanged_samples=11,
                change_ratio=0.08,
                rollout_strategy="very limited rollout",
                impact_level="low",
            )
        }

        with (
            patch(
                "app.policy_portfolio_service.evaluate_apply_readiness",
                side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
            ),
            patch(
                "app.policy_portfolio_service.create_rollout_plan",
                side_effect=self._build_portfolio_rollout_side_effect(rollout_map),
            ),
        ):
            response = self.client.get("/api/v1/admin/policy/portfolio-report")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(recommendation_id, {item["id"] for item in payload["top_priority_recommendations"]})

    def test_policy_portfolio_endpoint_is_read_only_and_does_not_mutate_state(self) -> None:
        recommendation_id = self._insert_policy_recommendation(
            condition="contains_code == true",
            route_to="code_specialist",
            status="approved",
            review_status="compatible",
            priority=1,
        )
        readiness_map = {
            recommendation_id: self._build_apply_readiness_response(
                recommendation_id=recommendation_id,
                condition="contains_code == true",
                route_to="code_specialist",
                current_status="approved",
                readiness="ready_for_future_apply",
                has_simulation=True,
                has_rollout_plan=True,
            )
        }
        rollout_map = {
            recommendation_id: self._build_rollout_plan_response(
                recommendation_id=recommendation_id,
                condition="contains_code == true",
                route_to="code_specialist",
                total_samples=10,
                changed_samples=1,
                unchanged_samples=9,
                change_ratio=0.1,
                rollout_strategy="very limited rollout",
                impact_level="low",
            )
        }

        before_events = self._get_audit_events_for_recommendation(recommendation_id)

        with (
            patch(
                "app.policy_portfolio_service.evaluate_apply_readiness",
                side_effect=self._build_dashboard_readiness_side_effect(readiness_map),
            ),
            patch(
                "app.policy_portfolio_service.create_rollout_plan",
                side_effect=self._build_portfolio_rollout_side_effect(rollout_map),
            ),
            patch(
                "app.models.litellm_gateway.LiteLLMGateway.generate",
                side_effect=AssertionError("policy portfolio must not call the model"),
            ),
        ):
            response = self.client.get("/api/v1/admin/policy/portfolio-report")

        self.assertEqual(response.status_code, 200)

        after_events = self._get_audit_events_for_recommendation(recommendation_id)
        self.assertEqual(len(before_events), len(after_events))

        db = SessionLocal()
        try:
            recommendation = db.get(PolicyRecommendation, recommendation_id)
        finally:
            db.close()

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.status, "approved")

    def test_chat_survives_rag_context_failure(self) -> None:
        with (
            patch.dict("os.environ", {"MOCK_LLM_RESPONSES": "true"}, clear=False),
            patch("app.agent.orchestrator.RAGPipeline.prepare_context", side_effect=RuntimeError("rag unavailable")),
        ):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            get_orchestrator.cache_clear()

            response = self.client.post(
                "/api/v1/chat",
                json={"message": "Search the knowledge base and retrieve the auth section from spec.pdf"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["task_type"], "rag")
            self.assertEqual(payload["selected_model"], "long_context_rag")
            self.assertIn("Sources:", payload["answer"])

        get_settings.cache_clear()
        get_model_registry.cache_clear()
        get_orchestrator.cache_clear()

    def test_model_registry_provider_follows_model_prefix(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FAST_GENERAL_MODEL": "anthropic/claude-3-5-haiku-latest",
                "STRONG_REASONING_MODEL": "openai/gpt-4o-mini",
                "CODE_SPECIALIST_MODEL": "openai/gpt-4.1-mini",
                "RAG_MODEL": "openai/gpt-4o-mini",
                "LOCAL_FALLBACK_MODEL": "mock/local-fallback",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            get_model_registry.cache_clear()
            registry = get_model_registry()

            self.assertEqual(registry.get("fast_general").provider, "anthropic")
            self.assertEqual(registry.get("strong_reasoning").provider, "openai")
            self.assertEqual(registry.get("long_context_rag").provider, "openai")

        get_settings.cache_clear()
        get_model_registry.cache_clear()

    def test_load_env_file_sets_missing_variables_only(self) -> None:
        env_path = Path(__file__).resolve().parent / f".tmp-{uuid4().hex}.env"
        try:
            env_path.write_text(
                'OPENAI_API_KEY="test-openai-key"\nMOCK_LLM_RESPONSES=false\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                _load_env_file(env_path)
                self.assertEqual("test-openai-key", os.environ["OPENAI_API_KEY"])
                self.assertEqual("false", os.environ["MOCK_LLM_RESPONSES"])

            with patch.dict("os.environ", {"OPENAI_API_KEY": "already-set"}, clear=True):
                _load_env_file(env_path)
                self.assertEqual("already-set", os.environ["OPENAI_API_KEY"])
        finally:
            if env_path.exists():
                env_path.unlink()

    def test_gateway_recognizes_dashscope_deepseek_and_volcengine_keys(self) -> None:
        gateway = LiteLLMGateway()

        with patch.dict(
            "os.environ",
            {
                "DASHSCOPE_API_KEY": "dashscope-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "ARK_API_KEY": "ark-key",
            },
            clear=True,
        ):
            self.assertTrue(gateway._has_provider_credentials("dashscope"))
            self.assertTrue(gateway._has_provider_credentials("deepseek"))
            self.assertTrue(gateway._has_provider_credentials("volcengine"))

        with patch.dict(
            "os.environ",
            {
                "VOLCENGINE_API_KEY": "volcengine-key",
            },
            clear=True,
        ):
            self.assertTrue(gateway._has_provider_credentials("volcengine"))


if __name__ == "__main__":
    unittest.main()
