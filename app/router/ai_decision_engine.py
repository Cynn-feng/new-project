from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.models import LiteLLMGateway, ModelRegistry
from app.router.rule_router import RuleRouter
from app.schemas import DecisionTrace, LLMRequest, Message, ModelSelection, TaskClassification, TaskFeatures

SUPPORTED_ROUTE_ALIASES = (
    "fast_general",
    "strong_reasoning",
    "code_specialist",
    "long_context_rag",
)

_DECISION_SYSTEM_PROMPT = """You are an AI routing policy model.
Your only job is to choose the best route for a request.
Return strict JSON with keys: route_to, confidence, reason.
Allowed route_to values: fast_general, strong_reasoning, code_specialist, long_context_rag.
confidence must be a number between 0 and 1.
Do not include markdown fences or extra text."""


class AIDecisionEngine:
    def __init__(self, settings: Settings, registry: ModelRegistry):
        self.settings = settings
        self.registry = registry
        self.gateway = LiteLLMGateway(settings)

    def select_model(
        self,
        *,
        message: str,
        classification: TaskClassification,
        features: TaskFeatures,
        router: RuleRouter,
    ) -> tuple[ModelSelection, DecisionTrace]:
        rule_selection = router.choose_model(classification, features)

        if not self.settings.enable_ai_decision_engine:
            return rule_selection, self._rule_trace(
                rule_selection,
                notes=["AI decision engine is disabled, so the live route stayed rule-based."],
            )

        try:
            decision_result = self._request_decision(message, classification, features, rule_selection)
        except Exception as exc:
            return rule_selection, self._rule_trace(
                rule_selection,
                notes=[f"DeepSeek decision request failed before inference ({exc}), so the live route stayed rule-based."],
            )

        if not decision_result["success"]:
            return rule_selection, self._rule_trace(
                rule_selection,
                notes=[decision_result["note"]],
            )

        try:
            prediction = self._parse_decision_response(decision_result["content"])
        except Exception as exc:
            return rule_selection, self._rule_trace(
                rule_selection,
                notes=[
                    decision_result["note"],
                    f"DeepSeek returned invalid decision JSON ({exc}), so the live route stayed rule-based.",
                ],
            )

        ai_route = str(prediction["route_to"])
        ai_confidence = float(prediction["confidence"])
        ai_model_version = str(decision_result["model_used"] or self.settings.ai_decision_model)
        notes = [decision_result["note"]]

        if ai_route not in SUPPORTED_ROUTE_ALIASES or not self.registry.has(ai_route):
            notes.append(
                f"DeepSeek predicted unsupported route {ai_route}, so the live route stayed on {rule_selection.model_alias}."
            )
            return rule_selection, self._rule_trace(
                rule_selection,
                ai_route=ai_route,
                ai_confidence=ai_confidence,
                ai_model_version=ai_model_version,
                notes=notes,
            )

        threshold = self.settings.ai_decision_min_confidence
        if ai_confidence < threshold and ai_route != rule_selection.model_alias:
            notes.append(
                f"DeepSeek predicted {ai_route} at confidence {ai_confidence:.2f}, below the apply threshold {threshold:.2f}."
            )
            return rule_selection, self._rule_trace(
                rule_selection,
                ai_route=ai_route,
                ai_confidence=ai_confidence,
                ai_model_version=ai_model_version,
                notes=notes,
            )

        if ai_route == rule_selection.model_alias:
            notes.append(
                f"DeepSeek agreed with the rule route at confidence {ai_confidence:.2f}, so the selected route remains {ai_route}."
            )
            applied_reason = (
                f"DeepSeek decision model {ai_model_version} confirmed {ai_route} at confidence {ai_confidence:.2f}. "
                f"{prediction['reason']}"
            )
            selection = router.selection_for_alias(ai_route, applied_reason)
            return selection, DecisionTrace(
                source="ai_model",
                applied_route_to=selection.model_alias,
                applied_reason=selection.reason,
                rule_route_to=rule_selection.model_alias,
                ai_route_to=ai_route,
                ai_confidence=round(ai_confidence, 3),
                ai_model_version=ai_model_version,
                notes=notes,
            )

        notes.append(
            f"DeepSeek overrode the rule route {rule_selection.model_alias} with {ai_route} at confidence {ai_confidence:.2f}."
        )
        applied_reason = (
            f"DeepSeek decision model {ai_model_version} selected {ai_route} at confidence {ai_confidence:.2f}, "
            f"overriding rule route {rule_selection.model_alias}. {prediction['reason']}"
        )
        selection = router.selection_for_alias(ai_route, applied_reason)
        return selection, DecisionTrace(
            source="ai_model",
            applied_route_to=selection.model_alias,
            applied_reason=selection.reason,
            rule_route_to=rule_selection.model_alias,
            ai_route_to=ai_route,
            ai_confidence=round(ai_confidence, 3),
            ai_model_version=ai_model_version,
            notes=notes,
        )

    def _request_decision(
        self,
        message: str,
        classification: TaskClassification,
        features: TaskFeatures,
        rule_selection: ModelSelection,
    ) -> dict[str, Any]:
        provider = self._provider_from_model(self.settings.ai_decision_model)
        request = LLMRequest(
            model_alias="router_decision_deepseek",
            provider=provider,
            model=self.settings.ai_decision_model,
            messages=[
                Message(role="system", content=_DECISION_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=self._build_decision_prompt(message, classification, features, rule_selection),
                ),
            ],
            temperature=0.0,
            max_tokens=220,
            metadata={
                "router_decision": True,
                "needs_json": True,
                "task_type": classification.task_type.value,
                "rule_route_to": rule_selection.model_alias,
                "contains_code": features.contains_code,
                "needs_json_flag": features.needs_json,
                "needs_tools": features.needs_tools,
                "code_signal": features.code_signal,
                "retrieval_signal": features.retrieval_signal,
                "reasoning_signal": features.reasoning_signal,
                "summarization_signal": features.summarization_signal,
            },
        )
        response = self.gateway.generate(request)
        if response.success and response.content.strip():
            return {
                "success": True,
                "content": response.content,
                "model_used": response.model,
                "note": f"DeepSeek decision model {response.model} returned a routing decision.",
            }

        error = response.error_message or "empty_response"
        return {
            "success": False,
            "content": "",
            "model_used": response.model,
            "note": f"DeepSeek decision model failed ({error}), so the live route stayed rule-based.",
        }

    def _build_decision_prompt(
        self,
        message: str,
        classification: TaskClassification,
        features: TaskFeatures,
        rule_selection: ModelSelection,
    ) -> str:
        payload = {
            "message": message,
            "classification": classification.model_dump(),
            "features": features.model_dump(),
            "rule_route_suggestion": {
                "route_to": rule_selection.model_alias,
                "reason": rule_selection.reason,
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _parse_decision_response(self, raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()

        payload = json.loads(content)
        route_to = str(payload["route_to"]).strip()
        confidence = float(payload["confidence"])
        reason = str(payload.get("reason") or "No explicit reason was returned by the decision model.").strip()
        confidence = max(0.0, min(1.0, confidence))
        return {
            "route_to": route_to,
            "confidence": confidence,
            "reason": reason,
        }

    def _rule_trace(
        self,
        selection: ModelSelection,
        *,
        ai_route: str | None = None,
        ai_confidence: float | None = None,
        ai_model_version: str | None = None,
        notes: list[str] | None = None,
    ) -> DecisionTrace:
        return DecisionTrace(
            source="rule_router",
            applied_route_to=selection.model_alias,
            applied_reason=selection.reason,
            rule_route_to=selection.model_alias,
            ai_route_to=ai_route,
            ai_confidence=round(ai_confidence, 3) if ai_confidence is not None else None,
            ai_model_version=ai_model_version,
            notes=notes or [],
        )

    def _provider_from_model(self, model_name: str, default_provider: str = "deepseek") -> str:
        if "/" not in model_name:
            return default_provider
        provider = model_name.split("/", 1)[0].strip().lower()
        return provider or default_provider
