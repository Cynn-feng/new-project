import json
from time import perf_counter
import os

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.provider_adapter import ProviderAdapter
from app.schemas import LLMRequest, LLMResponse, TaskType

logger = get_logger(__name__)
_LITELLM_LOADED = False
_COMPLETION = None
_COMPLETION_COST = None


class LiteLLMGateway(ProviderAdapter):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def generate(self, request: LLMRequest) -> LLMResponse:
        started_at = perf_counter()

        # 本地开发默认走 mock，保证没有 API key 时整条链路仍可联调。
        if self._should_mock(request.provider, request.model):
            return self._mock_response(request, started_at)

        completion, completion_cost = self._load_litellm()
        if completion is None:
            return self._failure_response(
                request=request,
                started_at=started_at,
                error_message="litellm is not installed in the current environment.",
            )

        try:
            kwargs = {
                "model": request.model,
                "messages": [message.model_dump() for message in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "timeout": self.settings.litellm_timeout_seconds,
            }
            if request.tools:
                kwargs["tools"] = request.tools

            response = completion(**kwargs)
            latency_ms = max(1, int((perf_counter() - started_at) * 1000))
            content = self._extract_content(response)
            prompt_tokens = self._usage_value(response, "prompt_tokens")
            completion_tokens = self._usage_value(response, "completion_tokens")
            total_tokens = self._usage_value(response, "total_tokens")
            estimated_cost = self._estimate_cost(response)
            finish_reason = None
            if getattr(response, "choices", None):
                finish_reason = getattr(response.choices[0], "finish_reason", None)

            return LLMResponse(
                content=content,
                provider=request.provider,
                model=request.model,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                finish_reason=finish_reason,
                success=True,
                simulated=False,
                raw=self._dump_response(response),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM invocation failed for %s: %s", request.model, exc)
            return self._failure_response(
                request=request,
                started_at=started_at,
                error_message=str(exc),
            )

    def _should_mock(self, provider: str, model: str) -> bool:
        if self.settings.mock_llm_responses:
            return True
        if provider == "mock" or model.startswith("mock/"):
            return True
        return not self._has_provider_credentials(provider)

    def _has_provider_credentials(self, provider: str) -> bool:
        env_map = {
            "openai": ("OPENAI_API_KEY",),
            "anthropic": ("ANTHROPIC_API_KEY",),
            "gemini": ("GEMINI_API_KEY",),
            "google": ("GOOGLE_API_KEY",),
            "dashscope": ("DASHSCOPE_API_KEY",),
            "deepseek": ("DEEPSEEK_API_KEY",),
            "volcengine": ("ARK_API_KEY", "VOLCENGINE_API_KEY"),
        }
        env_vars = env_map.get(provider.strip().lower())
        if env_vars is None:
            return False
        return any(os.getenv(env_var) for env_var in env_vars)

    def _failure_response(self, request: LLMRequest, started_at: float, error_message: str) -> LLMResponse:
        # 失败也返回统一结构，避免上层对成功/失败写两套处理逻辑。
        latency_ms = max(1, int((perf_counter() - started_at) * 1000))
        return LLMResponse(
            content="",
            provider=request.provider,
            model=request.model,
            latency_ms=latency_ms,
            success=False,
            error_message=error_message,
            simulated=False,
            raw={"error": error_message},
        )

    def _mock_response(self, request: LLMRequest, started_at: float) -> LLMResponse:
        # mock 响应尽量保留 usage / latency 字段，方便先验证日志链路。
        latency_ms = max(1, int((perf_counter() - started_at) * 1000))
        content = self._build_mock_content(request)
        prompt_tokens = max(1, len(request.messages[-1].content.split()))
        completion_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            provider=request.provider,
            model=request.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=0.0,
            finish_reason="stop",
            success=True,
            simulated=True,
            raw={"mode": "mock"},
        )

    def _build_mock_content(self, request: LLMRequest) -> str:
        task_type = request.metadata.get("task_type", TaskType.GENERAL_QA.value)
        prompt = request.messages[-1].content.strip()

        if request.metadata.get("admin_analysis"):
            return json.dumps(
                {
                    "summary": "Mock admin analysis completed for the current platform snapshot.",
                    "observations": [
                        "Traffic and model usage were summarized from recent execution logs.",
                        "This mock response keeps the endpoint usable without live provider credentials.",
                    ],
                    "anomalies": [
                        "No live anomaly detection was run because the gateway is in mock mode."
                    ],
                    "recommendations": [
                        "Review fallback and error trends before adjusting routing policies.",
                        "Keep this endpoint read-only and use it for operational analysis only.",
                    ],
                }
            )

        if request.metadata.get("router_recommendation"):
            return json.dumps(
                {
                    "summary": "Mock router recommendation generated from recent telemetry.",
                    "recommended_changes": [
                        {
                            "priority": 1,
                            "condition": "If requests frequently fail validation due to invalid_json.",
                            "route_to": "structured_output_model",
                            "reason": "A structure-focused route would better serve JSON-constrained outputs.",
                        }
                    ],
                    "notes": [
                        "This is a read-only recommendation.",
                        "No router policy was modified automatically.",
                    ],
                }
            )

        if request.metadata.get("router_decision"):
            rule_route = request.metadata.get("rule_route_to", "fast_general")
            confidence = 0.86
            if request.metadata.get("contains_code"):
                rule_route = "code_specialist"
                confidence = 0.91
            elif request.metadata.get("needs_tools"):
                rule_route = "long_context_rag"
                confidence = 0.88
            elif float(request.metadata.get("reasoning_signal", 0.0) or 0.0) >= 0.5:
                rule_route = "strong_reasoning"
                confidence = 0.84

            return json.dumps(
                {
                    "route_to": rule_route,
                    "confidence": confidence,
                    "reason": "Mock DeepSeek routing aligned with the current request signals.",
                }
            )

        if request.metadata.get("needs_json"):
            return '{"status":"ok","route":"json","summary":"Mock JSON response generated."}'

        if task_type == TaskType.CODING.value:
            return (
                "```python\n"
                f"# simulated response for: {prompt}\n"
                "def suggested_fix():\n"
                "    return 'inspect the failing branch and add a regression test'\n"
                "```"
            )

        if task_type == TaskType.SUMMARIZATION.value:
            return f"Summary: {prompt[:120]} ..."

        if task_type == TaskType.REASONING.value:
            return (
                f"Analysis: the request '{prompt}' was routed to a stronger reasoning path "
                "because it suggests multi-step comparison or explanation."
            )

        if task_type == TaskType.RAG.value:
            return (
                f"Answer: mock retrieval flow handled '{prompt}'.\n"
                "Sources: mock-doc-001, mock-doc-002"
            )

        return f"Answer: mock general response for '{prompt}'."

    def _extract_content(self, response: object) -> str:
        if not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _usage_value(self, response: object, key: str) -> int:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        return int(getattr(usage, key, 0) or 0)

    def _estimate_cost(self, response: object) -> float:
        _, completion_cost = self._load_litellm()
        if completion_cost is None:
            return 0.0
        try:
            return float(completion_cost(completion_response=response))
        except Exception:
            return 0.0

    def _dump_response(self, response: object) -> dict:
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
            if isinstance(raw, dict):
                return raw
        return {"response_type": type(response).__name__}

    def _load_litellm(self):
        global _LITELLM_LOADED
        global _COMPLETION
        global _COMPLETION_COST

        if not _LITELLM_LOADED:
            # 延迟导入可避免纯 mock 模式下触发 LiteLLM 的额外初始化开销。
            try:
                from litellm import completion, completion_cost
            except ImportError:  # pragma: no cover
                completion = None
                completion_cost = None

            _COMPLETION = completion
            _COMPLETION_COST = completion_cost
            _LITELLM_LOADED = True

        return _COMPLETION, _COMPLETION_COST
