from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.models import LiteLLMGateway
from app.schemas import LLMRequest, Message


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    content: str
    latency_ms: int
    model_used: str
    fallback_triggered: bool
    success: bool
    error_type: str | None = None


def call_model(
    prompt: str,
    *,
    model_name: str = "gpt-4o-mini",
    system_prompt: str | None = None,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    metadata: dict[str, Any] | None = None,
) -> ModelCallResult:
    settings = get_settings()
    gateway = LiteLLMGateway(settings)
    normalized_primary = _normalize_model_name(model_name)
    attempts = [normalized_primary]

    fallback_model = settings.local_fallback_model
    normalized_fallback = _normalize_model_name(fallback_model) if fallback_model else None
    if normalized_fallback and normalized_fallback != normalized_primary:
        attempts.append(normalized_fallback)

    last_error_type: str | None = None
    last_latency_ms = 0
    fallback_triggered = False
    request_metadata = dict(metadata or {})

    for index, attempt_model in enumerate(attempts):
        provider = _provider_from_model(attempt_model)
        request = LLMRequest(
            model_alias=attempt_model,
            provider=provider,
            model=attempt_model,
            messages=[
                Message(
                    role="system",
                    content=system_prompt or "You are a precise and helpful assistant.",
                ),
                Message(role="user", content=prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=request_metadata,
        )
        response = gateway.generate(request)
        last_latency_ms = response.latency_ms
        last_error_type = _error_type_from_response(response)
        if response.success and response.content.strip():
            return ModelCallResult(
                content=response.content,
                latency_ms=response.latency_ms,
                model_used=response.model,
                fallback_triggered=fallback_triggered,
                success=True,
                error_type=None,
            )
        if index == 0 and len(attempts) > 1:
            fallback_triggered = True

    return ModelCallResult(
        content="",
        latency_ms=last_latency_ms,
        model_used=attempts[-1],
        fallback_triggered=fallback_triggered,
        success=False,
        error_type=last_error_type or "generation_error",
    )


def _normalize_model_name(model_name: str, default_provider: str = "openai") -> str:
    normalized = model_name.strip()
    if "/" in normalized:
        return normalized
    return f"{default_provider}/{normalized}"


def _provider_from_model(model_name: str, default_provider: str = "openai") -> str:
    if "/" not in model_name:
        return default_provider
    provider = model_name.split("/", 1)[0].strip().lower()
    return provider or default_provider


def _error_type_from_response(response) -> str | None:
    if response.success and response.content.strip():
        return None

    if response.success and not response.content.strip():
        return "empty_response"

    message = (response.error_message or "").casefold()
    if "timeout" in message:
        return "timeout"
    if "rate limit" in message or "429" in message:
        return "rate_limit"
    if any(token in message for token in ("unauthorized", "forbidden", "api key", "auth")):
        return "auth_error"
    return "generation_error"
