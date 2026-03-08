from dataclasses import dataclass
from functools import lru_cache

from app.core.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class ModelProfile:
    # 用统一能力描述不同模型，路由层只依赖别名和能力，不直接绑死 provider SDK。
    alias: str
    provider: str
    model: str
    capabilities: tuple[str, ...]
    max_context: int
    cost_level: str
    speed_level: str
    fallback_model_alias: str | None = None


class ModelRegistry:
    def __init__(self, profiles: dict[str, ModelProfile]):
        self._profiles = profiles

    def get(self, alias: str) -> ModelProfile:
        if alias not in self._profiles:
            raise KeyError(f"Unknown model alias: {alias}")
        return self._profiles[alias]

    def has(self, alias: str) -> bool:
        return alias in self._profiles

    def list_profiles(self) -> list[ModelProfile]:
        return list(self._profiles.values())


def _provider_from_model(model: str, default: str) -> str:
    if "/" not in model:
        return default
    provider = model.split("/", 1)[0].strip().lower()
    return provider or default


def _build_registry(settings: Settings) -> ModelRegistry:
    # 这里维护的是平台内的“逻辑模型别名”，不是直接暴露给上层的供应商细节。
    return ModelRegistry(
        {
            "fast_general": ModelProfile(
                alias="fast_general",
                provider=_provider_from_model(settings.fast_general_model, default="openai"),
                model=settings.fast_general_model,
                capabilities=("chat", "summary", "classification"),
                max_context=32000,
                cost_level="low",
                speed_level="high",
                fallback_model_alias="local_fallback",
            ),
            "strong_reasoning": ModelProfile(
                alias="strong_reasoning",
                provider=_provider_from_model(settings.strong_reasoning_model, default="anthropic"),
                model=settings.strong_reasoning_model,
                capabilities=("reasoning", "analysis", "long_context"),
                max_context=200000,
                cost_level="high",
                speed_level="medium",
                fallback_model_alias="fast_general",
            ),
            "code_specialist": ModelProfile(
                alias="code_specialist",
                provider=_provider_from_model(settings.code_specialist_model, default="openai"),
                model=settings.code_specialist_model,
                capabilities=("coding", "debugging"),
                max_context=128000,
                cost_level="medium",
                speed_level="medium",
                fallback_model_alias="fast_general",
            ),
            "long_context_rag": ModelProfile(
                alias="long_context_rag",
                provider=_provider_from_model(settings.rag_model, default="anthropic"),
                model=settings.rag_model,
                capabilities=("rag", "retrieval", "tool_use", "long_context"),
                max_context=200000,
                cost_level="high",
                speed_level="medium",
                fallback_model_alias="strong_reasoning",
            ),
            "local_fallback": ModelProfile(
                alias="local_fallback",
                provider=_provider_from_model(settings.local_fallback_model, default="mock"),
                model=settings.local_fallback_model,
                capabilities=("fallback", "offline"),
                max_context=16000,
                cost_level="low",
                speed_level="high",
            ),
        }
    )


@lru_cache
def get_model_registry() -> ModelRegistry:
    return _build_registry(get_settings())
