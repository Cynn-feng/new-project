from app.models import ModelRegistry
from app.schemas import ModelSelection, TaskClassification, TaskFeatures, TaskType


class RuleRouter:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def choose_model(
        self,
        classification: TaskClassification,
        features: TaskFeatures,
    ) -> ModelSelection:
        route_scores = self._route_scores(classification, features)
        best_alias, best_score = max(route_scores.items(), key=lambda item: item[1])
        reason = (
            f"Route scores favored {best_alias} ({best_score:.2f}). "
            f"code={route_scores['code_specialist']:.2f}, rag={route_scores['long_context_rag']:.2f}, "
            f"reasoning={route_scores['strong_reasoning']:.2f}, general={route_scores['fast_general']:.2f}."
        )
        return self.selection_for_alias(best_alias, reason)

    def selection_for_alias(self, alias: str, reason: str) -> ModelSelection:
        # 统一从注册表解析 provider 和真实模型名，避免规则里重复写配置。
        profile = self.registry.get(alias)
        return ModelSelection(
            model_alias=profile.alias,
            provider=profile.provider,
            model=profile.model,
            reason=reason,
            fallback_model_alias=profile.fallback_model_alias,
        )

    def _route_scores(
        self,
        classification: TaskClassification,
        features: TaskFeatures,
    ) -> dict[str, float]:
        task_bonus = 0.18 * classification.confidence
        specialized_penalty = max(
            features.code_signal,
            features.retrieval_signal,
            features.reasoning_signal,
        ) * 0.28

        fast_general_score = (
            0.58
            + features.summarization_signal * 0.18
            + (task_bonus if classification.task_type is TaskType.SUMMARIZATION else 0.0)
            - specialized_penalty
            - (0.4 if features.needs_long_context else 0.0)
        )

        return {
            "code_specialist": min(
                features.code_signal * 0.72
                + (0.1 if features.contains_code else 0.0)
                + (task_bonus if classification.task_type is TaskType.CODING else 0.0),
                1.0,
            ),
            "long_context_rag": min(
                features.retrieval_signal * 0.65
                + (0.1 if features.needs_tools else 0.0)
                + (0.28 if features.needs_long_context else 0.0)
                + (task_bonus if classification.task_type is TaskType.RAG else 0.0),
                1.0,
            ),
            "strong_reasoning": min(
                features.reasoning_signal * 0.68
                + (0.1 if features.needs_json else 0.0)
                + (task_bonus if classification.task_type is TaskType.REASONING else 0.0),
                1.0,
            ),
            "fast_general": min(max(fast_general_score, 0.18), 1.0),
        }
