from app.core.config import Settings, get_settings
from app.models import get_model_registry
from app.router import AIDecisionEngine, FeatureExtractor, RuleRouter, TaskClassifier
from app.schemas import DecisionTrace, ModelSelection, RouterInspectResponse, TaskClassification, TaskFeatures, TaskType


def inspect_route(message: str) -> dict:
    settings = get_settings()
    registry = get_model_registry()
    features = extract_features(message)
    classification = classify_task(message, features=features)
    selection, decision_trace = choose_model(
        message,
        classification.task_type,
        features,
        confidence=classification.confidence,
        settings=settings,
        registry=registry,
    )
    selected_model_name = registry.get(selection.model_alias).model
    return RouterInspectResponse(
        task_type=classification.task_type,
        features=features,
        selected_model_key=selection.model_alias,
        selected_model_name=selected_model_name,
        reason=selection.reason,
        decision_trace=decision_trace,
    ).model_dump()


def classify_task(message: str, features: TaskFeatures | None = None) -> TaskClassification:
    classifier = TaskClassifier()
    resolved_features = features or extract_features(message)
    return classifier.classify(resolved_features)


def extract_features(message: str) -> TaskFeatures:
    extractor = FeatureExtractor()
    return extractor.extract(message)


def choose_model(
    message: str,
    task_type: TaskType,
    features: TaskFeatures,
    *,
    confidence: float = 0.75,
    settings: Settings | None = None,
    registry=None,
) -> tuple[ModelSelection, DecisionTrace]:
    resolved_settings = settings or get_settings()
    resolved_registry = registry or get_model_registry()
    router = RuleRouter(resolved_registry)
    classification = TaskClassification(
        task_type=task_type,
        confidence=confidence,
        reason="Read-only route inspection synthesized a task classification.",
    )
    decision_engine = AIDecisionEngine(resolved_settings, resolved_registry)
    return decision_engine.select_model(
        message=message,
        classification=classification,
        features=features,
        router=router,
    )
