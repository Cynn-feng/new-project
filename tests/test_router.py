from dataclasses import replace
import unittest
from unittest.mock import patch

from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.models import get_model_registry
from app.router import AIDecisionEngine, FeatureExtractor, RuleRouter, TaskClassifier
from app.schemas import ChatRequest, LLMResponse, ModelSelection, TaskClassification, TaskFeatures, TaskType
from app.storage.task_repo import TaskRepository


class RouterFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = FeatureExtractor()
        cls.classifier = TaskClassifier()
        cls.router = RuleRouter(get_model_registry())

    def test_code_shape_routes_without_keyword_only_matching(self) -> None:
        message = """
def normalize_items(items):
    cleaned = []
    for item in items:
        cleaned.append(item.strip())
    return cleaned
""".strip()

        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        selection = self.router.choose_model(classification, features)

        self.assertTrue(features.contains_code)
        self.assertGreaterEqual(features.code_signal, 0.5)
        self.assertEqual(classification.task_type, TaskType.CODING)
        self.assertEqual(selection.model_alias, "code_specialist")

    def test_retrieval_request_uses_rag_route(self) -> None:
        message = "Search the knowledge base and retrieve the auth section from https://example.com/spec.pdf, then cite the source."

        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        selection = self.router.choose_model(classification, features)

        self.assertTrue(features.needs_tools)
        self.assertGreaterEqual(features.retrieval_signal, 0.55)
        self.assertEqual(classification.task_type, TaskType.RAG)
        self.assertEqual(selection.model_alias, "long_context_rag")

    def test_reasoning_request_prefers_reasoning_model(self) -> None:
        message = (
            "Compare Redis and Postgres for a task queue. "
            "What trade-offs matter most for durability and ops cost? "
            "Return JSON with a short recommendation."
        )

        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        selection = self.router.choose_model(classification, features)

        self.assertTrue(features.needs_json)
        self.assertGreaterEqual(features.reasoning_signal, 0.5)
        self.assertEqual(classification.task_type, TaskType.REASONING)
        self.assertEqual(selection.model_alias, "strong_reasoning")

    def test_summary_request_stays_on_fast_general(self) -> None:
        message = (
            "请总结下面这段产品更新，提炼成三点重点，并保持简洁："
            "本周我们上线了新的权限配置页，补齐了审计日志，缩短了工单回执时间，"
            "同时修复了移动端表单提交失败的问题。"
        )

        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        selection = self.router.choose_model(classification, features)

        self.assertGreaterEqual(features.summarization_signal, 0.45)
        self.assertEqual(classification.task_type, TaskType.SUMMARIZATION)
        self.assertEqual(selection.model_alias, "fast_general")

    def test_feature_extractor_handles_blank_input_safely(self) -> None:
        features = self.extractor.extract("   ")

        self.assertEqual(features.input_length, 3)
        self.assertFalse(features.contains_code)
        self.assertEqual(features.code_signal, 0.0)

    def test_task_repository_truncates_reason_fields(self) -> None:
        init_db()
        db = SessionLocal()
        long_reason = "r" * 400
        try:
            record = TaskRepository().create(
                db=db,
                payload=ChatRequest(message="Explain routing"),
                classification=TaskClassification(
                    task_type=TaskType.GENERAL_QA,
                    confidence=0.6,
                    reason=long_reason,
                ),
                features=TaskFeatures(input_length=15),
                selection=ModelSelection(
                    model_alias="fast_general",
                    provider="openai",
                    model="openai/gpt-4o-mini",
                    reason=long_reason,
                ),
            )
        finally:
            db.close()

        self.assertEqual(len(record.classification_reason), 255)
        self.assertEqual(len(record.routing_reason), 255)

    def test_ai_decision_engine_uses_deepseek_decision_when_json_is_valid(self) -> None:
        message = "Compare Redis and Postgres for queue durability and explain the trade-offs."
        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        settings = replace(
            get_settings(),
            ai_decision_model="deepseek/deepseek-chat",
            enable_ai_decision_engine=True,
            ai_decision_min_confidence=0.4,
        )
        engine = AIDecisionEngine(settings, get_model_registry())

        with patch(
            "app.router.ai_decision_engine.LiteLLMGateway.generate",
            return_value=LLMResponse(
                content='{"route_to":"strong_reasoning","confidence":0.88,"reason":"DeepSeek prefers multi-step comparison."}',
                provider="deepseek",
                model="deepseek/deepseek-chat",
                latency_ms=21,
                success=True,
                raw={"mode": "test"},
            ),
        ):
            selection, trace = engine.select_model(
                message=message,
                classification=classification,
                features=features,
                router=self.router,
            )

        self.assertEqual(selection.model_alias, "strong_reasoning")
        self.assertEqual(trace.source, "ai_model")
        self.assertEqual(trace.applied_route_to, "strong_reasoning")
        self.assertEqual(trace.rule_route_to, "strong_reasoning")
        self.assertEqual(trace.ai_route_to, "strong_reasoning")
        self.assertEqual(trace.ai_model_version, "deepseek/deepseek-chat")

    def test_ai_decision_engine_falls_back_to_rules_when_deepseek_returns_invalid_json(self) -> None:
        message = "Search the knowledge base and retrieve the architecture document section with a source."
        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        settings = replace(
            get_settings(),
            ai_decision_model="deepseek/deepseek-chat",
            enable_ai_decision_engine=True,
        )
        engine = AIDecisionEngine(settings, get_model_registry())

        with patch(
            "app.router.ai_decision_engine.LiteLLMGateway.generate",
            return_value=LLMResponse(
                content="not-json",
                provider="deepseek",
                model="deepseek/deepseek-chat",
                latency_ms=19,
                success=True,
                raw={"mode": "test"},
            ),
        ):
            selection, trace = engine.select_model(
                message=message,
                classification=classification,
                features=features,
                router=self.router,
            )

        self.assertEqual(selection.model_alias, "long_context_rag")
        self.assertEqual(trace.source, "rule_router")
        self.assertEqual(trace.applied_route_to, "long_context_rag")
        self.assertIn("invalid decision json", " ".join(trace.notes).casefold())

    def test_ai_decision_engine_falls_back_to_rules_when_deepseek_request_fails(self) -> None:
        message = "Fix this Python function and explain the traceback."
        features = self.extractor.extract(message)
        classification = self.classifier.classify(features)
        engine = AIDecisionEngine(
            replace(
                get_settings(),
                ai_decision_model="deepseek/deepseek-chat",
                enable_ai_decision_engine=True,
                ai_decision_min_confidence=0.4,
            ),
            get_model_registry(),
        )
        with patch(
            "app.router.ai_decision_engine.LiteLLMGateway.generate",
            return_value=LLMResponse(
                content="",
                provider="deepseek",
                model="deepseek/deepseek-chat",
                latency_ms=17,
                success=False,
                error_message="forced deepseek outage",
                raw={"error": "forced deepseek outage"},
            ),
        ):
            selection, trace = engine.select_model(
                message=message,
                classification=classification,
                features=features,
                router=self.router,
            )

        self.assertEqual(selection.model_alias, "code_specialist")
        self.assertEqual(trace.source, "rule_router")
        self.assertIn("failed", " ".join(trace.notes).casefold())


if __name__ == "__main__":
    unittest.main()
