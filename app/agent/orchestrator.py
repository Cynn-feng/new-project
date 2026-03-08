from functools import lru_cache

from sqlalchemy.orm import Session

from app.agent.rag_pipeline import RAGPipeline
from app.agent.tool_executor import ToolExecutor
from app.core.config import Settings, get_settings
from app.core.database import init_db
from app.core.logging import get_logger
from app.evaluator.output_validator import OutputValidator
from app.models import LiteLLMGateway, get_model_registry
from app.router import AIDecisionEngine, FallbackPolicy, FeatureExtractor, RuleRouter, TaskClassifier
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DecisionTrace,
    LLMResponse,
    LLMRequest,
    LLMUsage,
    Message,
    ModelSelection,
    TaskClassification,
    TaskFeatures,
    TaskType,
    ValidationResult,
)
from app.storage.execution_repo import ExecutionRepository
from app.storage.task_repo import TaskRepository

logger = get_logger(__name__)


class AgentOrchestrator:
    def __init__(self, settings: Settings):
        # 编排层负责把“分类、路由、调用、校验、落库”串成一条稳定流水线。
        registry = get_model_registry()
        self.settings = settings
        self.classifier = TaskClassifier()
        self.feature_extractor = FeatureExtractor()
        self.router = RuleRouter(registry)
        self.ai_decision_engine = AIDecisionEngine(settings, registry)
        self.fallback_policy = FallbackPolicy(registry)
        self.gateway = LiteLLMGateway(settings)
        self.validator = OutputValidator()
        self.task_repo = TaskRepository()
        self.execution_repo = ExecutionRepository()
        self.tool_executor = ToolExecutor()
        self.rag_pipeline = RAGPipeline()

    def handle_chat(self, payload: ChatRequest, db: Session) -> ChatResponse:
        # 先做任务理解，再决定模型，不让网关层承担业务判断。
        features = self._safe_extract_features(payload.message)
        classification = self._safe_classify(features)
        primary_selection, decision_trace = self._safe_select_model(payload.message, classification, features)
        task_record = self.task_repo.create(
            db=db,
            payload=payload,
            classification=classification,
            features=features,
            selection=primary_selection,
        )

        # 请求会按“主模型 -> fallback 链”依次尝试，直到成功且校验通过。
        attempts = [primary_selection.model_alias, *self.fallback_policy.build_chain(primary_selection.model_alias)]
        final_response = None
        final_selection = primary_selection
        final_validation = ValidationResult()
        final_execution_id: int | None = None
        fallback_triggered = False

        for attempt_number, model_alias in enumerate(attempts, start=1):
            selection = primary_selection
            if attempt_number > 1:
                fallback_triggered = True
                # fallback 的原因需要明确记录，方便后续分析路由质量。
                selection = self.router.selection_for_alias(
                    model_alias,
                    "Fallback route selected after a previous model call failed or returned invalid output.",
                )

            llm_request = self._build_llm_request(
                payload=payload,
                task_type=classification.task_type,
                features=features,
                selection=selection,
                decision_trace=decision_trace,
            )
            llm_response = self._safe_generate(llm_request)
            validation = self.validator.validate(
                task_type=classification.task_type,
                features=features,
                output=llm_response.content,
            )
            execution_success = self._execution_success(llm_response, validation)
            error_type = self._execution_error_type(llm_response, validation)
            execution = self.execution_repo.create_execution(
                db=db,
                task_id=task_record.id,
                attempt_number=attempt_number,
                selected_model=primary_selection,
                actual_selection=selection,
                response=llm_response,
                fallback_triggered=attempt_number > 1,
                success=execution_success,
                error_type=error_type,
            )
            self.execution_repo.create_evaluation(
                db=db,
                execution_id=execution.id,
                validation=validation,
            )

            final_response = llm_response
            final_selection = selection
            final_validation = validation
            final_execution_id = execution.id

            # 只要本次调用成功且输出通过基础校验，就停止继续重试。
            if execution_success:
                break

        if final_response is None:
            raise RuntimeError("Router finished without producing a response.")

        task_record = self.task_repo.mark_final(
            db=db,
            task_record=task_record,
            final_selection=final_selection,
            fallback_triggered=fallback_triggered,
        )
        usage = LLMUsage(
            prompt_tokens=final_response.prompt_tokens,
            completion_tokens=final_response.completion_tokens,
            total_tokens=final_response.total_tokens,
            estimated_cost=final_response.estimated_cost,
        )
        return ChatResponse(
            task_id=task_record.id,
            execution_id=final_execution_id,
            task_type=classification.task_type,
            selected_model=final_selection.model_alias,
            resolved_model=final_selection.model,
            provider=final_selection.provider,
            reason=final_selection.reason,
            answer=final_response.content,
            latency_ms=final_response.latency_ms,
            fallback_triggered=fallback_triggered,
            classification=classification,
            features=features,
            decision_trace=decision_trace,
            validation=final_validation,
            usage=usage,
        )

    def _build_llm_request(
        self,
        payload: ChatRequest,
        task_type: TaskType,
        features: TaskFeatures,
        selection: ModelSelection,
        decision_trace: DecisionTrace,
    ) -> LLMRequest:
        # 统一在这里组装模型输入，避免 API 层或网关层各自拼 prompt。
        messages = [Message(role="system", content=self.settings.default_system_prompt)]

        if task_type is TaskType.RAG:
            messages.append(Message(role="system", content=self._rag_context_note(payload.message)))

        if features.needs_tools:
            messages.append(Message(role="system", content=self._tool_plan_note(payload.message)))

        messages.append(Message(role="user", content=payload.message))

        temperature, max_tokens = self._generation_params(task_type)
        # 关键特征写进 metadata，后续日志、mock 网关和观测层都能直接复用。
        metadata = dict(payload.metadata)
        metadata.update({
            "task_type": task_type.value,
            "needs_json": features.needs_json,
            "needs_tools": features.needs_tools,
            "contains_code": features.contains_code,
            "code_signal": features.code_signal,
            "retrieval_signal": features.retrieval_signal,
            "reasoning_signal": features.reasoning_signal,
            "summarization_signal": features.summarization_signal,
            "decision_trace": decision_trace.model_dump(),
        })
        return LLMRequest(
            model_alias=selection.model_alias,
            provider=selection.provider,
            model=selection.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=payload.tools,
            metadata=metadata,
        )

    def _generation_params(self, task_type: TaskType) -> tuple[float, int]:
        if task_type is TaskType.CODING:
            return 0.1, 1400
        if task_type is TaskType.REASONING:
            return 0.2, 1600
        if task_type is TaskType.RAG:
            return 0.2, 1800
        if task_type is TaskType.SUMMARIZATION:
            return 0.2, 800
        return 0.3, 1000

    def _safe_extract_features(self, message: str) -> TaskFeatures:
        try:
            return self.feature_extractor.extract(message)
        except Exception as exc:
            logger.warning("Feature extraction failed, using neutral defaults: %s", exc)
            return TaskFeatures(
                input_length=len(message),
                line_count=max(message.count("\n") + 1, 1),
                question_count=message.count("?") + message.count("？"),
            )

    def _safe_classify(self, features: TaskFeatures) -> TaskClassification:
        try:
            return self.classifier.classify(features)
        except Exception as exc:
            logger.warning("Task classification failed, falling back to general_qa: %s", exc)
            return TaskClassification(
                task_type=TaskType.GENERAL_QA,
                confidence=0.4,
                reason="Task classification failed, so the request fell back to the general_qa route.",
            )

    def _safe_select_model(
        self,
        message: str,
        classification: TaskClassification,
        features: TaskFeatures,
    ) -> tuple[ModelSelection, DecisionTrace]:
        try:
            return self.ai_decision_engine.select_model(
                message=message,
                classification=classification,
                features=features,
                router=self.router,
            )
        except Exception as exc:
            logger.warning("Model routing failed, using default fallback route: %s", exc)
            selection = self._fallback_selection(
                reason="Model routing failed, so the request fell back to the default general model.",
            )
            return selection, DecisionTrace(
                source="rule_router",
                applied_route_to=selection.model_alias,
                applied_reason=selection.reason,
                rule_route_to=selection.model_alias,
                notes=["AI-assisted routing failed, so the request fell back to the default general model."],
            )

    def _safe_generate(self, request: LLMRequest) -> LLMResponse:
        try:
            return self.gateway.generate(request)
        except Exception as exc:
            logger.warning("Gateway generate raised unexpectedly for %s: %s", request.model, exc)
            return LLMResponse(
                content="",
                provider=request.provider,
                model=request.model,
                success=False,
                error_message=str(exc),
                raw={"error": str(exc)},
            )

    def _fallback_selection(self, reason: str) -> ModelSelection:
        for alias in ("fast_general", "local_fallback"):
            if self.router.registry.has(alias):
                return self.router.selection_for_alias(alias, reason)
        raise RuntimeError("No fallback model is configured in the registry.")

    def _execution_success(
        self,
        response: LLMResponse,
        validation: ValidationResult,
    ) -> bool:
        return response.success and validation.valid

    def _execution_error_type(
        self,
        response: LLMResponse,
        validation: ValidationResult,
    ) -> str | None:
        if response.success and validation.valid:
            return None

        if not response.success:
            message = (response.error_message or "").casefold()
            if "timeout" in message:
                return "timeout"
            if "rate limit" in message or "429" in message:
                return "rate_limit"
            if any(token in message for token in ("unauthorized", "forbidden", "api key", "auth")):
                return "auth_error"
            return "generation_error"

        if validation.issues:
            return validation.issues[0]
        return "validation_failed"

    def _rag_context_note(self, message: str) -> str:
        try:
            rag_context = self.rag_pipeline.prepare_context(message)
            return f"RAG context note: {rag_context.note}"
        except Exception as exc:
            logger.warning("RAG context preparation failed, continuing without context: %s", exc)
            return "RAG context note: unavailable, proceed without retrieved context."

    def _tool_plan_note(self, message: str) -> str:
        try:
            tool_plan = self.tool_executor.plan(message, needs_tools=True)
            return f"Tool planning note: {tool_plan.note}"
        except Exception as exc:
            logger.warning("Tool planning failed, continuing without tool note: %s", exc)
            return "Tool planning note: unavailable, continue without a precomputed tool plan."


@lru_cache
def get_orchestrator() -> AgentOrchestrator:
    # 测试环境下也确保建表完成，避免未触发 lifespan 时写库失败。
    init_db()
    return AgentOrchestrator(get_settings())
