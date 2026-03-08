from sqlalchemy.orm import Session

from app.schemas import LLMResponse, ModelSelection, ValidationResult
from app.storage.entities import EvaluationRecord, ExecutionRecord


class ExecutionRepository:
    def create_execution(
        self,
        db: Session,
        task_id: int,
        attempt_number: int,
        selected_model: ModelSelection,
        actual_selection: ModelSelection,
        response: LLMResponse,
        fallback_triggered: bool,
        success: bool,
        error_type: str | None,
    ) -> ExecutionRecord:
        # 一次模型调用对应一条 execution，包含成本、时延和原始返回。
        execution = ExecutionRecord(
            task_id=task_id,
            attempt_number=attempt_number,
            selected_model_key=selected_model.model_alias,
            selected_model_name=selected_model.model,
            actual_model_used=actual_selection.model,
            provider=actual_selection.provider,
            model_alias=actual_selection.model_alias,
            model=actual_selection.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            estimated_cost=response.estimated_cost,
            success=success,
            error_type=error_type,
            error_message=response.error_message,
            finish_reason=response.finish_reason,
            fallback_triggered=fallback_triggered,
            used_fallback=fallback_triggered,
            raw_response_json=response.raw,
        )
        db.add(execution)
        return self._commit_and_refresh(db, execution)

    def create_evaluation(
        self,
        db: Session,
        execution_id: int,
        validation: ValidationResult,
    ) -> EvaluationRecord:
        # 校验结果单独存表，后续可以独立扩展更多质量指标。
        evaluation = EvaluationRecord(
            execution_id=execution_id,
            valid=validation.valid,
            issues_json=validation.issues,
            json_valid=validation.json_valid,
            contains_code_block=validation.contains_code_block,
            passed_basic_checks=validation.passed_basic_checks,
        )
        db.add(evaluation)
        return self._commit_and_refresh(db, evaluation)

    def _commit_and_refresh(self, db: Session, record: ExecutionRecord | EvaluationRecord):
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        db.refresh(record)
        return record
