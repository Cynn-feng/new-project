from sqlalchemy.orm import Session

from app.schemas import ChatRequest, ModelSelection, TaskClassification, TaskFeatures
from app.storage.entities import TaskRecord


class TaskRepository:
    _REASON_LIMIT = 255

    def create(
        self,
        db: Session,
        payload: ChatRequest,
        classification: TaskClassification,
        features: TaskFeatures,
        selection: ModelSelection,
    ) -> TaskRecord:
        # 先落一条任务主记录，后续执行日志和评估日志都通过 task_id 关联。
        task_record = TaskRecord(
            user_id=payload.user_id,
            user_message=payload.message,
            task_type=classification.task_type.value,
            classification_reason=self._truncate_reason(classification.reason),
            classification_confidence=classification.confidence,
            features_json=features.model_dump(),
            selected_model_alias=selection.model_alias,
            final_model_alias=selection.model_alias,
            selected_provider=selection.provider,
            final_provider=selection.provider,
            routing_reason=self._truncate_reason(selection.reason),
            fallback_triggered=False,
            request_metadata=payload.metadata,
        )
        db.add(task_record)
        return self._commit_and_refresh(db, task_record)

    def mark_final(
        self,
        db: Session,
        task_record: TaskRecord,
        final_selection: ModelSelection,
        fallback_triggered: bool,
    ) -> TaskRecord:
        # 执行结束后回填最终命中的模型，便于分析 primary route 的命中率。
        task_record.final_model_alias = final_selection.model_alias
        task_record.final_provider = final_selection.provider
        task_record.fallback_triggered = fallback_triggered
        db.add(task_record)
        return self._commit_and_refresh(db, task_record)

    def _commit_and_refresh(self, db: Session, task_record: TaskRecord) -> TaskRecord:
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        db.refresh(task_record)
        return task_record

    def _truncate_reason(self, reason: str) -> str:
        return reason[: self._REASON_LIMIT]
