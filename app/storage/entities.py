from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TaskRecord(Base):
    # tasks 记录“用户请求和路由决策”，是一条任务的总览。
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    classification_reason: Mapped[str] = mapped_column(String(255), nullable=False)
    classification_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    features_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selected_model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    final_model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    selected_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    final_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    routing_reason: Mapped[str] = mapped_column(String(255), nullable=False)
    fallback_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    request_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ExecutionRecord(Base):
    # executions 记录每一次模型调用尝试，包括 fallback 和成本信息。
    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False, index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_model_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    actual_model_used: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fallback_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    used_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class EvaluationRecord(Base):
    # evaluations 记录输出质量检查结果，便于后续统计 pass rate。
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    execution_id: Mapped[int] = mapped_column(ForeignKey("executions.id"), nullable=False, index=True)
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    issues_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    json_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    contains_code_block: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    passed_basic_checks: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PolicyRecommendation(Base):
    # policy_recommendations 只管理建议生命周期，不自动修改任何路由代码。
    __tablename__ = "policy_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    condition: Mapped[str] = mapped_column(Text, nullable=False)
    route_to: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    review_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="ai")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PolicyAuditEvent(Base):
    # policy_audit_events 是 append-only 审计表，用于追踪治理动作时间线。
    __tablename__ = "policy_audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    recommendation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_status: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_summary: Mapped[str] = mapped_column(Text, nullable=False)
    event_details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
