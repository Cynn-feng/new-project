from typing import Any

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str | None = Field(default=None, max_length=64)
    message: str = Field(..., min_length=1, description="User input that needs routing")
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None

    @field_validator("user_id", mode="before")
    @classmethod
    def normalize_user_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        if not isinstance(value, str):
            return value
        return value.strip()

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        return {} if value is None else value


class RouterInspectRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input that needs route inspection")

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        if not isinstance(value, str):
            return value
        return value.strip()


class LLMRequest(BaseModel):
    model_alias: str
    provider: str
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
