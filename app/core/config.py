from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        os.environ.setdefault(key, _strip_wrapping_quotes(value.strip()))


_load_env_file(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True, slots=True)
class Settings:
    # 把运行时配置收敛到一个对象里，避免业务代码直接散落读取环境变量。
    app_name: str
    api_prefix: str
    debug: bool
    database_url: str
    log_level: str
    mock_llm_responses: bool
    default_system_prompt: str
    litellm_timeout_seconds: int
    enable_ai_decision_engine: bool
    ai_decision_model: str
    ai_decision_min_confidence: float
    fast_general_model: str
    strong_reasoning_model: str
    code_specialist_model: str
    rag_model: str
    local_fallback_model: str


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@lru_cache
def get_settings() -> Settings:
    # 配置对象做缓存，整个进程内复用同一份解析结果。
    return Settings(
        app_name=os.getenv("APP_NAME", "Agent Router Platform"),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        debug=_to_bool(os.getenv("DEBUG"), default=False),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./agent_router.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        mock_llm_responses=_to_bool(os.getenv("MOCK_LLM_RESPONSES"), default=True),
        default_system_prompt=os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are a precise and helpful assistant. Keep answers grounded and concise.",
        ),
        litellm_timeout_seconds=_to_int(os.getenv("LITELLM_TIMEOUT_SECONDS"), default=30),
        enable_ai_decision_engine=_to_bool(os.getenv("ENABLE_AI_DECISION_ENGINE"), default=True),
        ai_decision_model=os.getenv("AI_DECISION_MODEL", "deepseek/deepseek-chat"),
        ai_decision_min_confidence=_to_float(os.getenv("AI_DECISION_MIN_CONFIDENCE"), default=0.62),
        fast_general_model=os.getenv("FAST_GENERAL_MODEL", "openai/gpt-4o-mini"),
        strong_reasoning_model=os.getenv(
            "STRONG_REASONING_MODEL",
            "anthropic/claude-3-5-sonnet-latest",
        ),
        code_specialist_model=os.getenv("CODE_SPECIALIST_MODEL", "openai/gpt-4o-mini"),
        rag_model=os.getenv("RAG_MODEL", "anthropic/claude-3-5-sonnet-latest"),
        local_fallback_model=os.getenv("LOCAL_FALLBACK_MODEL", "mock/local-fallback"),
    )
