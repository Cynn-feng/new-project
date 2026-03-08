from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()
# SQLite 需要关闭同线程限制，才能让 FastAPI 的依赖注入安全复用连接。
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    # 每个请求分配一个独立 Session，请求结束后统一关闭。
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # 延迟导入实体，避免数据库层和存储层在模块加载时循环依赖。
    from app.storage import entities  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if settings.database_url.startswith("sqlite"):
        _ensure_sqlite_execution_columns()


def _ensure_sqlite_execution_columns() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("executions"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("executions")}
    required_columns = {
        "selected_model_key": "VARCHAR(64)",
        "selected_model_name": "VARCHAR(128)",
        "actual_model_used": "VARCHAR(128)",
        "error_type": "VARCHAR(64)",
        "fallback_triggered": "BOOLEAN NOT NULL DEFAULT 0",
    }
    missing_columns = {name: ddl for name, ddl in required_columns.items() if name not in existing_columns}
    if not missing_columns:
        return

    try:
        with engine.begin() as connection:
            for column_name, ddl in missing_columns.items():
                connection.execute(text(f"ALTER TABLE executions ADD COLUMN {column_name} {ddl}"))

            connection.execute(
                text(
                    """
                    UPDATE executions
                    SET selected_model_key = COALESCE(selected_model_key, model_alias),
                        selected_model_name = COALESCE(selected_model_name, model),
                        actual_model_used = COALESCE(actual_model_used, model),
                        error_type = CASE
                            WHEN error_type IS NOT NULL THEN error_type
                            WHEN success = 0 THEN 'generation_error'
                            ELSE NULL
                        END,
                        fallback_triggered = COALESCE(fallback_triggered, used_fallback, 0)
                    """
                )
            )
    except Exception as exc:
        raise RuntimeError(
            "SQLite schema mismatch detected. Delete and recreate the local DB file: "
            f"{_sqlite_db_path_hint()}"
        ) from exc


def _sqlite_db_path_hint() -> str:
    prefix = "sqlite:///"
    if not settings.database_url.startswith(prefix):
        return settings.database_url
    raw_path = settings.database_url[len(prefix):]
    return str(Path(raw_path).resolve())
