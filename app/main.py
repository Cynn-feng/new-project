from contextlib import asynccontextmanager
from functools import lru_cache
from html import escape
import json
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.core.config import Settings, get_settings
from app.core.database import init_db
from app.core.logging import configure_logging

# 入口启动时先配置日志，避免后续模块初始化时丢失日志格式。
configure_logging()
settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = STATIC_DIR / "assets"
TEMPLATE_PATHS = {
    "home": STATIC_DIR / "landing.html",
    "workspace": STATIC_DIR / "workspace.html",
    "classification": STATIC_DIR / "classification.html",
    "governance": STATIC_DIR / "governance.html",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 应用启动时初始化数据库表，确保 API 首次请求即可落库。
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# 所有业务接口统一挂到可配置的 API 前缀下。
app.include_router(api_router, prefix=settings.api_prefix)
app.mount("/static", StaticFiles(directory=ASSETS_DIR), name="static")


@lru_cache
def _load_template(template_name: str) -> str:
    return TEMPLATE_PATHS[template_name].read_text(encoding="utf-8")


def _build_bootstrap(current_settings: Settings) -> str:
    mode_label = "Mock responses enabled" if current_settings.mock_llm_responses else "Live provider mode"
    return json.dumps(
        {
            "appName": current_settings.app_name,
            "apiPrefix": current_settings.api_prefix,
            "chatEndpoint": f"{current_settings.api_prefix}/chat",
            "inspectEndpoint": f"{current_settings.api_prefix}/router/inspect",
            "healthEndpoint": f"{current_settings.api_prefix}/health",
            "modelsEndpoint": f"{current_settings.api_prefix}/models",
            "adminStatsEndpoint": f"{current_settings.api_prefix}/admin/stats",
            "policySnapshotEndpoint": f"{current_settings.api_prefix}/admin/policy/snapshot",
            "policyDashboardEndpoint": f"{current_settings.api_prefix}/admin/policy/dashboard",
            "policyPortfolioEndpoint": f"{current_settings.api_prefix}/admin/policy/portfolio-report",
            "policyAuditOverviewEndpoint": f"{current_settings.api_prefix}/admin/policy/audit/overview",
            "docsPath": "/docs",
            "homePath": "/",
            "workspacePath": "/workspace",
            "classificationPath": "/classification",
            "governancePath": "/governance",
            "modeLabel": mode_label,
            "mockResponses": current_settings.mock_llm_responses,
        }
    )


def _render_page_html(template_name: str, current_settings: Settings) -> str:
    bootstrap = _build_bootstrap(current_settings)
    return (
        _load_template(template_name)
        .replace("__APP_NAME__", escape(current_settings.app_name))
        .replace("__MODE_LABEL__", escape("Mock responses enabled" if current_settings.mock_llm_responses else "Live provider mode"))
        .replace("__BOOTSTRAP__", bootstrap)
    )


@app.get("/system/info", tags=["system"])
def read_system_info() -> dict[str, str]:
    return {
        "app_name": settings.app_name,
        "api_prefix": settings.api_prefix,
        "docs": "/docs",
        "home": "/",
        "workspace": "/workspace",
        "classification": "/classification",
        "governance": "/governance",
        "console": "/console",
        "chat_endpoint": f"{settings.api_prefix}/chat",
    }


@app.get("/", tags=["site"], response_class=HTMLResponse)
def read_home(current_settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return HTMLResponse(content=_render_page_html("home", current_settings))


@app.get("/workspace", tags=["system"], response_class=HTMLResponse)
@app.get("/console", tags=["system"], response_class=HTMLResponse)
def read_workspace(current_settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return HTMLResponse(content=_render_page_html("workspace", current_settings))


@app.get("/classification", tags=["system"], response_class=HTMLResponse)
def read_classification(current_settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return HTMLResponse(content=_render_page_html("classification", current_settings))


@app.get("/governance", tags=["system"], response_class=HTMLResponse)
def read_governance(current_settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return HTMLResponse(content=_render_page_html("governance", current_settings))
