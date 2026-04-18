from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

import backend.config  # noqa: F401 — configure logging
from backend.api.routes.classify import router as classify_router
from backend.api.routes.escalations import router as escalations_router
from backend.api.routes.health import router as health_router
from backend.api.routes.tools import router as tools_router


def create_app() -> FastAPI:
    app = FastAPI(title="BitBot API", version="0.1.0")

    origins_env = os.getenv("CORS_ORIGINS", "")
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(health_router)
    app.include_router(classify_router)
    app.include_router(tools_router)
    app.include_router(escalations_router)
    return app


app = create_app()
