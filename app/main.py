"""FastAPI application entry point for the Customer Churn Prediction API."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.routes import router


LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "fastapi_app.log"),
    ],
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready FastAPI service for predicting telecom customer churn "
        "using the trained Scikit-Learn model and preprocessing artifacts."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


UI_TEMPLATE_PATH = PROJECT_ROOT / "app" / "templates" / "index.html"


def render_interface() -> HTMLResponse:
    """Render the browser interface used to interact with the prediction API."""

    if not UI_TEMPLATE_PATH.exists():
        return HTMLResponse(
            content="<h1>Interface template not found</h1>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return HTMLResponse(
        content=UI_TEMPLATE_PATH.read_text(encoding="utf-8"),
        status_code=status.HTTP_200_OK,
    )


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> HTMLResponse:
    """Serve the customer churn prediction web interface."""

    return render_interface()


@app.get("/interface", response_class=HTMLResponse, include_in_schema=False)
async def interface() -> HTMLResponse:
    """Alias for the customer churn prediction web interface."""

    return render_interface()


@app.on_event("startup")
async def startup_event() -> None:
    """Log application startup details."""

    logger.info("Starting Customer Churn Prediction API")
    logger.info("Project root resolved to: %s", PROJECT_ROOT)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a clean response for request validation failures."""

    logger.warning(
        "Validation error for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Invalid request payload. Please verify the submitted fields.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a generic response for unexpected application errors."""

    logger.exception(
        "Unhandled error for %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected server error occurred.",
            "details": str(exc),
        },
    )