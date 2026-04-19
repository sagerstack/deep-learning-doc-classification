import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path

from app.src.config import (
    APP_HOST,
    APP_PORT,
    CORS_ORIGINS,
    FORCE_DEVICE,
    LOG_LEVEL,
    MONITORING_DB_PATH,
    STATIC_DIR,
)

GCN_THUMBS_DIR = Path(__file__).resolve().parent / "data" / "gcn_thumbs"
from app.src.logging_config import configure_logging
from app.src.middleware.logging import LoggingMiddleware
from app.src.monitoring.store import init_db
from app.src.routes.classify import router as classify_router
from app.src.routes.monitoring import router as monitoring_router
from app.src.services.model_registry import load_all_models

logger = logging.getLogger(__name__)


def _detect_device() -> torch.device:
    if FORCE_DEVICE:
        return torch.device(FORCE_DEVICE)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()

    # Initialise monitoring DB before model loading
    init_db(MONITORING_DB_PATH)

    device = _detect_device()
    logger.info("Starting model loading on device: %s", device)

    t0 = time.perf_counter()
    registry = load_all_models(device)
    elapsed = time.perf_counter() - t0

    application.state.registry = registry
    application.state.device = device

    num_gnn = len(registry["models"])
    logger.info(
        "Models loaded: 1 CNN baseline + %d GNN models in %.1fs on %s",
        num_gnn,
        elapsed,
        device,
    )

    yield

    del application.state.registry
    logger.info("Model registry cleaned up")


app = FastAPI(
    title="Document Classification: CNN vs GNN",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if GCN_THUMBS_DIR.exists():
    app.mount("/gcn-thumbs", StaticFiles(directory=str(GCN_THUMBS_DIR)), name="gcn-thumbs")

app.include_router(classify_router)
app.include_router(monitoring_router)
