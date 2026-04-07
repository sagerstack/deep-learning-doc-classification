from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.src.config import STATIC_DIR, TEMPLATES_DIR
from app.src.routes.classify import router as classify_router


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup: model loading will be added in plan 03
    yield
    # Shutdown: cleanup will be added in plan 03


app = FastAPI(
    title="Document Classification: CNN vs GNN",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(classify_router)
