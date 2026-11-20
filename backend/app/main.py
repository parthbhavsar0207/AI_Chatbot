"""FastAPI application entrypoint for LPU CSE-AIML RAG Chatbot."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.health import router as health_router
from app.api.endpoints.retrieval import router as retrieval_router
from app.core.config import settings
from app.services.retrieval_service import RetrievalServiceManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing long-lived singleton services at startup."""
    # Initialize singleton RetrievalService (ChromaDB + FastEmbed) once at startup
    RetrievalServiceManager.get_instance().initialize()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for the LPU CSE-AIML Student Assistance RAG Chatbot.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration for frontend communication (explicit origins only; no wildcard with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
)

# Include API routers
app.include_router(health_router)
app.include_router(retrieval_router)
app.include_router(chat_router)


@app.get("/", summary="Root Index")
async def root():
    """Welcome endpoint pointing to documentation and service status."""
    return {
        "message": "Welcome to LPU CSE-AIML RAG Chatbot Backend API",
        "docs": "/docs",
        "health": "/health",
        "version": settings.app_version,
    }
