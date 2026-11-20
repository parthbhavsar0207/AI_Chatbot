"""Health check endpoint handler."""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    service: str
    version: str
    environment: str
    timestamp: str


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
async def health_check() -> HealthResponse:
    """Returns the operational status of the backend API service."""
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
