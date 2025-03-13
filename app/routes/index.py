from fastapi import APIRouter, responses
import logging
from app.routes.vector_store_routes import vector_store_router
from app.routes.agent_routes import agent_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
def root() -> responses.RedirectResponse:
    """Redirect to /docs"""
    try:
        logger.info("🔄 Redirecting to /docs")
        return responses.RedirectResponse("/docs")
    except Exception as e:
        logger.error(f"❌ Error redirecting to /docs: {e}")
    
@router.get("/health")
def health() -> dict:
    """Health check endpoint"""
    try:
        logger.info("🩺 Health check")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"❌ Error in health check: {e}")
        return {"status": "error"}


# Include all routers here
router.include_router(vector_store_router, prefix="/vector_store", tags=["vector_store"])
router.include_router(agent_router, prefix="/agent", tags=["agent"])
