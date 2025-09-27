from fastapi import APIRouter, HTTPException
import time
import logging
logger = logging.getLogger(__name__)
router = APIRouter()
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        from main import model_service
        health = await model_service.health_check()
        return {
            'status': health['status'],
            'timestamp': time.time(),
            'models_loaded': health['models_loaded'],
            'uptime': health.get('uptime', 0),
            'message': 'Service is running'
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }