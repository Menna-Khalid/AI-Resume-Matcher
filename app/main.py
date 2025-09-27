from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging, os, sys
from pathlib import Path
from contextlib import asynccontextmanager
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)
from app.api.predict import router as predict_router
from app.api.health import router as health_router
from app.services.model_service import ModelService
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Create FastAPI app with custom docs location
app = FastAPI(
    title="AI Resume Matcher",
    version="1.0.0",
    docs_url="/api/docs",      # Swagger UI
    redoc_url="/api/redoc"     # ReDoc
)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global model service
model_service = ModelService()
@asynccontextmanager
async def lifespan(app):
    logger.info("Starting AI Resume Matcher...")
    await model_service.initialize()
    yield
    logger.info("Shutting down...")

app.router.lifespan_context = lifespan
# Include routers
app.include_router(predict_router, prefix="/api", tags=["predictions"])
app.include_router(health_router, prefix="/api", tags=["health"])
# Serve favicon to avoid 404 warning
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(current_dir, "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"status": "no favicon"}
# Routes for UI pages
@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("app/templates/index.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return FileResponse("app/templates/dashboard.html")
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)