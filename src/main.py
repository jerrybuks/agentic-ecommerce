"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.routes.admin import router as admin_router
from src.data.database.connection import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="Ecommerce platform with AI agents - Admin API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(admin_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic Ecommerce API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.head("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

