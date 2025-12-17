"""Main FastAPI application."""
import sys
from pathlib import Path

# Add src to path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.routes.admin import router as admin_router
from src.routes.user import router as user_router
from data.database.connection import engine, Base
# Import order models to ensure tables are created
from data.database.order_models import Order, OrderItem, Voucher, ShippingInfo
from src.indexing.embeddings import EmbeddingStore

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="Ecommerce platform with AI agents - Admin & Query API"
)

# Initialize vector stores at startup
print("Initializing vector stores...")
handbook_store = EmbeddingStore(
    persist_directory="data/vector_store",
    collection_name="general_handbook",
    clear_existing=False
)
products_store = EmbeddingStore(
    persist_directory="data/vector_store",
    collection_name="products",
    clear_existing=False
)
# Pre-load vector stores to avoid initialization delay on first query
handbook_vectorstore = handbook_store.get_vectorstore()
products_vectorstore = products_store.get_vectorstore()
print("✓ Vector stores initialized")

# Store vector stores in app state for reuse
app.state.handbook_vectorstore = handbook_vectorstore
app.state.products_vectorstore = products_vectorstore

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
app.include_router(user_router)


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

