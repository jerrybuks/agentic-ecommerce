"""Database connection and session management."""
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from src.config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1


def get_db_session():
    """
    Get a database session with retry logic.
    Retries up to 3 times on connection failure.
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            db = SessionLocal()
            # Test the connection with a simple query
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            return db
        except OperationalError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"[DB] Connection attempt {attempt + 1} failed, retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"[DB] All {MAX_RETRIES} connection attempts failed")
    
    raise last_error


def get_db():
    """Dependency for getting database session with retry logic."""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()

