from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Define the local database URL (SQLite is perfect for this local hub)
SQLALCHEMY_DATABASE_URL = "sqlite:///../neurosync.db"

# Create the SQLAlchemy engine that connects to the database
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a sessionmaker to spawn database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all our database models (tables)
Base = declarative_base()

# --- Database Schema / Models ---

class TaskLog(Base):
    """
    Represents a single task delegated to an AI node.
    """
    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_prompt = Column(Text, nullable=False)
    assigned_node = Column(String, index=True) # e.g. "Claude", "Ollama"
    response_text = Column(Text)
    status = Column(String, default="pending") # "pending", "success", "error"

# Create the tables in the database
Base.metadata.create_all(bind=engine)
