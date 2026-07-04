import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Bug #7 fix: the previous relative path put the DB wherever uvicorn
# happened to be launched from. Anchor it to this file's location so it's
# always <project_root>/neurosync.db regardless of working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "neurosync.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

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
