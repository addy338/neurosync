import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import database
from connectors import (
    GeminiConnector, OpenAIConnector, ClaudeConnector,
    OllamaConnector, PythonExecutorConnector, HiveOrchestrator
)

# Initialize the FastAPI app
app = FastAPI(title="NeuroSync Omni-AI Hub", version="2.0.0")

# Setup CORS to allow Next.js (port 3000) to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate all connectors once at startup
gemini = GeminiConnector()
openai_conn = OpenAIConnector()
claude = ClaudeConnector()
ollama = OllamaConnector()
executor = PythonExecutorConnector()
hive = HiveOrchestrator()

# Model name → connector mapping
MODEL_REGISTRY = {
    "Gemini 2.5 Flash (Cloud)": gemini,
    "GPT-4o (Cloud)": openai_conn,
    "Claude Sonnet (Cloud)": claude,
    "Llama 3.2 (Local)": ollama,
    "Python Executor (Local)": executor,
    "🐝 Auto Hive Mode": hive,
}

# Dependency to get a database session for each request
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic schema for the incoming request
class PromptRequest(BaseModel):
    prompt: str
    model: str = "🐝 Auto Hive Mode"

class TaskResponse(BaseModel):
    id: int
    original_prompt: str
    assigned_node: str | None
    status: str
    response_text: str | None = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the NeuroSync Omni-AI Orchestration Hub API v2.0"}

@app.post("/tasks/", response_model=TaskResponse)
def create_task(request: PromptRequest, db: Session = Depends(get_db)):
    """
    Receive a complex prompt, route it to the correct node, and log the result.
    """
    connector = MODEL_REGISTRY.get(request.model, hive)
    node_name, response_text = connector.query(request.prompt)

    # Save to database
    new_task = database.TaskLog(
        original_prompt=request.prompt,
        status="success",
        assigned_node=node_name,
        response_text=response_text
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task) 
    
    return new_task

@app.get("/tasks/", response_model=List[TaskResponse])
def get_tasks(db: Session = Depends(get_db), limit: int = 10):
    """
    Retrieve recent tasks from the hive mind memory.
    """
    tasks = db.query(database.TaskLog).order_by(database.TaskLog.id.desc()).limit(limit).all()
    return tasks
