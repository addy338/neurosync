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
from connectors import LocalConnectors, CloudConnectors

# Initialize the FastAPI app
app = FastAPI(title="NeuroSync Omni-AI Hub", version="1.0.0")

# Setup CORS to allow Next.js (port 3000) to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_connector = LocalConnectors()
cloud_connector = CloudConnectors()

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
    model: str = "Llama 3.2 (Local)"

class TaskResponse(BaseModel):
    id: int
    original_prompt: str
    assigned_node: str | None
    status: str
    response_text: str | None = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the NeuroSync Omni-AI Orchestration Hub API"}

@app.post("/tasks/", response_model=TaskResponse)
def create_task(request: PromptRequest, db: Session = Depends(get_db)):
    """
    Receive a complex prompt, route it to a node, and log the result.
    """
    # Route the prompt
    if request.model == "Gemini 2.5 Pro (Cloud)":
        node_name, response_text = cloud_connector.query(request.prompt)
    elif request.model == "Python Executor (Local)":
        node_name, response_text = local_connector.execute_python(request.prompt)
    else:
        node_name, response_text = local_connector.query(request.prompt)

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
