from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import database

# Initialize the FastAPI app
app = FastAPI(title="NeuroSync Omni-AI Hub", version="1.0.0")

# Dependency to get a database session for each request
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define the Pydantic schema for our API requests
# This ensures that any incoming JSON matches this structure
class PromptRequest(BaseModel):
    prompt: str

class TaskResponse(BaseModel):
    id: int
    original_prompt: str
    assigned_node: str | None
    status: str

@app.get("/")
def read_root():
    return {"message": "Welcome to NeuroSync - The Omni-AI Orchestration Hub"}

@app.post("/tasks/", response_model=TaskResponse)
def create_task(request: PromptRequest, db: Session = Depends(get_db)):
    """
    Receive a complex prompt, log it to the database, and begin orchestration.
    """
    # 1. Save the initial task to the database
    new_task = database.TaskLog(
        original_prompt=request.prompt,
        status="routing",
        assigned_node="pending"
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task) # Refresh to get the generated ID

    # 2. (Future) Route the prompt to the correct AI node based on complexity
    # For example: If it's a code task -> Aider/Cursor. If it's local bash -> Open Interpreter.
    
    return new_task

@app.get("/tasks/", response_model=List[TaskResponse])
def get_tasks(db: Session = Depends(get_db), limit: int = 10):
    """
    Retrieve recent tasks from the hive mind memory.
    """
    tasks = db.query(database.TaskLog).order_by(database.TaskLog.id.desc()).limit(limit).all()
    return tasks
