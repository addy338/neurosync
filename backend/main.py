import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Generator
from fastapi.middleware.cors import CORSMiddleware
import database
from connectors import (
    GeminiConnector, CodeSpecialistConnector, WritingSpecialistConnector,
    OllamaConnector, PythonExecutorConnector, HiveOrchestrator
)
from memory import memory

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
gemini   = GeminiConnector()           # 🧠 Orchestrator + computation
coder    = CodeSpecialistConnector()   # 💻 Code generation (Gemini 2.0 Flash)
writer   = WritingSpecialistConnector()# ✍️ Writing & analysis (Gemini 2.5 Lite)
ollama   = OllamaConnector()           # 🦙 100% local
executor = PythonExecutorConnector()   # ⚡ Arithmetic only
hive     = HiveOrchestrator()          # 🐝 Auto routing

# Model name → connector mapping
MODEL_REGISTRY = {
    "🐝 Auto Hive Mode":              hive,
    "Gemini 2.5 Flash (Cloud)":       gemini,
    "Code Specialist (Gemini 2.0)":   coder,
    "Writing Specialist (Gemini Lite)": writer,
    "Llama 3.2 (Local)":              ollama,
    "Python Executor (Local)":        executor,
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
    use_memory: bool = True  # Bug #3 fix: memory is no longer forced-on

class TaskResponse(BaseModel):
    id: int
    original_prompt: str
    assigned_node: str | None
    status: str
    response_text: str | None = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the NeuroSync Omni-AI Orchestration Hub API v2.0"}

# ─────────────────────────────────────────────
# 📦 BATCH ENDPOINT (unchanged, kept for fallback)
# Returns the full response as a single JSON object.
# Used when streaming is not needed or not supported.
# ─────────────────────────────────────────────
@app.post("/tasks/", response_model=TaskResponse)
def create_task(request: PromptRequest, db: Session = Depends(get_db)):
    """
    Receive a complex prompt, retrieve past memory, route it, and save the result.
    Returns the complete response once generation is finished.
    """
    # 🧠 Step 1: Memory Recall (RAG) — now skippable via use_memory
    past_context = memory.recall_memory(request.prompt) if request.use_memory else ""
    enriched_prompt = past_context + request.prompt if past_context else request.prompt

    # 🐝 Step 2: Routing & Execution
    connector = MODEL_REGISTRY.get(request.model, hive)
    try:
        node_name, response_text = connector.query(enriched_prompt)
    except Exception as e:
        node_name, response_text = "System-Error", f"⚠️ Unhandled failure: {e}"

    status = "error" if node_name in ("System-Error", "Ollama-Offline") else "success"

    # 🧠 Step 3: Memory Storage (error responses are filtered inside memory.py)
    memory.add_memory(request.prompt, response_text, node_name)

    # 💾 Step 4: UI Database Storage
    new_task = database.TaskLog(
        original_prompt=request.prompt,
        status=status,
        assigned_node=node_name,
        response_text=response_text
    )
    try:
        db.add(new_task)
        db.commit()
        db.refresh(new_task)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to persist task: {e}")

    return new_task


# ─────────────────────────────────────────────
# 📡 STREAMING ENDPOINT
#
# 💡 LEARNING NOTE: StreamingResponse lets FastAPI
# keep the HTTP connection open and drip data as
# it's generated. We use Server-Sent Events (SSE)
# format — each chunk is "event: X\ndata: Y\n\n".
#
# The frontend consumes this with a ReadableStream
# reader, appending each token to the chat bubble
# in real-time — exactly like ChatGPT or Claude.
#
# We persist to DB *after* the stream finishes so
# we have the complete text to store.
# ─────────────────────────────────────────────
@app.post("/tasks/stream/")
def stream_task(request: PromptRequest, db: Session = Depends(get_db)):
    """
    Streaming version of create_task. Returns an SSE stream.
    Each event is either:
      event: node  → the routing decision header (markdown)
      event: chunk → a text token to append to the bubble
      event: done  → signals stream end; data is the node name
    """
    # Recall memory before opening the stream (fast, synchronous)
    past_context = memory.recall_memory(request.prompt) if request.use_memory else ""
    enriched_prompt = past_context + request.prompt if past_context else request.prompt

    connector = MODEL_REGISTRY.get(request.model, hive)

    def generate() -> Generator[str, None, None]:
        full_text = ""
        node_name = "Unknown"

        try:
            gen = connector.stream_query(enriched_prompt)
            try:
                while True:
                    sse_chunk = next(gen)
                    # Accumulate text from chunk events for DB storage
                    if sse_chunk.startswith("event: chunk\n"):
                        token = sse_chunk.split("data: ", 1)[1].rstrip("\n")
                        full_text += token.replace("\\n", "\n")
                    elif sse_chunk.startswith("event: node\n"):
                        # Routing header — prepend to full_text too
                        header = sse_chunk.split("data: ", 1)[1].rstrip("\n")
                        full_text = header.replace("\\n", "\n") + full_text
                    elif sse_chunk.startswith("event: done\n"):
                        node_name = sse_chunk.split("data: ", 1)[1].rstrip("\n")
                    yield sse_chunk
            except StopIteration:
                pass
        except Exception as e:
            err_msg = f"⚠️ Unhandled stream failure: {e}"
            full_text = err_msg
            node_name = "System-Error"
            yield f"event: chunk\ndata: {err_msg}\n\n"
            yield f"event: done\ndata: System-Error\n\n"

        # Persist to DB after stream completes
        status = "error" if "System-Error" in node_name or "Offline" in node_name else "success"
        memory.add_memory(request.prompt, full_text, node_name)

        new_task = database.TaskLog(
            original_prompt=request.prompt,
            status=status,
            assigned_node=node_name,
            response_text=full_text
        )
        try:
            db.add(new_task)
            db.commit()
        except Exception:
            db.rollback()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind a proxy
        }
    )


@app.get("/tasks/", response_model=List[TaskResponse])
def get_tasks(db: Session = Depends(get_db), limit: int = 10):
    """
    Retrieve recent tasks from the hive mind memory.
    """
    tasks = db.query(database.TaskLog).order_by(database.TaskLog.id.desc()).limit(limit).all()
    return tasks
