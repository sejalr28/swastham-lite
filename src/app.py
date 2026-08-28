"""
app.py
------
FastAPI backend wrapping the Agent (safety -> tools/RAG router) with
session-based conversation history persisted in SQLite.

Run locally:
    pip install -r ../requirements.txt
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /session              -> create a new conversation session
    POST /chat                 -> send a message, get the agent's response
    GET  /history/{session_id} -> fetch full conversation history

Note: this file needs `fastapi`/`uvicorn`/`pydantic` installed. It could
not be executed inside the sandbox used to build this project (no
outbound network to pip-install them), so run it locally per the README's
"How to run" section and report back if anything doesn't match.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
from agent import Agent


agent: Optional[Agent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    db.init_db()
    agent = Agent()  # loads the RAG index + compiles tool patterns once at startup
    yield


app = FastAPI(title="Swastham-lite API", lifespan=lifespan)

# Permissive CORS for local development (e.g. a simple frontend on a
# different port). Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    mode: str
    tool_name: Optional[str] = None
    sources: List[dict] = []


class HistoryMessage(BaseModel):
    role: str
    content: str
    mode: Optional[str] = None
    created_at: str


@app.post("/session", response_model=SessionResponse)
def create_session():
    session_id = db.create_session()
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    if not db.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="unknown session_id - call POST /session first")

    db.add_message(req.session_id, role="user", content=req.message)

    resp = agent.handle(req.message)

    db.add_message(req.session_id, role="assistant", content=resp.answer, mode=resp.mode)

    return ChatResponse(
        answer=resp.answer,
        mode=resp.mode,
        tool_name=resp.tool_name,
        sources=resp.sources,
    )


@app.get("/history/{session_id}", response_model=List[HistoryMessage])
def history(session_id: str):
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")
    return db.get_history(session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
