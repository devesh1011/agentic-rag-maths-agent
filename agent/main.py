from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime
import asyncio

import uuid

# Import the core agent
from agent import math_routing_agent, AgentState
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import dotenv

agent_dir = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(agent_dir, ".env"))


# DSPy feedback data being stored here
FEEDBACK_FILE_PATH = "./data/feedback.json"

math_routing_agent = None
AgentState = None


# agent init
async def init_agent():
    """Initialize the agent asynchronously to avoid asyncio conflicts"""
    global math_routing_agent, AgentState
    if math_routing_agent is None:
        from agent import math_routing_agent as agent, AgentState as AS

        math_routing_agent = agent
        AgentState = AS


# Configuration
FEEDBACK_FILE_PATH = "data/feedback.json"

# Initialize FastAPI app
app = FastAPI(title="Math Routing Agent API", version="1.0.0")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str
    original_answer: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str
    proposed_answer: Optional[str] = None


# Feedback storage function
def save_feedback(session_id: str, user_message: str, agent_answer: str, feedback: str):
    """Save user feedback to JSON file"""
    os.makedirs(os.path.dirname(FEEDBACK_FILE_PATH), exist_ok=True)

    feedback_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "agent_answer": agent_answer,
        "user_feedback": feedback,
    }

    # Load existing feedback
    try:
        with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
            # If it's a dict (old format), convert to list
            if isinstance(feedback_data, dict):
                feedback_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    feedback_data.append(feedback_entry)

    # Save back to file
    with open(FEEDBACK_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)


# API Routes
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Initialize agent if not already done
        await init_agent()

        session_id = request.session_id or str(uuid.uuid4())

        initial_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            context="",
            end=False,
            answer="",
            user_feedback="",
        )

        config = {"configurable": {"thread_id": session_id}}

        # Run the graph with interrupt handling
        try:
            final_state = None
            for step_output in math_routing_agent.stream(initial_state, config=config):
                final_state = step_output

            # If we reach here, the graph completed without interruption
            # Get the final state
            state_snapshot = math_routing_agent.get_state(config)
            final_values = state_snapshot.values

        except Exception as e:
            # Handle interrupt or other exceptions
            if "interrupt" in str(e).lower() or "graph paused" in str(e).lower():
                # Graph was interrupted for feedback
                state_snapshot = math_routing_agent.get_state(config)
                final_values = state_snapshot.values
            else:
                # Re-raise other exceptions
                raise e

        # Check if guardrails ended the conversation
        if final_values.get("end"):
            final_message = final_values["messages"][-1]
            return ChatResponse(
                response=final_message.content,
                session_id=session_id,
                status="completed",
            )

        # If we have an answer, return it directly (API mode)
        proposed_answer = final_values.get("answer", "")
        if proposed_answer:
            return ChatResponse(
                response=proposed_answer,
                session_id=session_id,
                status="completed",
                proposed_answer=proposed_answer,
            )

        # Fallback: return the last message
        final_message = final_values["messages"][-1] if final_values["messages"] else None
        if final_message:
            return ChatResponse(
                response=final_message.content,
                session_id=session_id,
                status="completed",
            )

        # If nothing else, return a generic response
        return ChatResponse(
            response="I have processed your request but couldn't generate a response.",
            session_id=session_id,
            status="completed",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback and get final response"""
    try:
        # Initialize agent if not already done
        await init_agent()

        config = {"configurable": {"thread_id": request.session_id}}

        # Get the current state to extract the original user message
        state_snapshot = math_routing_agent.get_state(config)
        current_state = state_snapshot.values

        # Find the original user message
        user_message = ""
        for msg in current_state["messages"]:
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Save feedback to file
        save_feedback(
            session_id=request.session_id,
            user_message=user_message,
            agent_answer=request.original_answer,
            feedback=request.feedback,
        )

        # Resume the graph with feedback
        math_routing_agent.invoke(Command(resume=request.feedback), config=config)

        # Get final state
        final_state_snapshot = math_routing_agent.get_state(config)
        final_values = final_state_snapshot.values

        # Extract the final answer from the conversation
        final_answer = ""
        messages = final_values.get("messages", [])
        for msg in reversed(messages):  # Get the last AI message
            if (
                hasattr(msg, "content")
                and msg.content
                and not isinstance(msg, HumanMessage)
            ):
                final_answer = msg.content
                break

        return {
            "message": "Feedback received and processed successfully",
            "user_feedback": final_values.get("user_feedback", ""),
            "session_id": request.session_id,
            "final_answer": final_answer,  # Include the final answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Math Routing Agent API"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Math Routing Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Chat with the math agent",
            "/feedback": "POST - Submit feedback for agent answer",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
