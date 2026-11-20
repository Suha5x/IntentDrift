import sys
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.temporal_analysis import TemporalDriftAnalyzer
from backend.request_models import AnalyzeRequest, AnalyzeResponse, ResetRequest


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

app = FastAPI(
    title="IntentDrift API",
    description="API for detecting goal drift in autonomous AI agents",
    version="1.0.0"
)

# Allow the React frontend to connect later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("IntentDrift model loaded successfully!")


# Store separate monitoring sessions
sessions = {}


# --------------------------------------------------
# HOME / HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "IntentDrift",
        "message": "IntentDrift API is running",
        "status": "active"
    }


# --------------------------------------------------
# ANALYZE AGENT ACTION
# --------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_action(request: AnalyzeRequest):

    session_id = request.session_id

    # Create a new monitoring session if necessary
    if session_id not in sessions:

        goal_embedding = model.encode(
            request.goal,
            convert_to_tensor=True
        )

        sessions[session_id] = {
            "goal": request.goal,
            "goal_embedding": goal_embedding,
            "analyzer": TemporalDriftAnalyzer(),
            "history": []
        }

    session = sessions[session_id]

    # Do not allow a different goal in an existing session
    if session["goal"] != request.goal:
        raise HTTPException(
            status_code=400,
            detail="This session already has a different goal. Reset it or use a new session_id."
        )

    # Convert the agent action into an embedding
    action_embedding = model.encode(
        request.action,
        convert_to_tensor=True
    )

    # Calculate semantic similarity
    similarity = float(
        util.cos_sim(
            session["goal_embedding"],
            action_embedding
        )[0][0].item()
    )

    # Calculate instantaneous drift
    instantaneous_drift = max(
        0.0,
        min(1.0, 1.0 - similarity)
    )

    # Perform temporal drift analysis
    analysis = session["analyzer"].analyze(
        instantaneous_drift
    )

    # Save action history
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goal": request.goal,
        "action": request.action,
        "action_type": request.action_type,
        "similarity": similarity,
        "instantaneous_drift": instantaneous_drift,
        **analysis
    }

    session["history"].append(record)

    return {
        "session_id": session_id,
        "goal": request.goal,
        "action": request.action,
        "similarity": similarity,
        "instantaneous_drift": instantaneous_drift,
        "recent_mean": analysis["recent_mean"],
        "persistence": analysis["persistence"],
        "cumulative_drift": analysis["cumulative"],
        "drift_score": analysis["drift_score"],
        "status": analysis["status"],
        "intervention": analysis["intervention"]
    }


# --------------------------------------------------
# GET ACTION HISTORY
# --------------------------------------------------

@app.get("/history")
def get_history(session_id: str = "default"):

    if session_id not in sessions:
        return {
            "session_id": session_id,
            "total_actions": 0,
            "history": []
        }

    history = sessions[session_id]["history"]

    return {
        "session_id": session_id,
        "total_actions": len(history),
        "history": history
    }


# --------------------------------------------------
# GET MONITORING STATISTICS
# --------------------------------------------------

@app.get("/stats")
def get_stats(session_id: str = "default"):

    if session_id not in sessions:
        return {
            "session_id": session_id,
            "total_actions": 0,
            "status_counts": {},
            "average_drift_score": 0.0
        }

    history = sessions[session_id]["history"]

    status_counts = {}

    for record in history:
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    average_score = (
        sum(record["drift_score"] for record in history) / len(history)
        if history else 0.0
    )

    return {
        "session_id": session_id,
        "total_actions": len(history),
        "status_counts": status_counts,
        "average_drift_score": average_score
    }


# --------------------------------------------------
# RESET MONITORING SESSION
# --------------------------------------------------

@app.post("/reset")
def reset_session(request: ResetRequest):

    session_id = request.session_id

    if session_id in sessions:
        del sessions[session_id]

    return {
        "session_id": session_id,
        "message": "Monitoring session reset successfully"
    }