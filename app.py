import os
import json
import time
import uuid
from typing import List, Dict

from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Optional Redis
try:
    import redis
except Exception:
    redis = None

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN", "")
REDIS_URL = os.getenv("REDIS_URL", "")
MODEL_ID = os.getenv("MODEL_ID", "gemma2-9b-it")
PORT = int(os.getenv("PORT", 8000))

print(f"Using model: {MODEL_ID}")

# Initialize Groq model (re-usable)
# We keep verify=False as in your local case; remove in prod
model = ChatGroq(model=MODEL_ID, api_key=GROQ_API_KEY, http_client=httpx.Client(verify=False))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Session storage: Redis if configured, else in-memory dict
if REDIS_URL and redis:
    redis_client = redis.from_url(REDIS_URL)
    use_redis = True
    print("Using Redis for session memory:", REDIS_URL)
else:
    redis_client = None
    use_redis = False
    print("Redis not configured or unavailable — using in-memory sessions (not persistent).")

# In-memory fallback (per-process)
in_memory_sessions: Dict[str, List[Dict]] = {}

# Config for how much history to keep (you can tune)
MAX_HISTORY_ITEMS = 40  # each item is one message (user or bot)
SYSTEM_PROMPT = "You are a helpful, concise assistant. Keep responses short and friendly."

def _get_session_key(session_id: str) -> str:
    return f"chat:session:{session_id}"

def load_history(session_id: str) -> List[Dict]:
    """Return list of messages in the form [{'role':'user'|'bot','content':...}, ...]"""
    if use_redis and redis_client:
        raw = redis_client.get(_get_session_key(session_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return []
    else:
        return in_memory_sessions.get(session_id, [])

def save_history(session_id: str, history: List[Dict]):
    # trim to MAX_HISTORY_ITEMS
    if len(history) > MAX_HISTORY_ITEMS:
        history = history[-MAX_HISTORY_ITEMS:]
    if use_redis and redis_client:
        redis_client.set(_get_session_key(session_id), json.dumps(history))
    else:
        in_memory_sessions[session_id] = history

def build_prompt_from_history(history: List[Dict], user_message: str) -> str:
    """
    Build a single prompt string that includes system prompt, recent history (User/Bot),
    and final user message, so the model receives context.
    """
    parts = [f"SYSTEM: {SYSTEM_PROMPT}", ""]
    # Include last N messages as `User:` / `Bot:`
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Bot: {content}")
    parts.append(f"User: {user_message}")
    parts.append("Bot:")  # prompt model to finish as Bot
    return "\n".join(parts)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    """
    Expect JSON:
    {
      "message": "Hello",
      "session_id": "<client-session-id>"   # optional; server returns new if missing
    }
    Response:
    {
      "reply": "...",
      "session_id": "<session-id-used>"
    }
    """
    payload = await request.json()
    user_message: str = payload.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "empty message"}, status_code=400)

    session_id: str = payload.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    # Load and update history
    history = load_history(session_id)  # list of {"role","content"}

    # Build prompt including history
    prompt = build_prompt_from_history(history, user_message)

    # Call model (we pass the prompt as a single HumanMessage)
    try:
        response_msg = model.invoke([HumanMessage(content=prompt)])
        bot_reply = response_msg.content.strip()
    except Exception as e:
        # On error, return helpful error to client (and keep session)
        print("Model error:", repr(e))
        return JSONResponse({"error": "model error", "details": str(e)}, status_code=500)

    # Append both messages to history & save
    history.append({"role": "user", "content": user_message})
    history.append({"role": "bot", "content": bot_reply})
    save_history(session_id, history)

    return {"reply": bot_reply, "session_id": session_id, "timestamp": int(time.time())}

@app.post("/session/clear")
async def clear_session(request: Request):
    payload = await request.json()
    session_id = payload.get("session_id")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    if use_redis and redis_client:
        redis_client.delete(_get_session_key(session_id))
    else:
        in_memory_sessions.pop(session_id, None)
    return {"ok": True}
