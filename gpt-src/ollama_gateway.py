import os
import requests
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# ===== 設定（用環境變數覆寫）=====
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")  # 你的 Ollama API
API_KEY = os.getenv("API_KEY", "YKH-415")                        # 你的 API Key
API_KEY_HEADER_NAME = os.getenv("API_KEY_HEADER", "MYAPI-Key")    # 用哪個 header 傳 Key
# =================================

# Swagger / OpenAPI 基本資訊
app = FastAPI(
    title="Ollama Gateway (with API Key & Swagger)",
    version="1.1.0",
    description="A thin proxy in front of Ollama with API key authentication, CORS and Swagger UI.",
)

# CORS（上線請改為白名單）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生產環境建議改成你的網域白名單
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全性：API Key Header（Swagger 會出現 Authorize 按鈕）
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

def require_api_key(x_api_key: Optional[str] = Depends(api_key_header)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ---------- Pydantic 請求模型 ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None
    stream: Optional[bool] = False  # 本 gateway 先回非串流

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None
    stream: Optional[bool] = False

class PullRequest(BaseModel):
    name: str      # 例如 "gpt-oss:20b"
    stream: Optional[bool] = False
# --------------------------------------

def forward_json(method: str, path: str, payload: Optional[dict] = None):
    url = f"{OLLAMA_BASE}{path}"
    try:
        if method == "POST":
            r = requests.post(url, json=payload, timeout=600)
        else:
            r = requests.get(url, timeout=120)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@app.get("/health", summary="Gateway health")
def health():
    return {"ok": True, "ollama_base": OLLAMA_BASE}

@app.get("/models", summary="List local models (proxy to /api/tags)", dependencies=[Depends(require_api_key)])
def list_models():
    return forward_json("GET", "/api/tags")

@app.post("/pull", summary="Pull a model (proxy to /api/pull)", dependencies=[Depends(require_api_key)])
def pull_model(body: PullRequest):
    return forward_json("POST", "/api/pull", body.dict(exclude_none=True))

@app.post("/chat", summary="Chat (proxy to /api/chat)", dependencies=[Depends(require_api_key)])
def chat(body: ChatRequest):
    return forward_json("POST", "/api/chat", body.dict(exclude_none=True))

@app.post("/generate", summary="Single-turn generate (proxy to /api/generate)", dependencies=[Depends(require_api_key)])
def generate(body: GenerateRequest):
    return forward_json("POST", "/api/generate", body.dict(exclude_none=True))
