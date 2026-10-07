import json
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:9b"
BASE_DIR = Path(__file__).parent

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = DEFAULT_MODEL
    think: bool = False


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    payload = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "think": request.think,
    }

    def stream():
        try:
            with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=300) as response:
                for line in response.iter_lines():
                    if line:
                        yield line + "\n"
        except httpx.HTTPError as error:
            yield json.dumps({"error": f"Could not reach Ollama: {error}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")