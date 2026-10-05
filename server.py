from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"
BASE_DIR = Path(__file__).parent

app = FastAPI()


class Question(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/hello/{name}")
def hello(name: str):
    return {"greeting": f"Hello {name}!"}


@app.post("/chat")
def chat(question: Question):
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": question.text}],
            "stream": False,
        },
        timeout=120,
    )
    data = response.json()
    if "message" not in data:
        return {"error": data}
    return {"answer": data["message"]["content"]}