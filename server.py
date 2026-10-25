import json
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import benchmark
import database

OLLAMA = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
BASE_DIR = Path(__file__).parent

app = FastAPI()
database.init_db()

# Avoids asking Ollama for capabilities on every request. Restart the server to clear it.
_capabilities_cache: dict[str, list[str]] = {}


def get_capabilities(model: str) -> list[str]:
    if model in _capabilities_cache:
        return _capabilities_cache[model]
    try:
        response = httpx.post(OLLAMA + "/api/show", json={"model": model}, timeout=10)
        response.raise_for_status()
        capabilities = response.json().get("capabilities", [])
    except httpx.HTTPError:
        # Don't cache failures: if Ollama was down we want to ask again next time.
        return []
    _capabilities_cache[model] = capabilities
    return capabilities


def supports_thinking(model: str) -> bool:
    return "thinking" in get_capabilities(model)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = DEFAULT_MODEL
    think: bool | None = None
    conversation_id: int | None = None  # None = start a new conversation


class NewConversationRequest(BaseModel):
    model: str = DEFAULT_MODEL


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/models")
def models():
    try:
        tags = httpx.get(OLLAMA + "/api/tags", timeout=10).json()
        result = []
        for item in tags.get("models", []):
            name = item["name"]
            capabilities = get_capabilities(name)
            if capabilities and "completion" not in capabilities:
                continue
            result.append({"name": name, "thinking": "thinking" in capabilities})
        return {"models": result}
    except httpx.HTTPError as error:
        return {"error": f"Could not reach Ollama: {error}"}


# --- Conversations -------------------------------------------------

@app.get("/conversations")
def get_conversations():
    return {"conversations": database.list_conversations()}


@app.post("/conversations")
def new_conversation(request: NewConversationRequest):
    # Placeholder title; replaced when the first message is saved.
    conversation_id = database.create_conversation(title="New conversation", model=request.model)
    return {"id": conversation_id}


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: int):
    return {"messages": database.get_messages(conversation_id)}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: int):
    database.delete_conversation(conversation_id)
    return {"ok": True}


# --- Benchmark --------------------------------------------------------

class BenchmarkRequest(BaseModel):
    prompt: str
    models: list[str]
    runs: int = 1
    think: bool = False


@app.post("/benchmark/run")
def benchmark_run(request: BenchmarkRequest):
    runs = max(1, min(request.runs, 5))

    def stream():
        for event in benchmark.run_benchmark(
            request.prompt, request.models, runs, request.think, supports_thinking
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/benchmark/history")
def benchmark_history():
    return {"runs": database.list_benchmark_runs()}


@app.delete("/benchmark/{batch_id}")
def benchmark_delete(batch_id: str):
    database.delete_benchmark_batch(batch_id)
    return {"ok": True}


# --- Chat ------------------------------------------------------------

CLOSE_TAG = "</think>"


def make_title(text: str) -> str:
    # Placeholder title, also the fallback if title generation fails.
    text = text.strip().replace("\n", " ")
    return text[:40] + ("…" if len(text) > 40 else "")


# Titles are an easy job, so the small fast model does it.
TITLE_MODEL = "llama3.2:3b"
TITLE_MAX_CHARS = 60
TITLE_PROMPT = (
    "You write short titles for chat conversations. "
    "Reply with only the title: 3 to 6 words, in the same language as the user's message. "
    "No quotes, no explanation, no punctuation at the end."
)


def clean_title(raw: str) -> str | None:
    # Model output is untrusted: keep one short, clean line.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    title = lines[0]
    if title.lower().startswith("title:"):
        title = title[len("title:"):]
    title = title.strip().strip("\"'`*#").strip().rstrip(".!:;,").strip()
    title = " ".join(title.split())
    if not title:
        return None
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS].rstrip() + "…"
    return title


def generate_title(question: str) -> str | None:
    payload = {
        "model": TITLE_MODEL,
        "messages": [
            {"role": "system", "content": TITLE_PROMPT},
            {"role": "user", "content": question[:1000]},  # a title doesn't need the whole question
        ],
        "stream": False,
        # Short keep_alive so llama doesn't sit in RAM next to qwen.
        "keep_alive": "2m",
        "options": {"temperature": 0, "num_predict": 24, "num_ctx": 2048},
    }
    try:
        response = httpx.post(OLLAMA + "/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json().get("message", {}).get("content", "")
    except (httpx.HTTPError, ValueError):
        return None  # keep the placeholder title
    return clean_title(raw)


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    payload = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "options": {
            # Ollama's default cap can end a thinking model before it answers. -1 = no cap.
            "num_predict": -1,
            # History + reasoning + answer must fit, or the answer gets cut off mid-way.
            "num_ctx": 8192,
        },
        # Default is 5 min. Keeps the model warm through a session instead of reloading from disk.
        "keep_alive": "30m",
    }
    if request.think is not None:
        payload["think"] = request.think

    conversation_id = request.conversation_id
    is_new = conversation_id is None
    if is_new:
        conversation_id = database.create_conversation(title="New conversation", model=request.model)

    # Only the newest message; earlier ones are already saved.
    last_user_message = messages[-1]["content"] if messages else ""
    database.add_message(conversation_id, "user", last_user_message)
    if is_new:
        # Placeholder so the sidebar shows something while the answer streams.
        database.rename_conversation(conversation_id, make_title(last_user_message))

    def stream():
        # Only expect thinking if it was requested; with think=false the model skips it.
        assume_thinking = supports_thinking(request.model) and bool(request.think)

        # First line tells the frontend which conversation this is (matters for new ones).
        yield json.dumps({"conversation_id": conversation_id}) + "\n"

        # If Ollama reports a separate thinking field, trust it. Otherwise the model writes
        # <think>...</think> into the content and we split it ourselves. If </think> never
        # arrives, everything is reclassified as the answer at the end instead of guessing
        # a split point mid-stream.
        using_native_split = False
        custom_phase = "thinking" if assume_thinking else "content"
        custom_buffer = ""
        thinking_accum = []  # what we streamed as thinking when splitting ourselves
        saw_close_tag = False
        answer_parts = []
        final_line_data = None

        try:
            with httpx.stream("POST", OLLAMA + "/api/chat", json=payload, timeout=300) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    msg = data.get("message", {}) or {}
                    native_thinking = msg.get("thinking") or ""
                    content = msg.get("content") or ""

                    if native_thinking:
                        using_native_split = True
                        yield line + "\n"
                    elif using_native_split:
                        # Ollama handles the split for this request, so pass content through as-is.
                        if content:
                            answer_parts.append(content)
                        yield line + "\n"
                    elif content:
                        if custom_phase == "content":
                            answer_parts.append(content)
                            yield json.dumps({"message": {"content": content, "thinking": None}}) + "\n"
                        else:
                            custom_buffer += content
                            idx = custom_buffer.find(CLOSE_TAG)
                            if idx != -1:
                                pre = custom_buffer[:idx]
                                if pre:
                                    thinking_accum.append(pre)
                                    yield json.dumps({"message": {"thinking": pre, "content": ""}}) + "\n"
                                # Some models leave whitespace right after the tag.
                                rest = custom_buffer[idx + len(CLOSE_TAG):].lstrip("\n ")
                                custom_buffer = ""
                                custom_phase = "content"
                                saw_close_tag = True
                                if rest:
                                    answer_parts.append(rest)
                                    yield json.dumps({"message": {"content": rest, "thinking": None}}) + "\n"
                            else:
                                # Hold back enough characters to catch a tag split across two chunks.
                                hold_back = len(CLOSE_TAG) - 1
                                if len(custom_buffer) > hold_back:
                                    emit, custom_buffer = custom_buffer[:-hold_back], custom_buffer[-hold_back:]
                                    if emit:
                                        thinking_accum.append(emit)
                                        yield json.dumps({"message": {"thinking": emit, "content": ""}}) + "\n"
                    else:
                        yield line + "\n"

                    if data.get("done"):
                        final_line_data = data
        except httpx.HTTPError as error:
            yield json.dumps({"error": f"Could not reach Ollama: {error}"}) + "\n"
            return

        if custom_buffer:
            thinking_accum.append(custom_buffer)
            custom_buffer = ""

        if (not using_native_split) and assume_thinking and not saw_close_tag:
            # No </think> ever came, so it was all answer. Tell the frontend to move it.
            full_text = "".join(thinking_accum)
            if full_text:
                yield json.dumps({"reclassify_as_answer": full_text}) + "\n"
                answer_parts = [full_text]

        # Final line (with total_duration) goes last, so the frontend knows the stream is done.
        if final_line_data is not None:
            yield json.dumps(final_line_data) + "\n"

        full_answer = "".join(answer_parts)
        if full_answer:
            database.add_message(conversation_id, "assistant", full_answer)

        # After the answer, so the user never waits and the models don't share the CPU.
        if is_new and full_answer:
            title = generate_title(last_user_message)
            if title:
                database.rename_conversation(conversation_id, title)
                yield json.dumps({"title": title}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")