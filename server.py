import json
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import database

OLLAMA = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
BASE_DIR = Path(__file__).parent

app = FastAPI()
database.init_db()  # skapar tabellerna om de inte redan finns

# Cache så vi inte behöver fråga Ollama om modellens capabilities (thinking,
# tool-stöd, m.m.) på varje enda chattförfrågan. Rensas genom att starta om
# servern, t.ex. om du laddar ner en ny version av en modell.
_capabilities_cache: dict[str, list[str]] = {}


def get_capabilities(model: str) -> list[str]:
    if model not in _capabilities_cache:
        try:
            info = httpx.post(OLLAMA + "/api/show", json={"model": model}, timeout=10).json()
            _capabilities_cache[model] = info.get("capabilities", [])
        except httpx.HTTPError:
            _capabilities_cache[model] = []
    return _capabilities_cache[model]


def supports_thinking(model: str) -> bool:
    return "thinking" in get_capabilities(model)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = DEFAULT_MODEL
    think: bool | None = None
    conversation_id: int | None = None  # None = ny konversation ska skapas


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


# --- Konversationer -------------------------------------------------

@app.get("/conversations")
def get_conversations():
    return {"conversations": database.list_conversations()}


@app.post("/conversations")
def new_conversation(request: NewConversationRequest):
    # Titeln sätts till ett platshållarvärde och byts ut när första
    # meddelandet sparas (se spara-logiken i /chat nedan).
    conversation_id = database.create_conversation(title="Ny konversation", model=request.model)
    return {"id": conversation_id}


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: int):
    return {"messages": database.get_messages(conversation_id)}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: int):
    database.delete_conversation(conversation_id)
    return {"ok": True}


# --- Chatt ------------------------------------------------------------

def split_thinking(full_text: str) -> tuple[str, str]:
    """
    Delar en komplett textsträng i (tankedel, svarsdel) baserat på en
    </think>-tagg, OM en sådan finns. Körs först när HELA svaret är i hand,
    inte textbit för textbit under strömningen - en modell som inte
    konsekvent skriver ut sin stängningstagg gör det omöjligt att veta i
    förväg var gränsen går, och att gissa mitt i strömmen riskerade att
    klippa texten (och till och med enskilda ord) mitt itu.
    """
    idx = full_text.find("</think>")
    if idx == -1:
        return "", full_text
    thinking_text = full_text[:idx]
    # Vissa modeller lämnar whitespace eller enstaka skräptecken direkt
    # efter stängningstaggen - trimma bort det så svaret börjar rent.
    answer_text = full_text[idx + len("</think>"):].lstrip("\n ")
    return thinking_text, answer_text


def make_title(text: str) -> str:
    # Enkel titel: de första ~40 tecknen av användarens första meddelande.
    text = text.strip().replace("\n", " ")
    return text[:40] + ("…" if len(text) > 40 else "")


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    payload = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "options": {
            # Ollamas standardgräns för genererad text är ofta så låg (~128
            # tokens) att en tankemodell hinner klart med resonemanget men
            # aldrig når fram till själva svaret. -1 betyder "ingen gräns,
            # modellen stoppar själv när den är klar".
            "num_predict": -1,
            # Kontextfönstret (historik + resonemang + svar, allt ihop).
            # Ollamas standard (ofta 2048-4096) kan fyllas fort av en
            # tankemodell med lång chain-of-thought, särskilt när historiken
            # växer. Nås gränsen kapas generering mitt i, vilket ser ut som
            # ett slumpmässigt avbrott. Höj den här om du har RAM/VRAM för
            # det och fortfarande ser avbrutna svar.
            "num_ctx": 8192,
        },
    }
    if request.think is not None:
        payload["think"] = request.think

    conversation_id = request.conversation_id
    is_new = conversation_id is None
    if is_new:
        conversation_id = database.create_conversation(title="Ny konversation", model=request.model)

    # Spara användarens senaste meddelande (det sista i listan) direkt.
    last_user_message = messages[-1]["content"] if messages else ""
    database.add_message(conversation_id, "user", last_user_message)
    if is_new:
        # Döp om konversationen baserat på det första meddelandet.
        conn = database.get_connection()
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (make_title(last_user_message), conversation_id),
        )
        conn.commit()
        conn.close()

    def stream():
        # Viktigt: det räcker inte att modellen KAN tänka (supports_thinking).
        # Om think=false faktiskt skickades med den här förfrågan hoppar
        # modellen ofta över hela <think>...</think>-steget och svarar direkt.
        assume_thinking = supports_thinking(request.model) and bool(request.think)

        # Skicka med conversation_id som FÖRSTA raden så frontend vet
        # vilken konversation den nu pratar med (viktigt för nya konversationer).
        yield json.dumps({"conversation_id": conversation_id}) + "\n"

        if assume_thinking:
            # Eftersom vi inte längre strömmar tanketexten live (se nedan)
            # skulle frontend annars bara visa "..." under hela väntetiden.
            # Den här signalen är separat från "message.thinking" så den
            # inte råkar blandas in i den riktiga resonemangstexten senare.
            yield json.dumps({"status": "thinking"}) + "\n"

        raw_content_parts = []  # allt content vi fått från Ollama, oklippt
        final_line_data = None  # den sista raden (har total_duration m.m.)

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
                        # Ollama separerar redan tanke/svar åt oss (fungerar
                        # för modeller med korrekt konfigurerat stöd, t.ex.
                        # standard-qwen3.5). Då finns ingen anledning att vi
                        # själva gissar något - skicka vidare direkt, live.
                        yield line + "\n"

                    if content:
                        raw_content_parts.append(content)

                    if data.get("done"):
                        final_line_data = data
        except httpx.HTTPError as error:
            yield json.dumps({"error": f"Could not reach Ollama: {error}"}) + "\n"
            return

        full_content = "".join(raw_content_parts)

        if assume_thinking:
            # Nu har vi HELA svaret i hand - dela upp det i ett enda steg
            # istället för att gissa oss fram bit för bit medan det strömmar.
            thinking_text, answer_text = split_thinking(full_content)
        else:
            thinking_text, answer_text = "", full_content

        if thinking_text:
            yield json.dumps({"message": {"thinking": thinking_text, "content": ""}}) + "\n"
        if answer_text:
            yield json.dumps({"message": {"content": answer_text, "thinking": None}}) + "\n"

        # Skicka den sista raden (total_duration m.m.) sist, så frontend vet
        # att allt är klart och kan visa tiden.
        if final_line_data is not None:
            yield json.dumps(final_line_data) + "\n"

        if answer_text:
            database.add_message(conversation_id, "assistant", answer_text)

    return StreamingResponse(stream(), media_type="application/x-ndjson")