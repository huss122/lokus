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
database.init_db()  # skapar tabellerna om de inte redan finns

# Cache så vi inte behöver fråga Ollama om modellens capabilities (thinking,
# tool-stöd, m.m.) på varje enda chattförfrågan. Rensas genom att starta om
# servern, t.ex. om du laddar ner en ny version av en modell.
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


# --- Chatt ------------------------------------------------------------

CLOSE_TAG = "</think>"


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
        # Ollama lägger annars modellen i vila efter 5 minuters inaktivitet
        # (standardvärdet) och måste då ladda in den från disk igen vid
        # nästa fråga, vilket känns som att appen är "seg" jämfört med en
        # modell som redan är varm i minnet. 30 minuter håller den laddad
        # genom en hel arbetssession. Kostnaden är att den upptar
        # RAM/VRAM längre även när du inte använder den just då.
        "keep_alive": "30m",
    }
    if request.think is not None:
        payload["think"] = request.think

    conversation_id = request.conversation_id
    is_new = conversation_id is None
    if is_new:
        conversation_id = database.create_conversation(title="New conversation", model=request.model)

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

        # using_native_split blir True om Ollama SJÄLV rapporterar en separat
        # "thinking"-del (fungerar för modeller med korrekt konfigurerat stöd,
        # t.ex. standard-qwen3.5). Då litar vi helt på Ollama och kör aldrig
        # vår egen tagg-tolkning på den förfrågan.
        #
        # Om det aldrig händer tolkar vi istället själva: modellen bakar då
        # in <think>...</think> som vanlig text i "content"-fältet. Vi
        # strömmar den texten live till "tänkande"-rutan tills vi hittar
        # </think>, och växlar då till att strömma resten live som svar.
        #
        # Specialfall: om </think> ALDRIG dyker upp (modellen glömmer
        # stänga sin tagg) har vi då av misstag visat hela svaret i
        # tänkande-rutan. Det korrigeras i EFTERHAND, i en enda klumpvis
        # ombyggnad när strömmen är slut - INTE genom att gissa mitt i
        # strömmen var gränsen "borde" gå. Det var just den gissningen
        # (en godtycklig gräns i textbufferten, t.ex. efter N tecken) som
        # tidigare klippte enstaka ord mitt itu mellan de två rutorna.
        using_native_split = False
        custom_phase = "thinking" if assume_thinking else "content"
        custom_buffer = ""
        thinking_accum = []  # allt vi visat som "tänkande" via vår egen tolkning
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
                        # Ollama har redan visat sig sköta uppdelningen
                        # korrekt för den här förfrågan - lita på den även
                        # för rena content-rader (t.ex. själva svarstexten).
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
                                # Vissa modeller lämnar kvar whitespace eller
                                # enstaka skräptecken direkt efter taggen.
                                rest = custom_buffer[idx + len(CLOSE_TAG):].lstrip("\n ")
                                custom_buffer = ""
                                custom_phase = "content"
                                saw_close_tag = True
                                if rest:
                                    answer_parts.append(rest)
                                    yield json.dumps({"message": {"content": rest, "thinking": None}}) + "\n"
                            else:
                                # Håll kvar så många tecken att en tagg som
                                # delats itu mellan två chunkar inte missas.
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
            # </think> dök aldrig upp. Allt vi strömmade till tänkande-rutan
            # var i själva verket det riktiga svaret - klassa om det i en
            # enda klumpvis operation nu när vi har allt, och be frontend
            # flytta det till svarsrutan istället för att gömma det.
            full_text = "".join(thinking_accum)
            if full_text:
                yield json.dumps({"reclassify_as_answer": full_text}) + "\n"
                answer_parts = [full_text]

        # Skicka den sista raden (total_duration m.m.) sist, så frontend vet
        # att allt är klart och kan visa tiden.
        if final_line_data is not None:
            yield json.dumps(final_line_data) + "\n"

        full_answer = "".join(answer_parts)
        if full_answer:
            database.add_message(conversation_id, "assistant", full_answer)

    return StreamingResponse(stream(), media_type="application/x-ndjson")