import uuid

import httpx

import database

OLLAMA = "http://localhost:11434"
MAX_TOKENS = 1024  # cap per answer, so a rambling model can't run forever


def _chat_once(model: str, prompt: str, think: bool | None, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 8192,
            "num_predict": max_tokens,
            "temperature": 0,  # same prompt -> (nearly) same answer, comparable runs
            "seed": 42,
        },
    }
    if think is not None:
        payload["think"] = think
    response = httpx.post(OLLAMA + "/api/chat", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _get_memory(model: str) -> tuple[float | None, float | None]:
    # /api/ps lists models currently loaded. "size" = total bytes the model
    # occupies, "size_vram" = how many of those bytes sit on the GPU.
    try:
        loaded = httpx.get(OLLAMA + "/api/ps", timeout=10).json().get("models", [])
    except httpx.HTTPError:
        return None, None
    for m in loaded:
        if m.get("name") == model or m.get("model") == model:
            return m.get("size", 0) / 1e6, m.get("size_vram", 0) / 1e6
    return None, None


def run_benchmark(prompt: str, models: list[str], runs: int, think: bool, supports_thinking):
    prompt = prompt.strip()
    if not prompt or not models:
        yield {"error": "Prompt and at least one model are required."}
        return

    batch_id = uuid.uuid4().hex
    try:
        for model in models:
            # Only send the think flag to models that support it.
            think_arg = think if supports_thinking(model) else None

            # Warm-up: loads the model into memory so load time doesn't
            # pollute the measured runs. Result is discarded.
            yield {"status": f"Warming up {model}…"}
            _chat_once(model, "Hi", False if think_arg is not None else None, 1)

            for run_index in range(1, runs + 1):
                yield {"status": f"{model}: run {run_index}/{runs}…"}
                data = _chat_once(model, prompt, think_arg, MAX_TOKENS)

                message = data.get("message", {}) or {}
                content = message.get("content") or ""
                thinking = message.get("thinking") or ""

                # Ollama reports all durations in nanoseconds.
                load_ns = data.get("load_duration", 0)
                prompt_ns = data.get("prompt_eval_duration", 0)
                eval_ns = data.get("eval_duration", 0)
                eval_count = data.get("eval_count", 0)

                mem_total, mem_vram = _get_memory(model)

                row = {
                    "batch_id": batch_id,
                    "model": model,
                    "prompt": prompt,
                    "think": 1 if think_arg else 0,
                    "run_index": run_index,
                    "load_ms": load_ns / 1e6,
                    "ttft_ms": (load_ns + prompt_ns) / 1e6,
                    "total_ms": data.get("total_duration", 0) / 1e6,
                    "eval_count": eval_count,
                    "tokens_per_s": eval_count / (eval_ns / 1e9) if eval_ns else 0,
                    "response_chars": len(content),
                    "thinking_chars": len(thinking),
                    "mem_total_mb": mem_total,
                    "mem_vram_mb": mem_vram,
                    "response": content,
                }
                database.add_benchmark_run(row)
                yield {"result": True}
    except httpx.HTTPError as error:
        yield {"error": f"Benchmark failed: {error}"}
        return

    yield {"done": True}