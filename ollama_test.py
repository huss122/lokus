import httpx

svar = httpx.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "Vad är Python? Svara i en mening."}],
        "stream": False,
    },
    timeout=300,
)
data = svar.json()
print(data["message"]["content"])
print("tokens/s:", data["eval_count"] / (data["eval_duration"] / 1e9))
