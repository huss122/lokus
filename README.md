# Lokus

Local, privacy-first AI document assistant and model benchmarking tool.

> **Status:** Active Development

Lokus is an offline AI assistant built with FastAPI and Ollama. It provides a local web interface for streaming responses, inspecting reasoning chains, and benchmarking model performance without relying on external APIs.

## Features

- **Local Inference:** Fully offline operation using local models via Ollama.
- **Streaming & Reasoning:** Real-time token streaming with automatic extraction of model reasoning (`<think>` blocks).
- **Benchmarking Suite:** Measures generation speed (tokens/sec), time-to-first-token (TTFT), total duration, and RAM/VRAM utilization.
- **Persistence:** Local SQLite database for chat history and benchmark logs.

## Tech Stack

- **Backend:** Python, FastAPI, HTTPX, SQLite
- **Runtime / Models:** Ollama (Qwen 3.5, Llama 3.2, etc.)
- **Frontend:** HTML, CSS, Vanilla JavaScript

## Roadmap

- [x] Core chat interface and streaming setup
- [x] Thinking/reasoning visualization
- [x] Benchmark tracking module
- [ ] RAG pipeline for local document querying
- [ ] Security evaluation suite for prompt injection testing

## Quickstart

### Prerequisites

1. Python 3.10+
2. [Ollama](https://ollama.com/) running locally:
   ```bash
   ollama pull qwen3.5:9b
   ```

### Setup

1. Install dependencies:
   ```bash
   pip install fastapi httpx uvicorn
   ```

2. Run the application:
   ```bash
   uvicorn server:app --reload
   ```

3. Open `http://127.0.0.1:8000` in your browser.