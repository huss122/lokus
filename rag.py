import io
import re

import httpx
import numpy as np
from pypdf import PdfReader

import database

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
# ~200 tokens: small enough for a sharp embedding, big enough to keep a fact with its context.
CHUNK_SIZE = 800
# ~20% overlap so a sentence on a chunk boundary is whole in at least one chunk.
CHUNK_OVERLAP = 150
EMBED_BATCH = 32
SUPPORTED_TYPES = (".pdf", ".txt", ".md")

# nomic-embed-text was trained with these task prefixes; leaving them out hurts retrieval.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class DocumentError(Exception):
    pass


def read_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    # Returns (page number, text). Text files count as a single page 1.
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as error:
            raise DocumentError(f"Could not read the PDF: {error}")
    if name.endswith((".txt", ".md")):
        return [(1, data.decode("utf-8", errors="replace"))]
    raise DocumentError(f"Unsupported file type. Use {', '.join(SUPPORTED_TYPES)}.")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Packs whole sentences into chunks, so a chunk rarely starts or ends mid-sentence.
    chunks, current = [], ""
    for piece in _sentences(text, size):
        if current and len(current) + 1 + len(piece) > size:
            chunks.append(current)
            tail = _tail(current, overlap)
            # Skip the overlap when it would push the next chunk over the size limit.
            current = tail if len(tail) + 1 + len(piece) <= size else ""
        current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def _sentences(text: str, size: int):
    for piece in re.split(r"(?<=[.!?])\s+|\n\s*\n", text):
        piece = " ".join(piece.split())
        # A "sentence" longer than a chunk (tables, lists without full stops) is cut at word boundaries.
        while len(piece) > size:
            cut = piece.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            yield piece[:cut]
            piece = piece[cut:].strip()
        if piece:
            yield piece


def _tail(chunk: str, overlap: int) -> str:
    # The end of the previous chunk, starting at a word boundary.
    tail = chunk[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 and len(chunk) > overlap else ""


def embed(texts: list[str], prefix: str) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = [prefix + t for t in texts[start:start + EMBED_BATCH]]
        try:
            response = httpx.post(
                OLLAMA + "/api/embed", json={"model": EMBED_MODEL, "input": batch}, timeout=300
            )
        except httpx.HTTPError as error:
            raise DocumentError(f"Could not reach Ollama: {error}")
        if response.status_code == 404:
            raise DocumentError(f"Embedding model missing. Run: ollama pull {EMBED_MODEL}")
        if response.status_code != 200:
            raise DocumentError(f"Embedding failed: {response.text[:200]}")
        vectors.extend(response.json()["embeddings"])
    matrix = np.array(vectors, dtype=np.float32)
    # Unit length means cosine similarity is just a dot product.
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def add_document(filename: str, data: bytes) -> dict:
    if database.get_document_by_name(filename):
        raise DocumentError(f'"{filename}" is already added. Delete it first to re-add it.')
    pages = read_pages(filename, data)

    chunks = []  # (page, position, text)
    for page, text in pages:
        for position, chunk in enumerate(split_text(text)):
            chunks.append((page, position, chunk))
    if not chunks:
        raise DocumentError("No text found. Scanned PDFs (images of text) are not supported.")

    vectors = embed([c[2] for c in chunks], DOCUMENT_PREFIX)
    document_id = database.add_document(
        filename, EMBED_MODEL, len(pages), sum(len(t) for _, t in pages),
        [(page, position, text, vector.tobytes()) for (page, position, text), vector in zip(chunks, vectors)],
    )
    return {"id": document_id, "filename": filename, "pages": len(pages), "chunks": len(chunks)}


def search(query: str, k: int = 4) -> list[dict]:
    rows = database.get_chunks(EMBED_MODEL)
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    query_vector = embed([query], QUERY_PREFIX)[0]
    scores = matrix @ query_vector
    best = np.argsort(scores)[::-1][:k]
    return [
        {"filename": rows[i]["filename"], "page": rows[i]["page"], "text": rows[i]["text"], "score": float(scores[i])}
        for i in best
    ]
