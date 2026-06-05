from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows so Vietnamese / Unicode text prints correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_LLM_MODEL,
    GeminiEmbedder,
    GeminiLLM,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/sports_articles_multisource_clean.md",
    "data/sports_dataset.md",
    "data/web-the-thao-paper1.txt",
    "data/tin-tuc-DaNang-Dragons.txt",
    "data/espn-paper-2.txt",
    "data/espn-paper-1.txt",
    "data/articles_clean.md"
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    llm_fn = demo_llm

    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
            llm_fn = GeminiLLM(model_name=os.getenv("GEMINI_LLM_MODEL", GEMINI_LLM_MODEL))
        except Exception as exc:
            print(f"Gemini init failed ({exc}), falling back to mock.")
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"LLM backend:       {getattr(llm_fn, '_backend_name', llm_fn.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))

    # Interactive query loop — runs when no CLI argument was given
    if question is None:
        _interactive_loop(agent, store)

    return 0


def _interactive_loop(agent: KnowledgeBaseAgent, store) -> None:
    print("\n" + "=" * 60)
    print("Interactive RAG — nhập query, 'quit' hoặc Ctrl-C để thoát")
    print("=" * 60)
    while True:
        try:
            raw = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        print("\n--- Top-3 Retrieved Chunks ---")
        for i, r in enumerate(store.search(raw, top_k=3), 1):
            preview = r["content"][:150].replace("\n", " ")
            print(f"[{i}] score={r['score']:.4f}  src={r['metadata'].get('source', '?')}")
            print(f"     {preview}...")

        print("\n--- Agent Answer ---")
        print(agent.answer(raw, top_k=3))


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
