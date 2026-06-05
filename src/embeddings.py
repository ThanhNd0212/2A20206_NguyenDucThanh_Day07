from __future__ import annotations

import hashlib
import math
import os
import time

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


_EMBED_INTERVAL = 2.0   # seconds between embedding calls (free-tier safe)
_last_embed_call: float = 0.0


def _throttle_embed() -> None:
    """Ensure at most 1 embedding call every _EMBED_INTERVAL seconds."""
    global _last_embed_call
    gap = time.time() - _last_embed_call
    if gap < _EMBED_INTERVAL:
        time.sleep(_EMBED_INTERVAL - gap)
    _last_embed_call = time.time()


def _gemini_retry(fn, *, max_retries: int = 6, label: str = "Gemini"):
    """Call fn(), retrying on 503 (overload) and 429 (rate limit) with backoff."""
    from google.genai.errors import ClientError, ServerError

    for attempt in range(max_retries):
        try:
            return fn()
        except ServerError as exc:           # 503
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [{label}] 503 – retrying in {wait}s ({attempt+1}/{max_retries})...")
            time.sleep(wait)
        except ClientError as exc:           # 429 rate-limit; re-raise others
            if getattr(exc, "code", 0) != 429:
                raise
            if attempt == max_retries - 1:
                raise
            wait = 30 * (attempt + 1)        # 30 → 60 → 90 → 120 → 150 s
            print(f"  [{label}] 429 rate-limit – waiting {wait}s ({attempt+1}/{max_retries})...")
            time.sleep(wait)


class GeminiEmbedder:
    """Google Gemini Embeddings API-backed embedder."""

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        from google import genai

        self.model_name = model_name
        self._backend_name = model_name
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, text: str) -> list[float]:
        from google.genai import types

        _throttle_embed()
        result = _gemini_retry(
            lambda: self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            ),
            label="GeminiEmbedder",
        )
        return list(result.embeddings[0].values)


class GeminiLLM:
    """Google Gemini generative model — callable as an llm_fn for KnowledgeBaseAgent."""

    def __init__(self, model_name: str = GEMINI_LLM_MODEL) -> None:
        from google import genai

        self.model_name = model_name
        self._backend_name = model_name
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, prompt: str) -> str:
        response = _gemini_retry(
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            ),
            label="GeminiLLM",
        )
        return response.text


_mock_embed = MockEmbedder()
