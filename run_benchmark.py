"""
Chạy similarity predictions + 5 benchmark queries trên sports corpus.
Usage: python run_benchmark.py
"""
from __future__ import annotations
import io, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(override=False)

from src import (
    Document, EmbeddingStore, RecursiveChunker,
    GeminiEmbedder, GeminiLLM, KnowledgeBaseAgent, compute_similarity,
)

FILES = [
    ("data/sports_articles_multisource_clean.md", "vi"),
    ("data/sports_dataset.md",                    "vi"),
    ("data/web-the-thao-paper1.txt",              "vi"),
    ("data/tin-tuc-DaNang-Dragons.txt",           "vi"),
    ("data/espn-paper-2.txt",                     "en"),
    ("data/espn-paper-1.txt",                     "en"),
]

SIMILARITY_PAIRS = [
    ("Alcaraz vô địch Roland Garros 2025.",
     "Carlos Alcaraz bảo vệ danh hiệu tại Roland Garros.",
     "HIGH"),
    ("Thunder vô địch NBA 2025.",
     "Leon Marchand phá kỷ lục bơi lội.",
     "LOW"),
    ("Verstappen vô địch F1 lần thứ tư.",
     "Verstappen giành chức vô địch thế giới F1.",
     "HIGH"),
    ("Da Nang Dragons thi đấu tại ABL.",
     "Alcaraz thắng Sinner trong trận chung kết dài nhất lịch sử.",
     "LOW"),
    ("Chunking chia tài liệu thành các phần nhỏ hơn.",
     "Segmentation splits text into smaller sections.",
     "HIGH"),
]

QUERIES = [
    ("Q1",
     "Đối thủ cuối cùng của tuyển Việt Nam ở Asian Cup 2027 là đội nào và họ thắng Lebanon với tỷ số bao nhiêu?",
     "Yemen là đối thủ cuối cùng; Yemen thắng Lebanon 2-0."),
    ("Q2",
     "Knicks thắng Game 1 NBA Finals 2026 trước Spurs với tỷ số bao nhiêu và họ đã lội ngược dòng cách biệt bao nhiêu điểm?",
     "Knicks thắng 105-95 sau khi bị dẫn 14 điểm."),
    ("Q3",
     "Jannik Sinner vô địch Roland Garros 2026 sau khi thắng ai và hoàn tất cột mốc Grand Slam nào?",
     "Sinner thắng Carlos Alcaraz 3-1 và hoàn tất Career Grand Slam."),
    ("Q4",
     "Nguồn sinh khí mới nào có thể giúp Danang Dragons tạo bất ngờ trước Saigon Heat?",
     "Karachi Edo."),
    ("Q5",
     "Tay vợt 17 tuổi nào thắng Marin Cilic ở vòng một Roland Garros 2026?",
     "Moise Kouame."),
]

out_lines: list[str] = []

def log(line: str = "") -> None:
    print(line)
    out_lines.append(line)

embedder = GeminiEmbedder()
llm      = GeminiLLM()

# ── Similarity Pairs ──────────────────────────────────────────────────────────
log("=" * 70)
log("SECTION 5 — SIMILARITY PREDICTIONS (gemini-embedding-2)")
log("=" * 70)
log("")

sim_results = []
for i, (sa, sb, pred) in enumerate(SIMILARITY_PAIRS, 1):
    va = embedder(sa)
    vb = embedder(sb)
    score = compute_similarity(va, vb)
    correct = (pred == "HIGH" and score >= 0.7) or (pred == "LOW" and score < 0.7)
    status = "OK" if correct else "WRONG"
    log(f"Pair {i}: {score:.4f}  pred={pred}  [{status}]")
    log(f"  A: {sa}")
    log(f"  B: {sb}")
    log("")
    sim_results.append((i, sa, sb, pred, score, status))

# ── Load & chunk ──────────────────────────────────────────────────────────────
log("")
log("=" * 70)
log("SECTION 6 — BENCHMARK QUERIES")
log("=" * 70)
log("")
log("=== Loading & chunking documents ===")

chunker = RecursiveChunker(chunk_size=200)
docs: list[Document] = []
for path_str, lang in FILES:
    path = Path(path_str)
    if not path.exists():
        log(f"  SKIP (missing): {path_str}")
        continue
    text = path.read_text(encoding="utf-8")
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            id=f"{path.stem}_c{i}",
            content=chunk,
            metadata={"source": path.name, "lang": lang},
        ))
    log(f"  {path.name}: {len(text):,} chars → {len(chunks)} chunks")

log(f"\nTotal chunks to embed: {len(docs)}")
log(f"Estimated time (~2s/call): {len(docs)*2//60} min {len(docs)*2%60} s\n")

# ── Embed ─────────────────────────────────────────────────────────────────────
log("=== Embedding (throttled at 30 RPM) ===")
store = EmbeddingStore("sports_bench", embedding_fn=embedder)

t0 = time.time()
store.add_documents(docs)
elapsed = time.time() - t0
log(f"\nEmbedded {store.get_collection_size()} chunks in {elapsed:.0f}s")

# ── Benchmark ─────────────────────────────────────────────────────────────────
agent = KnowledgeBaseAgent(store=store, llm_fn=llm)

log("")
log("─" * 70)
log("BENCHMARK RESULTS — RecursiveChunker(chunk_size=200) + gemini-embedding-2")
log("─" * 70)

for qid, query, gold in QUERIES:
    log(f"\n{'─'*70}")
    log(f"[{qid}] {query}")
    log(f"Gold answer: {gold}")
    log("")

    results = store.search(query, top_k=3)
    for i, r in enumerate(results, 1):
        preview = r["content"][:200].replace("\n", " ")
        log(f"  [{i}] score={r['score']:.4f}  src={r['metadata']['source']}")
        log(f"       {preview}...")
        log("")

    log("Agent answer:")
    answer = agent.answer(query, top_k=3)
    log(answer)
    log("")

log("=" * 70)
log(f"Benchmark complete. Total chunks in store: {store.get_collection_size()}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("report/benchmark_results.txt")
out_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"\nResults saved to {out_path}")
