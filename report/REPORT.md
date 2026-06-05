# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Thành
**Nhóm:** Đặng Minh Hải - 2A202600713; Hoàng Phúc Quân - 2A202600560; Nguyễn Đức Thành - 2A202600838 
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai văn bản có cosine similarity cao khi các vector embedding của chúng hướng về cùng một phương trong không gian nhiều chiều, tức là chúng biểu đạt ý nghĩa hoặc chủ đề tương tự nhau, bất kể độ dài văn bản.

**Ví dụ HIGH similarity:**
- Sentence A: "Alcaraz vô địch Roland Garros 2025 sau trận chung kết kéo dài 5 tiếng."
- Sentence B: "Carlos Alcaraz bảo vệ thành công danh hiệu tại Roland Garros năm 2025."
- Tại sao tương đồng: Cùng chủ thể (Alcaraz), cùng sự kiện (Roland Garros 2025), cùng kết quả (vô địch) → embedding sẽ gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Oklahoma City Thunder vô địch NBA 2025 với Shai Gilgeous-Alexander xuất sắc."
- Sentence B: "Leon Marchand phá kỷ lục thế giới 200m hỗn hợp tại World Championships 2025."
- Tại sao khác: Khác hoàn toàn về môn thể thao (bóng rổ vs bơi lội), chủ thể và bối cảnh → embedding xa nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector, không bị ảnh hưởng bởi độ dài vector — hai đoạn văn cùng nội dung nhưng khác độ dài vẫn có cosine gần 1. Euclidean distance lại bị ảnh hưởng bởi magnitude, khiến văn bản dài luôn cho khoảng cách lớn hơn dù nội dung giống nhau.

---

### Chunking Math (Ex 1.2)

**Document 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Theo công thức:
```
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
           = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23 chunks
```

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào?**
```
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = ceil(24.75)
           = 25 chunks
```

> Tăng overlap làm tăng số lượng chunks (từ 23 → 25) vì mỗi bước tiến ngắn hơn. Người ta muốn overlap nhiều hơn khi câu trả lời quan trọng có thể nằm ở ranh giới giữa hai chunks — overlap đảm bảo context không bị mất khi cắt.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Thể thao đa môn (Tennis, Bóng rổ NBA, Bơi lội, F1, Da Nang Dragons)

**Tại sao nhóm chọn domain này?**
> Thể thao là domain có dữ liệu phong phú, cập nhật liên tục, và dễ verify kết quả retrieval (tên VĐV, tỷ số, sự kiện đều rõ ràng). Queries thể thao thường cụ thể ("Ai vô địch?", "Kết quả trận X như thế nào?") — rất phù hợp để đánh giá precision của RAG. Ngoài ra, nhóm có sẵn nhiều nguồn song ngữ (tiếng Việt + English) cho phép thử nghiệm metadata filter theo ngôn ngữ.

### Data Inventory

Nguồn chính là các file thể thao trong `data/`. Tôi không dùng các file không thuộc domain thể thao như `python_intro.txt`, `rag_system_design.md`, `vector_store_notes.md`, `vi_retrieval_notes.md`, `customer_support_playbook.txt`. Tôi cũng không đưa `data/articles_clean.md` vào default corpus vì nó là bản markdown duplicate của `data/articles_clean.jsonl`.

| # | File nguồn | Số document sau khi load | Nội dung chính | Metadata đã gán |
|---|------------|--------------------------|----------------|-----------------|
| 1 | `data/articles_clean.jsonl` | 10 | Dân trí thể thao: Asian Cup, World Cup 2026, U19 Việt Nam, Premier League | `source`, `url`, `title`, `published_at`, `author`, `category`, `topic`, `tags`, `entities` |
| 2 | `data/sports_articles_multisource_clean.md` | 13 | Dataset multi-source: Roland Garros, Australian Open, World Cup 2026, bóng chuyền, ASIAD | `source`, `title`, `topic`, `category`, `entities`, `published` |
| 3 | `data/sports_dataset.md` | 10 | Dataset tổng hợp nhiều môn: tennis, NBA, bơi, F1, cầu lông, điền kinh, cycling, boxing, gymnastics | `source`, `title`, `topic`, `tournament`, `tags`, `author` |
| 4 | `data/espn-paper-1.txt` | 1 | ESPN NBA Finals: ký ức Knicks thập niên 1990 và Jalen Brunson | `source`, `title`, `author`, `published_at`, `category`, `language`, `url` |
| 5 | `data/espn-paper-2.txt` | 1 | ESPN NBA Finals Game 1: Knicks thắng Spurs 105-95 | `source`, `title`, `author`, `published_at`, `category`, `language`, `url` |
| 6 | `data/tin-tuc-DaNang-Dragons.txt` | 1 | Bài về Danang Dragons, Saigon Heat và Karachi Edo tại VBA 2026 | `source`, `title`, `author`, `published_at`, `category`, `language`, `url` |
| 7 | `data/web-the-thao-paper1.txt` | 1 | Web Thể Thao NBA Finals: CĐV lao vào sân chụp ảnh với Victor Wembanyama | `source`, `title`, `author`, `published_at`, `category`, `language`, `url` |

Tổng corpus sau khi chạy `python3 main.py`: **37 documents** và được lưu embedding vào `data/embedding_store.json`.

### Metadata Schema (của Nguyễn Đức Thành,tôi)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| title | string | "Alcaraz thắng chung kết Roland Garros dài nhất lịch sử" | Giúp filter theo bài viết cụ thể |
| author | string | "Thục Quyên", "VnExpress Thể thao" | Xác định nguồn tác giả |
| source | string | URL bài gốc hoặc tên file | Traceability — chỉ ra xuất xứ chunk |
| date | string | "4/6/2026", "2025-06-08" | Filter theo thời gian (tin cũ vs mới) |
| lan | string | "vi" / "en" | Filter theo ngôn ngữ — quan trọng khi có corpus song ngữ |
| cats | list | ["thể thao", "bóng rổ"] | Filter theo môn thể thao hoặc chủ đề |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=200)` trên 2 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| sports_dataset.md (27.843 chars) | fixed_size | 186 | 196 | Thấp — cắt giữa câu thường xuyên |
| sports_dataset.md | by_sentences | 62 | 443 | Cao — giữ câu nguyên vẹn nhưng chunk to |
| sports_dataset.md | recursive | 209 | 132 | Trung bình — bám theo paragraph |
| web-the-thao-paper1.txt (3.218 chars) | fixed_size | 21 | 196 | Thấp |
| web-the-thao-paper1.txt | by_sentences | 5 | 638 | Cao nhưng chunk quá to |
| web-the-thao-paper1.txt | recursive | 27 | 118 | Trung bình-Cao |

### Strategy Của Tôi

**Loại:** `RecursiveChunker(chunk_size=200)`

**Mô tả cách hoạt động:**
> RecursiveChunker thử các separator theo thứ tự ưu tiên: `\n\n` (ngăn cách đoạn văn) → `\n` (ngăn dòng) → `. ` (ranh giới câu) → ` ` (từ) → ký tự. Với mỗi đoạn text, nó dừng lại ở separator nào giữ được chunk ≤ chunk_size. Nếu một phần vẫn quá dài, nó đệ quy xuống separator tiếp theo. Base case là khi text đã đủ ngắn (≤ chunk_size) hoặc hết separator.

**Tại sao tôi chọn strategy này cho domain thể thao?**
> Bài báo thể thao viết theo cấu trúc đoạn văn rõ ràng — mỗi đoạn thường tường thuật một pha bóng, một sự kiện, hoặc một trích dẫn. RecursiveChunker tôn trọng ranh giới `\n\n` (đoạn văn) trước, đảm bảo mỗi chunk là một đơn vị thông tin hoàn chỉnh. FixedSizeChunker sẽ cắt giữa câu, SentenceChunker tạo chunk quá to (400-600 chars/chunk), còn RecursiveChunker cân bằng giữa kích thước và ngữ nghĩa.

**Code snippet:**
```python
from src import RecursiveChunker

chunker = RecursiveChunker(chunk_size=200)
chunks = chunker.chunk(text)
# separators mặc định: ["\n\n", "\n", ". ", " ", ""]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality |
|-----------|----------|-------------|------------|--------------------|
| sports_dataset.md | FixedSizeChunker (baseline) | 186 | 196 | Kém — cắt giữa tên VĐV, tỷ số |
| sports_dataset.md | SentenceChunker (baseline) | 62 | 443 | Trung bình — chunk to, giảm precision |
| sports_dataset.md | **RecursiveChunker (của tôi)** | **209** | **132** | **Tốt — chunk nhỏ, bám đoạn văn** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------|-----------|----------|
| Nguyễn Đức Thành (tôi) | RecursiveChunker(200) + gemini-embedding-2| 10/10 | Bám ranh giới đoạn văn, chunk nhỏ | Chunk count lớn → chi phí embedding cao |
| Đặng Minh Hải - 2A202600713 | Article-level chunks + `source/topic` filter + OpenAI embeddings | 10 / 10 | Top-1 đúng ở 5/5 queries; metadata giảm nhiễu giữa nhiều file nguồn | Bài dài như ESPN cần đọc evidence kỹ hơn |
| Hoàng Phúc Quân - 2A202600560 | SentenceChunker (`by_sentences`, `max_sentences_per_chunk=3`) + OpenAI embeddings | 10 / 10 | Top-1 đúng ở 5/5 queries; chunk ngắn, dễ đọc, giữ evidence rõ | Corpus tăng từ 37 docs lên 144 chunks, chi phí embed/search cao hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
Hai strategy tốt nhất trong nhóm là của Đặng Minh Hải và Hoàng Phúc Quân vì đều đạt 10/10. Với corpus đa nguồn hiện tại, article-level chunking có lợi thế đơn giản và tận dụng metadata `source/topic`, còn SentenceChunker có lợi thế evidence rõ hơn, đặc biệt ở bài ESPN dài. Strategy của Nguyễn Đức Thành tăng lên 10/10 nhờ filter theo source/topic, nhưng Chunk count lớn → chi phí embedding cao.

** Cá nhân tôi còn triển khai thêm `SectionChunker` tách text tại ranh giới section header: **
- Markdown headers: `^#{1,6} ` (e.g. `## Overview`)
- Underlined headers: dòng text theo sau bởi `===` hoặc `---`

Dùng lookahead regex để giữ header trong chunk của nó, merge section ngắn hơn `min_section_length` vào chunk tiếp theo.

Tôi nhận thấy:
- **Điểm mạnh:** Mỗi chunk là một section hoàn chỉnh — giữ nguyên header + nội dung tương ứng, không cắt giữa ý. Rất phù hợp với tài liệu có cấu trúc rõ ràng như `sports_dataset.md` (mỗi `## sport_001` là một bài tin trọn vẹn). Embedding của toàn section có ngữ nghĩa tập trung hơn so với FixedSize hay Recursive, giúp retrieval chính xác hơn với query theo chủ đề.
- **Điểm yếu:** Phụ thuộc hoàn toàn vào cấu trúc markdown — file plain text hoặc văn xuôi liên tục (như `espn-paper-1.txt`) sẽ không tìm thấy header, trả về toàn bộ document là một chunk duy nhất, kém hữu ích. Chunk size không kiểm soát được: section dài có thể vượt giới hạn context của LLM hoặc làm loãng embedding vector.**
---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'\. |! |\? |\.\n', text)` để tách câu theo 4 pattern kết thúc câu phổ biến. Sau đó lọc chuỗi rỗng, strip whitespace, rồi gom nhóm `max_sentences_per_chunk` câu thành một chunk bằng `' '.join(group)`. Edge case xử lý: text rỗng trả về `[]`, không tìm thấy câu trả về `[text.strip()]`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy với base case: nếu `len(text) <= chunk_size` → trả về `[text]`; nếu hết separator → cắt theo ký tự. Bước đệ quy: split text theo separator hiện tại, cộng dồn các phần vào buffer, khi buffer vượt quá chunk_size thì flush bằng cách gọi đệ quy với separator tiếp theo. Đảm bảo separator nào cũng được thử trước khi xuống separator thô hơn.

**`compute_similarity`** — approach:
> Áp dụng công thức `cos(a,b) = dot(a,b) / (‖a‖ × ‖b‖)`. Guard zero-vector: nếu magnitude của a hoặc b = 0.0 thì trả về 0.0 ngay để tránh chia cho 0. Mock embedder trả về unit vector nên dot product = cosine similarity.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `_make_record` embed content thành vector, copy metadata và inject thêm `doc_id`. `add_documents` vòng lặp đơn giản append vào `self._store` (list of dicts). `search` embed query rồi gọi `_search_records` — tính dot product giữa query_vec và từng stored embedding, sort giảm dần theo score, trả về top_k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter trước: lọc `self._store` giữ lại những record có `metadata[k] == v` cho mọi `(k,v)` trong `metadata_filter`, rồi mới chạy similarity search trên danh sách đã lọc. `delete_document` dùng list comprehension loại bỏ records có `metadata["doc_id"] == doc_id`, so sánh size trước/sau để trả về `True/False`.

### KnowledgeBaseAgent

**`answer`** — approach:
> 3 bước RAG: (1) `store.search(question, top_k)` lấy top-k chunks liên quan nhất; (2) join nội dung các chunks thành `context` bằng `"\n\n"`; (3) build prompt theo template `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` rồi gọi `llm_fn(prompt)`. Context đặt trước question để LLM ưu tiên knowledge base thay vì training data.

### Test Results

```
platform win32 -- Python 3.10.11, pytest-9.0.3
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

42 passed in 1.01s
```

**Số tests pass: 42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

> **Lưu ý:** Các cặp dưới đây được chạy với `gemini-embedding-2` — embedder thật phản ánh semantic meaning.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Alcaraz vô địch Roland Garros 2025." | "Carlos Alcaraz bảo vệ danh hiệu tại Roland Garros." | HIGH | **0.9002** | ✓ |
| 2 | "Thunder vô địch NBA 2025." | "Leon Marchand phá kỷ lục bơi lội." | LOW | **0.6175** | ✓ |
| 3 | "Verstappen vô địch F1 lần thứ tư." | "Verstappen giành chức vô địch thế giới F1." | HIGH | **0.9192** | ✓ |
| 4 | "Da Nang Dragons thi đấu tại ABL." | "Alcaraz thắng Sinner trong trận chung kết dài nhất lịch sử." | LOW | **0.4971** | ✓ |
| 5 | "Chunking chia tài liệu thành các phần nhỏ hơn." | "Segmentation splits text into smaller sections." | HIGH | **0.6855** | ✗ |

**Dự đoán đúng: 4 / 5**

**Kết quả nào bất ngờ nhất?**
> **Pair 5** là trường hợp thú vị nhất. Tôi dự đoán HIGH vì hai câu có cùng nghĩa — "chunking" và "segmentation" đều là "chia nhỏ văn bản". Tuy nhiên câu A là tiếng Việt, câu B là tiếng Anh. Score thực tế là 0.6855 — thấp hơn ngưỡng 0.7 (HIGH). `gemini-embedding-2` xử lý được cross-lingual semantics nhưng vẫn tạo ra penalty nhỏ khi hai ngôn ngữ khác nhau. Bài học: với corpus song ngữ, nên điều chỉnh threshold theo ngữ cảnh ngôn ngữ, và nên query bằng ngôn ngữ phù hợp với corpus.

---

## 6. Results — Cá nhân (10 điểm)

**Setup:** 6 tài liệu thể thao → `RecursiveChunker(chunk_size=200)` → `EmbeddingStore` với `gemini-embedding-2` → `KnowledgeBaseAgent` với `gemini-2.5-flash-lite`

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Đối thủ cuối cùng của tuyển Việt Nam ở Asian Cup 2027 là đội nào và họ thắng Lebanon với tỷ số bao nhiêu? | Yemen là đối thủ cuối cùng; Yemen thắng Lebanon 2-0. |
| 2 | Knicks thắng Game 1 NBA Finals 2026 trước Spurs với tỷ số bao nhiêu và họ đã lội ngược dòng cách biệt bao nhiêu điểm? | Knicks thắng 105-95 sau khi bị dẫn 14 điểm. |
| 3 | Jannik Sinner vô địch Roland Garros 2026 sau khi thắng ai và hoàn tất cột mốc Grand Slam nào? | Sinner thắng Carlos Alcaraz 3-1 và hoàn tất Career Grand Slam. |
| 4 | Nguồn sinh khí mới nào có thể giúp Danang Dragons tạo bất ngờ trước Saigon Heat? | Karachi Edo. |
| 5 | Tay vợt 17 tuổi nào thắng Marin Cilic ở vòng một Roland Garros 2026? | Moise Kouame. |


### Kết Quả Của Tôi

**Setup:** 7 tài liệu thể thao → `RecursiveChunker(chunk_size=200)` → `gemini-embedding-2` → `gemini-2.5-flash-lite`

| # | Query (tóm tắt) | Top-1 src | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-----------------|-----------|-------|-----------|------------------------|
| Q1 | Asian Cup 2027 — Yemen vs Lebanon | articles_clean.md | 0.6351 | ✓ | "Yemen là đối thủ cuối cùng; Yemen thắng Lebanon 2-0" |
| Q2 | Knicks 105-95, lội ngược dòng 14 điểm | espn-paper-2.txt | 0.7398 | ✓ | "Knicks thắng 105-95, lội ngược dòng 14 điểm" |
| Q3 | Sinner Career Grand Slam RG 2026 | sports_articles_multisource_clean.md | 0.6382 | ✗ Không có data | "Không có thông tin Sinner vô địch RG 2026 trong corpus" |
| Q4 | Karachi Edo — Da Nang Dragons | tin-tuc-DaNang-Dragons.txt | 0.6963 | ✓ | "Karachi Edo" |
| Q5 | Moise Kouame 17 tuổi — Cilic RG 2026 | sports_articles_multisource_clean.md | 0.6371 | ✓ | "Moise Kouame" |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **4 / 5**

> **Phân tích:**
> - **Q1**: Data về Asian Cup 2027 có trong `articles_clean.md` (thêm trong quá trình chạy). Score 0.6351 thấp hơn các truy vấn khác nhưng agent trả lời đúng.
> - **Q2**: `espn-paper-2.txt` chứa đúng thông tin Knicks/Spurs NBA Finals 2026. Agent trích xuất được tỷ số và cách biệt điểm chính xác.
> - **Q3**: Corpus không có data về kết quả Roland Garros 2026 (chỉ có RG 2025 Alcaraz vs Sinner). Agent grounded đúng — từ chối hallucinate, liệt kê rõ những gì corpus có và không có.
> - **Q4, Q5**: Retrieval và agent đều chính xác 100%.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Qua quá trình so sánh trong nhóm, tôi nhận ra rằng cùng một bộ tài liệu thể thao nhưng chunking strategy khác nhau dẫn đến kết quả retrieval rất khác nhau. Điều thú vị là không có strategy nào "thắng tuyệt đối" — mỗi người có điểm mạnh ở một loại query khác nhau. Từ đó tôi hiểu rằng việc chọn strategy phải gắn liền với kiểu câu hỏi thực tế, không chỉ với cấu trúc tài liệu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua phần demo liên nhóm, tôi học được rằng metadata design đóng vai trò quan trọng không kém chunking. Nhóm nào thiết kế metadata schema cẩn thận (phân loại theo môn thể thao, ngôn ngữ, thời gian) có thể dùng `search_with_filter()` để tăng precision đáng kể, đặc biệt với corpus lớn và đa dạng.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ parse và inject đầy đủ các trường metadata (title, author, date, cats, topic) vào từng chunk thay vì chỉ lưu tên file. Điều này cho phép filter theo môn thể thao (`cats=bóng rổ`) hoặc theo thời gian (`date >= 2025-01-01`) để loại bỏ tin cũ. Ngoài ra tôi sẽ thử `SectionChunker` cho file `.md` có header rõ ràng như `sports_dataset.md` — mỗi section `## sport_001` là một đơn vị tin tức trọn vẹn, rất phù hợp để chunk.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Có Demo thể hiện sự khách biệt giữa các approach | 5 / 5 |
| **Tổng** | | **≈ 88+ / 100** |
