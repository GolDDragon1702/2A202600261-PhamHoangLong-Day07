# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Hoàng Long

**Nhóm:** [Tên nhóm]

**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Hai vector có hướng gần giống nhau, tức là nội dung ngữ nghĩa của hai câu rất tương đồng dù có thể khác từ vựng.*

**Ví dụ HIGH similarity:**
- Sentence A: "Học bổng tài năng dành cho sinh viên mới là gì?"
- Sentence B: "Điều kiện học bổng tài năng cho sinh viên năm nhất?"
- Tại sao tương đồng: Cả hai câu đều hỏi về học bổng tài năng cho sinh viên mới, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "Học bổng tài năng dành cho sinh viên mới là gì?"
- Sentence B: "Giá vé xe buýt là bao nhiêu?"
- Tại sao khác: Hai câu hỏi hoàn toàn khác nhau về chủ đề, không có liên quan ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Cosine similarity đo lường góc giữa các vector, tập trung vào hướng (ý nghĩa) thay vì độ lớn, phù hợp với bản chất của text embeddings. Euclidean distance nhạy cảm với độ lớn của vector, có thể dẫn đến kết quả sai lệch khi so sánh các câu có độ dài khác nhau.*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
Stride = chunk_size - overlap = 500 - 50 = 450
Số chunks ≈ ⌈(10000 - 500) / 450⌉ + 1 = ⌈9500 / 450⌉ + 1 = ⌈21.11⌉ + 1 = 22 + 1 = 23

> *Đáp án: 23 chunks*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Nếu overlap tăng lên 100, chunk count sẽ giảm xuống còn 21 chunks. Overlap nhiều hơn giúp tăng cường ngữ cảnh liên kết giữa các chunk, cải thiện khả năng retrieval khi câu hỏi liên quan đến ranh giới giữa các chunk.*

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Quy định tài chính và Biểu phí của Đại học VinUni (VinUni Financial Regulations & Fees).

**Tại sao nhóm chọn domain này?**
> Đây là bộ dữ liệu thực tế, có cấu trúc phân cấp markdown rõ ràng (#, ##, bullet points) và chứa nhiều thông tin con số, điều kiện chi tiết. Việc sử dụng RAG trên domain này giúp thử nghiệm khả năng giải thích các truy vấn phức tạp của sinh viên về học phí, học bổng.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | data1.md (Đại học chính quy) | VinUni | 13,897 | doc_id, source, chunk_index |
| 2 | data2.md (Bác sỹ nội trú) | VinUni | 1,081 | doc_id, source, chunk_index |
| 3 | data3.md (Pathway English) | VinUni | 3,556 | doc_id, source, chunk_index |
| 4 | data4.md (Học bổng & HTTC) | VinUni | 13,214 | doc_id, source, chunk_index |
| 5 | data5.md (Quy định chung) | VinUni | 6,635 | doc_id, source, chunk_index |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id | string | "data1" | Dùng để lọc (filter) kết quả theo từng loại văn bản hoặc chương. |
| source | string | "data/data1.md" | Cung cấp nguồn trích dẫn (citation) cho câu trả lời của Agent. |
| chunk_index | integer | 5 | Giúp xác định vị trí tương đối và lấy ngữ cảnh xung quanh nếu cần. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
|data2.md | FixedSizeChunker (`fixed_size`) | 6 | 182.8 |Không |
|data2.md | SentenceChunker (`by_sentences`) | 4 | 209.0 |Tương đối |
|data2.md | RecursiveChunker (`recursive`) | 7 | 118.9 |Tốt |
|data2.md | MarkdownChunker (`markdown`) (custom)| 3 | 313.7 |Tốt |

### Strategy Của Tôi

**Loại:** MarkdownChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

Chunker phân tách tài liệu Markdown theo cấu trúc heading (#, ##, …) để giữ ngữ cảnh phân cấp, sau đó tiếp tục chia nhỏ nội dung thành các “semantic blocks” như đoạn văn, danh sách bullet, và bảng. Mỗi chunk luôn được gắn kèm chuỗi heading cha đã được làm sạch để đảm bảo tự chứa đủ context khi truy xuất. Các block có tính nguyên tử như bảng hoặc danh sách được giữ nguyên, không bị cắt giữa chừng. Cuối cùng, hệ thống áp dụng overlap nhẹ giữa các chunk để duy trì tính liên tục ngữ nghĩa và cải thiện khả năng retrieval. 

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

Chunker phân tách tài liệu Markdown theo cấu trúc heading (#, ##, …) để giữ ngữ cảnh phân cấp, sau đó tiếp tục chia nhỏ nội dung thành các “semantic blocks” như đoạn văn, danh sách bullet, và bảng. Mỗi chunk luôn được gắn kèm chuỗi heading cha đã được làm sạch để đảm bảo tự chứa đủ context khi truy xuất. Các block có tính nguyên tử như bảng hoặc danh sách được giữ nguyên, không bị cắt giữa chừng. Cuối cùng, hệ thống áp dụng overlap nhẹ giữa các chunk để duy trì tính liên tục ngữ nghĩa và cải thiện khả năng retrieval.

**Code snippet (nếu custom):**
```python
class MarkdownChunker:
    # Trích đoạn logic chính: Tách theo heading và semantic blocks
    def chunk(self, text: str) -> List[str]:
        sections = self._parse_sections(text)
        chunks = []
        for section in sections:
            context_prefix = self._format_context(section["heading_context"])
            blocks = self._split_semantic_blocks(section["body"])
            chunks.extend(self._merge_blocks(blocks, context_prefix))
        return chunks

    def _split_semantic_blocks(self, text: str) -> List[str]:
        # Tách riêng Bảng, Danh sách và Đoạn văn để không bị cắt vụn
        # ... logic regex cho Table và Bullet ...
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| data2.md | RecursiveChunker | 7 | 118.9 | Tốt cho text thô, nhưng có thể cắt giữa heading. |
| data2.md | **MarkdownChunker** | 3 | 313.7 | Rất tốt: Mỗi chunk đều mang context cha và giữ nguyên bảng biểu. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Phạm Hoàng Long | MarkdownChunker | 10 | Bảo toàn ngữ cảnh heading, xử lý tốt bảng và bullet groups. | Thực hiện logic phức tạp hơn, tốn tài nguyên xử lý ban đầu. |


**Strategy nào tốt nhất cho domain này? Tại sao?**
> `MarkdownChunker` là tốt nhất cho domain này vì tài liệu quy định tài chính có cấu trúc heading rất sâu (đến 4-5 cấp). Nếu dùng chunker bình thường, khi retrieve được một bullet point, ta sẽ không biết nó thuộc về mục "Phí thư viện" hay "Phí Ký túc xá". `MarkdownChunker` giải quyết triệt để việc này bằng cách đính kèm context heading vào mỗi chunk.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `re.split(r'(?<=[.!?])\s+|\.\n', text)` giúp phát hiện các dấu câu kết thúc nhưng không làm mất nội dung sau dấu chấm. Sau đó gộp các câu lại thành nhóm `max_sentences_per_chunk` để tăng độ bao phủ ngữ nghĩa.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Áp dụng thuật toán chia để trị, thử nghiệm tách văn bản theo thứ tự ưu tiên các dấu phân tách: `\n\n`, `\n`, ` `, và cuối cùng là ký tự trống. Nếu một khối vẫn lớn hơn `chunk_size`, nó sẽ được đệ quy xuống cấp separator thấp hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Sử dụng `ChromaDB` làm backend để lưu trữ vector. Khi không có ChromaDB, hệ thống tự động fallback về in-memory storage. Việc tìm kiếm được thực hiện bằng cách tính `cosine_similarity` giữa query embedding và tất cả document embeddings trong bộ nhớ hoặc dùng chỉ mục của ChromaDB.

**`search_with_filter` + `delete_document`** — approach:
> Filter được thực hiện thông qua metadata query. Trong in-memory fallback, tôi thực hiện filtering trước khi tính similarity để tiết kiệm chi phí tính toán. Việc xóa tài liệu sử dụng `doc_id` hoặc ID cụ thể của chunk.

### KnowledgeBaseAgent

**`answer`** — approach:
> Inject context bằng cách lấy `top_k` chunks relevant nhất, concat chúng thành một khối văn bản và đưa vào system prompt. Prompt được thiết kế để Agent chỉ trả lời dựa trên thông tin đã cung cấp, nếu không có sẽ trả lời là không tìm thấy thông tin.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 11.34s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Tôi muốn nộp học phí" | "Khi nào đóng tiền học?" | high | 0.85 | Có |
| 2 | "Học bổng tài năng" | "Điều kiện miễn học phí" | high | 0.72 | Có |
| 3 | "Ký túc xá có máy lạnh không?" | "Mức phí phòng 8 người" | low | 0.45 | Có |
| 4 | "Bảo hiểm y tế bắt buộc" | "Khám sức khỏe Vinmec" | low | 0.38 | Có |
| 5 | "Học phí bác sỹ nội trú" | "Phí xét tuyển hồ sơ đại học" | low | 0.32 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp số 2 ("Học bổng tài năng" và "Điều kiện miễn học phí"). Mặc dù hai cụm từ không trùng lặp nhiều về từ vựng, nhưng embedding model vẫn nhận diện được sự tương đồng rất cao (0.72). Điều này cho thấy embeddings biểu diễn nghĩa thông qua không gian vector đa chiều, nơi các khái niệm liên quan đến "tài chính" và "ưu đãi giáo dục" nằm gần nhau.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "Học phí cử nhân điều dưỡng là bao nhiêu?" | 349.650.000 đồng /năm. |
| 2 | "Phí xét tuyển hồ sơ là bao nhiêu và khi nào nộp?" | 2.000.000 đồng/lần, nộp trước ngày phỏng vấn. |
| 3 | "Điều kiện duy trì học bổng tài năng 100%?" | CGPA từ 3.2 trở lên, kỷ luật tốt, hoàn thành EXCEL đánh giá. |
| 4 | "Phí phạt mượn sách quá hạn 1 ngày là bao nhiêu?" | 20.000 đồng /ngày /tài liệu. |
| 5 | "Học phí bác sỹ nội trú có được miễn không?" | Có cơ hội được miễn tùy chính sách hàng năm của Vinmec. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Quy định học phí điều dưỡng | Chunk chứa biểu phí 349.650.000đ | 0.92 | Có | 349.650.000 đồng/năm. |
| 2 | Phí xét tuyển hồ sơ | Chunk mục II.1 phí 2.000.000đ | 0.88 | Có | 2 triệu, nộp trước phỏng vấn. |
| 3 | Duy trì học bổng 100% | Bảng điều kiện duy trì học bổng | 0.95 | Có | CGPA > 3.2 và các đ/k khác. |
| 4 | Phí phạt mượn sách | Mục 3.1 phí phạt 20.000đ/ngày | 0.91 | Có | 20.000 đồng/ngày. |
| 5 | Miễn học phí nội trú | Mục B.1 về học bổng nội trú | 0.87 | Có | Có thể được miễn học phí. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách thiết kế Metadata Schema tối ưu để hỗ trợ lọc (filtering) dữ liệu trước khi thực hiện similarity search, giúp tăng tốc độ phản hồi đáng kể.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm đã sử dụng thêm kỹ thuật "HyDE" (Hypothetical Document Embeddings) để tạo một câu trả lời giả định từ query trước khi search, giúp cải thiện độ chính xác khi query quá ngắn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thực hiện tiền xử lý dữ liệu kỹ hơn, ví dụ như trích xuất schema của các bảng biểu phức tạp thành định dạng JSON trước khi chunking để AI có thể hiểu cấu trúc bảng tốt hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
