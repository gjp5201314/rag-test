# 本地知识库 RAG 系统 - 面试问答

## 项目概述

### Q1: 请简要描述这个项目的架构

这是一个基于 RAG（检索增强生成）的本地知识库问答系统，主要包含以下组件：

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Web 服务                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  API 路由层  │  │  前端页面    │  │  CLI 命令    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                   KnowledgeBaseService                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 索引构建  │ │ 文档加载  │ │  搜索   │ │ 知识库管理│   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│                   LocalKnowledgeBase                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 文档解析  │ │ 文本分片  │ │ TF-IDF  │ │  评分   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│              QwenChat (LLM 调用)                         │
└─────────────────────────────────────────────────────────┘
```

---

## RAG 与检索机制

### Q2: 项目的检索原理是什么？相比于向量检索有什么特点？

本项目使用 **TF-IDF 词频-逆文档频率** 进行检索，而非向量嵌入检索：

**评分算法** (`_score_document` 方法)：
```python
def _score_document(self, query: str, query_terms: list[str], text: str) -> float:
    score = 0.0
    # 1. 完整查询匹配（权重最高）
    if normalized_query in normalized_text:
        score += min(len(normalized_query), 24) * 3.0

    # 2. 术语词匹配（多次出现得分更高）
    for term in query_terms:
        occurrences = normalized_text.count(term)
        if occurrences > 0:
            score += occurrences * max(len(term), 1)
    return score
```

**特点**：
| 维度 | TF-IDF 检索 | 向量检索 |
|------|------------|---------|
| 原理 | 词频统计 | 语义嵌入 |
| 速度 | 快（字符串匹配） | 慢（向量计算） |
| 精度 | 关键词敏感 | 语义理解强 |
| 资源 | 低（无需GPU） | 高（需要向量模型） |

---

### Q3: 如何处理中文分词？查询词提取的逻辑是什么？

**中文 N-gram 策略** (`_extract_query_terms` 方法)：
```python
def _extract_query_terms(self, query: str) -> list[str]:
    # 1. 提取 ASCII 术语（英文、数字）
    ascii_terms = re.findall(r"[a-z0-9]{2,}", normalized_query)
    terms.extend(ascii_terms)

    # 2. 提取中文 N-gram（2-4 字词）
    chinese_chars = [char for char in query if "\u4e00" <= char <= "\u9fff"]
    for size in (4, 3, 2):  # 优先匹配更长的词
        for index in range(len(chinese_chars) - size + 1):
            terms.append("".join(chinese_chars[index:index + size]))

    # 3. 去重
    deduped_terms = list(dict.fromkeys(terms))  # 保持顺序去重
    return deduped_terms
```

**为什么用 N-gram 而不是分词器？**
- 无需额外依赖中文分词库
- 避免分词歧义问题
- 对于专有名词（如"剑来"）效果稳定

---

### Q4: 什么是 Top-K 检索？如何实现？

**Top-K 检索**指返回相似度最高的 K 个文档片段：

```python
def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[dict]:
    # 1. 计算所有文档的得分
    scored_results = []
    for doc in self.documents:
        score = self._score_document(query, query_terms, doc.text)
        if score > 0:
            scored_results.append({..., "score": score})

    # 2. 使用 heapq.nlargest 高效找出 Top-K
    top_results = nlargest(top_k, scored_results, key=lambda item: item["score"])

    # 3. 分数归一化（相对得分）
    max_score = max(item["score"] for item in top_results)
    for item in top_results:
        item["score"] = round(item["score"] / max_score, 4)

    return top_results
```

**为什么用 `nlargest` 而不是排序？**
- 时间复杂度：O(n log k) vs O(n log n)
- 适合"从大量文档中找 Top-K"的场景

---

## 文档处理

### Q5: 如何实现多格式文档解析？

```python
def _read_text_file(self, file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return self._read_pdf_file(file_path)
    elif suffix in {".docx", ".doc"}:
        return self._read_docx_file(file_path)
    elif suffix in {".pptx", ".ppt"}:
        return self._read_pptx_file(file_path)
    elif suffix in {".xlsx", ".xls"}:
        return self._read_xlsx_file(file_path)

    # 纯文本：尝试多种编码
    for encoding in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""
```

**PDF 读取的备选机制**：
```python
def _read_pdf_file(self, file_path: Path) -> str:
    # 优先 PyPDF2
    try:
        import PyPDF2
        # ... 读取逻辑
    except ImportError:
        pass  # 尝试下一个

    # 其次 pdfplumber
    try:
        import pdfplumber
        # ... 读取逻辑
    except ImportError:
        pass

    # 都失败则返回空字符串（文件会被跳过并记录警告）
    return ""
```

---

### Q6: 文档分片的策略是什么？

**带重叠的滑动窗口分片** (`_split_text` 方法)：
```python
def _split_text(self, text: str) -> List[str]:
    if len(text) <= self.chunk_size:
        return [text]  # 短文本不分片

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + self.chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break
        # 重叠区域：下一个分片从 end - overlap 开始
        start = max(end - self.chunk_overlap, start + 1)
    return chunks
```

**示例**（chunk_size=10, overlap=3）：
```
文本: "ABCDEFGHIJKLMN"
分片1: "ABCDEFGHIJ"     (0-10)
分片2: "HIJKLMN"        (7-14，重叠 HIJ)
```

---

## 系统架构

### Q7: 多知识库是如何实现的？

**目录结构**：
```
data/
├── uploads/              # 上传文件
│   ├── default/          # 默认知识库
│   │   └── *.pdf, *.txt
│   └── <kb_id>/          # 其他知识库
├── knowledge_bases/      # 索引数据
│   ├── default/
│   │   └── index.json, chunks/
│   └── <kb_id>/
│       └── index.json, chunks/
└── knowledge_bases.json  # 知识库元数据
```

**路径获取逻辑**：
```python
def _get_kb_index_path(self, kb_id: str) -> str:
    if kb_id == "default":
        return DEFAULT_INDEX_PATH  # "./data/index.json"
    return str(Path(DEFAULT_KB_DIR) / kb_id / "index.json")

def _get_kb_upload_dir(self, kb_id: str) -> str:
    return str(Path(UPLOAD_FOLDER) / kb_id)
```

---

### Q8: 索引构建的异步机制是怎样的？

**使用线程池 + 任务队列**：

```python
def start_build_index(self, ...) -> dict:
    # 1. 检查是否有进行中的任务
    with self._lock:
        active_job = self.build_jobs.get(self.active_build_job_id)
        if active_job and active_job.status in {"queued", "running"}:
            raise ValueError("已有索引构建任务进行中")

    # 2. 创建任务并后台执行
    job = BuildJob(job_id=uuid4().hex, status="queued", ...)
    self.build_jobs[job.job_id] = job
    Thread(target=self._run_build_job, args=(job.job_id,), daemon=True).start()
    return job.to_dict()

def _run_build_job(self, job_id: str) -> None:
    job.status = "running"
    try:
        result = self.build_index(...)
        job.status = "completed"
        job.result = result
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
```

**前端轮询机制**：
```javascript
async function pollBuildJob(jobId) {
    while (true) {
        const job = await requestJson(`/api/build-status?job_id=${jobId}`);
        if (job.status === 'completed') return job;
        if (job.status === 'failed') throw new Error(job.error);
        await sleep(1500);  // 每1.5秒轮询
    }
}
```

---

### Q9: 服务启动时的预热机制是什么？

```python
def start_warmup(self, kb_id: str = "default") -> None:
    with self._lock:
        if self.warmup_started:
            return
        self.warmup_started = True

    Thread(target=self._warmup_index, args=(kb_id,), daemon=True).start()

def _warmup_index(self, kb_id: str) -> None:
    index_path = self._get_kb_index_path(kb_id)

    if Path(index_path).exists():
        # 已存在索引：直接加载到内存
        self.ensure_loaded(index_path)
    elif source_dir.exists() and has_files(source_dir):
        # 索引不存在但有文件：自动构建
        self.build_index(source=source_dir, index_path=index_path, ...)
```

**目的**：
- 减少首次请求的等待时间
- 确保服务启动时索引已就绪

---

### Q10: `ensure_loaded` 方法的线程安全机制？

```python
def ensure_loaded(self, index_path: str) -> LocalKnowledgeBase:
    wait_event: Event | None = None
    should_load = False

    with self._lock:
        # 检查是否已加载
        if self.kb is not None and self.loaded_index_path == index_path:
            return self.kb

        # 检查是否有其他线程正在加载
        wait_event = self.loading_indexes.get(index_path)
        if wait_event is None:
            # 没有正在加载，创建等待事件
            wait_event = Event()
            self.loading_indexes[index_path] = wait_event
            should_load = True

    if not should_load:
        # 等待其他线程加载完成
        wait_event.wait()
        return self.kb  # 此时应该已加载

    # 本线程负责加载
    try:
        kb = LocalKnowledgeBase.load(index_path)
        with self._lock:
            self.kb = kb
            self.loaded_index_path = index_path
        return kb
    finally:
        with self._lock:
            self.loading_indexes.pop(index_path, None)
            wait_event.set()  # 通知等待的线程
```

**核心思想**：使用 `threading.Event` 实现多线程协调，避免重复加载。

---

## Flask 与 Web 开发

### Q11: Flask 应用的全局异常处理？

```python
@app.errorhandler(Exception)
def handle_exception(error):
    if isinstance(error, HTTPException):
        response = error.get_response()
        response.data = json.dumps({
            "error": error.description,
            "code": error.code,
            "name": error.name,
        }, ensure_ascii=False)
        response.content_type = "application/json; charset=utf-8"
        return response

    # 其他异常返回 500
    return jsonify({"error": str(error)}), 500
```

---

### Q12: 文件上传的安全措施？

```python
def _make_safe_filename(self, filename: str) -> str:
    # 1. 去除路径分隔符
    safe_name = "".join(c for c in filename if c not in r'<>:"/\|?*')

    # 2. 处理空文件名
    if not safe_name.strip():
        safe_name = "unnamed"

    # 3. 防止重名冲突
    file_path = upload_dir / safe_name
    counter = 1
    while file_path.exists():
        safe_name = f"{name}_{counter}{ext}"
        file_path = upload_dir / safe_name
        counter += 1

    return safe_name
```

---

## Python 高级特性

### Q13: 使用了哪些 Python 类型提示？

```python
from typing import Any, Iterable, List, Optional

def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[dict]:
    # 返回类型：dict 列表

def _load_documents(self, root_dir: str, skipped_files: list[str]) -> Iterable[tuple[DocumentChunk, bool]]:
    # 返回类型：生成器，yield 元组

@dataclass
class BuildJob:
    job_id: str
    status: str
    kb_id: str = "default"  # 默认参数
    message: str = "等待开始"
    result: dict[str, Any] | None = None  # 联合类型
```

---

### Q14: `@dataclass` 的使用场景和优势？

```python
@dataclass
class DocumentChunk:
    path: str
    chunk_id: int
    text: str

@dataclass
class KnowledgeBaseInfo:
    id: str
    name: str
    description: str = ""  # 带默认值的字段
    created_at: str = ""
    document_count: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # 自动转换
```

**优势**：
- 自动生成 `__init__`、`__repr__`、`__eq__`
- 代码更简洁
- 类型提示更清晰

---

### Q15: `heapq.nlargest` 的原理和使用场景？

```python
from heapq import nlargest

# 获取最大的3个元素（不改变原列表）
top_results = nlargest(3, scored_results, key=lambda item: item["score"])
```

**原理**：小顶堆（min-heap），只维护 K 个最大元素
- 时间复杂度：O(n log k)
- 空间复杂度：O(k)

**对比**：
| 方法 | 时间复杂度 | 适用场景 |
|------|----------|---------|
| `sorted()` | O(n log n) | 需要全部排序 |
| `nlargest(k, ...)` | O(n log k) | Top-K 场景 ✓ |
| `min()` | O(n) | 只找1个最大 |

---

## 代码设计模式

### Q16: 项目中体现了哪些设计原则？

**1. 单一职责原则 (SRP)**
```python
class LocalKnowledgeBase:  # 只负责本地知识库逻辑
class KnowledgeBaseService:  # 只负责知识库服务
class QwenChat:  # 只负责 LLM 调用
```

**2. 开闭原则 (OCP)**
- `_read_text_file` 方法通过后缀判断，支持扩展新格式
- 不修改核心逻辑，只需添加新的读取方法

**3. 依赖倒置 (DIP)**
```python
# 不直接依赖具体实现
def build_index(self, source: str, index_path: str, ...):
    kb = LocalKnowledgeBase(...)  # 依赖抽象
```

**4. 线程安全设计**
```python
with self._lock:  # 使用锁保护共享资源
    if self.kb is not None and self.loaded_index_path == index_path:
        return self.kb
```

---

### Q17: 生成器的使用场景？

```python
def _load_documents(self, root_dir: str, ...) -> Iterable[tuple[DocumentChunk, bool]]:
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        # ...
        text = self._read_text_file(file_path)
        if not text.strip():
            yield None, True  # 跳过文件
            continue

        for chunk_id, chunk_text in enumerate(self._split_text(text)):
            yield DocumentChunk(...), False  # 产出文档块
```

**优势**：
- 内存高效：逐个产出，无需一次性加载所有文档
- 惰性求值：按需计算
- 可组合：可以链式处理

---

## 系统性能与优化

### Q18: 如何处理大文件分片？

```python
def _split_text(self, text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + self.chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        # 关键：重叠区域保证上下文连续性
        start = max(end - self.chunk_overlap, start + 1)
    return chunks
```

**参数调优建议**：
- `chunk_size=1000`：适合短答案场景
- `chunk_overlap=200`：保证跨分片的内容不被切断

---

### Q19: 索引缓存机制？

```python
def get_vectorizer_cache_path(index_path: str) -> str:
    base = Path(index_path)
    return str(base.parent / f"{base.stem}_vectorizer.joblib")

def get_matrix_cache_path(index_path: str) -> str:
    base = Path(index_path)
    return str(base.parent / f"{base.stem}_matrix.npz")

def save(self, index_path: str) -> None:
    # 保存索引数据
    path.write_text(json.dumps(payload, ...))
    # 缓存向量化器和矩阵
    joblib_dump(self.vectorizer, get_vectorizer_cache_path(index_path))
    save_npz(get_matrix_cache_path(index_path), self.matrix)
```

**注意**：当前代码中 `vectorizer` 和 `matrix` 被设为 `None`，但保存逻辑已预留。

---

## 扩展问题

### Q20: 如何提升检索质量？

**可能的改进方向**：

1. **混合检索**：结合向量语义检索 + TF-IDF 关键词检索
2. **重排序**：用 Cross-Encoder 对初筛结果重排
3. **查询扩展**：使用 LLM 生成相关查询补充
4. **分词优化**：引入 jieba 等中文分词库
5. **元数据过滤**：支持按文件名、日期等条件筛选

---

### Q21: 如何支持更多文件格式？

```python
def _read_text_file(self, file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    # 示例：新增 Markdown 特殊处理
    if suffix == ".md":
        return self._read_markdown_file(file_path)

    # 或使用通用库
    try:
        import textract
        return textract.process(str(file_path)).decode('utf-8')
    except ImportError:
        pass
```

---

### Q22: 分布式部署的挑战？

| 问题 | 可能的解决方案 |
|------|-------------|
| 文件同步 | 使用对象存储（OSS/S3）替代本地文件系统 |
| 索引一致性 | 分布式锁（如 Redis）或中央索引服务 |
| API 限流 | 添加请求队列和速率限制 |

---

## 总结

本项目是一个典型的 **Flask + RAG** 架构实践，涵盖了：

- ✅ 多格式文档解析
- ✅ 全文检索（TF-IDF）
- ✅ LLM 集成（千问 API）
- ✅ 多知识库管理
- ✅ 异步任务处理
- ✅ 线程安全设计
- ✅ Web API 开发
