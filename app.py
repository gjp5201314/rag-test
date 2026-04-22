import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Iterable, List
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from openai import OpenAI
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.exceptions import HTTPException


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".csv",
    ".html",
    ".css",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sql",
    ".log",
}

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
}

DEFAULT_SOURCE_DIR = "./data"
DEFAULT_INDEX_PATH = "./data/index.json"
DEFAULT_EXCLUDED_FILE_NAMES = {
    "index.json",
}
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_vectorizer_cache_path(index_path: str) -> Path:
    path = Path(index_path)
    return path.with_name(f"{path.name}.vectorizer.joblib")


def get_matrix_cache_path(index_path: str) -> Path:
    path = Path(index_path)
    return path.with_name(f"{path.name}.matrix.npz")


@dataclass
class DocumentChunk:
    path: str
    chunk_id: int
    text: str


@dataclass
class BuildJob:
    job_id: str
    status: str
    source: str
    index_path: str
    chunk_size: int
    chunk_overlap: int
    message: str = "等待开始"
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "source": self.source,
            "index": self.index_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "message": self.message,
            "result": self.result,
            "error": self.error,
        }


class LocalKnowledgeBase:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[DocumentChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, root_dir: str) -> int:
        self.documents = list(self._load_documents(root_dir))
        if not self.documents:
            raise ValueError(f"在目录 {root_dir} 中没有找到可索引的文本文件。")

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
        )
        self.matrix = self.vectorizer.fit_transform(doc.text for doc in self.documents).astype("float32")
        return len(self.documents)

    def save(self, index_path: str) -> None:
        if self.vectorizer is None or self.matrix is None:
            raise ValueError("索引尚未构建，不能保存。")

        payload = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents": [asdict(doc) for doc in self.documents],
        }

        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib_dump(self.vectorizer, get_vectorizer_cache_path(index_path), compress=3)
        save_npz(get_matrix_cache_path(index_path), self.matrix, compressed=True)

    @classmethod
    def load(cls, index_path: str) -> "LocalKnowledgeBase":
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        kb = cls(
            chunk_size=payload.get("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=payload.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        )
        kb.documents = [DocumentChunk(**item) for item in payload["documents"]]
        vectorizer_path = get_vectorizer_cache_path(index_path)
        matrix_path = get_matrix_cache_path(index_path)
        if vectorizer_path.exists() and matrix_path.exists():
            kb.vectorizer = joblib_load(vectorizer_path)
            kb.matrix = load_npz(matrix_path)
        else:
            kb.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                lowercase=False,
            )
            kb.matrix = kb.vectorizer.fit_transform(doc.text for doc in kb.documents).astype("float32")
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib_dump(kb.vectorizer, vectorizer_path, compress=3)
            save_npz(matrix_path, kb.matrix, compressed=True)
        return kb

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[dict]:
        if self.vectorizer is None or self.matrix is None:
            raise ValueError("知识库未初始化，请先构建或加载索引。")

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "path": doc.path,
                    "chunk_id": doc.chunk_id,
                    "score": score,
                    "text": doc.text,
                }
            )
        return results

    def _load_documents(self, root_dir: str) -> Iterable[DocumentChunk]:
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"目录不存在: {root_dir}")

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in file_path.parts):
                continue
            if file_path.name in DEFAULT_EXCLUDED_FILE_NAMES:
                continue
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            text = self._read_text_file(file_path)
            if not text.strip():
                continue

            relative_path = str(file_path.relative_to(root))
            for chunk_id, chunk_text in enumerate(self._split_text(text)):
                yield DocumentChunk(path=relative_path, chunk_id=chunk_id, text=chunk_text)

    def _read_text_file(self, file_path: Path) -> str:
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except OSError:
                return ""
        return ""

    def _split_text(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks


class QwenChat:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def answer(self, question: str, contexts: List[dict]) -> str:
        context_text = "\n\n".join(
            [
                f"[文件: {item['path']} | 分片: {item['chunk_id']} | 相似度: {item['score']:.4f}]\n{item['text']}"
                for item in contexts
            ]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个本地知识库问答助手。"
                    "只能优先依据提供的检索上下文回答。"
                    "如果上下文不足，请明确说明‘根据当前知识库内容无法确定’，不要编造事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"检索上下文：\n{context_text}\n\n"
                    "请基于以上内容给出中文回答，并在结尾列出引用到的文件路径。"
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or "模型没有返回内容。"


class KnowledgeBaseService:
    def __init__(self) -> None:
        self._lock = Lock()
        self.kb: LocalKnowledgeBase | None = None
        self.loaded_index_path: str | None = None
        self.build_jobs: dict[str, BuildJob] = {}
        self.active_build_job_id: str | None = None
        self.loading_indexes: dict[str, Event] = {}
        self.loading_errors: dict[str, str] = {}
        self.warmup_started = False
        self.warmup_error: str | None = None

    def build_index(
        self,
        source: str,
        index_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> dict:
        kb = LocalKnowledgeBase(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        count = kb.build(source)
        kb.save(index_path)
        with self._lock:
            self.kb = kb
            self.loaded_index_path = index_path
        return {
            "source": source,
            "index": index_path,
            "chunks": count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def start_build_index(
        self,
        source: str,
        index_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> dict[str, Any]:
        with self._lock:
            active_job = self.build_jobs.get(self.active_build_job_id or "")
            if active_job and active_job.status in {"queued", "running"}:
                same_request = (
                    active_job.source == source
                    and active_job.index_path == index_path
                    and active_job.chunk_size == chunk_size
                    and active_job.chunk_overlap == chunk_overlap
                )
                if same_request:
                    return active_job.to_dict()
                raise ValueError("已有索引构建任务进行中，请等待当前任务完成后再试。")

            job = BuildJob(
                job_id=uuid4().hex,
                status="queued",
                source=source,
                index_path=index_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                message="任务已创建，等待开始构建。",
            )
            self.build_jobs[job.job_id] = job
            self.active_build_job_id = job.job_id

        Thread(target=self._run_build_job, args=(job.job_id,), daemon=True).start()
        return job.to_dict()

    def _run_build_job(self, job_id: str) -> None:
        with self._lock:
            job = self.build_jobs[job_id]
            job.status = "running"
            job.message = "正在扫描文件并构建索引，请稍候..."

        try:
            result = self.build_index(
                source=job.source,
                index_path=job.index_path,
                chunk_size=job.chunk_size,
                chunk_overlap=job.chunk_overlap,
            )
            with self._lock:
                current_job = self.build_jobs[job_id]
                current_job.status = "completed"
                current_job.message = f"索引构建完成：共 {result['chunks']} 个分片。"
                current_job.result = result
                current_job.error = None
        except Exception as error:
            with self._lock:
                current_job = self.build_jobs[job_id]
                current_job.status = "failed"
                current_job.message = f"构建失败：{error}"
                current_job.error = str(error)
                current_job.result = None
        finally:
            with self._lock:
                if self.active_build_job_id == job_id:
                    self.active_build_job_id = None

    def get_build_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.build_jobs.get(job_id)
            if job is None:
                raise FileNotFoundError(f"构建任务不存在: {job_id}")
            return job.to_dict()

    def ensure_loaded(self, index_path: str) -> LocalKnowledgeBase:
        wait_event: Event | None = None
        should_load = False
        with self._lock:
            if self.kb is not None and self.loaded_index_path == index_path:
                return self.kb

            wait_event = self.loading_indexes.get(index_path)
            if wait_event is None:
                wait_event = Event()
                self.loading_indexes[index_path] = wait_event
                self.loading_errors.pop(index_path, None)
                should_load = True

        if not should_load:
            wait_event.wait()
            with self._lock:
                if self.kb is not None and self.loaded_index_path == index_path:
                    return self.kb
                error_message = self.loading_errors.get(index_path)
            if error_message:
                raise RuntimeError(error_message)
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        try:
            kb = LocalKnowledgeBase.load(index_path)
            with self._lock:
                self.kb = kb
                self.loaded_index_path = index_path
            return kb
        except Exception as error:
            with self._lock:
                self.loading_errors[index_path] = str(error)
            raise
        finally:
            with self._lock:
                current_event = self.loading_indexes.pop(index_path, None)
                if current_event is not None:
                    current_event.set()

    def start_warmup(self, index_path: str = DEFAULT_INDEX_PATH) -> None:
        with self._lock:
            if self.warmup_started:
                return
            self.warmup_started = True

        Thread(target=self._warmup_index, args=(index_path,), daemon=True).start()

    def _warmup_index(self, index_path: str) -> None:
        try:
            if Path(index_path).exists():
                self.ensure_loaded(index_path)
        except Exception as error:
            with self._lock:
                self.warmup_error = str(error)

    def ask(self, question: str, index_path: str, top_k: int = DEFAULT_TOP_K) -> dict:
        load_dotenv()
        kb = self.ensure_loaded(index_path)
        results = kb.search(question, top_k=top_k)
        if not results:
            return {
                "answer": "未检索到相关内容，请尝试更换问题或重新建立索引。",
                "results": [],
            }

        api_key = get_env("DASHSCOPE_API_KEY")
        model = get_env("QWEN_MODEL", DEFAULT_QWEN_MODEL)
        base_url = get_env("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)

        chat = QwenChat(api_key=api_key, model=model, base_url=base_url)
        answer = chat.answer(question, results)
        return {
            "answer": answer,
            "results": results,
        }

    def status(self, index_path: str = DEFAULT_INDEX_PATH) -> dict:
        index_file = Path(index_path)
        index_exists = index_file.exists()
        loaded = self.kb is not None and self.loaded_index_path == index_path
        document_count = len(self.kb.documents) if loaded and self.kb is not None else 0
        active_job = self.build_jobs.get(self.active_build_job_id or "")
        warming_up = index_path in self.loading_indexes
        return {
            "index_exists": index_exists,
            "index_path": index_path,
            "loaded": loaded,
            "document_count": document_count,
            "api_key_configured": bool(os.getenv("DASHSCOPE_API_KEY")),
            "build_job": active_job.to_dict() if active_job else None,
            "warming_up": warming_up,
            "warmup_error": self.warmup_error if index_path == DEFAULT_INDEX_PATH else None,
        }


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


service = KnowledgeBaseService()
service.start_warmup()
app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_manifest() -> Response:
    return Response(status=204)


@app.get("/api/status")
def api_status():
    load_dotenv()
    index_path = request.args.get("index", DEFAULT_INDEX_PATH)
    return jsonify(service.status(index_path=index_path))


@app.post("/api/build")
def api_build():
    load_dotenv()
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", DEFAULT_SOURCE_DIR)
    index_path = payload.get("index", DEFAULT_INDEX_PATH)
    chunk_size = int(payload.get("chunk_size", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(payload.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))

    result = service.start_build_index(
        source=source,
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return jsonify(result), 202


@app.get("/api/build-status")
def api_build_status():
    job_id = (request.args.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"error": "缺少 job_id 参数"}), 400
    return jsonify(service.get_build_job(job_id))


@app.post("/api/ask")
def api_ask():
    load_dotenv()
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    index_path = payload.get("index", DEFAULT_INDEX_PATH)
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    result = service.ask(question=question, index_path=index_path, top_k=top_k)
    return jsonify(result)


@app.errorhandler(Exception)
def handle_exception(error):
    if isinstance(error, HTTPException):
        response = error.get_response()
        response.data = json.dumps(
            {
                "error": error.description,
                "code": error.code,
                "name": error.name,
            },
            ensure_ascii=False,
        )
        response.content_type = "application/json; charset=utf-8"
        return response

    return jsonify({"error": str(error)}), 500


def build_command(args: argparse.Namespace) -> None:
    result = service.build_index(
        source=args.source,
        index_path=args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"已建立索引，共 {result['chunks']} 个文本分片，保存到 {result['index']}"
    )


def ask_command(args: argparse.Namespace) -> None:
    result = service.ask(question=args.question, index_path=args.index, top_k=args.top_k)
    print("=" * 80)
    print("检索结果：")
    for item in result["results"]:
        print(f"- {item['path']}#chunk-{item['chunk_id']} | score={item['score']:.4f}")
    print("=" * 80)
    print("回答：")
    print(result["answer"])


def serve_command(args: argparse.Namespace) -> None:
    load_dotenv()
    app.run(host=args.host, port=args.port, debug=False)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地知识库 + 千问远程模型问答")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="扫描目录并建立本地索引")
    build_parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="需要建立知识库的目录")
    build_parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="索引输出文件")
    build_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="分片长度")
    build_parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="分片重叠长度")
    build_parser.set_defaults(func=build_command)

    ask_parser = subparsers.add_parser("ask", help="基于本地索引进行问答")
    ask_parser.add_argument("question", help="用户问题")
    ask_parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="已有索引文件")
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="检索返回分片数")
    ask_parser.set_defaults(func=ask_command)

    serve_parser = subparsers.add_parser("serve", help="启动 Web 服务")
    serve_parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=7860, help="监听端口")
    serve_parser.set_defaults(func=serve_command)

    return parser


def main() -> None:
    load_dotenv()
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
