import pickle
import shutil
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_embeddings import LocalTfidfEmbeddings

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents_from_data_dir() -> list[Document]:
    documents: list[Document] = []

    if not DATA_DIR.exists():
        raise FileNotFoundError("未找到 data 文件夹，请先创建并放入 .txt 或 .md 文件")

    for file_path in sorted(DATA_DIR.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(file_path)},
            )
        )

    if not documents:
        raise ValueError("data 文件夹中没有可用的 .txt 或 .md 文件")

    return documents


def remove_vector_store_with_retry(retries: int = 5, delay: float = 1.0) -> None:
    if not CHROMA_DIR.exists():
        return

    last_error = None
    for _ in range(retries):
        try:
            shutil.rmtree(CHROMA_DIR)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)

    raise PermissionError(
        "无法删除 chroma_db，通常是因为 app.py 或其他进程正在占用向量库文件。"
        "请先停止正在运行的检索/Web 服务后再执行 python main.py。"
    ) from last_error



def rebuild_vector_store() -> None:
    # 1. 读取 data 文件夹中的文档
    documents = load_documents_from_data_dir()

    # 2. 文档切分（更适合中文与长文本，避免超大块文本影响检索）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    docs = text_splitter.split_documents(documents)

    # 对超短文档保留原文作为独立片段，避免在海量长文片段中被稀释
    extra_docs: list[Document] = []
    for doc in documents:
        content = doc.page_content.strip()
        if content and len(content) <= 1000:
            extra_docs.append(
                Document(
                    page_content=content,
                    metadata={**doc.metadata, "chunk_type": "full_document"},
                )
            )

    docs.extend(extra_docs)

    # 3. 完全本地 embedding（不依赖 Hugging Face 在线下载）
    embeddings = LocalTfidfEmbeddings()
    texts = [doc.page_content for doc in docs]
    embeddings.fit(texts)

    # 4. 重建向量数据库，避免旧 embedding 维度残留
    remove_vector_store_with_retry()

    if VECTORIZER_PATH.exists():
        VECTORIZER_PATH.unlink()

    Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # 5. 保存向量器，供 query.py 和 app.py 查询时复用
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(embeddings.vectorizer, f)

    print(
        f"向量库构建完成，共加载 {len(documents)} 个文件，"
        f"切分并补充得到 {len(docs)} 个片段（含 {len(extra_docs)} 个短文档整篇片段）"
    )


if __name__ == "__main__":
    rebuild_vector_store()