import os
import pickle
from typing import List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from local_embeddings import LocalTfidfEmbeddings

load_dotenv()

app = Flask(__name__)


def load_vector_store():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    embeddings = LocalTfidfEmbeddings()
    embeddings.vectorizer = vectorizer
    embeddings._is_fitted = True

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )


def load_llm():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY，请先在 .env 中填写通义千问密钥")

    return ChatOpenAI(
        model="qwen-turbo",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def build_prompt(question: str, docs: List[Document]) -> str:
    context = "\n\n".join(
        doc.page_content for doc in docs if isinstance(doc, Document) or hasattr(doc, "page_content")
    )

    return f"""你是一个知识库问答助手。请严格根据提供的知识库内容回答问题。
如果知识库中没有明确答案，请直接回答“我不知道”。
如果知识库内容里已经出现了明确事实，请直接提炼作答，不要因为措辞不同就拒答。
回答尽量简洁、准确，优先输出结论，不要复述无关内容。

知识库内容：
{context}

用户问题：
{question}
"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "message 不能为空"}), 400

    try:
        db = load_vector_store()
        llm = load_llm()

        docs = db.similarity_search(question, k=8)

        if not docs:
            return jsonify(
                {
                    "answer": "我不知道",
                    "sources": [],
                }
            )

        prompt = build_prompt(question, docs)
        response = llm.invoke(prompt)

        answer = (response.content or "").strip()
        if answer in {"我不知道", "不知道", "抱歉，我不知道", "抱歉，我不清楚"}:
            fallback_context = "\n\n".join(doc.page_content for doc in docs[:4])
            fallback_prompt = f"""请严格根据下面检索到的知识库内容回答问题，不要脱离内容自由发挥。
如果内容中有明确答案，就直接用一句中文回答，不要说'我不知道'。
如果内容中确实没有答案，才回答“我不知道”。

知识库内容：
{fallback_context}

问题：{question}
"""
            fallback_response = llm.invoke(fallback_prompt)
            fallback_answer = (fallback_response.content or "").strip()
            if fallback_answer:
                answer = fallback_answer

        sources = [
            f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content[:200]}"
            for doc in docs
        ]
        return jsonify(
            {
                "answer": answer or "我不知道",
                "sources": sources,
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "缺少 tfidf_vectorizer.pkl，请先运行 python main.py 构建知识库"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
