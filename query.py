import os
import pickle

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from local_embeddings import LocalTfidfEmbeddings

load_dotenv()

# 1. 加载本地 embedding 状态并恢复向量器
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

embeddings = LocalTfidfEmbeddings()
embeddings.vectorizer = vectorizer
embeddings._is_fitted = True

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# 2. 检索器
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. 通义千问 LLM（OpenAI 兼容接口）
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 4. 检索并拼接上下文
question = "你的文档内容是什么？"
docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in docs if isinstance(doc, Document) or hasattr(doc, "page_content"))

prompt = f"""请仅根据以下知识库内容回答问题；如果知识库中没有答案，就明确说不知道。

知识库内容：
{context}

问题：
{question}
"""

# 5. 调用通义千问生成答案
response = llm.invoke(prompt)
print("回答:", response.content)