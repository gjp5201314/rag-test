from typing import List

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class LocalTfidfEmbeddings(Embeddings):
    """完全本地的 TF-IDF 向量实现，不依赖外网下载模型。"""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._is_fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._is_fitted:
            self.fit(texts)
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._is_fitted:
            raise ValueError("Embedding 模型尚未拟合，请先运行 main.py 构建向量库")
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()
