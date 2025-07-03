from langchain.embeddings import HuggingFaceEmbeddings
from functools import lru_cache

class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)

    @lru_cache(maxsize=128)
    def embed_single(self, text: str) -> list[float]:
        return self.embedding.embed([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embedding.embed(texts)
