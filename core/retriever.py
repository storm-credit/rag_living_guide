from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from core.config import Settings

class Retriever:
    def __init__(self, settings: Settings):
        embed_model = settings.EMBEDDING_MODEL
        index_path = settings.VECTORSTORE_PATH
        self.embedding = HuggingFaceEmbeddings(model_name=embed_model)
        self.db = FAISS.load_local(index_path, self.embedding)

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        query: 검색어
        top_k: 반환할 문서 수
        """
        return self.db.similarity_search(query, k=top_k)
