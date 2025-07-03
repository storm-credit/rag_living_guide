from langchain.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import os

class VectorStore:
    def __init__(self, embedding_model_path: str, index_path: str):
        self.embedding_model_path = embedding_model_path
        self.index_path = index_path
        self.model = SentenceTransformer(embedding_model_path)
        self.db = None

    def build(self, docs: list[Document]):
        """
        docs: langchain.schema.Document list
        """
        self.db = FAISS.from_documents(docs, self.model)
        os.makedirs(self.index_path, exist_ok=True)
        self.db.save_local(self.index_path)

    def load(self) -> FAISS:
        """
        로컬에 저장된 인덱스를 불러옵니다.
        """
        self.db = FAISS.load_local(self.index_path, self.model)
        return self.db
