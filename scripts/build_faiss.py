from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os

EMBED_PATH = "trained_models/embedding_model"
DATA_DIR = "data/processed"
OUT_DIR = "vectorstore/index"

def load_docs():
    docs = []
    for fn in os.listdir(DATA_DIR):
        if fn.endswith(".txt"):
            text = open(os.path.join(DATA_DIR, fn), encoding="utf-8").read()
            docs.append(Document(page_content=text, metadata={"source": fn}))
    return docs

if __name__ == "__main__":
    model = SentenceTransformer(EMBED_PATH)
    docs = load_docs()
    db = FAISS.from_documents(docs, model)
    os.makedirs(OUT_DIR, exist_ok=True)
    db.save_local(OUT_DIR)
    print("✅ FAISS 인덱스 생성됨:", OUT_DIR)
