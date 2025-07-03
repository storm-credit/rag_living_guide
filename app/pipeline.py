from core.embedder import Embedder
from core.retriever import Retriever
from core.llm.base import BaseLLM
from core.config import Settings
from core.logger import logger

class RAGPipeline:
    def __init__(self, llm: BaseLLM, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.EMBEDDING_MODEL)
        self.retriever = Retriever(settings)
        self.llm = llm

    def run(self, query: str) -> str:
        logger.info(f"Received query: {query}")
        docs = self.retriever.search(query)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        answer = self.llm.generate(prompt)
        logger.info(f"Generated answer, length={len(answer)}")
        return answer
