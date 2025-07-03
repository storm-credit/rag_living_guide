import pytest
from app.pipeline import RAGPipeline
from langchain.schema import Document

class DummyLLM:
    def generate(self, prompt):
        return "dummy answer"

class DummyRetriever:
    def search(self, query, top_k=5):
        return [Document(page_content="fake context")]

@pytest.fixture
def settings():
    from core.config import Settings
    return Settings()

def test_pipeline_run(settings):
    dummy_llm = DummyLLM()
    pipeline = RAGPipeline(llm=dummy_llm, settings=settings)
    pipeline.retriever = DummyRetriever()
    answer = pipeline.run("test query")
    assert answer == "dummy answer"
