import pytest
from core.config import Settings
from core.llm.base import BaseLLM

class DummyLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        return "dummy answer"
    def identify(self) -> str:
        return "dummy"

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def dummy_llm():
    return DummyLLM()
