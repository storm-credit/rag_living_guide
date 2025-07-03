import pytest
from core.retriever import Retriever
from core.config import Settings

@pytest.fixture
def settings():
    return Settings()

def test_search_returns_list(settings):
    r = Retriever(settings)
    results = r.search("test")
    assert isinstance(results, list)
