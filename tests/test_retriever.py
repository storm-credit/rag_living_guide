def test_search():
    from core.retriever import Retriever
    r = Retriever()
    assert isinstance(r.search('hello'), list)