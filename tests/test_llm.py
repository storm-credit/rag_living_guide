import pytest
from core.llm.mistral import MistralLLM
from core.llm.koalpaca import KoAlpacaLLM

@pytest.mark.parametrize("cls, name", [
    (MistralLLM, "mistral"),
    (KoAlpacaLLM, "koalpaca"),
])
def test_identify_and_generate(cls, name):
    llm = cls()
    assert llm.identify() == name
    # generate()의 외부 호출은 Mocking 필요하므로 단순 식별만 테스트
