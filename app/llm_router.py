from core.llm.koalpaca import KoAlpacaLLM
from core.llm.mistral import MistralLLM
from core.retriever import Retriever

class LLMRouter:
    """
    Detect language and route to the proper LLM.
    Korean -> KoAlpaca, Other -> Mistral
    """
    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()
        self.ko = KoAlpacaLLM()
        self.en = MistralLLM()

    def detect_language(self, text: str) -> str:
        return "ko" if any("\uac00" <= ch <= "\ud7a3" for ch in text) else "en"

    def generate(self, user_input: str, top_k: int = 3) -> str:
        lang = self.detect_language(user_input)
        docs = self.retriever.search(user_input, top_k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}\n\nAnswer:"
        return self.ko.generate(prompt) if lang == "ko" else self.en.generate(prompt)
