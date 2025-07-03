from core.llm.base import BaseLLM
import requests
import os
import logging

logger = logging.getLogger("RAG.KoAlpaca")

class KoAlpacaLLM(BaseLLM):
    """KoAlpaca API 호출 모듈"""

    def __init__(self):
        self.api_url = os.getenv("KOALPACA_API_URL", "http://localhost:8001/generate")
        self.api_key = os.getenv("KOALPACA_API_KEY", None)

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7
        }
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except requests.RequestException as e:
            logger.error(f"KoAlpaca 오류: {e}")
            return f"[KoAlpaca 오류] {e}"

    def identify(self) -> str:
        return "koalpaca"
