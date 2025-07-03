from core.llm.base import BaseLLM
import subprocess
import logging

logger = logging.getLogger("RAG.Mistral")

class MistralLLM(BaseLLM):
    """Ollama 기반 Mistral 실행 모듈"""

    def generate(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", "mistral", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            if result.returncode != 0:
                logger.error(f"Ollama error: {result.stderr.strip()}")
                return f"[오류] ollama 실행 실패"
            return result.stdout.strip()
        except Exception as e:
            logger.exception("MistralLLM 예외 발생")
            return f"[예외 발생] {e}"

    def identify(self) -> str:
        return "mistral"
