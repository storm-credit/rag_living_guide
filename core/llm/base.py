from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """LLM 추상 인터페이스"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """프롬프트를 받아 텍스트를 생성합니다."""
        pass

    @abstractmethod
    def identify(self) -> str:
        """모델명을 반환합니다."""
        pass
