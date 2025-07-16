from abc import ABC, abstractmethod
from PIL import Image

class VisionLLM(ABC):
    @abstractmethod
    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        pass

class TextLLM(ABC):
    @abstractmethod
    def call_text_llm(self, prompt: str) -> str:
        pass