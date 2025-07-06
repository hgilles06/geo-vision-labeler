from abc import ABC, abstractmethod

class VLMAgent(ABC):
    @abstractmethod
    def run(self, images):
        pass