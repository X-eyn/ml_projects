from abc import ABC, abstractmethod
from typing import Optional

class SpeechRecognizer(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def transcribe(self, audio_data) -> str:
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        pass
