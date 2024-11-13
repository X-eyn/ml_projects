import whisper
import numpy as np

class WhisperRecognizer(SpeechRecognizer):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self.load_model()
    
    def load_model(self):
        return whisper.load_model(self.config.model_path)
    
    @property
    def model_type(self) -> str:
        return "whisper"
    
    def transcribe(self, audio_data) -> str:
        result = self.model.transcribe(audio_data, language=self.config.config.get('language', 'en'))
        return result['text'].strip()