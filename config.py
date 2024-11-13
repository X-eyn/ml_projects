from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    model_type: str
    model_path: str
    config: Dict[str, Any] = None

class AISystemConfig:
    def __init__(self):
        self.face_detection = ModelConfig(
            model_type="mtcnn",
            model_path="default",
            config={"confidence_threshold": 0.9}
        )
        self.speech_recognition = ModelConfig(
            model_type="whisper",
            model_path="base",
            config={"language": "en"}
        )