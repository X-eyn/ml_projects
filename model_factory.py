class ModelFactory:
    @staticmethod
    def get_face_detector(config: ModelConfig) -> FaceDetector:
        if config.model_type.lower() == "mtcnn":
            return MTCNNDetector(config)
        # Add more models here
        raise ValueError(f"Unsupported face detection model: {config.model_type}")
    
    @staticmethod
    def get_speech_recognizer(config: ModelConfig) -> SpeechRecognizer:
        if config.model_type.lower() == "whisper":
            return WhisperRecognizer(config)
        elif config.model_type.lower() == "canary":
            return CanaryRecognizer(config)
        # Add more models here
        raise ValueError(f"Unsupported speech recognition model: {config.model_type}")
