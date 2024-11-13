# speech_recognition_canary.py
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
from speech_recognition_base import SpeechRecognizer
from config import ModelConfig
import numpy as np

class CanaryRecognizer(SpeechRecognizer):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = self.load_model()
        
    def load_model(self):
        processor = AutoProcessor.from_pretrained(self.config.model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        return processor, model
    
    @property
    def model_type(self) -> str:
        return "canary"
    
    def transcribe(self, audio_data) -> str:
        # Convert audio data to the correct format (ensure it's normalized between -1 and 1)
        if audio_data.dtype == np.float32:
            # Already in correct format
            audio_np = audio_data
        else:
            # Convert to float32 and normalize if needed
            audio_np = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
        # Process the audio
        inputs = self.processor(
            audio_np,
            sampling_rate=16000,  # Ensure this matches your audio capture rate
            return_tensors="pt"
        ).to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                max_length=256,
                **self.config.config.get("generation_config", {})
            )
            
        # Decode the generated tokens
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()

