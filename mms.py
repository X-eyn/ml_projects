from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
import sounddevice as sd
import numpy as np

# Load the dataset and process audio (Bengali example)
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "bn", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
bn_sample = next(iter(stream_data))["audio"]["array"]

# Play audio preview
sd.play(bn_sample, 16000)
sd.wait()  # Wait until the audio finishes playing

# Load the model and processor
model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# Set the target language to Bengali (ISO 639-3 code: 'ben')
processor.tokenizer.set_target_lang("ben")
model.load_adapter("ben")

# Prepare input and transcribe
inputs = processor(bn_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)

# Write the transcription to a text file
with open("transcription_output.txt", "w", encoding="utf-8") as f:
    f.write("Transcription: " + transcription + "\n")

print("Transcription written to 'transcription_output.txt'")
