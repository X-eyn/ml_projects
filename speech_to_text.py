import whisper

import pyaudio

import numpy as np

import torch



# Initialize Whisper model

model = whisper.load_model("base")



def transcribe_real_time():

    # Set up microphone stream using PyAudio

    audio_format = pyaudio.paInt16

    channels = 1

    rate = 16000

    chunk_size = 1024



    audio = pyaudio.PyAudio()

    stream = audio.open(format=audio_format, channels=channels,

                        rate=rate, input=True, frames_per_buffer=chunk_size)

    print("Listening...")



    try:

        while True:

            # Capture audio data

            audio_data = stream.read(chunk_size)

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0



            # Process and transcribe audio

            if len(audio_np) > 0:

                mel = whisper.log_mel_spectrogram(torch.tensor(audio_np))

                result = model.transcribe(mel)

                print("Transcription:", result['text'])



    except KeyboardInterrupt:

        print("Stopping...")



    # Clean up

    stream.stop_stream()

    stream.close()

    audio.terminate()
