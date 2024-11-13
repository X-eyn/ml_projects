import pyaudio
import queue
import threading
import numpy as np
import time

class AudioTranscriber:
    def __init__(self, recognizer: SpeechRecognizer):
        self.recognizer = recognizer
        self.audio_queue = queue.Queue()
        self.keep_running = True
        
        # Audio configuration
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 1024 * 4
        self.audio = pyaudio.PyAudio()
        
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_seconds = 3
        self.samples_per_buffer = int(self.rate * self.buffer_seconds)

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        while self.keep_running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                self.buffer = np.append(self.buffer, audio_data)

                if len(self.buffer) >= self.samples_per_buffer:
                    audio_data = self.buffer[:self.samples_per_buffer]
                    self.buffer = self.buffer[self.samples_per_buffer:]

                    try:
                        text = self.recognizer.transcribe(audio_data)
                        if text:
                            print("Transcription:", text)
                    except Exception as e:
                        print(f"Transcription error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process_audio: {e}")

    def start_transcription(self):
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        print("Started audio stream")
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
        
        print("Listening... (Press Ctrl+C to stop)")

    def stop_transcription(self):
        self.keep_running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()