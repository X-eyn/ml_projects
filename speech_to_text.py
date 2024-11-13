import whisper
import pyaudio
import numpy as np
import queue
import threading
import time

class AudioTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.audio_queue = queue.Queue()
        self.keep_running = True
        
        # Audio configuration
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 1024 * 4  # Increased chunk size
        self.audio = pyaudio.PyAudio()
        
        # Buffer to accumulate audio
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_seconds = 3  # Process 3 seconds of audio at a time
        self.samples_per_buffer = int(self.rate * self.buffer_seconds)

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        while self.keep_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1)
                self.buffer = np.append(self.buffer, audio_data)

                # Process when buffer reaches desired size
                if len(self.buffer) >= self.samples_per_buffer:
                    # Convert audio to format expected by Whisper
                    audio_data = self.buffer[:self.samples_per_buffer]
                    self.buffer = self.buffer[self.samples_per_buffer:]

                    # Transcribe
                    try:
                        result = self.model.transcribe(audio_data, language='en')
                        if result['text'].strip():  # Only print if there's actual text
                            print("Transcription:", result['text'].strip())
                    except Exception as e:
                        print(f"Transcription error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process_audio: {e}")

    def start_transcription(self):
        # Start audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        print("Started audio stream")
        
        # Start processing thread
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

def transcribe_real_time():
    transcriber = AudioTranscriber()
    try:
        transcriber.start_transcription()
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        transcriber.stop_transcription()

if __name__ == "__main__":
    transcribe_real_time()