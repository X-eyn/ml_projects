import cv2
import threading

def main():
    # Initialize configuration with Canary model
    config = AISystemConfig()
    config.speech_recognition = ModelConfig(
        model_type="canary",
        model_path=r"C:\Users\Admin\Downloads\canary-1b.nemo",  # or your specific Canary model path
        config={
            "generation_config": {
                "temperature": 0.2,
                "do_sample": False,
                "num_beams": 1,
            }
        }
    )
     
    # Create models using factory
    face_detector = ModelFactory.get_face_detector(config.face_detection)
    speech_recognizer = ModelFactory.get_speech_recognizer(config.speech_recognition)
    
    # Initialize audio transcriber
    transcriber = AudioTranscriber(speech_recognizer)
    
    # Start speech recognition in a separate thread
    speech_thread = threading.Thread(target=transcriber.start_transcription)
    speech_thread.start()
    
    # Main face detection loop
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces and get processed frame
            faces, processed_frame = face_detector.detect(frame)
            
            cv2.imshow("Face Detection", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        transcriber.stop_transcription()
        speech_thread.join()

if __name__ == "__main__":
    main()