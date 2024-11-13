import threading
import cv2
from speech_to_text import transcribe_real_time
from face_recognition import detect_and_recognize_faces


def main():

    running = threading.Event()
    running.set()

    transcriber = None

    def speech_recognition_thread():
        nonlocal transcriber
        print("Starting real-time speech recognition...")
        transcriber = transcribe_real_time()

    speech_thread = threading.Thread(target=speech_recognition_thread)
    speech_thread.start()

    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_and_recognize_faces(frame)
        cv2.imshow("Face Detection", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping all processes...")
            running.clear()
            if transcriber:
                transcriber.stop_transcription()
            break

    cap.release()
    cv2.destroyAllWindows()

    speech_thread.join()
    print("All processes stopped successfully")


if __name__ == "__main__":
    main()
