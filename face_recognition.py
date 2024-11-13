import cv2
from mtcnn import MTCNN
import dlib

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize DLIB for face recognition
face_recognizer = dlib.get_frontal_face_detector()

def detect_and_recognize_faces(frame):
    detected_faces = detector.detect_faces(frame)
    if detected_faces:
        for face in detected_faces:
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            # Optionally, use dlib for further processing if needed
    return frame

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_and_recognize_faces(frame)
        cv2.imshow("Face Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
