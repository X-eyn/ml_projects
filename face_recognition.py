import cv2
from mtcnn import MTCNN
import dlib


detector = MTCNN()


face_recognizer = dlib.get_frontal_face_detector()


def detect_and_recognize_faces(frame):
    detected_faces = detector.detect_faces(frame)
    if detected_faces:
        for face in detected_faces:
            x, y, width, height = face['box']
            confidence = face['confidence']

            cv2.rectangle(frame, (x, y), (x + width,
                          y + height), (255, 0, 0), 2)

            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
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
