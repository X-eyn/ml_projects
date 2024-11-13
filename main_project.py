import threading

from speech_to_text import transcribe_real_time  # Updated to import the real-time function

from face_recognition import detect_and_recognize_faces

import cv2



# Thread to handle real-time speech recognition

def speech_recognition_thread():

    print("Starting real-time speech recognition...")

    transcribe_real_time()  # Call the real-time transcription function



# Main function for face detection

def main_face_detection():

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



# Start the speech recognition in a separate thread

speech_thread = threading.Thread(target=speech_recognition_thread)

speech_thread.start()



# Run face detection

main_face_detection()
