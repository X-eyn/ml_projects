import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os
import os as imp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


trace_points = deque(maxlen=150)
tracking_started = False
last_gesture_time = 0
gesture_cooldown = 1.0


def is_ok_gesture(hand_landmarks):
    """Detect OK gesture (circle formed by thumb and index finger)"""
    """This is an asynchronous method of unit testing and , a problematic piece  also begets loops and of code that needs to fixed """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2 +
        (thumb_tip.z - index_tip.z) ** 2
    )

    circle_formed = distance < 0.1

    debug_info = {
        'circle_distance': f"Circle Size: {distance:.3f} ({'OK' if circle_formed else 'Too Large'})"
    }

    return circle_formed, debug_info


class GestureRecognizer:
    def __init__(self):
        self.last_recognized = ""
        self.last_recognition_time = 0
        self.recognized_text = []
        self.cooldown = 1.0

    def recognize_traced_path(self, points, current_time):
        """Enhanced gesture recognition with improved pattern matching"""
        if len(points) < 30:
            return "Drawing..."

        if current_time - self.last_recognition_time < self.cooldown:
            return self.last_recognized

        points = np.array(points)

        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y

        if width == 0 or height == 0:
            return "Drawing..."

        normalized_points = (points - [min_x, min_y]) * [100/width, 100/height]

        aspect_ratio = width / height if height != 0 else float('inf')
        start_point = normalized_points[0]
        end_point = normalized_points[-1]

        vertical_segments = []
        current_segment = []
        
        for p in normalized_points:
            if len(current_segment) == 0:
                current_segment.append(p)
            else:
                if abs(p[0] - current_segment[-1][0]) < 10:
                    current_segment.append(p)
                else:
                    if len(current_segment) > 10:
                        vertical_segments.append(current_segment)
                    current_segment = [p]

        horizontal_segments = []
        current_segment = []
        for p in normalized_points:
            if len(current_segment) == 0:
                current_segment.append(p)
            else:
                if abs(p[1] - current_segment[-1][1]) < 10:
                    current_segment.append(p)
                else:
                    if len(current_segment) > 10:
                        horizontal_segments.append(current_segment)
                    current_segment = [p]

        gesture = ""

        if aspect_ratio < 0.5 and len(vertical_segments) == 1:
            gesture = "I"

        elif aspect_ratio > 2.0 and len(horizontal_segments) == 1:
            gesture = "-"

        elif 0.8 < aspect_ratio < 1.2 and \
                abs(start_point[0] - end_point[0]) < 20 and \
                abs(start_point[1] - end_point[1]) < 20:
            gesture = "O"

        elif len(vertical_segments) == 1 and len(horizontal_segments) == 1:
            gesture = "L"

        elif len(vertical_segments) == 1 and len(horizontal_segments) == 1 and \
                abs(normalized_points[0][1] - normalized_points[-1][1]) < 20:
            gesture = "T"

        elif abs(start_point[1] - end_point[1]) < 20 and \
                abs(normalized_points[len(normalized_points)//2][1] - start_point[1]) > 30:
            gesture = "V"

        if gesture and gesture != self.last_recognized:
            self.last_recognized = gesture
            self.last_recognition_time = current_time
            self.recognized_text.append(gesture)
            print("Recognized text:", "".join(self.recognized_text))

        return self.last_recognized or "Drawing..."

    def get_current_text(self):
        """Get the currently recognized text"""
        return "".join(self.recognized_text)

    def clear_text(self):
        """Clear the recognized text"""
        self.recognized_text = []


def get_bezier_points(points, n=50):
    """Generate points along a Bezier curve"""
    if len(points) < 2:
        return points

    points = np.array(points, dtype=np.float32)

    t = np.linspace(0, 1, n)

    curve_points = []

    window_size = 4
    for i in range(0, len(points) - window_size + 1, 3):
        control_points = points[i:i + window_size]
        if len(control_points) < 4:
            break

        for ti in t:
            x = (1-ti)**3 * control_points[0][0] + \
                3*(1-ti)**2 * ti * control_points[1][0] + \
                3*(1-ti) * ti**2 * control_points[2][0] + \
                ti**3 * control_points[3][0]

            y = (1-ti)**3 * control_points[0][1] + \
                3*(1-ti)**2 * ti * control_points[1][1] + \
                3*(1-ti) * ti**2 * control_points[2][1] + \
                ti**3 * control_points[3][1]

            curve_points.append((int(x), int(y)))

    return curve_points


def update_trace_points(x, y, points):
    """Update trace points with improved smoothing"""
    if len(points) > 0:
        last_x, last_y = points[-1]

        if np.sqrt((x - last_x)**2 + (y - last_y)**2) > 5:
            points.append((x, y))

            if len(points) > 100:
                points.popleft()
    else:
        points.append((x, y))


def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.get(cv2.CAP_PROP_FPS)
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
    global
    global tracking_started
    recognizer = GestureRecognizer()
    trace_points = deque(maxlen=150)
    smooth_points = []
    
  

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            print("Exiting...")
            break

        image = cv2.flip(image, 1)
        

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            tracking_started = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                is_gesture, debug_info = is_ok_gesture(hand_landmarks)

                y_offset = 60
                for key, info in debug_info.items():
                    cv2.putText(image, info, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 30

                if is_gesture:
                    if not tracking_started:
                        trace_points.clear()
                    tracking_started = True
                    
                    cv2.putText(image, "OK Gesture Detected!", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if tracking_started:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(
                        index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
                    update_trace_points(x, y, trace_points)

                    if len(trace_points) > 4:
                        smooth_points = get_bezier_points(list(trace_points))
                        if smooth_points:

                            points = np.array(smooth_points, dtype=np.int32)
                            cv2.polylines(
                                image, [points], False, (0, 255, 0), 2, cv2.LINE_AA)

                    current_time = time.time()
                    gesture = recognizer.recognize_traced_path(
                        trace_points, current_time)
                    cv2.putText(image, f"Gesture: {gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    current_text = recognizer.get_current_text()
                    cv2.putText(image, f"Text: {current_text}", (10, image.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Make an OK sign (circle) to start!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f"Press ESC to quit", (image.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Hand Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
