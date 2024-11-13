from mtcnn import MTCNN
import cv2

class MTCNNDetector(FaceDetector):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self.load_model()
        self.confidence_threshold = config.config.get('confidence_threshold', 0.9)
    
    def load_model(self):
        return MTCNN()
    
    @property
    def model_type(self) -> str:
        return "mtcnn"
    
    def detect(self, frame):
        detected_faces = self.model.detect_faces(frame)
        processed_frame = frame.copy()
        
        filtered_faces = []
        for face in detected_faces:
            if face['confidence'] >= self.confidence_threshold:
                filtered_faces.append(face)
                x, y, width, height = face['box']
                confidence = face['confidence'] * 100
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                
                # Display confidence score
                score_text = f"Conf: {confidence:.2f}%"
                cv2.putText(processed_frame, score_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return filtered_faces, processed_frame