from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict

class FaceDetector(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def detect(self, frame) -> Tuple[List[Dict], np.ndarray]:
        """Returns: Tuple of (face_detections, processed_frame)"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        pass