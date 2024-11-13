import importlib
import torch
import torchaudio
import cv2
import dlib
from mtcnn import MTCNN

# Function to check installation
def check_import(library_name):
    try:
        importlib.import_module(library_name)
        print(f"{library_name} is installed correctly.")
    except ImportError:
        print(f"ERROR: {library_name} is NOT installed.")

# Verify OpenAI Whisper (if installed via openai-whisper)
def check_whisper():
    try:
        import whisper
        print("OpenAI Whisper is installed correctly.")
    except ImportError:
        print("ERROR: OpenAI Whisper is NOT installed.")

# Check OpenCV installation by verifying its version
def check_opencv():
    print("Checking OpenCV version...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print("OpenCV is installed correctly.")
    except Exception as e:
        print(f"ERROR with OpenCV: {e}")

# Check torch and torchaudio installations
def check_torch_torchaudio():
    print("Checking PyTorch and Torchaudio versions...")
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchaudio version: {torchaudio.__version__}")
        print("PyTorch and Torchaudio are installed correctly.")
    except Exception as e:
        print(f"ERROR with PyTorch or Torchaudio: {e}")

# Check dlib installation by creating a dlib face detector
def check_dlib():
    try:
        detector = dlib.get_frontal_face_detector()
        print("dlib is installed correctly. Face detector created.")
    except Exception as e:
        print(f"ERROR with dlib: {e}")

# Check MTCNN (Make sure `mtcnn` is installed correctly)
def check_mtcnn():
    try:
        mtcnn = MTCNN()
        print("MTCNN is installed correctly.")
    except Exception as e:
        print(f"ERROR with MTCNN: {e}")

# Run the checks
if __name__ == "__main__":
    check_import("whisper")
    check_opencv()
    check_torch_torchaudio()
    check_dlib()
    check_mtcnn()
