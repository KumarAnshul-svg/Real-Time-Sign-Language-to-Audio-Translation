import sys
import torch
import cv2
import mediapipe as mp
import pyttsx3
ex
def check_python():
    print("===== PYTHON CHECK =====")
    print("Python Version:", sys.version)
    if sys.version_info.major == 3 and sys.version_info.minor == 10:
        print("Python version is correct (3.10)")
    else:
        print("WARNING: Recommended Python version is 3.10")
    print()


def check_gpu():
    print("===== GPU CHECK =====")
    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)

    if cuda_available:
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
        print("GPU Memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    else:
        print("WARNING: CUDA not available. You are using CPU version of PyTorch.")
    print()


def check_opencv():
    print("===== OPENCV CHECK =====")
    print("OpenCV Version:", cv2.__version__)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("WARNING: Webcam not detected.")
    else:
        print("Webcam detected successfully.")
        cap.release()
    print()


def check_mediapipe():
    print("===== MEDIAPIPE CHECK =====")
    try:
        mp_hands = mp.solutions.hands
        print("MediaPipe loaded successfully.")
    except Exception as e:
        print("MediaPipe error:", e)
    print()


def check_tts():
    print("===== TEXT-TO-SPEECH CHECK =====")
    try:
        engine = pyttsx3.init()
        engine.say("Environment setup complete.")
        engine.runAndWait()
        print("TTS working correctly.")
    except Exception as e:
        print("TTS error:", e)
    print()


def main():
    print("\n=========== ENVIRONMENT TEST START ===========\n")
    check_python()
    check_gpu()
    check_opencv()
    check_mediapipe()
    check_tts()
    print("=========== ENVIRONMENT TEST COMPLETE ===========")


if __name__ == "__main__":
    main()