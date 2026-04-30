# utils/realtime_asl.py

import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
import time
import pyttsx3
import threading
from collections import Counter

from model import ASLModel
from inference_utils import normalize_landmarks, predict

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL + LABELS
# =========================

label_encoder = joblib.load("../labels/asl_label_encoder.pkl")

model = ASLModel(42, len(label_encoder.classes_)).to(device)
model.load_state_dict(torch.load("../models/asl_model.pth", map_location=device))
model.eval()

# =========================
# TEXT TO SPEECH
# =========================

engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 150)

def speak_text(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

# =========================
# LOGIC VARIABLES
# =========================

prediction_buffer = []
BUFFER_SIZE = 12
CONF_THRESHOLD = 0.85

current_word = ""
last_added_letter = ""
last_add_time = 0
ADD_DELAY = 1.2

hand_present = False
last_hand_time = 0
spoke_for_this_word = True   # start as True so it doesn’t speak empty words

prev_time = 0

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""
    confidence_text = ""

    # =========================
    # HAND DETECTED
    # =========================

    if results.multi_hand_landmarks:
        hand_present = True
        last_hand_time = time.time()

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = normalize_landmarks(hand_landmarks)
            input_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)

            predicted_index, confidence_value = predict(model, input_tensor, device)
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

            prediction_buffer.append(predicted_label)

            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)

            most_common = Counter(prediction_buffer).most_common(1)[0][0]

            if (
                confidence_value > CONF_THRESHOLD and
                prediction_buffer.count(most_common) > BUFFER_SIZE * 0.7
            ):
                prediction_text = most_common
                confidence_text = f"{confidence_value:.2f}"

                now = time.time()

                if (
                    most_common != last_added_letter or
                    now - last_add_time > ADD_DELAY
                ):
                    # If starting a new word, reset spoke flag
                    if current_word == "":
                        spoke_for_this_word = False

                    current_word += most_common
                    last_added_letter = most_common
                    last_add_time = now

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # HAND NOT DETECTED
    # =========================

    else:
        if hand_present:
            hand_present = False

        # Speak the full word after hand disappears
        if current_word and not spoke_for_this_word:
            if time.time() - last_hand_time > 2:
                print("Speaking full word:", current_word)
                speak_text(current_word)

                # Reset for next word
                current_word = ""
                prediction_buffer.clear()
                last_added_letter = ""
                last_add_time = 0
                spoke_for_this_word = True

    # =========================
    # FPS
    # =========================

    now = time.time()
    fps = 1 / (now - prev_time) if prev_time != 0 else 0
    prev_time = now

    # =========================
    # DISPLAY
    # =========================

    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence_text}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"Word: {current_word}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("ASL Real-Time Detection + Auto Speech", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()