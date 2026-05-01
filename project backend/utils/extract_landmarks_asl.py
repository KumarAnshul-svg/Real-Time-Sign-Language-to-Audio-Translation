import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATASET_PATH = "../datasets/ASL_raw"
OUTPUT_DIR = "../processed_landmarks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "asl_landmarks.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = []
labels = []

print("Starting ASL landmark extraction with wrist normalization...")

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nProcessing label: {label}")

    for image_name in tqdm(os.listdir(label_path)):
        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)

                data.append(landmarks[:42])  # 21 points × 2 coords
                labels.append(label)

print("\nSaving CSV...")

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(OUTPUT_FILE, index=False)

print("Extraction complete.")
print("Total samples:", len(df))