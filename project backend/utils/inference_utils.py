import torch
import numpy as np

def normalize_landmarks(hand_landmarks):
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)

    return np.array(landmarks, dtype=np.float32)


def predict(model, input_tensor, device):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item()