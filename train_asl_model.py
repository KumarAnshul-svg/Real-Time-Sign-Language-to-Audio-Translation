# utils/train_asl_model.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import joblib

from model import ASLModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_PATH = "../processed_landmarks/asl_landmarks.csv"
MODEL_PATH = "../models/asl_model.pth"
LABEL_PATH = "../labels/asl_label_encoder.pkl"

# Load Data
df = pd.read_csv(DATA_PATH)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Total samples:", len(X))
print("Feature size:", X.shape[1])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

os.makedirs("../labels", exist_ok=True)
joblib.dump(le, LABEL_PATH)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

model = ASLModel(X.shape[1], len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 25

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test.cpu(), predicted.cpu())

print("Test Accuracy:", acc)

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print("Model saved successfully.")