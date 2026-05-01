# generate_labels_json.py

import joblib
import json

def generate():
    label_encoder = joblib.load("labels/asl_label_encoder.pkl")
    labels = list(label_encoder.classes_)
    with open("web/labels.json", "w") as f:
        json.dump(labels, f)
    print(f"Generated web/labels.json with {len(labels)} labels.")

if __name__ == "__main__":
    generate()
