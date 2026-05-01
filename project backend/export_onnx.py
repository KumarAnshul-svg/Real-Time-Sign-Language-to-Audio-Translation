# export_onnx.py

import torch
import joblib
import os
import onnx
from utils.model import ASLModel

def export():
    # Load labels to get num_classes
    try:
        label_encoder = joblib.load("labels/asl_label_encoder.pkl")
        num_classes = len(label_encoder.classes_)
        print(f"Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # Initialize model
    model = ASLModel(42, num_classes)
    
    # Load weights
    try:
        model.load_state_dict(torch.load("models/asl_model.pth", map_location="cpu"))
        print("Loaded model weights.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 42)

    # Export to ONNX
    temp_onnx_path = "models/temp_asl_model.onnx"
    final_onnx_path = "models/asl_model.onnx"
    
    os.makedirs("models", exist_ok=True)

    # Export using torch
    print("Exporting with torch.onnx.export...")
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx_path,
        export_params=True,
        opset_version=11, # Using 11 for maximum compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    # Now use the onnx library to ensure it's a single file and formatted correctly
    print("Validating and re-saving with onnx library...")
    try:
        model_onnx = onnx.load(temp_onnx_path)
        # onnx.save will default to internal data for small models
        onnx.save(model_onnx, final_onnx_path)
        print(f"Final model saved to {final_onnx_path}")
        
        # Cleanup temp files and external data if created
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        temp_data = temp_onnx_path + ".data"
        if os.path.exists(temp_data):
            os.remove(temp_data)
        
        # Also remove any accidental asl_model.onnx.data
        accidental_data = final_onnx_path + ".data"
        if os.path.exists(accidental_data):
            os.remove(accidental_data)
            
        print(f"Final Size: {os.path.getsize(final_onnx_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"Error during onnx re-save: {e}")

if __name__ == "__main__":
    export()
