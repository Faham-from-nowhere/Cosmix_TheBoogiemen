from ultralytics import YOLO
import shutil
import os

def export():
    print("Loading YOLOv8 MVP model weights...")
    model = YOLO("runs/detect/train/weights/best.pt")
    print("Exporting model to ONNX Runtime format...")
    
    # Exporting for onnxruntime-web. 
    # Use opset=12 as it's structurally compatible across most web execution providers.
    onnx_path = model.export(format="onnx", imgsz=800, opset=12)
    print(f"ONNX Model built successfully at: {onnx_path}")
    
    # We will copy it to the 'web' folder so the JS app can fetch it easily
    os.makedirs('web', exist_ok=True)
    if onnx_path and os.path.exists(onnx_path):
        dest = 'web/best.onnx'
        shutil.copy(onnx_path, dest)
        print(f"Copied ONNX model to {dest}")
    else:
        print("Export somehow failed. ONNX path mismatch.")

if __name__ == '__main__':
    export()
