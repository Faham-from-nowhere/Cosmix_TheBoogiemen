from ultralytics import YOLO

def main():
    print("Initializing YOLOv8-nano...")
    # Load a pretrained model (recommended for training)
    model = YOLO('yolov8n.pt')

    print("Starting MVP training...")
    # Train the model for a few epochs
    results = model.train(data='data.yaml', epochs=3, imgsz=800, batch=4)

    print("Training complete! Model evaluates automatically.")

if __name__ == '__main__':
    main()
