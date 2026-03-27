from ultralytics import YOLO
import cv2
import glob
import numpy as np

def calculate_concept_scores(image, bbox):
    # bbox format: [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure chip dimensions are valid
    if x1 >= x2 or y1 >= y2:
        return 0, 0, 0
        
    chip = image[y1:y2, x1:x2]
    if chip.size == 0:
        return 0, 0, 0
    
    gray_chip = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    
    # Concept 1: Radar Cross-Section Intensity (normalized mean pixel brightness)
    intensity = np.mean(gray_chip) / 255.0
    
    # Concept 2: Edge Sharpness (Laplacian variance) roughly scaled
    laplacian = cv2.Laplacian(gray_chip, cv2.CV_64F)
    sharpness = np.var(laplacian) / 1000.0  
    
    # Concept 3: Geometry (Aspect Ratio)
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = float(w) / float(h) if h > 0 else 0
    
    return intensity, min(sharpness, 9.99), aspect_ratio

def main():
    print("Loading MVP model weights...")
    model = YOLO('runs/detect/train/weights/best.pt')

    sample_images = glob.glob('dataset/images/train/*.jpg')
    
    for sample_img in sample_images:
        original_img = cv2.imread(sample_img)
        # Lower confidence massively to force candidate boxes for MVP visualization
        results = model(original_img, verbose=False, conf=0.01)
        
        if len(results[0].boxes) > 0:
            print(f"Running Inference & White-Box Explainability on {sample_img}...")
            # Process YOLO detections
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    intensity, sharpness, aspect_ratio = calculate_concept_scores(original_img, b)
                    
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    panel_y = max(0, y1 - 65)
                    cv2.rectangle(original_img, (x1, panel_y), (x1 + 220, y1), (0, 0, 0), -1)
                    
                    text1 = f"Conf: {conf:.2f} | Int: {intensity:.2f}"
                    text2 = f"Shp: {sharpness:.2f} | AR: {aspect_ratio:.2f}"
                    
                    cv2.putText(original_img, text1, (x1 + 5, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.putText(original_img, text2, (x1 + 5, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            out_file = 'explainable_output.jpg'
            cv2.imwrite(out_file, original_img)
            print(f"White-Box Explainability complete! Render saved to {out_file}")
            return

if __name__ == '__main__':
    main()
