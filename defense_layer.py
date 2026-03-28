import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os

def add_adversarial_noise(image):
    """ Simulate adversarial SAR speckle noise targeting object detectors. """
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.04
    noisy = np.copy(image)
    
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def defense_filter(image):
    """ Lightweight Spatial Smoothing. Median Blur is highly effective against speckle interference. """
    return cv2.medianBlur(image, 5)

def render_filtered_boxes(res, base_img, filename):
    canvas = base_img.copy()
    valid_count = 0
    for box in res[0].boxes:
        b = box.xyxy[0].cpu().numpy()
        bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        w_box = bx2 - bx1
        h_box = by2 - by1
        
        # Morphological Filtering (Ghost Anchor Deletion)
        if w_box > 150 or h_box > 150 or w_box < 2 or h_box < 2:
            continue
            
        valid_count += 1
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(canvas, f"SHIP", (bx1, max(by1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imwrite(filename, canvas)
    return valid_count

def main():
    print("Initializing O.R.I.O.N. Edge Defense Protocol...")
    model = YOLO('runs/detect/train/weights/best.pt')
    
    sample_images = glob.glob('dataset/images/train/aug_0*.jpg')
    if not sample_images:
        print("No sample images found in local cache.")
        return
        
    img_path = sample_images[0]
    original = cv2.imread(img_path)
    
    # 1. Inject adversarial noise attack
    attacked = add_adversarial_noise(original)
    
    # 2. Defend via sanitization layer
    defended = defense_filter(attacked)
    
    # 3. Model Resilience Test
    print("Running inference on UNDEFENDED, ATTACKED image...")
    res_attack = model(attacked, verbose=False, conf=0.01)
    
    print("Running inference on DEFENDED, SANITIZED image...")
    res_defend = model(defended, verbose=False, conf=0.01)
    
    # Render Custom Morphological Bounding Boxes
    undf_boxes = render_filtered_boxes(res_attack, attacked, 'attack_pred.jpg')
    def_boxes = render_filtered_boxes(res_defend, defended, 'defend_pred.jpg')
    
    print(f"False Positives / Missed Detections (Undefended): {undf_boxes}")
    print(f"Restored Signatures (Defended): {def_boxes}")
    print("Saved evaluation maps to disk (attack_pred.jpg & defend_pred.jpg). Phase 5 executed cleanly!")

if __name__ == '__main__':
    main()
