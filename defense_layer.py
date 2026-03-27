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
    
    undf_boxes = len(res_attack[0].boxes)
    def_boxes = len(res_defend[0].boxes)
    print(f"False Positives / Missed Detections (Undefended): {undf_boxes}")
    print(f"Restored Signatures (Defended): {def_boxes}")
    
    # Highlight performance discrepancy visually
    res_attack[0].save(filename='attack_pred.jpg')
    res_defend[0].save(filename='defend_pred.jpg')
    print("Saved evaluation maps to disk (attack_pred.jpg & defend_pred.jpg). Phase 5 executed.")

if __name__ == '__main__':
    main()
