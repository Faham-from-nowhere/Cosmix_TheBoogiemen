import os
import cv2
import glob
import random

def augment():
    train_img_dir = 'dataset/images/train'
    train_lbl_dir = 'dataset/labels/train'
    synth_dir = 'dataset/synthetic_chips'
    
    synth_chips = glob.glob(os.path.join(synth_dir, '*.jpg'))
    if not synth_chips:
        print("No synthetic chips found.")
        return
    
    bg_images = glob.glob(os.path.join(train_img_dir, '*.jpg'))
    # Append "aug_" so we don't pick them up repeatedly inside this loop
    original_bg_images = [img for img in bg_images if 'aug_' not in os.path.basename(img)]
    
    count = 0
    # Create 20 augmented background pictures
    for i in range(20):
        bg_path = random.choice(original_bg_images)
        bg = cv2.imread(bg_path)
        if bg is None: continue
        h, w, _ = bg.shape
        
        chip_path = random.choice(synth_chips)
        chip = cv2.imread(chip_path)
        if chip is None: continue
        
        if w <= 64 or h <= 64: continue
        
        # Random location for the synthetic anomaly
        x = random.randint(0, w - 64)
        y = random.randint(0, h - 64)
        
        # Paste chip onto background
        bg[y:y+64, x:x+64] = chip
        
        new_basename = f"aug_{count}_{os.path.basename(bg_path)}"
        cv2.imwrite(os.path.join(train_img_dir, new_basename), bg)
        
        # Copy original labels to the new image, then append the synthetic label
        old_lbl_path = os.path.join(train_lbl_dir, os.path.basename(bg_path).replace('.jpg', '.txt'))
        new_lbl_path = os.path.join(train_lbl_dir, new_basename.replace('.jpg', '.txt'))
        
        lines = []
        if os.path.exists(old_lbl_path):
            with open(old_lbl_path, 'r') as f:
                lines = f.readlines()
            
        # YOLO format for new chip (normalized)
        x_c = (x + 32) / w
        y_c = (y + 32) / h
        box_w = 64 / w
        box_h = 64 / h
        
        lines.append(f"0 {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}\n")
        
        with open(new_lbl_path, 'w') as f:
            f.writelines(lines)
            
        count += 1
        
    print(f"Injected {count} synthesized augmented configurations into the Phase 1 training corpus.")

if __name__ == '__main__':
    augment()
