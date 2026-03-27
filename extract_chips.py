import os
import cv2
import glob

def extract():
    img_dir = 'dataset/images/train'
    label_dir = 'dataset/labels/train'
    out_dir = 'dataset/chips'
    os.makedirs(out_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    count = 0
    for img_path in img_files:
        basename = os.path.basename(img_path)
        label_path = os.path.join(label_dir, basename.replace('.jpg', '.txt'))
        if not os.path.exists(label_path): continue
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO: class, x_center, y_center, width, height (normalized)
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                
                # Crop 64x64 around center
                x_min = max(0, int(x_center - 32))
                y_min = max(0, int(y_center - 32))
                x_max = min(w, x_min + 64)
                y_max = min(h, y_min + 64)
                
                chip = img[y_min:y_max, x_min:x_max]
                if chip.shape[0] > 0 and chip.shape[1] > 0:
                    chip = cv2.resize(chip, (64, 64))
                    cv2.imwrite(os.path.join(out_dir, f'chip_{count}_{i}.jpg'), chip)
                    count += 1
    print(f"Extracted {count} 64x64 ship chips for GAN training.")

if __name__ == '__main__':
    extract()
