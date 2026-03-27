import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms

def dynamic_land_mask(tile):
    # Create a dynamic topological mask to black-out continuous coastal landmasses
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones_like(tile) * 255
    for cnt in contours:
        # If a bright structure is larger than 600 pixels, it's a coastline/city, not a ship. Mask it out!
        if cv2.contourArea(cnt) > 600:
            # Dilate the land mask slightly to catch the fading edges of the coastline
            cv2.drawContours(mask, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)
            mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)
            
    return cv2.bitwise_and(tile, mask)

def process_large_scene(image_path, model_path, tile_size=800, overlap=100, conf_thresh=0.25, iou_thresh=0.15):
    print(f"Loading Sentinel-1 Scene: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    h, w, _ = img.shape
    model = YOLO(model_path)
    
    all_boxes = []
    all_scores = []
    
    print("Slicing scene and running inference...")
    # Slide a window across the massive image
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            
            tile = img[y1:y2, x1:x2]
            
            # Skip purely black tiles but allow very dark radar ocean tiles
            if tile.mean() < 0.1: continue 
            
            # We bypass the median blur here because on massive Sentinel-1 scenes (10m resolution), 
            # small ships are literally 2-3 pixels wide. A 5x5 blur wipes them out completely!
            # We strictly unlock the raw MVP confidence (0.001) to pull all raw gradients.
            results = model(tile, verbose=False, conf=0.001, iou=iou_thresh)
            
            for box in results[0].boxes:
                # Convert tile-relative coordinates to global image coordinates
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                score = box.conf[0].item()
                
                # HACKATHON MVP HOTFIX: Morphological Box Filtering
                # Since the MVP is under-trained, lowering conf to 0.001 causes YOLO to spit out
                # massive 'ghost anchors' covering huge swaths of empty ocean. We can surgically remove
                # them by mathematical limits. Sentinel-1 is 10m/px. A ship will NEVER be 1000m (100px) wide.
                w_box = bx2 - bx1
                h_box = by2 - by1
                if w_box > 40 or h_box > 40 or w_box < 2 or h_box < 2:
                    continue  # Ghost Anchor isolated! Discard target.
                
                global_x1 = bx1 + x1
                global_y1 = by1 + y1
                global_x2 = bx2 + x1
                global_y2 = by2 + y1
                
                all_boxes.append([global_x1, global_y1, global_x2, global_y2])
                all_scores.append(score)

    if not all_boxes:
        print("No ships detected in this scene.")
        return

    print("Stitching detections and applying global Non-Maximum Suppression (NMS)...")
    boxes_tensor = torch.tensor(all_boxes)
    scores_tensor = torch.tensor(all_scores)
    
    # Remove overlapping duplicate boxes created at the tile seams
    keep_indices = nms(boxes_tensor, scores_tensor, iou_thresh)
    final_boxes = boxes_tensor[keep_indices].numpy()
    
    print(f"Total unique ships detected: {len(final_boxes)}")
    
    # Draw the final boxes on the massive original image
    for box in final_boxes:
        gx1, gy1, gx2, gy2 = map(int, box)
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
        cv2.putText(img, "Ship", (gx1, max(gy1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_path = "sentinel2_final_output.jpg"
    cv2.imwrite(output_path, img)
    print(f"Success! Final scene saved to {output_path}")

if __name__ == '__main__':
    # Make sure you place a Sentinel-1 image named 'sentinel1_scene.jpg' in the root folder
    process_large_scene('sentinel2_scene.jpg', 'runs/detect/train/weights/best.pt')
