import zipfile
import json
import os

zip_path = 'D:/Downloads/HRSID_jpg.zip'
extract_dir = 'dataset'
os.makedirs(extract_dir, exist_ok=True)
images_dir = os.path.join(extract_dir, 'images')
labels_dir = os.path.join(extract_dir, 'labels')

os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

print("Reading zip file and json...")
with zipfile.ZipFile(zip_path, 'r') as z:
    try:
        with z.open('HRSID_JPG/annotations/train2017.json') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        exit(1)

    images = coco_data['images']
    # Select subset for rapid MVP iteration
    train_imgs = images[:100]
    val_imgs = images[100:120]
    target_imgs = train_imgs + val_imgs
    
    img_id_to_split = {}
    for img in train_imgs: img_id_to_split[img['id']] = 'train'
    for img in val_imgs: img_id_to_split[img['id']] = 'val'
    
    target_ids = set(img_id_to_split.keys())
    
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in target_ids]
    
    print(f"Extracting {len(target_imgs)} images for MVP...")
    for img in target_imgs:
        filename = img['file_name']
        try:
            source = z.read(f"HRSID_JPG/{filename}")
        except KeyError:
            try:
                source = z.read(f"HRSID_JPG/JPEGImages/{filename}")
            except KeyError:
                print(f"Could not find {filename}")
                continue
                
        split = img_id_to_split[img['id']]
        out_path = os.path.join(images_dir, split, filename)
        with open(out_path, 'wb') as out_f:
            out_f.write(source)
            
    print("Writing YOLO annotations...")
    img_id_to_info = {img['id']: img for img in target_imgs}
    labels_dict = {img['id']: [] for img in target_imgs}
    
    for ann in annotations:
        img_id = ann['image_id']
        img_info = img_id_to_info[img_id]
        img_w = img_info.get('width', 800)  # HRSID is usually 800x800, fallback to 800
        img_h = img_info.get('height', 800)
        
        x_min, y_min, w, h = ann['bbox']
        x_center = x_min + w / 2.0
        y_center = y_min + h / 2.0
        
        yolo_x = x_center / img_w
        yolo_y = y_center / img_h
        yolo_w = w / img_w
        yolo_h = h / img_h
        
        # Make sure values are capped between 0 and 1
        yolo_x = max(0.0, min(1.0, yolo_x))
        yolo_y = max(0.0, min(1.0, yolo_y))
        yolo_w = max(0.0, min(1.0, yolo_w))
        yolo_h = max(0.0, min(1.0, yolo_h))
        
        labels_dict[img_id].append(f"0 {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")
        
    for img_id, lines in labels_dict.items():
        img_info = img_id_to_info[img_id]
        filename = img_info['file_name']
        split = img_id_to_split[img_id]
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, split, txt_filename)
        with open(label_path, 'w') as out_f:
            out_f.writelines(lines)

yaml_content = f"""
train: {os.path.abspath(os.path.join(images_dir, 'train')).replace('\\\\', '/')}
val: {os.path.abspath(os.path.join(images_dir, 'val')).replace('\\\\', '/')}

nc: 1
names: ['ship']
"""
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"Dataset successfully built in {extract_dir}/")
print("data.yaml successfully created.")
