import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "sam_vit_l_0b3195.pth"


MODEL_TYPE = "vit_l"
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Drywall-Join-Detect", "Drywall-Join-Detect")


sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def yolo_to_xyxy(label_path, img_w, img_h):
    
    if not os.path.exists(label_path):
        return None
        
    with open(label_path, "r") as f:
        line = f.readline().strip().split()
        if len(line) < 5:
            return None

       
        _, cx, cy, bw, bh = map(float, line[:5])

    
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)

    return [x1, y1, x2, y2]

def refine_with_sam(image_rgb, bbox):
   


    predictor.set_image(image_rgb)
    
    box = np.array(bbox)
    
   
    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True
    )

   
    best_mask = masks[np.argmax(scores)]
    return (best_mask.astype(np.uint8) * 255)

def run_refinement(root_path):
    
    print(f"Dataset at: {root_path}\n")

    splits = ["train_split", "val", "test"]
    
    for split in splits:
        split_path = os.path.join(root_path, split)
        img_dir = os.path.join(split_path, "images")
        lbl_dir = os.path.join(split_path, "labels")
        mask_dir = os.path.join(split_path, "masks")

        if not os.path.exists(img_dir):
            print(f"Skipping {split}: 'images' folder not found.")
            continue

        os.makedirs(mask_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(img_dir) 
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"Split: {split} | Found {len(image_files)} images")

        for fname in tqdm(image_files, desc=f"Processing {split}", ncols=100):
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname.rsplit('.', 1)[0] + ".txt")

        
        
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            
            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        
            bbox = yolo_to_xyxy(lbl_path, w, h)
            if bbox is None:
                continue

        
            mask = refine_with_sam(image_rgb, bbox)

            
            out_name = fname.rsplit('.', 1)[0] + "__segment_taping_area.png"
            cv2.imwrite(os.path.join(mask_dir, out_name), mask)

    print("\nAll Drywall masks have been refined successfully!")

if __name__ == "__main__":
    run_refinement(DATASET_ROOT)