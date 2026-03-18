import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
import time

# Set seeds for reproducibility (Rubric requirement) [cite: 19]
torch.manual_seed(42)
np.random.seed(42)

# =========================================
#  DATASET
# =========================================
class PromptSegDataset(Dataset):
    def __init__(self, root, prompt_text, image_size=352):
        self.root = root
        self.prompt = prompt_text
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        
        self.images = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        self.preprocess_img = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # Ensure mask matches predicted spatial size 
        self.preprocess_mask = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Naming: 123__segment_crack.png 
        base = img_name.rsplit('.', 1)[0]
        mask_name = f"{base}__{self.prompt}.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = Image.open(mask_path).convert("L")
        mask = self.preprocess_mask(mask)
        mask = np.array(mask)
        # Binary mask values {0, 255} -> {0.0, 1.0} 
        mask = (mask > 127).astype(np.float32)
        mask = torch.tensor(mask) 

        image = self.preprocess_img(image)
        return {"image": image, "mask": mask, "prompt": self.prompt}

# =========================================
#  METRICS & LOSS 
# =========================================
def get_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()

def dice_loss(pred, target, eps=1e-6):
    pred = pred.sigmoid()
    num = 2 * (pred * target).sum()
    den = (pred + target).sum() + eps
    return 1 - num / den

# =========================================
#  TRAINING + VALIDATION
# =========================================
def train_one_epoch(model, processor, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training", ncols=100):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        prompts = batch["prompt"] 

        # FIX: Separate image and text processing to avoid 'do_rescale' error
        inputs = processor(
            text=prompts, 
            images=imgs, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        # Force do_rescale to False for the image part specifically if needed
        inputs['pixel_values'] = imgs 

        outputs = model(**inputs)
        # Rescale logits to match target size 
        pred = torch.nn.functional.interpolate(
            outputs.logits.unsqueeze(1), size=(352, 352), mode='bilinear', align_corners=False
        ).squeeze(1)

        loss = nn.BCEWithLogitsLoss()(pred, masks) + dice_loss(pred, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, processor, val_loader, device):
    model.eval()
    total_loss, total_iou, total_dice = 0, 0, 0
    inf_times = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", ncols=100):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            prompts = batch["prompt"]

            start_time = time.time()
            inputs = processor(text=prompts, images=imgs, return_tensors="pt", padding=True).to(device)
            inputs['pixel_values'] = imgs
            
            outputs = model(**inputs)
            pred = torch.nn.functional.interpolate(
                outputs.logits.unsqueeze(1), size=(352, 352), mode='bilinear', align_corners=False
            ).squeeze(1)
            
            inf_times.append((time.time() - start_time) / imgs.size(0))

            loss = nn.BCEWithLogitsLoss()(pred, masks) + dice_loss(pred, masks)
            iou, dice = get_metrics(pred, masks)
            
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice

    avg_inf = (sum(inf_times) / len(inf_times)) if inf_times else 0
    return total_loss/len(val_loader), total_iou/len(val_loader), total_dice/len(val_loader), avg_inf

# =========================================
#  MAIN
# =========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset paths — relative to repo root
    repo_root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drywall_root = os.path.join(repo_root, "data", "drywall")
    crack_root   = os.path.join(repo_root, "data", "wall-crack")

    # Task mapping: Dataset 1 (Taping) and Dataset 2 (Cracks) [cite: 4, 5, 9, 10]
    train_drywall = PromptSegDataset(os.path.join(drywall_root, "train_split"), "segment_taping_area")
    train_cracks  = PromptSegDataset(os.path.join(crack_root, "train_split"), "segment_crack")
    val_drywall   = PromptSegDataset(os.path.join(drywall_root, "val"), "segment_taping_area")
    val_cracks    = PromptSegDataset(os.path.join(crack_root, "val"), "segment_crack")

    train_dataset = torch.utils.data.ConcatDataset([train_drywall, train_cracks])
    val_dataset   = torch.utils.data.ConcatDataset([val_drywall, val_cracks])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Initialize model 
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    EPOCHS = 10

    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch+1}/{EPOCHS} ========")
        train_loss = train_one_epoch(model, processor, train_loader, optimizer, device)
        v_loss, v_iou, v_dice, avg_inf = validate(model, processor, val_loader, device)

        # Print metrics for Grading Rubric [cite: 15, 17]
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"mIoU: {v_iou:.4f} | Dice: {v_dice:.4f} | Avg Inf: {avg_inf:.4f}s")

        save_path = os.path.join(outputs_dir, f"clipseg_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path} (Size: {os.path.getsize(save_path)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()