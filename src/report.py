import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import numpy as np
from torchvision import transforms

class PromptSegDataset(Dataset):
    def __init__(self, root, prompt_text, image_size=352):
        self.root = root
        self.prompt = prompt_text
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.preprocess_img = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        self.preprocess_mask = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask_name = f"{img_name.rsplit('.', 1)[0]}__{self.prompt}.png"
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        mask = torch.tensor((np.array(self.preprocess_mask(mask)) > 127).astype(np.float32))
        return {"image": self.preprocess_img(image), "mask": mask, "prompt": self.prompt}

def get_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()

device = "cuda" if torch.cuda.is_available() else "cpu"
_base = os.path.join(os.path.dirname(__file__), "..")
drywall_root = os.path.join(_base, "data", "Drywall-Join-Detect", "Drywall-Join-Detect")
crack_root   = os.path.join(_base, "data", "wall-crack", "wall-crack")

val_drywall = PromptSegDataset(os.path.join(drywall_root, "val"), "segment_taping_area")
val_cracks  = PromptSegDataset(os.path.join(crack_root, "val"), "segment_crack")

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
checkpoint_path = "clipseg_epoch_10.pth"

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")

def run_final_report():
    model.eval()
    tasks = [("Taping Area", val_drywall), ("Cracks", val_cracks)]
    
    print("\n" + "="*40)
    print(f"{'TASK':<15} | {'mIoU':<10} | {'Dice':<10}")
    print("-"*40)

    for name, ds in tasks:
        loader = DataLoader(ds, batch_size=1)
        miou, mdice = 0, 0
        with torch.no_grad():
            for batch in loader:
                imgs, masks, prompts = batch["image"].to(device), batch["mask"].to(device), batch["prompt"]
                inputs = processor(text=prompts, images=imgs, return_tensors="pt", padding=True).to(device)
                inputs['pixel_values'] = imgs
                outputs = model(**inputs)
                pred = torch.nn.functional.interpolate(outputs.logits.unsqueeze(1), size=(352, 352)).squeeze(1)
                iou, dice = get_metrics(pred, masks)
                miou += iou
                mdice += dice
        print(f"{name:<15} | {miou/len(loader):.4f}      | {mdice/len(loader):.4f}")

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    for i in range(4):
        ds = val_drywall if i % 2 == 0 else val_cracks
        sample = ds[torch.randint(0, len(ds), (1,)).item()]
        img_in = sample["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            inputs = processor(text=[sample["prompt"]], images=img_in, return_tensors="pt").to(device)
            inputs['pixel_values'] = img_in
            out = model(**inputs)
            pred = torch.nn.functional.interpolate(out.logits.unsqueeze(1), size=(352, 352)).sigmoid().squeeze().cpu().numpy()

        axes[i, 0].imshow(sample["image"].permute(1, 2, 0))
        axes[i, 0].set_title(f"Original ({sample['prompt']})")
        axes[i, 1].imshow(sample["mask"], cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred > 0.5, cmap='gray')
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig("report_visuals.png")
    plt.show()
    print("\nVisuals saved as 'report_visuals.png'")

if __name__ == "__main__":
    run_final_report()