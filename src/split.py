import os
import random
import shutil

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def split_dataset(root):
    # root example: "data/Drywall-Join-Detect/Drywall-Join-Detect"
    train_root = os.path.join(root, "train")
    img_dir = os.path.join(train_root, "images")
    lbl_dir = os.path.join(train_root, "labels")

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        raise Exception(f"Missing train/images or train/labels in {root}")

    # Create new split folders
    for split in ["train_split", "val", "test"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)

    # Get all image files
    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
    random.shuffle(images)

    total = len(images)
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)
    # remaining 15% is test

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train+n_val]
    test_imgs  = images[n_train+n_val:]


    def move_files(file_list, split_name):
        for img in file_list:
            lbl = img.rsplit(".", 1)[0] + ".txt"

            src_img = os.path.join(img_dir, img)
            src_lbl = os.path.join(lbl_dir, lbl)

            dst_img = os.path.join(root, split_name, "images", img)
            dst_lbl = os.path.join(root, split_name, "labels", lbl)

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                print("Missing label for:", img)

    move_files(train_imgs, "train_split")
    move_files(val_imgs, "val")
    move_files(test_imgs, "test")

    print(f"\nFinished splitting: {root}")
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}\n")


# Run for both datasets
split_dataset(os.path.join(_DATA_DIR, "Drywall-Join-Detect", "Drywall-Join-Detect"))
split_dataset(os.path.join(_DATA_DIR, "wall-crack", "wall-crack"))
