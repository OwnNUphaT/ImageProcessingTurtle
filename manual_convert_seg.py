import json
import os

# Root dataset path
dataset_root = "Turtle_Head_Detection"

# Go through all splits
splits = ["train", "valid", "test"]

for split in splits:
    json_path = os.path.join(dataset_root, split, "_annotations.coco.json")
    label_dir = os.path.join(dataset_root, split, "labels")
    os.makedirs(label_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"⚠️ No annotation file for {split}")
        continue

    print(f"🚀 Processing {split} annotations...")

    # Load COCO annotations
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img["file_name"] for img in coco["images"]}
    annotations = coco["annotations"]

    # Process each image
    for ann in annotations:
        img_id = ann["image_id"]
        img_name = images[img_id]
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        segmentation = ann.get("segmentation", [])
        if not segmentation or not isinstance(segmentation[0], list):
            continue  # skip if segmentation missing

        with open(label_path, "a", encoding="utf-8") as out:
            # Normalize coordinates (x/w, y/h)
            img_info = next((img for img in coco["images"] if img["id"] == img_id), None)
            if img_info:
                w, h = img_info["width"], img_info["height"]
                for seg in segmentation:
                    norm_seg = [f"{x/w:.6f} {y/h:.6f}" for x, y in zip(seg[0::2], seg[1::2])]
                    out.write(f"0 {' '.join(norm_seg)}\n")

    print(f"✅ Done {split}: saved labels to {label_dir}")

print("\n🎉 Conversion complete! Your YOLO segmentation labels are ready.")
