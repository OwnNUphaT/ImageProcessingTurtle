from ultralytics import YOLO

# 🐢 Load a pretrained segmentation model
model = YOLO("yolov8n-seg.pt")  # 'n' = nano (fast), you can try 's' for better accuracy

# 🧠 Train the model
model.train(
    data="data.yaml",      # path to your dataset config
    epochs=50,             # number of training epochs
    imgsz=640,             # image size (can try 512 or 640)
    batch=16,              # batch size
    workers=2,             # number of dataloader workers
    name="turtle_seg",     # this sets the folder name in runs/segment/
    device='cpu'               
)

# 📊 Evaluate the trained model
model.val()

# 💾 (Optional) Export the trained model to other formats (e.g. ONNX, TorchScript, etc.)
# model.export(format="onnx")
