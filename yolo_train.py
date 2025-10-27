from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Train the model on your dataset
model.train(
    data="dataset_turtle_preprocessed/data.yaml",  # <-- adjust path if needed
    epochs=50,
    imgsz=640
)
