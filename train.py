from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=r"E:\savor_vision\dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="Savor_training",
        name="mvp_run",
    )

    model.export(format="onnx", opset=11)

if __name__ == "__main__":
    main()