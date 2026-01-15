from ultralytics import YOLO
import os 

def main():
    model = YOLO('yolo11n-seg.pt')
    data_path = os.path.join(os.getcwd(), "amodal_model", "data.yaml")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        device=0,          
        project="amodal_model",
        name="amodal_nano_v1",
        batch=16,           
        mask_ratio=4,       
        overlap_mask=False, 
        mixup=0.0,          
        mosaic=0.5,        
        hsv_v=0.2,         
        fliplr=0.5,    
        exist_ok=True       
    )

    print("training complete, getting best.pt")

    best_model = YOLO(os.path.join(os.getcwd(), "amodal_model", "amodal_nano_v1", "weights", "best.pt"))

    print("training complete, getting best.pt")


    best_model.export(format="onnx", opset=12, nms=False, simplify=False, dynamic=False)


if __name__ == "__main__":
    main()

    