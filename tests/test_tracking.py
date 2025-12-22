from ultralytics import YOLO
import numpy as np
import cv2
from utils import find_model_path

MODEL_PATH = find_model_path()
PERSISTENCE_THRESHOLD = 10
DROPOUT_THRESHOLD = 5
tracking_history = {}

def main():
    model = YOLO(MODEL_PATH)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened:
        print("ERROR: Could not open camera")
        exit()

    while True:
        success, frame = cam.read() 
        if not success:
            print("ERROR: Could not read camera data")
            exit()

        results = model.track(source=frame, conf=0.7, tracker="savor_tracker.yaml", persist=True )
        current_frame_detections = []

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist() 
            classes = results[0].boxes.cls.int().cpu().tolist()

        for box, tracking_id, class_index in zip(boxes, ids, classes):
            current_frame_detections.append(tracking_id)

            x_center, y_center = box[0], box[1]
            class_name = model.names[class_index]

            if tracking_id not in tracking_history:
                tracking_history[tracking_id] = {
                "name": class_name,
                "center": (x_center, y_center),
                "seen_count": 1,
                "missed_count": 0
            }
                
            else:
                item = tracking_history[tracking_id]
                item["center"] = (x_center, y_center)
                item["seen_count"] += 1
                item["missed_count"] = 0

                if item["seen_count"] == PERSISTENCE_THRESHOLD:
                    print("ITEM CONFIRMED: {class_name} - {id} is on the shelf")

        missing_ids = [tracking_id for tracking_id in tracking_history if tracking_id not in current_frame_detections]
        items_to_forget = []

        for id in missing_ids:
            item = tracking_history[id]
            item["missed_count"] += 1
            cv2.circle(frame, item["center"], 10, (0, 0, 255), -1)
            cv2.putText(frame, "LIMBO", (item["center"][0]-20, item["center"][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if item["missed_count"] >= DROPOUT_THRESHOLD:
                print(f"DATABASE: {item['name']} removed from shelf.")
                items_to_forget.append[item]

        for tid in items_to_forget:
            del tracking_history[tid]

        hud_overlay = frame.copy()
        hud_height = 40 + (len(tracking_history) * 30)
        cv2.rectangle(hud_overlay, (0, 0), (450, hud_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(hud_overlay, 0.4, frame, 0.6, 0)
        y_offset = 30

        for tid, info in tracking_history.items():
            is_missing = tid in missing_ids
            status_text = f"LIMBO ({info['missed_count']})" if is_missing else "VISIBLE"
            color = (0, 0, 255) if is_missing else (0, 255, 0)
            
            display_str = f"ID {tid}: {info['name']} | {status_text} | Conf: {info['seen_count']}"
            cv2.putText(frame, display_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30

        cv2.imshow("Savor: Pantry Spatial Memory", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()