from ultralytics import YOLO
import cv2

model = YOLO(r"E:\savor_vision\Savor_training\mvp_run\weights\best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

print("Status: Camera Active.")

while True:
    success, frame = cap.read()
    
    if success:
        results = model.track(frame, conf=0.8, persist=True, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("Savor Vision - Live Feed", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()