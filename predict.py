from ultralytics import YOLO
import random
import os

model_path =r"E:\savor_vision\Savor_training\mvp_run\weights\best.pt"
model = YOLO(model_path)

image_folder = r"E:\Savor_Vision\dataset\train\images"

all_images = [image for image in os.listdir(image_folder) if image.endswith('jpg')]

random_image = random.choice(all_images)
image_path = os.path.join(image_folder, random_image)

print(f"Looking at: {random_image}")

results = model.predict(source=image_path, conf=0.7, save=True)

results[0].show()