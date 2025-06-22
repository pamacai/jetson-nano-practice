
import cv2
"""
shelf_inventory.py

This script performs inventory tracking using object detection and tracking on a video walkthrough of a shelf.
It utilizes YOLOv8 for object detection and Norfair for object tracking to count unique items seen in the video.
The inventory is outputted as a CSV file.

Modules:
- cv2: OpenCV for video processing.
- ultralytics: YOLO model for object detection.
- numpy: Numerical operations.
- norfair: Object tracking library.
- pandas: Data manipulation and CSV output.

Classes:
- None

Functions:
- euclidean_distance(detection, tracked_object): Calculates the Euclidean distance between a detection and a tracked object.

Variables:
- model: YOLO object detection model.
- class_map: Mapping of class IDs to human-readable names.
- tracker: Norfair tracker for object tracking.
- video_path: Path to the input video file.
- cap: OpenCV video capture object.
- fps: Frames per second of the video.
- frame_interval: Interval for processing frames.
- frame_idx: Current frame index.
- inventory: Dictionary to store item counts.
- seen_ids: Set to track unique object IDs.

Usage:
1. Place the video file in the specified path (`video_path`).
2. Run the script to process the video and generate an inventory CSV file (`inventory_output.csv`).
"""
from ultralytics import YOLO
import numpy as np
from norfair import Detection, Tracker
import pandas as pd

# Setup YOLO model
model = YOLO("yolov8n.pt")  # Use yolov8n for speed or replace with fine-tuned model

# Class ID to Name mapping (based on COCO)
class_map = {
    39: "Bottle",
    41: "Cup",
    56: "Chair",
}

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

video_path = "sample_shelf_walkthrough.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = max(fps // 2, 1)  # Sample every ~0.5s
frame_idx = 0

inventory = {}
seen_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        results = model.predict(source=frame, conf=0.4)
        boxes = results[0].boxes

        detections = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = class_map.get(class_id, f"Item_{class_id}")
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            detections.append(Detection(points=np.array([cx, cy]), scores=np.array([confidence]), label=class_id))

        tracked_objects = tracker.update(detections=detections)
        for obj in tracked_objects:
            item_id = obj.id
            class_id = obj.last_detection.label
            name = class_map.get(class_id, f"Item_{class_id}")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                inventory[name] = inventory.get(name, 0) + 1

    frame_idx += 1

cap.release()

# Output inventory to CSV
df = pd.DataFrame(list(inventory.items()), columns=["Item", "Count"])
df.to_csv("inventory_output.csv", index=False)
print(df)
