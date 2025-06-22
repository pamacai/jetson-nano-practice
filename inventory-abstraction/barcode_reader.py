
import cv2
"""
barcode_reader.py

This script processes a video file to detect objects using a YOLO model and extracts barcodes from detected regions.
The extracted barcodes are saved to a CSV file for further analysis.

Modules:
- cv2: OpenCV library for video processing and image manipulation.
- numpy: Library for numerical operations.
- ultralytics: YOLO model for object detection.
- pyzbar: Library for decoding barcodes.
- pandas: Library for data manipulation and saving results.

Workflow:
1. Load a YOLO model for object detection.
2. Open a video file and process frames at regular intervals.
3. Detect objects in each frame using the YOLO model.
4. Crop detected regions and decode barcodes using pyzbar.
5. Handle cases where multiple barcodes are detected in a single region.
6. Save the extracted barcodes to a CSV file.

Outputs:
- A CSV file named 'barcode_sequence.csv' containing the extracted barcodes.
- A printed DataFrame of the extracted barcodes.

Warnings:
- If multiple barcodes are detected in a single region, only the first barcode is used, and a warning is printed.
- If no barcode is detected in a region, a message is printed indicating the absence of barcodes.

Usage:
- Ensure the YOLO model file ('yolov8n.pt') is available in the working directory or replace it with a fine-tuned model.
- Provide a video file path ('shelf_walkthrough_with_barcodes.mp4') for processing.
- Install required libraries: OpenCV, NumPy, Ultralytics, Pyzbar, and Pandas.

"""
import numpy as np
from ultralytics import YOLO
from pyzbar.pyzbar import decode as decode_barcode
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")  # Download from Ultralytics or replace with fine-tuned model

# Set video source
video_path = "shelf_walkthrough_with_barcodes.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = max(fps // 2, 1)
frame_idx = 0

barcode_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        results = model.predict(source=frame, conf=0.4)
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cropped = frame[y1:y2, x1:x2]

            barcodes = decode_barcode(cropped)
            if barcodes:
                unique_barcodes = {barcode.data.decode("utf-8") for barcode in barcodes}
                if len(unique_barcodes) > 1:
                    print(f"Warning: Multiple barcodes detected in box at coordinates {x1}, {y1}, {x2}, {y2}. Using the first one.")
                barcode_list.append(next(iter(unique_barcodes)))
            else:
                print(f"No barcode detected in box at coordinates {x1}, {y1}, {x2}, {y2}.")

    frame_idx += 1

cap.release()

# Output barcode list
df = pd.DataFrame(barcode_list, columns=["Barcode"])
df.to_csv("barcode_sequence.csv", index=False)
print("Extracted barcodes saved to barcode_sequence.csv")
print(df)
