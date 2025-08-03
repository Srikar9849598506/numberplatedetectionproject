# app.py

import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from datetime import datetime
import os
import csv

# Load model and OCR once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

model = load_model()
reader = load_ocr()

# Create folders & CSV
screenshot_dir = "vehicle_screenshots"
log_file = "vehicle_log.csv"
os.makedirs(screenshot_dir, exist_ok=True)

# Create CSV log file with headers if not exist
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Number Plate', 'Image Filename'])

st.title("🚗 Vehicle Detection & Number Plate Logging")

# UI controls
start = st.button("▶️ Start Camera")
stop = st.button("⏹ Stop Camera")
frame_placeholder = st.empty()
plate_placeholder = st.empty()

# State handling
if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Open webcam
cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame.")
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]

        if class_name in ['car', 'motorbike', 'bus', 'truck']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_img = frame[y1:y2, x1:x2]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # OCR
            ocr_results = reader.readtext(vehicle_img)
            for detection in ocr_results:
                number_plate = detection[1]
                if len(number_plate) >= 6 and any(char.isdigit() for char in number_plate):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    filename = f"vehicle_{timestamp.replace(':', '-')}.jpg"
                    filepath = os.path.join(screenshot_dir, filename)
                    cv2.imwrite(filepath, vehicle_img)

                    # Save to CSV
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp, number_plate, filename])

                    plate_placeholder.success(f"📸 {timestamp}: {number_plate}")
                    break
            break

    # Show frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb_frame, channels="RGB")

cap.release()
