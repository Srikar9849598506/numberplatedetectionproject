import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from datetime import datetime
import os
import csv
from PIL import Image

# ========== Setup ==========
st.set_page_config(page_title="Vehicle Detection", layout="centered")

# Load model and OCR
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

model = load_model()
reader = load_ocr()

# File storage
screenshot_dir = "vehicle_screenshots"
log_file = "vehicle_log.csv"
os.makedirs(screenshot_dir, exist_ok=True)

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Number Plate', 'Image Filename'])

# ========== UI: Device Type ==========
st.title("🚗 Vehicle Number Plate Detection")
device_type = st.radio("Choose your device:", ["📱 Phone", "💻 Desktop"])

# ========== UI: Mode ==========
if device_type:
    mode = st.radio("Select Input Type:", ["📷 Live Camera", "🖼️ Upload Image", "🎞️ Upload Video"])

# ========== Shared Detection Function ==========
def detect_number_plate(frame):
    results = model(frame)[0]
    detected = False

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

                    st.success(f"📸 {timestamp}: {number_plate}")
                    detected = True
                    break
            if detected:
                break
    return frame

# ========== Live Camera ==========
if mode == "📷 Live Camera":
    run = st.button("▶️ Start Live Detection")
    stop = st.button("⏹ Stop")

    if "live" not in st.session_state:
        st.session_state.live = False

    if run:
        st.session_state.live = True
    if stop:
        st.session_state.live = False

    if st.session_state.live:
        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(0)

        while st.session_state.live:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible.")
                break
            processed = detect_number_plate(frame)
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB")
        cap.release()

# ========== Upload Image ==========
elif mode == "🖼️ Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        result_img = detect_number_plate(frame)
        st.image(result_img, channels="BGR", caption="Processed Image")

# ========== Upload Video ==========
elif mode == "🎞️ Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        tfile = f"temp_video.mp4"
        with open(tfile, 'wb') as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_img = detect_number_plate(frame)
            rgb_frame = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB")
        cap.release()
