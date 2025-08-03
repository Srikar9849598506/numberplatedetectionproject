import cv2
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')  # Make sure yolov8n.pt is available in your project directory

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Directory to save screenshots
screenshot_dir = "vehicle_screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

print("[INFO] Starting live video... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Detect objects in frame
    results = model(frame)[0]
    vehicle_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]

        # Detect only cars, bikes, buses, trucks
        if class_name in ['car', 'motorbike', 'bus', 'truck']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_detected = True

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # Crop and save vehicle image
            vehicle_img = frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshot_dir, f"vehicle_{timestamp}.jpg")
            cv2.imwrite(filename, vehicle_img)

            # Perform OCR on the cropped vehicle image
            print(f"[INFO] Vehicle detected: {class_name}. Performing OCR...")
            results_text = reader.readtext(vehicle_img)
            for detection in results_text:
                detected_text = detection[1]
                if len(detected_text) >= 6 and any(char.isdigit() for char in detected_text):
                    print("🔎 Detected Number Plate:", detected_text)

            break  # Process only the first detected vehicle per frame

    # Try showing the live frame
    try:
        cv2.imshow("Live Vehicle Detection", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
    except cv2.error:
        print("[WARNING] GUI not supported in this environment. Skipping display.")
        break

# Release video stream
cap.release()

# Try destroying any OpenCV windows (skip if GUI not supported)
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass
