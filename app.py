import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import os

# ================================
# Load YOLO model & EasyOCR
# ================================
MODEL_PATH = "model.pt"
model = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(['en'])

LOG_FILE = "logs.xlsx"


# ================================
# Function: Detect and OCR
# ================================
def detect_and_read_plate(image):
    results = model.predict(image)

    if len(results[0].boxes) == 0:
        return None, None, image

    # Get highest confidence detection
    boxes = results[0].boxes
    best_box = boxes[0]
    conf = float(best_box.conf[0].cpu().numpy())
    xyxy = best_box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]

    # Crop plate
    x1, y1, x2, y2 = xyxy
    plate_crop = image[y1:y2, x1:x2]

    # OCR
    ocr_results = ocr_reader.readtext(plate_crop)
    plate_text = None
    if len(ocr_results) > 0:
        plate_text = ocr_results[0][1]  # Extract text string

    # Draw bbox on image
    img_out = image.copy()
    cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{plate_text if plate_text else 'Plate'} ({conf:.2f})"
    cv2.putText(img_out, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return plate_text, conf, img_out


# ================================
# Function: Log to Excel
# ================================
def log_to_excel(plate, conf):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"Number Plate": plate, "Confidence %": round(conf * 100, 2), "DateTime": now}

    if os.path.exists(LOG_FILE):
        df = pd.read_excel(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_excel(LOG_FILE, index=False)


# ================================
# Streamlit UI
# ================================
st.title("🚗 License Plate Detection & Recognition")
st.write("Upload an image of a car to detect and read its license plate.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detect plate
    plate_text, conf, img_out = detect_and_read_plate(image)

    if plate_text is None:
        st.error("❌ No number plate detected.")
    else:
        st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
        st.success(f"**Detected Plate:** {plate_text}")
        st.info(f"**Confidence:** {conf * 100:.2f}%")

        # Log
        log_to_excel(plate_text, conf)
        st.write("✅ Prediction logged to logs.xlsx")
