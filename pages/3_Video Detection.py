import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client  # Twilio for SMS notifications
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Debugging: Check if secrets are loaded
st.write("Secrets available:", st.secrets.keys())

# Load Twilio credentials dynamically
if "twilio" in st.secrets:
    TWILIO_ACCOUNT_SID = st.secrets["twilio"].get("account_sid", "")
    TWILIO_AUTH_TOKEN = st.secrets["twilio"].get("auth_token", "")
    TWILIO_PHONE_NUMBER = st.secrets["twilio"].get("from_phone", "")
    TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])
else:
    st.error("Twilio secrets not found! Check your secrets.toml configuration.")
    TWILIO_ENABLED = False

# Define model path
MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = Path("models/YOLOv8_Small_RDD.pt")
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load YOLO model
net = YOLO(str(MODEL_LOCAL_PATH))

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

def send_sms_alert(user_phone_number, damage_type):
    """Send an SMS alert dynamically based on user input."""
    if not TWILIO_ENABLED:
        st.warning("Twilio is not configured correctly. Notifications are disabled.")
        return
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert: {damage_type} detected on the road! Immediate action may be required.",
            from_=TWILIO_PHONE_NUMBER,
            to=user_phone_number
        )
        logging.info(f"SMS alert sent: {message.sid}")
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")

# Ensure temp directory exists
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

def generate_pdf_report(damage_type, snapshot_path):
    """Generate a PDF report with damage details and snapshot."""
    pdf_path = TEMP_DIR / "Detection_Report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Road Damage Detection Report")
    c.drawString(100, 730, f"Damage Type: {damage_type}")
    c.drawString(100, 710, "Severity: Moderate/Severe")
    c.drawString(100, 690, "Location: Unknown")
    c.drawString(100, 670, "Date & Time: (Generated at runtime)")
    if os.path.exists(snapshot_path):
        img = Image.open(snapshot_path)
        img.thumbnail((400, 300))
        img.save(TEMP_DIR / "snapshot_resized.jpg")
        c.drawImage(str(TEMP_DIR / "snapshot_resized.jpg"), 100, 450, width=400, height=300)
    c.save()
    return pdf_path

st.title("Road Damage Detection - Image")
st.write("Upload an image to detect road damage and receive notifications for detected issues.")

user_phone_number = st.text_input("Enter your phone number for alerts:")
image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], disabled=False)
score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

def process_image(image_file, score_threshold, user_phone_number):
    """Processes uploaded image, detects road damage, and sends notification."""
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    results = net.predict(image_np, conf=score_threshold)
    annotated_image = results[0].plot()
    snapshot_path = TEMP_DIR / "snapshot.jpg"
    cv2.imwrite(str(snapshot_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for _box in boxes:
            detection = {
                "class_id": int(_box.cls),
                "label": CLASSES[int(_box.cls)],
                "score": float(_box.conf),
                "box": _box.xyxy[0].astype(int),
            }
            if detection["score"] > score_threshold:
                send_sms_alert(user_phone_number, detection["label"])
                break  # Only send one alert per image

    st.image(annotated_image, caption="Detected Image", use_column_width=True)
    pdf_path = generate_pdf_report(detection["label"], snapshot_path)
    with open(pdf_path, "rb") as f:
        st.download_button("Download Detection Report", data=f, file_name="Detection_Report.pdf", mime="application/pdf")

if image_file and user_phone_number and st.button("Process Image"):
    st.warning(f"Processing Image: {image_file.name}")
    process_image(image_file, score_threshold, user_phone_number)
