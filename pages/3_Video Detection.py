import os
import logging
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
from collections import Counter
from sample_utils.download import download_file
from fpdf import FPDF  # For generating PDF reports

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define model path
MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = Path("./models/YOLOv8_Small_RDD.pt")

# Download model if it doesn't exist
if not MODEL_LOCAL_PATH.exists():
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load YOLO model
model = YOLO(str(MODEL_LOCAL_PATH))

# Class labels
CLASSES = ["alligator", "longitudinal", "transversal", "pothole"]

# Load Twilio credentials dynamically
if "twilio" in st.secrets:
    TWILIO_ACCOUNT_SID = st.secrets["twilio"].get("account_sid", "")
    TWILIO_AUTH_TOKEN = st.secrets["twilio"].get("auth_token", "")
    TWILIO_PHONE_NUMBER = st.secrets["twilio"].get("from_phone", "")
    TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])
else:
    st.error("Twilio secrets not found! Check your secrets.toml configuration.")
    TWILIO_ENABLED = False

def send_sms_alert(user_phone_number, most_frequent_damage):
    """Send an SMS alert for the most frequently detected damage."""
    if not TWILIO_ENABLED:
        st.warning("Twilio is not configured correctly. Notifications are disabled.")
        return

    if not user_phone_number.startswith("+"):
        st.error("Invalid phone number! Use international format (e.g., +1234567890)")
        return

    if not most_frequent_damage.strip():
        logging.error("Empty message detected. Skipping SMS.")
        return
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message_body = f"Alert: '{most_frequent_damage}' crack detected. Immediate action may be required."
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=user_phone_number
        )
        logging.info(f"SMS alert sent: {message.sid}")
        st.success(f"SMS Sent Successfully: {message_body}")
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")
        st.error(f"Twilio Error: {e}")

# Ensure temp directory exists
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

temp_file_input = TEMP_DIR / "video_input.mp4"
temp_file_infer = TEMP_DIR / "video_infer.mp4"
pdf_report_path = TEMP_DIR / "damage_report.pdf"

st.title("Road Damage Detection - Video")
st.write("Upload a video to detect road damage and receive notifications for detected issues.")

user_phone_number = st.text_input("Enter your phone number for alerts:")
video_file = st.file_uploader("Upload Video", type=["mp4"], disabled=False)
score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

def generate_pdf_report(damage_counter):
    """Generates a PDF report of detected damage types and their counts."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Road Damage Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for damage_type, count in damage_counter.items():
        pdf.cell(200, 10, f"{damage_type.capitalize()} Cracks: {count}", ln=True)
    
    pdf.output(str(pdf_report_path))

def process_video(video_file, score_threshold, user_phone_number):
    """Processes uploaded video, detects road damage, and sends an alert for the most common type."""
    temp_file_input.write_bytes(video_file.read())
    video_capture = cv2.VideoCapture(str(temp_file_input))

    if not video_capture.isOpened():
        st.error('Error opening the video file')
        return

    _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_bar = st.progress(0, text="Processing video...")
    image_location = st.empty()

    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    cv2_writer = cv2.VideoWriter(str(temp_file_infer), fourcc_mp4, _fps, (_width, _height))

    _frame_counter = 0
    damage_counter = Counter()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame, conf=score_threshold)
        annotated_frame = results[0].plot()

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for _box in boxes:
                class_id = int(_box.cls)

                if 0 <= class_id < len(CLASSES):
                    label = CLASSES[class_id]
                    damage_counter[label] += 1  

        _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation=cv2.INTER_AREA)
        cv2_writer.write(cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR))
        image_location.image(_image_pred)
        _frame_counter += 1
        inference_bar.progress(_frame_counter / _frame_count, text="Processing video...")

    inference_bar.empty()
    video_capture.release()
    cv2_writer.release()

    # Generate and save PDF report
    generate_pdf_report(damage_counter)

    # Determine most frequently detected damage type
    if damage_counter:
        most_frequent_damage = damage_counter.most_common(1)[0][0]
        send_sms_alert(user_phone_number, most_frequent_damage)

    st.success("Video Processed!")

    # Download Processed Video
    with open(temp_file_infer, "rb") as f:
        st.download_button("Download Processed Video", data=f, file_name="RDD_Prediction.mp4", mime="video/mp4")

    # Download PDF Report
    with open(pdf_report_path, "rb") as pdf_file:
        st.download_button("Download PDF Report", data=pdf_file, file_name="Road_Damage_Report.pdf", mime="application/pdf")

if video_file and user_phone_number and st.button("Process Video"):
    st.warning(f"Processing Video: {video_file.name}")
    process_video(video_file, score_threshold, user_phone_number)
