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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = Path("./models/YOLOv8_Small_RDD.pt")

if not MODEL_LOCAL_PATH.exists():
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

model = YOLO(str(MODEL_LOCAL_PATH))

CLASSES = ["alligator", "longitudinal", "transversal", "pothole"]

if "twilio" in st.secrets:
    TWILIO_ACCOUNT_SID = st.secrets["twilio"].get("account_sid", "")
    TWILIO_AUTH_TOKEN = st.secrets["twilio"].get("auth_token", "")
    TWILIO_PHONE_NUMBER = st.secrets["twilio"].get("from_phone", "")
    TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])
else:
    TWILIO_ENABLED = False

def send_sms_alert(user_phone_number, most_frequent_damage):
    if not TWILIO_ENABLED:
        return
    if not user_phone_number.startswith("+"):
        st.error("Invalid phone number format!")
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message_body = f"Alert: '{most_frequent_damage}' detected. Immediate action required."
        client.messages.create(body=message_body, from_=TWILIO_PHONE_NUMBER, to=user_phone_number)
        st.success("SMS Sent Successfully!")
    except Exception as e:
        st.error(f"Twilio Error: {e}")

TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

temp_file_input = TEMP_DIR / "video_input.mp4"
temp_file_infer = TEMP_DIR / "video_infer.mp4"
pdf_report_path = TEMP_DIR / "damage_report.pdf"

st.title("Road Damage Detection - Video")
user_phone_number = st.text_input("Enter phone number for alerts:")
video_file = st.file_uploader("Upload Video", type=["mp4"])
score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

def generate_pdf_report(damage_counter):
    c = canvas.Canvas(str(pdf_report_path), pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Road Damage Detection Report")
    c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    y_position = 700
    for damage, count in damage_counter.items():
        c.drawString(100, y_position, f"{damage.capitalize()} damage: {count}")
        y_position -= 20
    
    c.save()

def process_video(video_file, score_threshold, user_phone_number):
    temp_file_input.write_bytes(video_file.read())
    video_capture = cv2.VideoCapture(str(temp_file_input))
    if not video_capture.isOpened():
        st.error('Error opening video file')
        return
    
    _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_bar = st.progress(0)
    image_location = st.empty()
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    cv2_writer = cv2.VideoWriter(str(temp_file_infer), fourcc_mp4, _fps, (_width, _height))

    damage_counter = Counter()
    _frame_counter = 0
    
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
                    damage_counter[CLASSES[class_id]] += 1
        
        cv2_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        image_location.image(annotated_frame)
        _frame_counter += 1
        inference_bar.progress(_frame_counter / _frame_count)
    
    inference_bar.empty()
    video_capture.release()
    cv2_writer.release()
    
    if damage_counter:
        most_frequent_damage = damage_counter.most_common(1)[0][0]
        send_sms_alert(user_phone_number, most_frequent_damage)
        generate_pdf_report(damage_counter)
    
    st.success("Video Processed!")
    with open(temp_file_infer, "rb") as f:
        st.download_button("Download Processed Video", data=f, file_name="RDD_Prediction.mp4", mime="video/mp4")
    with open(pdf_report_path, "rb") as f:
        st.download_button("Download PDF Report", data=f, file_name="Damage_Report.pdf", mime="application/pdf")

if video_file and user_phone_number and st.button("Process Video"):
    process_video(video_file, score_threshold, user_phone_number)
