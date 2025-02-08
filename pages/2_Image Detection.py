import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from fpdf import FPDF

from sample_utils.download import download_file

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

def send_email(user_email, severity, pdf_path):
    sender_email = "21071A6612@vnrvjiet.in"
    sender_password = "rsyr xula xiuy nmjo"
    
    subject = "Road Damage Alert!"
    body = f"\n⚠️ WARNING: Dangerous road damage detected!\nSeverity Level: {severity}\nPlease take necessary precautions."
    
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = user_email
    
    msg.attach(MIMEText(body, "plain"))
    
    with open(pdf_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=Damage_Report.pdf")
        msg.attach(part)
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, msg.as_string())
        st.success(f"Email alert with PDF report sent to {user_email}")
    except Exception as e:
        st.error(f"Error sending email: {e}")

def generate_pdf(image, detections, severity):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Road Damage Detection Report", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Severity Level: {severity}", ln=True)
    pdf.ln(10)
    
    if detections:
        for det in detections:
            pdf.cell(200, 10, f"Detected: {det.label} (Confidence: {det.score:.2f})", ln=True)
    else:
        pdf.cell(200, 10, "No damage detected", ln=True)
    
    pdf.ln(10)
    pdf.image("temp_pred.png", x=10, y=None, w=180)
    pdf_path = "Damage_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in an image. Upload the image and start detecting. This section can be useful for examining baseline data.")

user_email = st.sidebar.text_input("Enter your email for alerts:")
image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there are false predictions.")

if image_file is not None:
    # Load the image
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)
    
    # Perform inference
    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]
    
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
    
    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
    
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)
        
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save("temp_pred.png")
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()
        
        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
        
        severity = "No damage detected"
        if detections:
            max_confidence = max([det.score for det in detections])
            severity = "Severe" if max_confidence > 0.6 else "Moderate" if max_confidence > 0.4 else "Mild"
            st.write(f"### Severity Level: {severity}")
        
        pdf_path = generate_pdf(_image_pred, detections, severity)
        st.download_button("Download PDF Report", data=open(pdf_path, "rb").read(), file_name="Damage_Report.pdf", mime="application/pdf")
        
        if severity in ["Severe", "Moderate"] and user_email:
            send_email(user_email, severity, pdf_path)
