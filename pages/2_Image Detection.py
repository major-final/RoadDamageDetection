import os
import logging
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

def send_email(user_email, severity):
    sender_email = "21071A6612@vnrvjiet.in"
    sender_password = "rsyr xula xiuy nmjo"
    
    subject = "Road Damage Alert!"
    body = f"\n⚠️ WARNING: Dangerous road damage detected!\nSeverity Level: {severity}\nPlease take necessary precautions."
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = user_email
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, msg.as_string())
        st.success(f"Email alert sent to {user_email}")
    except Exception as e:
        st.error(f"Error sending email: {e}")

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in an image. Upload the image and start detecting. This section can be useful for examining baseline data.")

# Retrieve user email from session state
user_email = st.session_state.get("user_email", "")
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
    
    detections = []  # Store detections
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

    # Original Image
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )

        if detections:
            image_area = w_ori * h_ori  
        
            severity_weights = {
                "Potholes": 1.5,
                "Alligator Crack": 1.2,
                "Transverse Crack": 1.0,
                "Longitudinal Crack": 0.8
            }
        
            severity_score = 0
            num_detections = len(detections)
        
            for det in detections:
                box_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
                box_ratio = box_area / image_area  
                weight = severity_weights.get(det.label, 1.0)  
        
                # Apply a stronger boost for higher confidence
                confidence_boost = (det.score ** 2)  
        
                # Increase severity score based on size, confidence, and weight
                severity_score += confidence_boost * box_ratio * weight  
        
            # Normalize the severity score
            severity_score /= max(1, num_detections)  # Prevent division by zero
        
            # Adjusted thresholds
            if severity_score > 0.25 or num_detections > 3:
                severity = "Severe"
            elif severity_score > 0.12 or num_detections > 1:
                severity = "Moderate"
            else:
                severity = "Mild"
        
            st.write(f"### Severity Level: {severity}")



           # Function to play sound
            import streamlit.components.v1 as components

            def play_alert_sound():
                sound_file = "mixkit-signal-alert-771.wav"
                if os.path.exists(sound_file):
                    # Convert to Base64 to embed in HTML
                    import base64
                    with open(sound_file, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode()
            
                    # Auto-play sound using JavaScript
                    sound_html = f"""
                    <audio autoplay>
                        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                    </audio>
                    """
                    components.html(sound_html, height=0)
                else:
                    st.warning("Sound file not found! Please check the path.")

            
            # Inside the severity detection block:
            if severity in ["Severe","Moderate"]:
                st.error("⚠️ WARNING: Dangerous road damage detected! Drive cautiously! ⚠️", icon="⚠️")
                play_alert_sound()  # Play alert sound when damage is detected

            
           

        def generate_pdf(detections, severity, annotated_image):
            buffer = BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            pdf.setTitle("Road Damage Detection Report")
            
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(200, 750, "Road Damage Detection Report")
            
            pdf.setFont("Helvetica", 12)
            pdf.drawString(50, 700, f"Severity Level: {severity}")
            
            pdf.drawString(50, 670, "Detected Objects:")
            y_position = 650
            for det in detections:
                label_text = f"  - {det.label} | Score: {det.score:.2f} | Box: {det.box.tolist()}"
                pdf.drawString(50, y_position, label_text)
                y_position -= 20
            
            pdf.save()
            buffer.seek(0)
            return buffer
        
        if detections:
            pdf_buffer = generate_pdf(detections, severity, _downloadImages)
        
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name="Road_Damage_Report.pdf",
                mime="application/pdf"
            )
