import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from behavior_detection import detect_behavior, FIGHTING_THRESHOLD
from datetime import datetime

st.set_page_config(page_title="ThreatSense AI DVR", layout="wide")

st.title("🎥 ThreatSense AI DVR")
st.write("AI-powered fight detection system")

# Load YOLO model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload CCTV Video", type=["mp4","avi","mov"])

frame_window = st.image([])
status_box = st.empty()

alert_count = 0
frame_count = 0
last_behavior = None

if uploaded_file:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    st.success("Video loaded successfully")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(frame)
        person_detected = False

        for r in results:

            boxes = r.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = model.names[cls]

                if label == "person" and conf > 0.5:
                    person_detected = True

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(frame,
                                f"Person ({conf:.2f})",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,255,0),
                                2)

        # Behavior detection
        behavior, confidence, probabilities = detect_behavior(frame)

        if behavior == "Fighting":
            color = (0,0,255)
            alert_count += 1
            status = f"⚠️ FIGHTING DETECTED ({confidence:.1f}%)"
        else:
            color = (0,255,0)
            status = f"✓ Normal ({confidence:.1f}%)"

        cv2.putText(frame,
                    status,
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

        cv2.putText(frame,
                    f"Frame: {frame_count}",
                    (30,frame.shape[0]-50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        cv2.putText(frame,
                    f"Alerts: {alert_count}",
                    (30,frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        frame_window.image(frame, channels="BGR")

        status_box.write(
            f"""
            **Frame:** {frame_count}  
            **Behavior:** {behavior}  
            **Confidence:** {confidence:.1f}%  
            **Alerts:** {alert_count}
            """
        )

    cap.release()

    st.success("Video processing completed")
