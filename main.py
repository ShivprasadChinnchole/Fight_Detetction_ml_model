import cv2
from ultralytics import YOLO
from behavior_detection import detect_behavior, FIGHTING_THRESHOLD
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

print("\n" + "="*60)
print("🎥 ThreatSense AI DVR - STARTED")
print("="*60)
print("📹 Camera is now monitoring...")
print(f"⚙️  Fighting Detection Threshold: {FIGHTING_THRESHOLD}% (SENSITIVE MODE)")
print("⚠️  Press 'q' to quit")
print("="*60 + "\n")

frame_count = 0
last_behavior = None
alert_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame from camera")
        break

    frame_count += 1

    # YOLO person detection
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

    # Behavior Detection
    behavior, confidence, probabilities = detect_behavior(frame)

    # Print behavior status to terminal
    if behavior != last_behavior:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if behavior == "Fighting":
            alert_count += 1
            print(f"\n{'='*60}")
            print(f"⚠️  ALERT #{alert_count} - FIGHTING DETECTED!")
            print(f"{'='*60}")
            print(f"🕒 Time: {timestamp}")
            print(f"👤 Person Detected: {'Yes' if person_detected else 'No'}")
            print(f"📊 Frame: {frame_count}")
            print(f"🎯 Confidence: {confidence:.1f}%")
            print(f"📈 Normal: {probabilities[0]*100:.1f}% | Fighting: {probabilities[1]*100:.1f}%")
            print(f"{'='*60}\n")
        else:
            print(f"✅ [{timestamp}] Behavior: {behavior} ({confidence:.1f}%) | Frame: {frame_count}")
        
        last_behavior = behavior

    # Display behavior status on frame
    if behavior == "Fighting":
        color = (0,0,255)
        status_text = f"⚠️ ALERT: FIGHTING DETECTED! ({confidence:.1f}%)"
    else:
        color = (0,255,0)
        status_text = f"✓ Status: {behavior} ({confidence:.1f}%)"

    # Main status text
    cv2.putText(frame,
                status_text,
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    # Add additional info
    # Show threshold and probabilities
    threshold_text = f"Normal: {probabilities[0]*100:.1f}% | Fighting: {probabilities[1]*100:.1f}% (Threshold: {FIGHTING_THRESHOLD}%)"
    cv2.putText(frame,
                threshold_text,
                (30,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2)
    
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

    # Show timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame,
                current_time,
                (frame.shape[1]-150,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    cv2.imshow("ThreatSense AI DVR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("🛑 ThreatSense AI DVR - STOPPED")
print(f"📊 Total Frames Processed: {frame_count}")
print(f"⚠️  Total Alerts: {alert_count}")
print("="*60 + "\n")