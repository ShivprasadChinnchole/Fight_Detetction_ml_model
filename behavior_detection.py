import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("violence_model.h5")

labels = ["Normal", "Fighting"]

# ULTRA SENSITIVE MODE - Very easy to trigger fighting detection
FIGHTING_THRESHOLD = 5.0  # Change this: 5=ULTRA SENSITIVE, 10=VERY SENSITIVE, 20=SENSITIVE, 50=STRICT

def detect_behavior(frame):
    # Resize frame to match training data (64x64)
    processed_frame = cv2.resize(frame, (64, 64))
    processed_frame = processed_frame / 255.0
    
    # Add batch dimension
    input_data = np.expand_dims(processed_frame, axis=0)
    
    # Make prediction
    prediction = model.predict(input_data, verbose=0)
    
    # Get probabilities
    normal_prob = prediction[0][0] * 100
    fighting_prob = prediction[0][1] * 100
    
    # Use threshold instead of argmax for easier detection
    if fighting_prob >= FIGHTING_THRESHOLD:
        label = "Fighting"
        confidence = fighting_prob
    else:
        label = "Normal"
        confidence = normal_prob
    
    # Return label, confidence, and all probabilities
    return label, confidence, prediction[0]