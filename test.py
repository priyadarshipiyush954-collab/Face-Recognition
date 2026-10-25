"""
Command-Line Live Face Recognition & Attendance Testing (Enhanced)
Uses OpenCV YuNet Deep Neural Detector and the AURA Enhanced Face Recognition Ensemble.
Press 'o' to log attendance, 'q' to quit.
"""

import cv2
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
YUNET_PATH = os.path.join(DATA_DIR, "face_detection_yunet_2023mar.onnx")

sys.path.append(BASE_DIR)
from engine.ml_engine import FaceRecognitionPipeline
from engine.attendance_manager import AttendanceManager

try:
    from win32com.client import Dispatch
    speaker = Dispatch("SAPI.SpVoice")
    def speak(text):
        try: speaker.Speak(text)
        except Exception: pass
except Exception:
    def speak(text): print(f"[Voice] {text}")

pipeline = FaceRecognitionPipeline(data_dir=DATA_DIR)
pipeline.load_or_train()
attendance_mgr = AttendanceManager()

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("[Error] Camera inaccessible.")
    sys.exit(1)

print("\n" + "=" * 65)
print("     AURA NEURAL LIVE RECOGNITION (CONSOLE MODE)")
print("=" * 65)
print("  - Looking for faces via YuNet Deep Neural Detector")
print("  - Predicting identity via Calibrated RBF-SVM + Cosine-KNN")
print("  - Press 'o' to log attendance for the recognized face")
print("  - Press 'q' to quit")
print("=" * 65 + "\n")

last_prediction = None
last_conf = 0.0

while True:
    ret, frame = video.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    results = pipeline.detect_and_recognize(frame)

    if results:
        top = results[0]
        b = top["box"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        name = top["name"]
        conf = top["confidence"]
        verified = top["verified"]
        last_prediction = name
        last_conf = conf

        color = (0, 245, 160) if verified else (71, 71, 255)

        # Draw box and name tag
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, max(0, y - 35)), (x + w, y), color, -1)
        tag = f"{name} ({conf}%)"
        cv2.putText(frame, tag, (x + 6, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        last_prediction = None

    cv2.imshow("AURA Face Recognition — Press 'o' for Attendance, 'q' to Quit", frame)
    k = cv2.waitKey(1)

    if k == ord('o'):
        if last_prediction and last_prediction.lower() not in ["unknown", "unverified"]:
            res = attendance_mgr.punch_attendance(last_prediction, last_conf)
            if res["success"]:
                speak(f"Attendance recorded for {last_prediction}")
                print(f"[✓] {res['message']}")
            else:
                print(f"[!] {res['message']}")
        else:
            print("[!] Cannot log attendance: No verified face detected.")

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
