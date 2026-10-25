"""
Face Capture & Enrollment Utility (OpenCV 5.0 & YuNet Compatible)
Captures face samples from webcam, normalizes to 50x50, updates dataset,
and triggers automatic multi-condition data augmentation.
"""

import cv2
import pickle
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
YUNET_PATH = os.path.join(DATA_DIR, "face_detection_yunet_2023mar.onnx")

os.makedirs(DATA_DIR, exist_ok=True)

name = input("Enter Student / User Name: ").strip()
if not name:
    print("Name cannot be empty. Exiting.")
    sys.exit(1)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access camera.")
    sys.exit(1)

# Initialize YuNet Deep Neural Network face detector
detector = None
if os.path.exists(YUNET_PATH):
    try:
        detector = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (640, 480), score_threshold=0.6, nms_threshold=0.3, top_k=5000
        )
    except Exception as e:
        print(f"[Warning] Failed to load YuNet: {e}")

faces_data = []
target_samples = 50
frame_count = 0

print("\n" + "=" * 60)
print(f"Enrolling: {name}")
print("Please look at the camera. Turn slightly left, right, and smile.")
print("Press 'q' at any time to cancel.")
print("=" * 60 + "\n")

while True:
    ret, frame = video.read()
    if not ret:
        print("Camera frame drop. Retrying...")
        continue

    frame = cv2.flip(frame, 1)
    h_f, w_f = frame.shape[:2]

    detected_box = None

    if detector is not None:
        detector.setInputSize((w_f, h_f))
        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            detected_box = (max(0, x), max(0, y), min(w_f - x, w), min(h_f - y, h))
    else:
        # Center region fallback
        side = int(min(w_f, h_f) * 0.5)
        detected_box = (int((w_f - side) / 2), int((h_f - side) / 2), side, side)

    if detected_box is not None:
        x, y, w, h = detected_box
        if w > 20 and h > 20:
            crop_img = frame[y : y + h, x : x + w]
            if crop_img.size > 0:
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) < target_samples and frame_count % 3 == 0:
                    faces_data.append(resized_img)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 245, 160), 2)
                cv2.putText(
                    frame,
                    f"Samples: {len(faces_data)}/{target_samples}",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 245, 160),
                    2,
                )

    frame_count += 1
    cv2.imshow("AURA Enrollment Chamber — Press 'q' to Quit", frame)
    k = cv2.waitKey(1)
    if k == ord("q") or len(faces_data) >= target_samples:
        break

video.release()
cv2.destroyAllWindows()

if len(faces_data) < target_samples:
    print(f"\nEnrollment cancelled or incomplete ({len(faces_data)}/{target_samples} samples). No changes saved.")
    sys.exit(0)

# Reshape into (N, 7500)
new_faces = np.asarray(faces_data, dtype=np.uint8).reshape(target_samples, -1)
new_names = [name] * target_samples

names_path = os.path.join(DATA_DIR, "names.pkl")
faces_path = os.path.join(DATA_DIR, "faces_data.pkl")

if os.path.exists(names_path) and os.path.exists(faces_path):
    with open(names_path, "rb") as f:
        existing_names = pickle.load(f)
    with open(faces_path, "rb") as f:
        existing_faces = pickle.load(f)
    all_names = existing_names + new_names
    all_faces = np.vstack([existing_faces, new_faces])
else:
    all_names = new_names
    all_faces = new_faces

with open(names_path, "wb") as f:
    pickle.dump(all_names, f)
with open(faces_path, "wb") as f:
    pickle.dump(all_faces, f)

print(f"\n[✓] Successfully saved {target_samples} raw samples for '{name}'.")
print("[*] Running multi-condition data augmentation and retraining model...")

try:
    sys.path.append(BASE_DIR)
    from engine.ml_engine import FaceRecognitionPipeline
    pipeline = FaceRecognitionPipeline(data_dir=DATA_DIR)
    metrics = pipeline.train_enhanced_pipeline()
    print(f"[✓] Model retraining complete! Total dataset: {metrics['augmented_samples']} augmented samples.")
    print(f"[✓] New Enhanced Accuracy: {metrics['enhanced_model']['accuracy']}%")
except Exception as e:
    print(f"[Warning] Automatic retraining notice: {e}")

print("\nReady! You can now start the web application by running 'python run_website.py'\n")
