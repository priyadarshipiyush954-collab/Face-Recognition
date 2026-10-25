---
name: codebase-map
description: Key files and their roles in the architecture
metadata:
  type: reference
---

### Core Engine
- **engine/ml_engine.py** (599 lines)
  - `DataAugmentor`: 16x augmentation pipeline (rotations, CLAHE, gamma, brightness, scale, blur)
  - `FeatureExtractor`: Multi-domain features (spatial intensities, gradients, texture) → 3,750 dims → PCA → 64 dims
  - `EnhancedFaceModel`: PCA + SVM (RBF, C=10.0) + KNN (cosine, k=5), soft-voting ensemble
  - `FaceRecognitionPipeline`: Load dataset, detect/recognize, enroll, train
  - YuNet detector (ONNX) in `data/face_detection_yunet_2023mar.onnx`

- **engine/attendance_manager.py** (131 lines)
  - `AttendanceManager`: Debounced punch logging (45s cooldown), CSV export
  - CSV schema: `NAME, TIME, CONFIDENCE, STATUS`

### Server & API
- **server.py** (271 lines)
  - FastAPI endpoints: `/api/recognize`, `/api/attendance/punch`, `/api/students/enroll`, `/api/train`, `/api/benchmark`
  - Base64 image decode/encode pipeline
  - Static file mounting

### Frontend
- **static/index.html**: Liquid glass layout
- **static/styles.css**: Glassmorphism, waterflow animations
- **static/waterflow.js**: Canvas fluid simulation
- **static/app.js**: Webcam HUD, API calls, Web Audio chimes

### Data
- **data/**: Pickled faces/names, trained model, YuNet ONNX, metrics JSON
- **Attendance/**: Daily CSV logs

### Utils
- **run_website.py**: 1-click launcher
- **add_faces.py**: Webcam enrollment
- **train_enhanced.py**: CLI training
- **test.py**: Console tester

**Tech Stack**: FastAPI, OpenCV 5.0, scikit-learn, numpy, pydantic, Jinja2
