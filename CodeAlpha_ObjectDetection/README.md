# 🎯 CodeAlpha — Object Detection & Tracking

> **Task 4** of the CodeAlpha AI/ML Internship  
> Real-time multi-object detection and tracking using **YOLOv8** + **SORT**

---

## 📌 Overview

This project delivers a complete, production-quality pipeline for detecting and tracking objects in real time from a webcam or video file.

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | SORT — Kalman Filter + Hungarian Algorithm |
| GUI Launcher | CustomTkinter |
| Display | OpenCV |
| Logging | CSV |

---

## 🗂️ Project Structure

```
CodeAlpha_ObjectDetection/
│
├── app.py            # GUI launcher (CustomTkinter)
├── detector.py       # Core detection + tracking engine
├── sort_tracker.py   # Full SORT implementation (no external sort lib)
├── requirements.txt  # Pinned dependencies
└── README.md
```

---

## ✨ Features

### Detection (`detector.py`)
- **YOLOv8** inference on every frame (n / s / m weight variants)
- **SORT** tracker assigns persistent IDs across frames
- **Corner-accent bounding boxes** colour-coded by tracking ID
- **HUD overlay** with FPS (colour-coded green/orange/red), frame counter, and per-class object counts
- **CSV detection log** auto-saved to `detection_logs/`

### Keyboard Controls (OpenCV window)
| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `S` | Save screenshot (`screenshot_YYYYMMDD_HHMMSS.jpg`) |
| `P` | Pause / Resume |

### GUI Launcher (`app.py`)
- Select **Webcam** or **Video File** source
- Browse-button file picker
- **Model selector** (YOLOv8 n / s / m)
- **Confidence slider** 0.10 – 0.90
- **Start / Stop** toggle button — spawns & terminates the detector subprocess

### SORT Tracker (`sort_tracker.py`)
- Kalman Filter state: `[cx, cy, scale, ratio, vx, vy, vs]`
- **Hungarian algorithm** (via `scipy.optimize.linear_sum_assignment`) for optimal detection-to-track assignment
- IoU-based matching with configurable threshold
- Configurable `max_age` (frames before track deletion) and `min_hits` (frames before track confirmation)
- **100 % standalone** — no external `sort` library required

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> YOLOv8 weights (`yolov8n.pt`, etc.) are downloaded automatically on first run.

### 2 — Launch the GUI

```bash
python app.py
```

### 3 — Or run the detector directly

```bash
# Webcam
python detector.py

# Custom video file
python detector.py --source path/to/video.mp4

# Higher-accuracy model, stricter confidence
python detector.py --source 0 --model yolov8s.pt --conf 0.5
```

---

## ⚙️ CLI Arguments (`detector.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | `0` = webcam, or path to video file |
| `--model` | `yolov8n.pt` | YOLO weights file |
| `--conf` | `0.4` | Detection confidence threshold (0–1) |

---

## 📦 Dependencies

```
ultralytics==8.3.27   # YOLOv8
opencv-python==4.10.0.84
numpy==1.26.4
scipy==1.14.1          # Hungarian algorithm
pillow==10.4.0
customtkinter==5.2.2
```

---

## 🧠 How SORT Works

```
Frame N arrives
      │
      ▼
┌─────────────┐     ┌─────────────────────┐
│  YOLOv8     │────▶│  Kalman Predict      │  (each existing track)
│  Detections │     └────────┬────────────┘
└─────────────┘              │
                             ▼
                    ┌─────────────────────┐
                    │  IoU Matrix         │
                    │  Hungarian Assign   │
                    └────────┬────────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
          Matched        Unmatched     Unmatched
          → Update       Detections    Tracks
          Kalman         → New Track   → Age + 1
                                       (delete if > max_age)
```

---

## 📊 Output

- **Live OpenCV window** with annotated video feed
- **`detection_logs/detections_YYYYMMDD_HHMMSS.csv`** — one row per tracked object per frame

CSV columns: `frame, track_id, class, confidence, x1, y1, x2, y2`

---

## 👨‍💻 Author

**CodeAlpha Intern**  
CodeAlpha AI/ML Internship — Task 4: Object Detection and Tracking

---

*Built with ❤️ using YOLOv8, OpenCV, and SORT.*
