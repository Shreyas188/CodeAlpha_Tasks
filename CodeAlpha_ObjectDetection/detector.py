"""
CodeAlpha - Real-time Object Detection and Tracking
Task 4: Object Detection and Tracking
Using YOLOv8 + SORT (Simple Online and Realtime Tracking)

Author: CodeAlpha Intern
"""
import cv2
import numpy as np
import time
import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from sort_tracker import Sort

# ── Color palette (one per track ID, cycles through) ──────────────────────────
COLORS = [
    (255,  56,  56), ( 56, 255,  56), ( 56,  56, 255), (255, 255,  56),
    (255,  56, 255), ( 56, 255, 255), (128,   0, 128), (255, 165,   0),
    (  0, 200,  83), (  0,  64, 255), (200, 200,   0), (200,   0,   0),
    (  0, 200, 200), (255, 160, 200), ( 80,   0, 200), (220, 100, 220),
]


def get_color(track_id: int):
    """Return a consistent BGR color for a given tracking ID."""
    return COLORS[int(track_id) % len(COLORS)]


def draw_bounding_box(frame, x1, y1, x2, y2, track_id, label, confidence, color):
    """
    Draw a professional bounding box with corner accents and a label banner.
    """
    # Main rectangle (thin)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Corner accent lines for a sci-fi HUD feel
    L = 15   # length of each corner tick
    T = 3    # thickness
    corners = [
        ((x1, y1), ( L, 0), (0,  L)),   # top-left
        ((x2, y1), (-L, 0), (0,  L)),   # top-right
        ((x1, y2), ( L, 0), (0, -L)),   # bottom-left
        ((x2, y2), (-L, 0), (0, -L)),   # bottom-right
    ]
    for (ox, oy), (dx1, dy1), (dx2, dy2) in corners:
        cv2.line(frame, (ox, oy), (ox + dx1, oy + dy1), color, T)
        cv2.line(frame, (ox, oy), (ox + dx2, oy + dy2), color, T)

    # Label banner
    text = f"ID:{track_id}  {label}  {confidence:.0%}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

    banner_y1 = max(y1 - th - 12, 0)
    banner_y2 = max(y1, th + 12)
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 12, banner_y2), color, -1)
    cv2.putText(frame, text, (x1 + 6, banner_y2 - 5),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def draw_stats_panel(frame, fps: float, object_counts: dict, frame_num: int, paused: bool):
    """
    Render a semi-transparent HUD panel in the top-left corner.
    Shows FPS, frame number, total objects, per-class counts, and key hints.
    """
    h, w = frame.shape[:2]
    n_classes = min(len(object_counts), 6)
    panel_h = 115 + n_classes * 22
    panel_w = 290

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title bar
    cv2.rectangle(frame, (8, 8), (8 + panel_w, 34), (0, 80, 120), -1)
    cv2.putText(frame, "CodeAlpha  |  YOLO + SORT",
                (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 212, 255), 2, cv2.LINE_AA)

    # FPS with colour-coded health indicator
    fps_color = (0, 230, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 235)
    cv2.putText(frame, f"FPS : {fps:5.1f}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 2, cv2.LINE_AA)

    # Frame counter + pause indicator
    status_text = f"Frame: {frame_num}" + ("  [PAUSED]" if paused else "")
    cv2.putText(frame, status_text, (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Object counts
    total = sum(object_counts.values())
    cv2.putText(frame, f"Objects: {total}", (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 50), 2, cv2.LINE_AA)

    y = 122
    for cls_name, count in list(object_counts.items())[:6]:
        cv2.putText(frame, f"  {cls_name:<14s}{count:>2d}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
        y += 22

    # Key-hint footer
    cv2.putText(frame, "Q: Quit   S: Screenshot   P: Pause",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)


def run_detection(source=0, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4):
    """
    Run real-time object detection + SORT tracking.

    Args:
        source         : 0 for webcam, or path string to a video file.
        model_path     : Path to a YOLOv8 weights file (.pt).
        conf_threshold : Minimum detection confidence (0–1).
    """
    print(f"\n{'='*52}")
    print(f"  CodeAlpha — Real-time Object Detection & Tracking")
    print(f"  Model: {model_path}   Conf: {conf_threshold:.2f}")
    print(f"{'='*52}")
    print("  Q → Quit   S → Save screenshot   P → Pause/Resume")
    print(f"{'='*52}\n")

    # ── Load YOLO model ────────────────────────────────────────────────────────
    print(f"Loading model '{model_path}' …", end=" ", flush=True)
    model = YOLO(model_path)
    print("OK")

    # ── Initialize SORT tracker ────────────────────────────────────────────────
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # ── Open video source ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: '{source}'")
        return

    # ── Prepare CSV detection log ──────────────────────────────────────────────
    log_dir  = Path("detection_logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"detections_{timestamp}.csv"
    log_file  = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["frame", "track_id", "class", "confidence",
                         "x1", "y1", "x2", "y2"])
    print(f"Detection log → {log_path}\n")

    frame_count = 0
    paused      = False
    prev_time   = time.time()
    fps         = 0.0
    frame       = None  # holds last rendered frame for pause display

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended / no more frames.")
                break

            frame_count += 1

            # Exponential-moving-average FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            # ── YOLOv8 inference ──────────────────────────────────────────────
            results = model(frame, conf=conf_threshold, verbose=False)

            detections   = []   # [x1, y1, x2, y2, conf]
            class_names  = {}   # det_idx → (cls_name, conf)
            object_counts = {}

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf     = float(box.conf[0])
                    cls_id   = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    detections.append([x1, y1, x2, y2, conf])
                    class_names[len(detections) - 1] = (cls_name, conf)
                    object_counts[cls_name] = object_counts.get(cls_name, 0) + 1

            # ── SORT tracker update ───────────────────────────────────────────
            dets_array    = np.array(detections) if detections else np.empty((0, 5))
            tracked_objs  = tracker.update(dets_array)

            # ── Draw tracked objects ──────────────────────────────────────────
            for obj in tracked_objs:
                x1, y1, x2, y2, track_id = map(int, obj)
                color = get_color(track_id)

                # Find best-matching detection for label/confidence
                label    = "object"
                conf_val = 0.0
                if detections:
                    best_iou = 0.0
                    for idx, det in enumerate(detections):
                        dx1, dy1, dx2, dy2 = det[:4]
                        inter = (max(0, min(x2, dx2) - max(x1, dx1)) *
                                 max(0, min(y2, dy2) - max(y1, dy1)))
                        union = ((x2-x1)*(y2-y1) + (dx2-dx1)*(dy2-dy1) - inter)
                        iou = inter / (union + 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            if idx in class_names:
                                label, conf_val = class_names[idx]

                draw_bounding_box(frame, x1, y1, x2, y2,
                                  track_id, label, conf_val, color)
                csv_writer.writerow([frame_count, track_id, label,
                                     f"{conf_val:.3f}", x1, y1, x2, y2])

            # ── HUD overlay ───────────────────────────────────────────────────
            draw_stats_panel(frame, fps, object_counts, frame_count, paused)

        # ── Show frame (always, even when paused) ─────────────────────────────
        if frame is not None:
            cv2.imshow("CodeAlpha — Object Detection & Tracking  [YOLOv8 + SORT]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # Q or Esc → quit
            break
        elif key == ord('s') and frame is not None:
            ss_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(ss_path, frame)
            print(f"[Screenshot] saved → {ss_path}")
        elif key == ord('p'):
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}]")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Frames processed: {frame_count}")
    print(f"Detection log saved → {log_path}")


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CodeAlpha — Real-time Object Detection & Tracking (YOLOv8 + SORT)"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: '0' for webcam, or path to a video file"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLO model weights (e.g. yolov8n.pt, yolov8s.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Detection confidence threshold (0.1 – 0.9)"
    )
    args = parser.parse_args()

    # Auto-convert numeric string sources to int (webcam index)
    source = int(args.source) if args.source.isdigit() else args.source
    run_detection(source=source, model_path=args.model, conf_threshold=args.conf)
