# main.py
from ultralytics import YOLO
import cv2
import numpy as np
import math
import imutils
import pygame
import threading
import time
import os

# ================== CONFIGURATION ==================
VIDEO_SOURCE = 'videos/road.mp4'     # Change here: 0 for webcam, or 'videos/road.mp4'
OUTPUT_PATH = 'outputs/result.avi'
CONF_THRESHOLD = 0.35
TARGET_CLASSES = ['person', 'car', 'motorbike', 'bus', 'truck']  # focus classes
ALERT_COOLDOWN = 3.0                 # seconds between audio alerts

# Create output folder if not exists
os.makedirs('outputs', exist_ok=True)

# ================== LOAD YOLO MODEL ==================
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

last_alert_time = 0.0

# ================== AUDIO ALERT FUNCTION ==================
def play_alert():
    def _play():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load('alert.wav')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print("Audio play error:", e)
    threading.Thread(target=_play, daemon=True).start()


# ================== PROXIMITY ESTIMATION ==================
def estimate_proximity(box_w, frame_w):
    """
    Heuristic: wider bounding box = closer object.
    Returns 0 (far) to 1 (very close).
    """
    rel = box_w / float(frame_w)
    score = min(1.0, (rel - 0.08) / (0.4 - 0.08))  # normalize
    score = max(0.0, score)
    return score

# ================== MAIN FUNCTION ==================
def main():
    global last_alert_time

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {VIDEO_SOURCE}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

    print("🚗 Processing started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video ended or no frame captured.")
            break

        frame = imutils.resize(frame, width=960)
        f_h, f_w = frame.shape[:2]

        # Run YOLO detection
        results = model(frame, stream=True, conf=CONF_THRESHOLD, verbose=False)
        warning_displayed = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                label = model.names[cls_idx]

                if label not in TARGET_CLASSES:
                    continue

                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 200, 20), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 200, 20), 2)

                # Proximity alert logic
                box_w = (x2 - x1)
                proximity = estimate_proximity(box_w, f_w)

                if proximity > 0.75:  # Object too close
                    cv2.putText(frame, "⚠️ COLLISION WARNING ⚠️",
                                (int(f_w * 0.2), 50),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                    warning_displayed = True

        # Alert with cooldown
        if warning_displayed:
            now = time.time()
            if now - last_alert_time > ALERT_COOLDOWN:
                if os.path.exists('alert.wav'):
                    play_alert()
                last_alert_time = now

        out.write(frame)
        cv2.imshow("ADAS-Lite", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Output saved to:", OUTPUT_PATH)

# ================== RUN ==================
if __name__ == '__main__':
    main()
