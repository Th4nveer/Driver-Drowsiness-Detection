"""
============================================================
 Real-Time Driver Drowsiness Detection Pipeline
============================================================
IMPORTANT: This script works with a model trained on
PRE-SCALED data (mean=0, std=1). Live webcam EAR/MAR/Head_Tilt
values are raw, so they are manually scaled here before
being fed into the model using the original dataset's
mean and std (from the unscaled dataset.csv).

Requirements:
    pip install opencv-python mediapipe==0.10.9 joblib numpy pygame scipy

Usage:
    python drowsiness_realtime.py
    python drowsiness_realtime.py --source 0
    python drowsiness_realtime.py --source video.mp4
    python drowsiness_realtime.py --no-alert
============================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import argparse
import time
import sys
import os
from collections import deque
from scipy.spatial import distance as dist

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH       = "drowsiness_model.pkl"

WINDOW_SIZE      = 300   # ~2 seconds at 30fps
DROWSY_THRESHOLD = 0.60  # 65% of window must be drowsy to alert
COOLDOWN_SECONDS = 1.0
DISPLAY_WIDTH    = 960
FPS_SMOOTH       = 15

# ─────────────────────────────────────────────
# RAW→SCALED CONVERSION
# ─────────────────────────────────────────────
# Your cleaned dataset was pre-scaled using StandardScaler.
# These are the mean and std of your ORIGINAL raw dataset
# (dataset.csv before scaling). The model expects scaled input,
# so live webcam values must be transformed: (raw - mean) / std
#
# !! UPDATE THESE with output of:
#    df_raw = pd.read_csv("dataset.csv")
#    print(df_raw[['EAR','MAR','Head_Tilt']].agg(['mean','std']))
#
# Default values below are typical MediaPipe EAR/MAR ranges —
# replace with your actual values for best accuracy.
RAW_MEAN = np.array([0.218414,0.088161,-0.621531])   # [EAR_mean, MAR_mean, HeadTilt_mean]
RAW_STD  = np.array([0.121520,0.143538,17.352403])   # [EAR_std,  MAR_std,  HeadTilt_std]

def scale_features(ear, mar, head_tilt):
    """Apply the same StandardScaler transform used during training."""
    raw = np.array([ear, mar, head_tilt])
    return (raw - RAW_MEAN) / (RAW_STD + 1e-8)


# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ─────────────────────────────────────────────
LEFT_EYE    = [362, 385, 387, 263, 373, 380]
RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
MOUTH_IDX   = [61, 291, 13, 14, 17, 0, 78, 308]
LEFT_EAR_P  = 234
RIGHT_EAR_P = 454


# ─────────────────────────────────────────────
# FEATURE COMPUTATION
# ─────────────────────────────────────────────

def _eucl(p1, p2):
    return dist.euclidean(p1, p2)

def compute_ear(landmarks, eye_indices, W, H):
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H]
                    for i in eye_indices])
    A = _eucl(pts[1], pts[5])
    B = _eucl(pts[2], pts[4])
    C = _eucl(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def compute_mar(landmarks, mouth_idx, W, H):
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H]
                    for i in mouth_idx])
    A = _eucl(pts[2], pts[3])
    B = _eucl(pts[0], pts[1])
    return A / (B + 1e-6)

def compute_head_tilt(landmarks, W, H):
    l  = landmarks[LEFT_EAR_P]
    r  = landmarks[RIGHT_EAR_P]
    dx = (r.x - l.x) * W
    dy = (r.y - l.y) * H
    return abs(np.degrees(np.arctan2(dy, dx)))


# ─────────────────────────────────────────────
# ALERT SYSTEM
# ─────────────────────────────────────────────

class AlertSystem:
    def __init__(self, enabled=True):
        self.enabled      = enabled
        self.last_alert_t = 0
        self._audio_ok    = False

        if self.enabled:
            try:
                import pygame
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self._pygame   = pygame
                self._beep_buf = self._make_beep(freq=880, duration=0.6)
                self._audio_ok = True
                print("[AlertSystem] Audio ready.")
            except Exception as e:
                print(f"[AlertSystem] Audio disabled ({e}). Visual alert only.")

    def _make_beep(self, freq=880, duration=0.6, volume=0.8, sample_rate=44100):
        t      = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave   = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
        stereo = np.column_stack([wave, wave])
        return self._pygame.sndarray.make_sound(stereo)

    def trigger(self):
        now = time.time()
        if now - self.last_alert_t < COOLDOWN_SECONDS:
            return False
        self.last_alert_t = now
        if self._audio_ok:
            try:
                self._beep_buf.play()
            except Exception:
                pass
        return True


# ─────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────

def draw_hud(frame, ear, mar, head_tilt, scaled, drowsy_ratio, is_alert, fps, window_filled):
    H, W = frame.shape[:2]

    # Semi-transparent top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 140), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    WHITE = (255, 255, 255)
    GREEN = (50,  220,  50)
    YELLOW= (0,   200, 255)
    RED   = (50,   50, 230)
    CYAN  = (255, 200,  50)
    GRAY  = (160, 160, 160)

    # Left: raw values
    cv2.putText(frame, f"EAR  : {ear:.3f}  (scaled: {scaled[0]:+.2f})",
                (12, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)
    cv2.putText(frame, f"MAR  : {mar:.3f}  (scaled: {scaled[1]:+.2f})",
                (12, 54),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)
    cv2.putText(frame, f"Tilt : {head_tilt:.1f}deg  (scaled: {scaled[2]:+.2f})",
                (12, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)
    cv2.putText(frame, f"FPS  : {fps:.1f}",
                (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)

    # Centre: status + bar
    ratio_col = RED if is_alert else (YELLOW if drowsy_ratio > 0.40 else GREEN)
    status    = "DROWSY" if is_alert else ("WARNING" if drowsy_ratio > 0.40 else "ALERT")
    cv2.putText(frame, f"{status}  {drowsy_ratio*100:.0f}%",
                (W//2 - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, ratio_col, 2)

    bx, by, bw, bh = W//2 - 90, 44, 180, 20
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    fp = int(drowsy_ratio * bw)
    if fp > 0:
        cv2.rectangle(frame, (bx, by), (bx + fp, by + bh),
                      (40, 40, 210) if is_alert else (40, 170, 40), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), WHITE, 1)

    # Threshold marker
    tx = bx + int(DROWSY_THRESHOLD * bw)
    cv2.line(frame, (tx, by - 4), (tx, by + bh + 4), (0, 210, 255), 2)
    cv2.putText(frame, f"{int(DROWSY_THRESHOLD*100)}%",
                (tx - 10, by + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 210, 255), 1)

    # Window progress
    cv2.putText(frame,
                f"Window: {window_filled}/{WINDOW_SIZE} frames  "
                f"({'ready' if window_filled==WINDOW_SIZE else 'filling...'})",
                (W//2 - 90, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY, 1)

    # Alert banner
    if is_alert:
        bann    = np.zeros((65, W, 3), dtype=np.uint8)
        bann[:] = (0, 0, 170)
        cv2.putText(bann, "!! DROWSINESS DETECTED  --  PLEASE PULL OVER !!",
                    (W//2 - 310, 42), cv2.FONT_HERSHEY_DUPLEX, 0.88, WHITE, 2)
        frame[H - 65:H] = cv2.addWeighted(bann, 0.88, frame[H - 65:H], 0.12, 0)

    return frame


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run(source=0, no_alert=False):

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: '{MODEL_PATH}'")
        print("        Run train_drowsiness.py first.")
        sys.exit(1)

    print(f"[INFO] Loading model from '{MODEL_PATH}' ...")
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model type : {type(model).__name__}")

    # Warn if still a pipeline (scaler double-applied)
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            print("[WARN] Model is a Pipeline — it contains a scaler internally.")
            print("       This will DOUBLE-SCALE the input. Re-run train_drowsiness.py")
            print("       to get a plain RandomForestClassifier without the pipeline.")
    except ImportError:
        pass

    # MediaPipe
    mp_face   = mp.solutions.face_mesh
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Camera FPS       : {native_fps:.1f}")
    print(f"[INFO] Window size      : {WINDOW_SIZE} frames (~{WINDOW_SIZE/native_fps:.1f}s)")
    print(f"[INFO] Drowsy threshold : {DROWSY_THRESHOLD*100:.0f}%")
    print(f"[INFO] RAW_MEAN used    : EAR={RAW_MEAN[0]}, MAR={RAW_MEAN[1]}, Tilt={RAW_MEAN[2]}")
    print(f"[INFO] RAW_STD  used    : EAR={RAW_STD[0]},  MAR={RAW_STD[1]},  Tilt={RAW_STD[2]}")
    print("[INFO] Press Q or ESC to quit.\n")

    alert_sys = AlertSystem(enabled=not no_alert)
    window    = deque(maxlen=WINDOW_SIZE)
    fps_times = deque(maxlen=FPS_SMOOTH)

    while True:
        t0     = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        h, w  = frame.shape[:2]
        frame = cv2.resize(frame, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)))
        H, W  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        ear = mar = head_tilt = 0.0
        scaled = np.array([0.0, 0.0, 0.0])

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            ear_l     = compute_ear(lm, LEFT_EYE,  W, H)
            ear_r     = compute_ear(lm, RIGHT_EYE, W, H)
            ear       = (ear_l + ear_r) / 2.0
            mar       = compute_mar(lm, MOUTH_IDX, W, H)
            head_tilt = compute_head_tilt(lm, W, H)

            # Scale raw values to match training distribution
            scaled   = scale_features(ear, mar, head_tilt)
            features = scaled.reshape(1, -1)

            pred = model.predict(features)[0]   # 0=Alert, 1=Drowsy
            window.append(int(pred))

            # Draw face mesh
            mp_draw.draw_landmarks(
                frame,
                result.multi_face_landmarks[0],
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
        else:
            window.append(0)   # no face = treat as alert (neutral)

        # Sliding window decision — only fire after window is fully filled
        drowsy_ratio = sum(window) / max(len(window), 1)
        is_alert     = (len(window) == WINDOW_SIZE) and (drowsy_ratio >= DROWSY_THRESHOLD)

        if is_alert:
            alert_sys.trigger()

        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-6)

        frame = draw_hud(frame, ear, mar, head_tilt, scaled,
                         drowsy_ratio, is_alert, fps, len(window))

        cv2.imshow("Driver Drowsiness Monitor", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[INFO] Pipeline stopped.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Driver Drowsiness Detection")
    parser.add_argument("--source",   default=0,
                        help="0 = webcam, or path to a video file")
    parser.add_argument("--no-alert", action="store_true",
                        help="Disable audio alert")
    args = parser.parse_args()

    try:
        src = int(args.source)
    except (ValueError, TypeError):
        src = args.source

    run(source=src, no_alert=args.no_alert)