# realtime_detection.py
# Driver Drowsiness Detection — Real-Time Pipeline
# NTCC Project | Amity School of Engineering & Technology

import os
import sys
import cv2
import math
import time
import json
import h5py
import numpy as np
from scipy.spatial import distance as dist

# ── 1. CLEAN ENVIRONMENT ──────────────────────────────────────────────────────
# Purge any legacy flags to force pure, native Keras 3 execution.
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import tensorflow as tf
import mediapipe as mp

print(f'[SYSTEM] TensorFlow version : {tf.__version__}')

# Native modern Keras imports
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# ── Config & Constants ────────────────────────────────────────────────────────
EYE_MODEL_PATH = os.path.join('models', 'eye_model.h5')
FACE_MODEL_PATH = os.path.join('models', 'face_model.h5')

IMG_SIZE     = 96
EAR_THRESH   = 0.25
MAR_THRESH   = 0.50
TILT_THRESH  = 20.0
SCORE_THRESH = 0.50
ALERT_SECS   = 2.0

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_PTS = [61, 291, 13, 14]

# ── HDF5 Scrubber (The Ultimate Fix) ──────────────────────────────────────────
def patch_h5_file(filepath):
    """
    Physically opens the .h5 file and translates legacy Keras 2 JSON metadata 
    into strict Keras 3 compliance before the TensorFlow loader even touches it.
    """
    try:
        with h5py.File(filepath, 'a') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Recursively scrub and translate incompatible keys
                def scrub(obj):
                    if isinstance(obj, dict):
                        # 1. Destroy the quantization artifact that crashes Dense layers
                        obj.pop('quantization_config', None)
                        
                        # 2. Translate the legacy batch_shape parameter for InputLayers
                        if 'batch_shape' in obj:
                            obj['batch_input_shape'] = obj.pop('batch_shape')
                            
                        for v in obj.values():
                            scrub(v)
                    elif isinstance(obj, list):
                        for i in obj:
                            scrub(i)
                            
                scrub(config)
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print(f"[SYSTEM] Metadata scrubbed and aligned for {filepath}")
    except Exception as e:
        print(f"[WARNING] Could not patch {filepath}: {e}")

# ── Model Loader ──────────────────────────────────────────────────────────────
def load_models_safe():
    for path in [EYE_MODEL_PATH, FACE_MODEL_PATH]:
        if not os.path.exists(path):
            print(f'\n[FATAL] Model file missing: {path}')
            print('Ensure your original .h5 models exist in the models/ directory.\n')
            sys.exit(1)
            
        # Apply the physical data patch to ensure compatibility
        patch_h5_file(path)

    print('[INFO] Loading networks natively into modern TensorFlow...')
    try:
        eye_m = tf.keras.models.load_model(EYE_MODEL_PATH, compile=False)
        face_m = tf.keras.models.load_model(FACE_MODEL_PATH, compile=False)
        
        for m in (eye_m, face_m):
            m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
        print('[INFO] Models loaded and compiled successfully.')
        return eye_m, face_m
        
    except Exception as e:
        print(f'\n[FATAL] Integration failed during model load.')
        print(f'Error Trace: {str(e)}\n')
        sys.exit(1)

# ── Mediapipe Initialization ──────────────────────────────────────────────────
def get_face_mesh():
    """
    Initializes MediaPipe FaceMesh using the standard public API.
    Refactored to eliminate fragile internal path guessing.
    """
    print('[INFO] Initializing MediaPipe Face Tracking...')
    try:
        mp_face_mesh = mp.solutions.face_mesh
        return mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
    except AttributeError as e:
        print('\n[FATAL] MediaPipe installation is corrupted or missing submodules.')
        print('Please run: pip uninstall -y mediapipe && pip install mediapipe==0.10.14\n')
        sys.exit(1)
        
# ── Camera Pipeline ───────────────────────────────────────────────────────────
def find_camera():
    print('[INFO] Scanning for cameras...')
    
    # We rearranged the search order to check 1, 2, and 3 BEFORE checking 0.
    # This bypasses the OBS Virtual Camera (which usually hijacks index 0).
    search_order = [1, 2, 3, 0] 
    
    for idx in search_order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f'[INFO] Locked onto Camera #{idx} @ {w}x{h}')
                return cap
            cap.release()
            
    print('\n[FATAL] No video stream detected. Ensure webcam is connected.\n')
    sys.exit(1)

# ── Feature Extraction ────────────────────────────────────────────────────────
def get_ear(lms, eye_idx, w, h):
    p = [(lms[i].x * w, lms[i].y * h) for i in eye_idx]
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return round((A + B) / (2.0 * max(C, 1e-6)), 4)

def get_mar(lms, mouth_idx, w, h):
    p = [(lms[i].x * w, lms[i].y * h) for i in mouth_idx]
    horiz = dist.euclidean(p[0], p[1])
    vert  = dist.euclidean(p[2], p[3])
    return round(vert / max(horiz, 1e-6), 4)

def get_tilt(lms, w, h):
    lx, ly = lms[33].x * w, lms[33].y * h
    rx, ry = lms[263].x * w, lms[263].y * h
    return round(abs(math.degrees(math.atan2(ry - ly, max(rx - lx, 1e-6)))), 2)

def crop_roi(frame, lms, indices, w, h, pad):
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices] if indices else [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    x1, y1 = max(0, min(p[0] for p in pts) - pad), max(0, min(p[1] for p in pts) - pad)
    x2, y2 = min(w, max(p[0] for p in pts) + pad), min(h, max(p[1] for p in pts) + pad)
    roi = frame[y1:y2, x1:x2]
    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) if roi.size > 0 else None

def prep(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_input(np.expand_dims(rgb.astype('float32'), 0))

def calculate_score(ear, mar, tilt, eye_p, yawn_p):
    ear_s  = max(0.0, (EAR_THRESH - ear) / max(EAR_THRESH, 1e-6))
    mar_s  = max(0.0, (mar - MAR_THRESH) / max(1 - MAR_THRESH, 1e-6))
    tilt_s = max(0.0, (tilt - TILT_THRESH) / max(90 - TILT_THRESH, 1e-6))
    return round(min(0.30 * eye_p + 0.30 * yawn_p + 0.20 * ear_s + 0.10 * mar_s + 0.10 * tilt_s, 1.0), 4)

# ── HUD / Overlay ─────────────────────────────────────────────────────────────
def draw_hud(frame, ear, mar, tilt, sc, st, col, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 52), col, -1)
    cv2.putText(frame, f'{st}   Score: {sc:.2f}', (12, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)
    metrics = [f'EAR  : {ear:.3f}', f'MAR  : {mar:.3f}', f'Tilt : {tilt:.1f} deg', f'FPS  : {fps:.0f}']
    for i, text in enumerate(metrics):
        cv2.putText(frame, text, (10, 88 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210,210,210), 1)
    return frame

def render_alert(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,h), (0,0,220), 14)
    cv2.putText(frame, '!! DROWSY ALERT !!', (w//2 - 185, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,220), 3)

# ── Core Loop ─────────────────────────────────────────────────────────────────
def run():
    eye_model, face_model = load_models_safe()
    cap = find_camera()
    face_mesh = get_face_mesh()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print('\n[SYSTEM] Pipeline active. Press Q to quit.\n')
    drowsy_since = None
    alert_on = False
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break

        h, w = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        fps = 1.0 / max(now - prev_t, 0.001)
        prev_t = now
        ear = mar = tilt = sc = ep = yp = 0.0
        st, col = 'NO FACE', (100, 100, 100)

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            ear = round((get_ear(lms, LEFT_EYE, w, h) + get_ear(lms, RIGHT_EYE, w, h)) / 2.0, 4)
            mar = get_mar(lms, MOUTH_PTS, w, h)
            tilt = get_tilt(lms, w, h)

            eye_crop = crop_roi(frame, lms, LEFT_EYE, w, h, pad=15)
            if eye_crop is not None: ep = float(eye_model.predict(prep(eye_crop), verbose=0)[0][0])

            face_crop = crop_roi(frame, lms, None, w, h, pad=30)
            if face_crop is not None: yp = float(face_model.predict(prep(face_crop), verbose=0)[0][0])

            sc = calculate_score(ear, mar, tilt, ep, yp)
            
            if sc < 0.30: st, col = 'AWAKE', (30, 180, 30)
            elif sc < 0.50: st, col = 'ALERT', (0, 165, 255)
            else: st, col = 'DROWSY', (0, 0, 220)

            if st == 'DROWSY':
                if drowsy_since is None: drowsy_since = time.time()
                elif time.time() - drowsy_since >= ALERT_SECS: alert_on = True
            else:
                drowsy_since = None
                alert_on = False

        frame = draw_hud(frame, ear, mar, tilt, sc, st, col, fps)
        if alert_on: render_alert(frame)
        cv2.imshow('Driver Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print('[SYSTEM] Pipeline terminated gracefully.')

if __name__ == '__main__':
    run()