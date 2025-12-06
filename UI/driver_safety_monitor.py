"""
Driver Safety Monitor - Aplicación Principal
"""

# --- IMPORTS ---
import cv2 as cv
import numpy as np
import mediapipe as mp
from time import time
from collections import deque
from dataclasses import dataclass
import math
import os
from ultralytics import YOLO

# --- UI ---
from ui_manager_v3 import (
    UIManager, 
    FusionWeights, 
    fuse_scores, 
    show_splash_screen,
    CAM_WIDTH,
    CAM_HEIGHT
)

# =======================================================
# 0. CARGAR MODELOS YOLO (celular y cinturón)
# =======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CELULAR_MODEL_PATH = os.path.join(SCRIPT_DIR, "celular.pt")
CINTURON_MODEL_PATH = os.path.join(SCRIPT_DIR, "cinturon.pt")

yolo_phone = YOLO(CELULAR_MODEL_PATH)
yolo_belt  = YOLO(CINTURON_MODEL_PATH)


# =======================================================
# 0.1. LÓGICA DEL CINTURÓN (grace + 5s ausencia)
# =======================================================

# Periodo de gracia
# Una vez detectado el cinturón, no volvemos a preguntar hasta después de X segundos.
# Esto es porque en la vida real, nadie maneja quitándose y poniéndose el cinturón a cada momento.
GRACE_AFTER_DETECT = 20
# Tiempo de nueva detección
# Una vez que detectamos el cinturón, pasamos unos segundos sin volver a preguntar si lo tiene puesto.
# Pasado ese tiempo, evaluamos durante una cantidad pequeña de segundos. 
# Para detectar que falta el cinturón, debe estar 5 segundos completos sin detectarlo.
MISSING_DURATION = 5
# Cuando se detecta el cinturón, los contadores se reinician. 

class BeltMonitor:
    def __init__(self, grace_after_detect=GRACE_AFTER_DETECT, require_missing_duration=MISSING_DURATION):
        self.grace_after_detect = grace_after_detect
        self.require_missing_duration = require_missing_duration

        self.last_belt_ok_time = None
        self.first_missing_time = None

    def update(self, belt_detected: bool, ts: float):
        if belt_detected:
            self.last_belt_ok_time = ts
            self.first_missing_time = None
            return 0

        if self.last_belt_ok_time is None:
            #return self._handle_missing(ts)
            return 100

        if (ts - self.last_belt_ok_time) < self.grace_after_detect:
            self.first_missing_time = None
            return 0

        return self._handle_missing(ts)

    def _handle_missing(self, ts: float):
        if self.first_missing_time is None:
            self.first_missing_time = ts
            return 0

        if (ts - self.first_missing_time) >= self.require_missing_duration:
            return 100

        return 0

# =======================================================
# 0.2. MONITOR DE ROSTRO — alerta si no hay rostro 5s
# =======================================================

class FaceMonitor:
    def __init__(self, missing_duration=5):
        self.missing_duration = missing_duration
        self.first_missing_time = None

    def update(self, face_detected: bool, ts: float):
        if face_detected:
            self.first_missing_time = None
            return 0  # sin alerta

        # si no se detecta rostro...
        if self.first_missing_time is None:
            self.first_missing_time = ts
            return 0

        if (ts - self.first_missing_time) >= self.missing_duration:
            return 100  # ALERTA

        return 0

# =======================================================
# 1. MEDIA PIPE HELPERS
# =======================================================

class MPHelpers:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(max_num_hands=2)
        self.pose = mp.solutions.pose.Pose(model_complexity=1)

    def process(self, bgr):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        face = self.face_mesh.process(rgb)
        hands = self.hands.process(rgb)
        pose = self.pose.process(rgb)
        return face, hands, pose


# =======================================================
# 2. SOMNOLENCIA: EAR / MAR / PERCLOS / BLINK RATE
# =======================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def compute_EAR(landmarks, eye_ids):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])
    A = euclidean(pts[eye_ids[1]], pts[eye_ids[5]])
    B = euclidean(pts[eye_ids[2]], pts[eye_ids[4]])
    C = euclidean(pts[eye_ids[0]], pts[eye_ids[3]])
    EAR = (A + B) / (2.0 * C)
    return EAR

def compute_MAR(landmarks, mouth_ids):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])
    A = euclidean(pts[mouth_ids[0]], pts[mouth_ids[1]])
    C = euclidean(pts[mouth_ids[2]], pts[mouth_ids[3]])
    return A / C


class PERCLOS:
    def __init__(self, window_seconds=60):
        self.window = window_seconds
        self.data = deque()

    def update(self, closed, t):
        self.data.append((t, closed))
        while self.data and self.data[0][0] < t - self.window:
            self.data.popleft()

    def value(self):
        if not self.data:
            return 0
        closed = sum(1 for _, c in self.data if c)
        return closed / len(self.data)


class BlinkRate:
    def __init__(self, window=15):
        self.window = window
        self.blinks = deque()
        self.prev_closed = False

    def update(self, closed, t):
        if closed and not self.prev_closed:
            self.blinks.append(t)
        self.prev_closed = closed
        while self.blinks and self.blinks[0] < t - self.window:
            self.blinks.popleft()

    def value(self):
        return len(self.blinks) / self.window


# =======================================================
# *** AGREGADO *** MANO TAPANDO CARA
# =======================================================

def hand_covers_face(face_landmarks, hands_results, frame_w, frame_h):
    if hands_results is None or not hands_results.multi_hand_landmarks:
        return False

    pts = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in face_landmarks])

    eye_mouth_y1 = int(min(pts[160][1], pts[13][1]))  # ojos
    eye_mouth_y2 = int(max(pts[160][1], pts[14][1]))  # boca
    eye_mouth_x1 = int(min(pts[33][0], pts[263][0]))
    eye_mouth_x2 = int(max(pts[33][0], pts[263][0]))

    for hand in hands_results.multi_hand_landmarks:
        for lm in hand.landmark:
            x = lm.x * frame_w
            y = lm.y * frame_h
            if eye_mouth_x1 <= x <= eye_mouth_x2 and eye_mouth_y1 <= y <= eye_mouth_y2:
                return True

    return False


# =======================================================
# 3. CINTURÓN — YOLO
# =======================================================

def detect_belt_yolo(frame, min_conf=0.5):
    """
    Devuelve:
    0.0  si encuentra cinturón
    100 si no lo encuentra
    """
    res = yolo_belt(frame, verbose=False)[0]

    for b in res.boxes:
        if float(b.conf[0]) >= min_conf:
            return 0.0

    return 100.0


# =======================================================
# 4. CELULAR — YOLO
# =======================================================

MINIMUM_CONFIDENCE_PHONE = 0.6

def detect_phone_yolo(frame, min_conf=MINIMUM_CONFIDENCE_PHONE):
    """
    cls=1 → celular
    """
    res = yolo_phone(frame, verbose=False)[0]

    for b in res.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])

        if cls == 1 and conf >= min_conf:
            return 100.0

    return 0.0


# =======================================================
# 5. MAIN
# =======================================================

def main():

    show_splash_screen(3000)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    mp_h = MPHelpers()
    perclos = PERCLOS()
    blinks = BlinkRate()
    belt_monitor = BeltMonitor()
    face_monitor = FaceMonitor(missing_duration=0.5)

    ui = UIManager(col1_x=16, col2_x=340)
    weights = FusionWeights(0.5, 0.25, 0.25)

    last_t = time()

    hand_on_face = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        now = time()
        fps = 1.0 / (now - last_t)
        last_t = now

        canvas = ui.create_canvas(frame)

        face, hands, pose = mp_h.process(frame)
        face_detected = bool(face.multi_face_landmarks)

        EAR, MAR = 0, 0
        eyes_closed = False
        yawn = False
        score_somn = 0

        # ----------------------------------------
        # SOMNOLENCIA REAL
        # ----------------------------------------
        if face.multi_face_landmarks:
            lm = face.multi_face_landmarks[0].landmark

            EAR_L = compute_EAR(lm, LEFT_EYE)
            EAR_R = compute_EAR(lm, RIGHT_EYE)
            EAR = (EAR_L + EAR_R) / 2

            MAR = compute_MAR(lm, MOUTH)

            eyes_closed = EAR < 0.21
            yawn = MAR > 0.7

            perclos.update(eyes_closed, now)
            # ⭐ Nueva escala de PERCLOS
            raw_perclos = perclos.value()
            #perclos_scaled = 100 / (1 + np.exp(-20*(raw_perclos - 0.15)))
            perclos_scaled = raw_perclos * 120
            
            blinks.update(eyes_closed, now)

            score_somn = (
                perclos_scaled +
                (20 if yawn else 0) +
                (20 if blinks.value() > 0.35 else 0)
            )
            score_somn = min(score_somn, 100)

            # *** AGREGADO: mano tapa ojos o boca ***
            hand_on_face = hand_covers_face(lm, hands, CAM_WIDTH, CAM_HEIGHT)
            if hand_on_face:
                score_somn = max(score_somn, 100)

        # ----------------------------------------
        # CINTURÓN — YOLO + monitoreo grace/5s
        # ----------------------------------------
        belt_detected = (detect_belt_yolo(frame) == 0)
        score_belt = belt_monitor.update(belt_detected, now)

        # ----------------------------------------
        # CELULAR — YOLO
        # ----------------------------------------
        score_phone = detect_phone_yolo(frame)
        
        # ----------------------------------------
        # ROSTRO NO DETECTADO
        # ----------------------------------------
        score_face = face_monitor.update(face_detected, now)

        # ----------------------------------------
        # FUSIÓN
        score_global = fuse_scores(score_somn, score_belt, score_phone, weights)

        if score_global > 50: state = 2
        elif score_global > 25: state = 1
        else: state = 0

        # ----------------------------------------
        # UI
        # ----------------------------------------
        ui.reset_positions()

        ui.draw_bar_col1(canvas, score_somn, "Somnolencia")
        ui.draw_bar_col1(canvas, score_belt, "Cinturon")
        ui.draw_bar_col1(canvas, score_phone, "Celular")
        ui.draw_bar_col1(canvas, score_global, "Riesgo Global")

        ui.put_kv_col2(canvas, f"FPS: {fps:.1f}")
        ui.put_kv_col2(canvas, f"EAR: {EAR:.3f} MAR: {MAR:.3f}")
        ui.put_kv_col2(canvas, f"PERCLOS: {perclos.value():.2f}")
        ui.put_kv_col2(canvas, f"Blink rate: {blinks.value():.2f}/s")

        # Alerta visual temporal por mano en la cara
        if hand_on_face:
            cv.putText(canvas, "Retire su mano de la cara", (12, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Alerta por si no se detecta rostro
        if score_face == 100:
            cv.putText(canvas, "PELIGRO: No se detecta rostro", (12, 90),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        ui.draw_alerts(canvas, state)

        cv.imshow(ui.window_name, canvas)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
