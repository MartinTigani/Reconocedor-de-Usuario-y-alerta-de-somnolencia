"""
Driver Safety Monitor – starter kit (real‑time)

Módulos incluidos:
 A) Somnolencia: EAR, MAR, PERCLOS con MediaPipe FaceMesh
 B) Cinturón de seguridad: heurística en ROI del torso (fallback)
 C) Celular al volante: mano cerca de la cara con MediaPipe Hands (fallback)

Hooks para detección con modelos:
 - Podés enchufar detectores ONNX/YOLO para {belt, phone}. Si no se pasan pesos,
   se usan heurísticas de bajo costo para que el script corra en cualquier PC.

Requisitos (pip):
  pip install opencv-python mediapipe numpy
Opcional: onnxruntime, ultralytics, opencv-contrib-python (si usás detectores extra)

Uso:
  python safety_monitor.py --camera 0 --log_csv logs.csv --show 1

Controles por teclado:
  [ESC] salir, [b] toggle overlay de cajas/ROIs, [p] pausar/reanudar
"""

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from time import time

import cv2 as cv
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # mensaje más claro si falta mediapipe
    raise SystemExit("Este script requiere 'mediapipe'. Instalá con: pip install mediapipe")

# =============================
# Utilidades geométricas
# =============================

def _euclid(a, b):
    a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def _bbox_from_landmarks(pts, pad=0):
    pts = np.array(pts, dtype=np.float32)
    x, y, w, h = cv.boundingRect(pts)
    return (x - pad, y - pad, w + 2 * pad, h + 2 * pad)


# =============================
# MediaPipe helpers (FaceMesh + Hands)
# =============================
class MPHelpers:
    def __init__(self, max_faces=1, max_hands=2, static=False):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, bgr):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        face_res = self.face_mesh.process(rgb)
        hands_res = self.hands.process(rgb)
        return face_res, hands_res


# =============================
# EAR/MAR sobre FaceMesh
# Índices de FaceMesh (468 + refines). Con refine_landmarks=True tenemos labios y ojos precisos.
# Referencia (algunos índices habituales):
#   Ojo izquierdo: 33, 160, 158, 133, 153, 144
#   Ojo derecho:   362, 385, 387, 263, 373, 380
#   Labios (MAR):  61, 291 (extremos); 13 (labio sup), 14 (labio inf)
# =============================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 291, 13, 14]  # left, right, top, bottom


def compute_EAR(eye_pts):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = eye_pts
    num = _euclid(p2, p6) + _euclid(p3, p5)
    den = 2.0 * _euclid(p1, p4)
    if den <= 1e-6:
        return 0.0
    return num / den


def compute_MAR(mouth_pts):
    # MAR simple con 4 puntos: (|top-bottom|)/(|left-right|)
    left, right, top, bottom = mouth_pts
    num = _euclid(top, bottom)
    den = _euclid(left, right)
    if den <= 1e-6:
        return 0.0
    return num / den


# =============================
# Heurísticas de cinturón y celular (fallbacks)
# =============================
@dataclass
class BeltHeuristicConfig:
    min_diag_len_frac: float = 0.35  # fracción mínima de la diagonal ROI para considerar "cinturón"
    hough_threshold: int = 30
    min_line_length_frac: float = 0.25
    max_line_gap: int = 10


def detect_belt_heuristic(bgr, roi_rect, cfg=BeltHeuristicConfig()):
    """Detecta una línea oblicua pronunciada dentro del ROI del torso como proxy del cinturón.
    Devuelve (bool_presencia, score_0a100).
    """
    x, y, w, h = roi_rect
    x = max(0, x); y = max(0, y)
    w = max(1, w); h = max(1, h)
    H, W = bgr.shape[:2]
    if x + w > W or y + h > H:
        w = min(w, W - x)
        h = min(h, H - y)
    roi = bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return False, 0.0
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    edges = cv.Canny(gray, 60, 150)
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=int(cfg.min_line_length_frac * math.hypot(w, h)),
        maxLineGap=cfg.max_line_gap,
    )
    if lines is None:
        return False, 0.0
    # Buscar líneas con ángulo entre 25° y 70° (oblicuas típicas del cinturón diagonal)
    good = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if 25.0 <= ang <= 70.0 and seg_len >= cfg.min_diag_len_frac * math.hypot(w, h):
            good += 1
    score = min(100.0, good * 25.0)
    return good > 0, score


@dataclass
class PhoneHeuristicConfig:
    hand_near_face_thresh_frac: float = 0.45  # mano dentro de 45% del ancho de la cara
    hold_seconds: float = 2.0  # persistencia para considerarlo uso indebido


class PhoneHeuristic:
    def __init__(self, cfg=PhoneHeuristicConfig()):
        self.cfg = cfg
        self._near_start_ts = None
        self._active = False

    def update(self, face_rect, hand_centroids, now):
        x, y, w, h = face_rect
        face_cx, face_cy = x + w / 2.0, y + h / 2.0
        risk_now = False
        for (hx, hy) in hand_centroids:
            d = math.hypot(hx - face_cx, hy - face_cy)
            if d <= self.cfg.hand_near_face_thresh_frac * w:
                risk_now = True
                break
        if risk_now:
            if self._near_start_ts is None:
                self._near_start_ts = now
            elif (now - self._near_start_ts) >= self.cfg.hold_seconds:
                self._active = True
        else:
            self._near_start_ts = None
            self._active = False
        score = 100.0 if self._active else (50.0 if risk_now else 0.0)
        return self._active, score


# =============================
# Fusión de riesgos
# =============================
@dataclass
class FusionWeights:
    somnolencia: float = 0.5
    cinturon: float = 0.25  # penaliza ausencia: usa (100 - score_cinturon)
    celular: float = 0.25


def fuse_scores(score_somn, score_belt, score_phone, w: FusionWeights):
    return (
            w.somnolencia * score_somn
            + w.cinturon * (100.0 - score_belt)
            + w.celular * score_phone
    )


# =============================
# PERCLOS con ventana de tiempo
# =============================
class PERCLOS:
    def __init__(self, window_seconds=60.0, fps_hint=30):
        self.window_seconds = float(window_seconds)
        self.events = deque()  # (ts, eyes_closed_bool)
        # estimación para prealloc
        self.maxlen = max(30, int(self.window_seconds * fps_hint))

    def update(self, eyes_closed: bool, ts: float):
        self.events.append((ts, eyes_closed))
        while self.events and (ts - self.events[0][0]) > self.window_seconds:
            self.events.popleft()

    def value(self):
        if not self.events:
            return 0.0
        closed = sum(1 for _, c in self.events if c)
        return closed / len(self.events)


# =============================
# Overlay y utilidades visuales
# =============================
class Overlay:
    @staticmethod
    def draw_bar(img, value_0_100, label, pos=(20, 20), size=(220, 16), color=(0, 255, 0)):
        x, y = pos; w, h = size
        cv.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
        ww = int(w * max(0.0, min(1.0, value_0_100 / 100.0)))
        # color según riesgo
        if value_0_100 < 25:
            col = (60, 200, 60)
        elif value_0_100 < 50:
            col = (0, 200, 255)
        else:
            col = (0, 0, 255)
        cv.rectangle(img, (x + 1, y + 1), (x + ww - 1, y + h - 1), col, -1)
        cv.putText(img, f"{label}: {value_0_100:.0f}", (x, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)

    @staticmethod
    def put_kv(img, x, y, text):
        cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv.LINE_AA)


# =============================
# Main
# =============================

def main():
    ap = argparse.ArgumentParser(description="Driver Safety Monitor – somnolencia + cinturón + celular")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--show", type=int, default=1, help="1=mostrar ventanas, 0=headless")
    ap.add_argument("--ear_thresh", type=float, default=0.23)
    ap.add_argument("--mar_thresh", type=float, default=0.65)
    ap.add_argument("--eyes_frames_closed", type=int, default=6, help="frames consecutivos para ojo cerrado")
    ap.add_argument("--yawn_frames", type=int, default=30, help="frames consecutivos para bostezo")
    ap.add_argument("--perclos_window", type=float, default=60.0)
    ap.add_argument("--skip_rate", type=int, default=1, help="procesar 1 de cada N frames (detectors)")
    ap.add_argument("--log_csv", type=str, default="")
    ap.add_argument("--weights_belt", type=str, default="", help="ruta a detector ONNX/YOLO para cinturón (opcional)")
    ap.add_argument("--weights_phone", type=str, default="", help="ruta a detector ONNX/YOLO para celular (opcional)")
    ap.add_argument("--wA", type=float, default=0.5)
    ap.add_argument("--wB", type=float, default=0.25)
    ap.add_argument("--wC", type=float, default=0.25)
    args = ap.parse_args()

    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    mp_helper = MPHelpers(max_faces=1, max_hands=2)
    perclos = PERCLOS(window_seconds=args.perclos_window, fps_hint=30)
    phone_heur = PhoneHeuristic()
    fusion = FusionWeights(args.wA, args.wB, args.wC)

    # Estados temporales
    eyes_closed_count = 0
    yawn_count = 0

    # Logging CSV
    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "ts",
            "fps",
            "EAR",
            "MAR",
            "perclos",
            "score_somn",
            "score_belt",
            "score_phone",
            "score_global",
        ])

    show_boxes = True
    paused = False

    t_prev = time()
    frame_idx = 0
    fps = 0.0

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                H, W = frame.shape[:2]

                now = time()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 1.0 / dt

                face_res, hands_res = mp_helper.process(frame)

                # Defaults (por si no hay cara)
                EAR = 0.0
                MAR = 0.0
                eyes_closed = False
                yawn = False
                score_somn = 0.0
                score_belt = 0.0
                score_phone = 0.0
                face_rect = (0, 0, 0, 0)

                # ========= Somnolencia (FaceMesh)
                if face_res.multi_face_landmarks:
                    lms = face_res.multi_face_landmarks[0].landmark
                    pts = np.array([(int(l.x * W), int(l.y * H)) for l in lms], dtype=np.int32)

                    # Ojos
                    leye = pts[LEFT_EYE]
                    reye = pts[RIGHT_EYE]
                    EAR_left = compute_EAR(leye)
                    EAR_right = compute_EAR(reye)
                    EAR = (EAR_left + EAR_right) / 2.0

                    # Boca
                    mouth = pts[MOUTH_POINTS]
                    MAR = compute_MAR(mouth)

                    # bbox de cara para overlays y ROIs
                    face_rect = _bbox_from_landmarks(pts[[33, 263, 1, 199]], pad=10)  # ojo izq, ojo der, nariz, barbilla aprox

                    # Eventos
                    if EAR < args.ear_thresh:
                        eyes_closed_count += 1
                    else:
                        eyes_closed_count = 0
                    if MAR > args.mar_thresh:
                        yawn_count += 1
                    else:
                        yawn_count = 0

                    eyes_closed = eyes_closed_count >= args.eyes_frames_closed
                    yawn = yawn_count >= args.yawn_frames

                    # PERCLOS
                    perclos.update(eyes_closed, now)
                    perclos_val = perclos.value()

                    # Scoring simple de somnolencia
                    score_somn = min(100.0, 100.0 * max(0.0, (args.ear_thresh - EAR)) / args.ear_thresh)
                    score_somn = max(score_somn, 100.0 * perclos_val)  # PERCLOS domina si alto
                    if yawn:
                        score_somn = max(score_somn, 60.0)

                    if args.show and show_boxes:
                        x, y, w, h = face_rect
                        cv.rectangle(frame, (x, y), (x + w, y + h), (50, 200, 50), 1)
                        for p in leye:
                            cv.circle(frame, tuple(p), 1, (0, 255, 255), -1)
                        for p in reye:
                            cv.circle(frame, tuple(p), 1, (0, 255, 255), -1)
                        for p in mouth:
                            cv.circle(frame, tuple(p), 1, (255, 200, 0), -1)

                # ========= Cinturón (heurística torso estimado desde cara)
                torso_roi = None
                if face_rect[2] > 0 and face_rect[3] > 0:
                    fx, fy, fw, fh = face_rect
                    # Torso: debajo de la cara, más ancho y más alto
                    rx = int(max(0, fx - 0.5 * fw))
                    ry = int(min(H - 1, fy + fh))
                    rw = int(min(W - rx, fw * 2.0))
                    rh = int(min(H - ry, fh * 1.5))
                    torso_roi = (rx, ry, rw, rh)
                    belt_ok, belt_score = detect_belt_heuristic(frame, torso_roi)
                    score_belt = belt_score if belt_ok else 0.0
                    if args.show and show_boxes and torso_roi is not None:
                        color = (0, 200, 0) if score_belt >= 30 else (0, 0, 255)
                        cv.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 1)

                # ========= Celular (mano cerca de la cara)
                hand_centroids = []
                if hands_res.multi_hand_landmarks:
                    for handLms in hands_res.multi_hand_landmarks:
                        pts_h = [(int(l.x * W), int(l.y * H)) for l in handLms.landmark]
                        cx = int(np.mean([p[0] for p in pts_h]))
                        cy = int(np.mean([p[1] for p in pts_h]))
                        hand_centroids.append((cx, cy))
                        if args.show and show_boxes:
                            cv.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                if face_rect[2] > 0:
                    phone_active, score_phone = phone_heur.update(face_rect, hand_centroids, now)
                else:
                    phone_active, score_phone = (False, 0.0)

                # ========= Fusión
                score_global = fuse_scores(score_somn, score_belt, score_phone, fusion)

                # ========= Overlay
                if args.show:
                    Overlay.draw_bar(frame, score_somn, "Somnolencia", (16, 24))
                    Overlay.draw_bar(frame, score_belt, "Cinturón (↑ mejor)", (16, 50))
                    Overlay.draw_bar(frame, score_phone, "Celular", (16, 76))
                    Overlay.draw_bar(frame, score_global, "Riesgo global", (16, 110))
                    Overlay.put_kv(frame, 16, 140, f"FPS: {fps:.1f}")
                    Overlay.put_kv(frame, 16, 160, f"EAR: {EAR:.3f}  MAR: {MAR:.3f}")
                    Overlay.put_kv(frame, 16, 180, f"PERCLOS: {perclos.value():.2f}")

                    # Banner de alerta
                    if score_global > 50:
                        cv.rectangle(frame, (0, 0), (W, 36), (0, 0, 255), -1)
                        cv.putText(frame, "ALERTA: Detengase y descanse", (12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # ========= Logging
                if csv_writer is not None:
                    csv_writer.writerow([
                        f"{now:.3f}",
                        f"{fps:.2f}",
                        f"{EAR:.4f}",
                        f"{MAR:.4f}",
                        f"{perclos.value():.3f}",
                        f"{score_somn:.1f}",
                        f"{score_belt:.1f}",
                        f"{score_phone:.1f}",
                        f"{score_global:.1f}",
                    ])

                frame_idx += 1

                if args.show:
                    cv.imshow("Driver Safety Monitor", frame)

            # Teclado / UI
            key = cv.waitKey(1) & 0xFF if args.show else 255
            if key == 27:  # ESC
                break
            elif key in (ord('b'), ord('B')):
                show_boxes = not show_boxes
            elif key in (ord('p'), ord('P')):
                paused = not paused

    finally:
        cap.release()
        cv.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
