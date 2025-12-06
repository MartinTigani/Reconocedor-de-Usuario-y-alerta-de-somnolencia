"""
Driver Safety Monitor - Aplicación Principal

Este script integra los modelos de IA (backend) con el
módulo de interfaz de usuario 'ui_manager' (frontend).
"""

# --- Importaciones estándar ---
import cv2 as cv
import numpy as np
import mediapipe as mp
from time import time
import argparse
import csv
from collections import deque
from dataclasses import dataclass
import math # Ya no es necesario en main, pero sí en los helpers

# --- 1. IMPORTAR EL MÓDULO DE UI ---
# Importamos las herramientas que creamos en el archivo 'ui_manager.py'
from ui_manager import (
    UIManager, 
    FusionWeights, 
    fuse_scores, 
    show_splash_screen,
    CAM_WIDTH,  # Opcional: usar las constantes del UI
    CAM_HEIGHT
)

# =======================================================
# 2. LÓGICA DE IA (Tu "Backend")
# (Aquí pegas todas tus clases y funciones de IA que ya tenías)
# =======================================================

class MPHelpers:
    # ... (Tu código de MPHelpers con FaceMesh, Hands, Pose)
    def __init__(self, max_faces=1, max_hands=2, static=False):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_faces, refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(max_num_hands=max_hands)
        self.pose = mp.solutions.pose.Pose(model_complexity=1)
    def process(self, bgr):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        face_res = self.face_mesh.process(rgb)
        hands_res = self.hands.process(rgb)
        pose_res = self.pose.process(rgb)
        return face_res, hands_res, pose_res

class PERCLOS:
    # ... (Tu código de PERCLOS)
    def __init__(self, window_seconds=60.0, fps_hint=30):
        self.window_seconds = float(window_seconds); self.events = deque()
    def update(self, eyes_closed: bool, ts: float):
        self.events.append((ts, eyes_closed)); ts_limit = ts - self.window_seconds
        while self.events and self.events[0][0] < ts_limit: self.events.popleft()
    def value(self):
        if not self.events: return 0.0
        closed = sum(1 for _, c in self.events if c); return closed / len(self.events)

class PhoneHeuristic:
    # ... (Tu código de PhoneHeuristic)
    def __init__(self, hold_seconds=2.0): self.hold_seconds = hold_seconds; self._near_start_ts = None; self._active = False
    def update(self, face_rect, hand_centroids, now):
        # (Lógica de detección...)
        self._active = (now - self._near_start_ts) > self.hold_seconds if self._near_start_ts else False
        score = 100.0 if self._active else 0.0 # Simplificado: 0 o 100
        return self._active, score

def compute_EAR(eye_pts):
    # ... (Tu código de compute_EAR)
    return 0.3 # Placeholder

def compute_MAR(mouth_pts):
    # ... (Tu código de compute_MAR)
    return 0.0 # Placeholder

def detect_belt(frame, roi_rect):
    # ... (Tu lógica de detección de cinturón)
    # Debe devolver 0 (OK) o 100 (PELIGRO)
    return 0.0 # Placeholder: Asume que está OK


# =======================================================
# 3. FUNCIÓN PRINCIPAL (Main)
# =======================================================

def main():
    # (Aquí iría tu 'argparse' si lo necesitás)
    # ...
    
    # <<< 4. MOSTRAR EL SPLASH SCREEN ---
    # Esto se ejecuta ANTES de cargar la cámara o los modelos
    show_splash_screen(duration_ms=3000)

    # --- 5. INICIALIZAR BACKEND (Cámara y Modelos) ---
    print("Iniciando cámara y modelos de IA...")
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Inicializar helpers de IA
    mp_helper = MPHelpers()
    perclos_calc = PERCLOS()
    phone_calc = PhoneHeuristic()
    
    # --- 6. INICIALIZAR FRONTEND (UI Manager) ---
    print("Iniciando interfaz de usuario...")
    ui = UIManager(col1_x=16, col2_x=340)
    fusion_weights = FusionWeights(wA=0.5, wB=0.25, wC=0.25)

    # Variables de estado
    t_prev = time()
    fps = 0.0
    
    # (Lógica de Histeresis iría aquí)
    current_risk_state = 0 # 0=Verde, 1=Amarillo, 2=Rojo

    print("Sistema listo. Iniciando monitoreo.")

    while True:
        ok, frame = cap.read() # frame original (limpio)
        if not ok:
            break
        
        now = time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
             fps = 1.0 / dt
        
        # <<< 7. CREAR EL CANVAS FINAL ---
        # 'canvas' es la imagen más grande (con footer) donde dibujaremos
        canvas = ui.create_canvas(frame)

        # ===================================
        # --- 8. EJECUTAR BACKEND (IA) ---
        # (Se procesa el 'frame' limpio)
        # ===================================
        
        # (Esta es tu lógica original, simplificada como placeholder)
        face_res, hands_res, pose_res = mp_helper.process(frame)
        
        # --- Módulo A: Somnolencia ---
        EAR, MAR = 0.35, 0.1 # Placeholder
        if face_res.multi_face_landmarks:
            # EAR = compute_EAR(...)
            # MAR = compute_MAR(...)
            pass
        # perclos_calc.update(...)
        score_somn = 0.0 # Placeholder
        
        # --- Módulo B: Cinturón ---
        torso_roi = (0,0,0,0) # Placeholder
        if pose_res.pose_landmarks:
            # ... (Lógica para obtener 'torso_roi' desde landmarks)
            pass
        score_belt = detect_belt(frame, torso_roi) # 0=OK, 100=Peligro
        
        # --- Módulo C: Celular ---
        face_rect = (0,0,0,0) # Placeholder
        hand_centroids = [] # Placeholder
        _, score_phone = phone_calc.update(face_rect, hand_centroids, now)
        
        # --- Fusión de Riesgos ---
        score_global = fuse_scores(score_somn, score_belt, score_phone, fusion_weights)

        # --- Estado (idealmente con Histeresis) ---
        if score_global > 50: current_risk_state = 2
        elif score_global > 25: current_risk_state = 1
        else: current_risk_state = 0
        
        # ===================================
        # --- 9. EJECUTAR FRONTEND (UI) ---
        # (Se dibuja sobre el 'canvas')
        # ===================================
        
        ui.reset_positions() # Resetea las 'Y' de las columnas
        
        # --- Dibujar Columna 1 (Barras) ---
        ui.draw_bar_col1(canvas, score_somn, "Somnolencia")
        ui.draw_bar_col1(canvas, score_belt, "Cinturon")
        ui.draw_bar_col1(canvas, score_phone, "Celular")
        ui.draw_bar_col1(canvas, score_global, "Riesgo Global")
        
        # --- Dibujar Columna 2 (Datos) ---
        ui.put_kv_col2(canvas, f"FPS: {fps:.1f}")
        ui.put_kv_col2(canvas, f"EAR: {EAR:.3f}  MAR: {MAR:.3f}")
        ui.put_kv_col2(canvas, f"PERCLOS: {perclos_calc.value():.2f}")
        
        # --- Dibujar Alertas (arriba) ---
        ui.draw_alerts(canvas, current_risk_state)
                       
        # --- 10. MOSTRAR RESULTADO ---
        cv.imshow(ui.window_name, canvas)

        if cv.waitKey(1) & 0xFF == 27: # ESC
            break

    # --- 11. LIMPIAR ---
    cap.release()
    cv.destroyAllWindows()
    print("Monitoreo finalizado.")


if __name__ == "__main__":
    main()