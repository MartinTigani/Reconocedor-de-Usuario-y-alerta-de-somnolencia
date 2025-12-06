# --- Este es el contenido de ui_manager.py ---

import cv2 as cv
import numpy as np
from dataclasses import dataclass
from time import time # <<< Necesitamos 'time' para un delay más preciso

# Constantes de layout
CAM_WIDTH = 640
CAM_HEIGHT = 360
FOOTER_HEIGHT = 200
WINDOW_HEIGHT = CAM_HEIGHT + FOOTER_HEIGHT
WINDOW_NAME = "Sistema de Asistencia al Conductor"

# =======================================================
# FUNCIÓN 1: SPLASH SCREEN (Modificada con barra de carga)
# =======================================================

def show_splash_screen(duration_ms=3000):
    """
    Muestra una pantalla de bienvenida con una barra de carga animada.
    """
    
    # 1. Crear el lienzo negro
    splash_canvas = np.zeros((WINDOW_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    
    # 2. Definir los textos (igual que antes)
    font_splash = cv.FONT_HERSHEY_SIMPLEX
    color_splash = (200, 200, 200)
    
    txt1 = "Vehiculo encendido"
    txt2 = "Iniciando sistema de asistencia al conductor"
    txt3 = "Espere..."
    
    def _get_center_x(text, font, scale, thickness):
        (w, h), _ = cv.getTextSize(text, font, scale, thickness)
        return (CAM_WIDTH - w) // 2
        
    y_center = WINDOW_HEIGHT // 2
    y1 = y_center - 40
    y2 = y_center
    y3 = y_center + 60
    
    x1 = _get_center_x(txt1, font_splash, 0.7, 2)
    x2 = _get_center_x(txt2, font_splash, 0.6, 1)
    x3 = _get_center_x(txt3, font_splash, 0.5, 1)

    # 3. Dibujar el texto estático en el lienzo
    cv.putText(splash_canvas, txt1, (x1, y1), font_splash, 0.7, color_splash, 2)
    cv.putText(splash_canvas, txt2, (x2, y2), font_splash, 0.6, color_splash, 1)
    cv.putText(splash_canvas, txt3, (x3, y3), font_splash, 0.5, (100,100,100), 1)

    # ======================================================
    # <<< INICIO: LÓGICA DE BARRA DE CARGA ANIMADA
    # ======================================================
    
    print("UI: Mostrando pantalla de bienvenida...")

    # 4. Definir propiedades de la barra de carga
    bar_width = 300
    bar_height = 15
    bar_x = (CAM_WIDTH - bar_width) // 2
    bar_y = y3 + 40 # Debajo del texto "Espere..."
    
    bar_bg_color = (40, 40, 40) # Fondo gris oscuro
    bar_fg_color = (200, 200, 200) # Relleno gris claro
    
    # 5. Loop de animación
    # Calculamos cuántos "pasos" tendrá la animación.
    # Queremos que cada frame dure aprox 30ms (unos 33 FPS)
    delay_ms = 30
    num_steps = duration_ms // delay_ms # (ej: 3000 / 30 = 100 pasos)
    
    for i in range(num_steps + 1):
        # Hacemos una copia del canvas (que ya tiene el texto)
        # para no tener que redibujar el texto 100 veces
        frame_anim = splash_canvas.copy()
        
        # a. Dibujar el fondo de la barra
        cv.rectangle(frame_anim, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), bar_bg_color, -1)
        
        # b. Calcular y dibujar el relleno de la barra
        progress = i / num_steps # Fracción de 0.0 a 1.0
        current_width = int(progress * bar_width)
        if current_width > 0:
            cv.rectangle(frame_anim, (bar_x, bar_y), (bar_x + current_width, bar_y + bar_height), bar_fg_color, -1)
        
        # c. Dibujar el porcentaje
        perc_text = f"{int(progress * 100)}%"
        (tw, th), _ = cv.getTextSize(perc_text, font_splash, 0.5, 1)
        tx = (CAM_WIDTH - tw) // 2
        ty = bar_y + bar_height + th + 15
        cv.putText(frame_anim, perc_text, (tx, ty), font_splash, 0.5, bar_fg_color, 1)

        # d. Mostrar el frame y esperar
        cv.imshow(WINDOW_NAME, frame_anim)
        cv.waitKey(delay_ms)

    # <<< FIN: LÓGICA DE BARRA DE CARGA ANIMADA
    # ======================================================


# =======================================================
# CLASE 1: UI MANAGER (Sin cambios)
# =======================================================
class UIManager:
    def __init__(self, col1_x=16, col2_x=340):
        # ... (exactamente igual que antes)
        self.footer_h = FOOTER_HEIGHT
        self.padding = 20
        self.x1, self.x2 = col1_x, col2_x
        self.y1, self.y2 = 0, 0
        self.col_y_step = 48
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thick = 1
        self.color = (240, 240, 240)
        self.canvas = np.zeros((WINDOW_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        self.window_name = WINDOW_NAME

    def reset_positions(self):
        # ... (exactamente igual que antes)
        start_y = CAM_HEIGHT + self.padding
        self.y1, self.y2 = start_y, start_y
        
    def create_canvas(self, frame):
        # ... (exactamente igual que antes)
        self.canvas[0:CAM_HEIGHT, 0:CAM_WIDTH] = frame
        self.canvas[CAM_HEIGHT:WINDOW_HEIGHT, 0:CAM_WIDTH] = (0, 0, 0)
        return self.canvas

    def draw_bar_col1(self, img, value_0_100, label, size=(220, 15)):
        # ... (exactamente igual que antes)
        x, y = self.x1, self.y1; self.y1 += self.col_y_step; w, h = size
        label_text = ""; col = (60, 200, 60)
        if label == "Cinturon":
            if value_0_100 > 70: col = (0, 0, 255); label_text = "Cinturon: PELIGRO"
            elif value_0_100 > 10: col = (0, 200, 255); label_text = "Cinturon: REVISAR"
            else: col = (60, 200, 60); label_text = "Cinturon: OK"
        else:
            # --- Caso especial para el detector de celular ---
            if label == "Celular":
                if value_0_100 >= 50:
                    col = (0, 0, 255)
                    label_text = "Celular: DETECTADO"
                else:
                    col = (60, 200, 60)
                    label_text = "Celular: No Detectado"
            else:
                # --- Comportamiento normal para otras barras ---
                if value_0_100 > 50: col = (0, 0, 255)
                elif value_0_100 > 25: col = (0, 200, 255)
                label_text = f"{label}: {value_0_100:.0f}"
        cv.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
        ww = int(w * max(0.0, min(1.0, value_0_100 / 100.0)))
        cv.rectangle(img, (x + 1, y + 1), (x + ww - 1, y + h - 1), col, -1)
        cv.putText(img, label_text, (x, y - 5), self.font, self.font_scale, self.color, self.font_thick, cv.LINE_AA)

    def put_kv_col2(self, img, text):
        # ... (exactamente igual que antes)
        x, y = self.x2, self.y2; self.y2 += self.col_y_step
        cv.putText(img, text, (x, y + 8), self.font, 0.6, self.color, 1, cv.LINE_AA)

    def draw_alerts(self, img, risk_state):
        # ... (exactamente igual que antes)
        if risk_state == 2:
            cv.rectangle(img, (0, 0), (CAM_WIDTH, 36), (0, 0, 255), -1)
            cv.putText(img, "ALERTA: Detengase y descanse", (12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif risk_state == 1:
            cv.rectangle(img, (0, 0), (CAM_WIDTH, 30), (0, 200, 255), -1)
            cv.putText(img, "PRECAUCION: Revise su estado", (12, 22), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


# =======================================================
# CLASE 2 y FUNCIÓN 2: Fusión (Sin cambios)
# =======================================================
@dataclass
class FusionWeights:
    # ... (exactamente igual que antes)
    somnolencia: float = 0.5
    cinturon: float = 0.25
    celular: float = 0.25

def fuse_scores(score_somn, score_belt, score_phone, w: FusionWeights):
    # ... (exactamente igual que antes)
    return (
        w.somnolencia * score_somn
        + w.cinturon * score_belt 
        + w.celular * score_phone
    )