"""
UI Testbed (Banco de Pruebas de Interfaz) v2
para el Driver Safety Monitor

OBJETIVO:
 Diseñar y probar la interfaz (overlays, barras, alertas, histeresis)
 de forma aislada, sin correr los modelos de IA.

CONTROLES:
  [ESC] - Salir
  [1]   - Simular Somnolencia LEVE (Sube score A)
  [2]   - Simular Bostezo (Sube score A al máximo)
  [3]   - Simular SIN cinturón (Score B=100, RIESGO ALTO)
  [4]   - Simular CON cinturón (Score B=0, RIESGO CERO)
  [5]   - Simular Celular (Sube score C)
  [0]   - Resetear todos los scores a 0
"""

import cv2 as cv
import numpy as np
from time import time # Ya lo usábamos para FPS, ahora también para el splash
from dataclasses import dataclass

# Constantes de la ventana
CAM_WIDTH = 640
CAM_HEIGHT = 360
FOOTER_HEIGHT = 200
WINDOW_HEIGHT = CAM_HEIGHT + FOOTER_HEIGHT
WINDOW_NAME = "Sistema de Asistencia al Conductor" # Nombre unificado
SPLASH_DURATION_MS = 3000 # 3 segundos

# =======================================================
# 1. CLASES DE UI (Iguales a v4/v5)
# =======================================================

class UIManager:
    def __init__(self, footer_height, col1_x=16, col2_x=340):
        # ... (Todo este código es idéntico al de la v5) ...
        self.footer_h = footer_height; self.padding = 20
        self.x1, self.x2 = col1_x, col2_x; self.y1, self.y2 = 0, 0
        self.col_y_step = 48; self.font = cv.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5; self.font_thick = 1; self.color = (240, 240, 240)
    def reset(self, camera_height):
        start_y = camera_height + self.padding; self.y1, self.y2 = start_y, start_y
    def draw_bar_col1(self, img, value_0_100, label, size=(220, 15)):
        x, y = self.x1, self.y1; self.y1 += self.col_y_step; w, h = size
        label_text = ""; col = (60, 200, 60)
        if label == "Cinturon":
            if value_0_100 > 70: col = (0, 0, 255); label_text = "Cinturon: PELIGRO"
            elif value_0_100 > 10: col = (0, 200, 255); label_text = "Cinturon: REVISAR"
            else: col = (60, 200, 60); label_text = "Cinturon: OK"
        else:
            if value_0_100 > 50: col = (0, 0, 255)
            elif value_0_100 > 25: col = (0, 200, 255)
            label_text = f"{label}: {value_0_100:.0f}"
        cv.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
        ww = int(w * max(0.0, min(1.0, value_0_100 / 100.0)))
        cv.rectangle(img, (x + 1, y + 1), (x + ww - 1, y + h - 1), col, -1)
        cv.putText(img, label_text, (x, y - 5), self.font, self.font_scale, self.color, self.font_thick, cv.LINE_AA)
    def put_kv_col2(self, img, text):
        x, y = self.x2, self.y2; self.y2 += self.col_y_step
        cv.putText(img, text, (x, y + 8), self.font, 0.6, self.color, 1, cv.LINE_AA)

@dataclass
class FusionWeights:
    somnolencia: float = 0.5; cinturon: float = 0.25; celular: float = 0.25

def fuse_scores(score_somn, score_belt, score_phone, w: FusionWeights):
    return (w.somnolencia * score_somn + w.cinturon * score_belt + w.celular * score_phone)


# =======================================================
# 2. MAIN LOOP DEL BANCO DE PRUEBAS
# =======================================================

def main_testbed():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    # --- Helpers de UI (sin cambios) ---
    ui_manager = UIManager(footer_height=FOOTER_HEIGHT, col1_x=16, col2_x=340)
    fusion_weights = FusionWeights(0.5, 0.25, 0.25)
    
    # ======================================================
    # <<< MODIFICADO: LÓGICA DEL SPLASH SCREEN ANIMADO
    # ======================================================
    
    # 1. Crear el lienzo negro del tamaño de la ventana final
    splash_canvas = np.zeros((WINDOW_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    
    # 2. Definir los textos y propiedades
    font_splash = cv.FONT_HERSHEY_SIMPLEX
    color_splash = (200, 200, 200) # Gris claro
    
    txt1 = "Vehiculo encendido"
    txt2 = "Iniciando sistema de asistencia al conductor"
    txt3 = "Espere..."
    
    # 3. Centrar el texto (calculando la posición X para cada línea)
    def get_center_x(text, font, scale, thickness):
        (w, h), _ = cv.getTextSize(text, font, scale, thickness)
        return (CAM_WIDTH - w) // 2
        
    y_center = WINDOW_HEIGHT // 2
    y1 = y_center - 40
    y2 = y_center
    y3 = y_center + 60
    
    x1 = get_center_x(txt1, font_splash, 0.7, 2)
    x2 = get_center_x(txt2, font_splash, 0.6, 1)
    x3 = get_center_x(txt3, font_splash, 0.5, 1)

    # 4. Dibujar el texto estático en el lienzo (se usará como fondo)
    cv.putText(splash_canvas, txt1, (x1, y1), font_splash, 0.7, color_splash, 2)
    cv.putText(splash_canvas, txt2, (x2, y2), font_splash, 0.6, color_splash, 1)
    cv.putText(splash_canvas, txt3, (x3, y3), font_splash, 0.5, (100,100,100), 1)

    # 5. Definir propiedades de la barra de carga
    bar_width = 300
    bar_height = 15
    bar_x = (CAM_WIDTH - bar_width) // 2
    bar_y = y3 + 40 # Debajo del texto "Espere..."
    bar_bg_color = (40, 40, 40)
    bar_fg_color = (200, 200, 200)
    
    print("UI: Mostrando pantalla de bienvenida...")

    # 6. Loop de animación
    delay_ms = 30
    num_steps = SPLASH_DURATION_MS // delay_ms
    
    for i in range(num_steps + 1):
        # Copiar el canvas con el texto
        frame_anim = splash_canvas.copy()
        
        # a. Dibujar el fondo de la barra
        cv.rectangle(frame_anim, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), bar_bg_color, -1)
        
        # b. Calcular y dibujar el relleno de la barra
        progress = i / num_steps
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
    
    # ======================================================
    # <<< FIN: LÓGICA DEL SPLASH SCREEN
    # ======================================================

    print("Iniciando banco de pruebas de UI v6 (con Splash Animado)...")
    print("Controles: [1-5] para simular eventos, [0] para reset, [ESC] para salir.")

    # --- Variables de simulación ---
    score_somn, score_belt, score_phone = 0.0, 0.0, 0.0
    
    t_prev = time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        
        # 1. Crear el lienzo principal (canvas)
        canvas = np.zeros((WINDOW_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        
        # 2. Pegar la cámara
        canvas[0:CAM_HEIGHT, 0:CAM_WIDTH] = frame
        
        now = time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
             fps = 1.0 / dt
        
        # --- Simulación (lógica de teclas) ---
        key = cv.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('1'): score_somn = min(100.0, score_somn + 25.0)
        elif key == ord('2'): score_somn = 100.0
        elif key == ord('3'): score_belt = 100.0
        elif key == ord('4'): score_belt = 0.0
        elif key == ord('5'): score_phone = min(100.0, score_phone + 34.0)
        elif key == ord('0'): score_somn = 0.0; score_belt = 0.0; score_phone = 0.0
            
        # --- Lógica de Fusión ---
        score_global = fuse_scores(score_somn, score_belt, score_phone, fusion_weights)
        
        # --- Lógica de Estado (para alertas) ---
        current_risk_state = 0
        if score_global > 50: current_risk_state = 2
        elif score_global > 25: current_risk_state = 1

        # --- Dibujar UI en el Footer ---
        ui_manager.reset(camera_height=CAM_HEIGHT) 
        ui_manager.draw_bar_col1(canvas, score_somn, "Somnolencia")
        ui_manager.draw_bar_col1(canvas, score_belt, "Cinturon")
        ui_manager.draw_bar_col1(canvas, score_phone, "Celular")
        ui_manager.draw_bar_col1(canvas, score_global, "Riesgo Global")
        
        ui_manager.put_kv_col2(canvas, f"FPS: {fps:.1f}")
        ui_manager.put_kv_col2(canvas, f"EAR: 0.350  MAR: 0.100")
        ui_manager.put_kv_col2(canvas, f"PERCLOS: 0.00")
        
        # --- Dibujar Alertas (arriba) ---
        if current_risk_state == 2:
            cv.rectangle(canvas, (0, 0), (CAM_WIDTH, 36), (0, 0, 255), -1)
            cv.putText(canvas, "ALERTA: Detengase y descanse", (12, 24), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif current_risk_state == 1:
            cv.rectangle(canvas, (0, 0), (CAM_WIDTH, 30), (0, 200, 255), -1)
            cv.putText(canvas, "PRECAUCION: Revise su estado", (12, 22), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                       
        cv.imshow(WINDOW_NAME, canvas)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main_testbed()