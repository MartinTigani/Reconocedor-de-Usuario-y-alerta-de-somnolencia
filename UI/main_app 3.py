import subprocess
import time
import os
import cv2
import face_recognition
import pickle
import numpy as np

# =============================
# RUTAS Y CONFIGURACIÓN
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Archivo de embeddings existente
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "owner_embeddings.pkl")

# Script a ejecutar si es exitoso
DRIVER_MONITOR = os.path.join(BASE_DIR, "driver_safety_monitor.py")

# ============================================
# FUNCIONES
# ============================================

def load_embeddings():
    """Carga los embeddings del archivo .pkl existente."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return None
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)

def show_denied_screen():
    """Muestra una pantalla roja de rechazo antes de cerrar."""
    # Crear una imagen negra (o roja oscura)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (0, 0, 150)  # Fondo rojo oscuro (B, G, R)

    cv2.putText(img, "ACCESO DENEGADO", (80, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img, "Persona NO reconocida", (100, 260), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Cerrando aplicacion...", (100, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("Seguridad", img)
    cv2.waitKey(3000) # Mostrar mensaje por 3 segundos
    cv2.destroyAllWindows()

def verify_user(known_embeddings_list):
    """
    Verifica identidad. Retorna True si coincide, False si no.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return False

    print("\n[INFO] Verificando identidad...")
    
    start_time = time.time()
    recognized = False
    MAX_DURATION = 8  # Segundos máximos para intentar reconocer

    while True:
        elapsed = time.time() - start_time
        # Si pasó el tiempo límite, cortamos
        if elapsed > MAX_DURATION:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar para velocidad
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        encs = face_recognition.face_encodings(rgb)
        
        match_found = False
        if encs:
            # Comparamos la primera cara detectada con la lista
            # Usamos tolerance=0.5 (puedes bajar a 0.45 si quieres ser más estricto)
            matches = face_recognition.compare_faces(known_embeddings_list, encs[0], tolerance=0.5)
            if True in matches:
                match_found = True

        # Interfaz visual
        if match_found:
            # Mensaje verde y éxito inmediato
            cv2.rectangle(frame, (0,0), (640, 80), (0,255,0), -1)
            cv2.putText(frame, "ACCESO CONCEDIDO", (100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.imshow("Seguridad", frame)
            cv2.waitKey(1000) # Mostrar éxito 1 segundo
            recognized = True
            break
        else:
            # Mensaje de búsqueda
            cv2.putText(frame, f"Escaneando... ({int(MAX_DURATION - elapsed)}s)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Seguridad", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return recognized

# =====================================================
# MAIN
# =====================================================
def main():
    print("==========================================")
    print("   SISTEMA DE SEGURIDAD - BIOMETRICO      ")
    print("==========================================\n")

    # 1. Cargar Embeddings
    embeddings = load_embeddings()

    if embeddings is None:
        print("[ERROR] No se encontró el archivo .pkl en la carpeta embeddings.")
        print("Cerrando sistema por falta de datos.")
        return

    # 2. Verificar Usuario
    is_valid_user = verify_user(embeddings)

    # 3. Toma de decisión
    if is_valid_user:
        print("\n[OK] Identidad Confirmada. Iniciando monitor de seguridad...")
        
        if os.path.exists(DRIVER_MONITOR):
            # =========== LANZAR DRIVER MONITOR ============
            subprocess.run(["python", DRIVER_MONITOR])
            # ==============================================
        else:
            print(f"[ERROR] Falta el archivo {DRIVER_MONITOR}")
    
    else:
        # =========== ACCESO DENEGADO ============
        print("\n[ALERTA] Usuario NO reconocido. Acceso denegado.")
        show_denied_screen()
        print("[INFO] Aplicación terminada por seguridad.")
        # El programa termina aquí, no ejecuta nada más.

if __name__ == "__main__":
    main()