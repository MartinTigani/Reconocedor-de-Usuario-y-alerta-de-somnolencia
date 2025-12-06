import cv2
import os
import time
from ultralytics import YOLO

# --- Obtener ruta absoluta del script actual ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

# --- Cargar modelo entrenado ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)

# --- Abrir cámara ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara iniciada. Presioná 'q' para salir.")

# --- Variables para control de detección persistente ---
phone_detected_since = None
DETECTION_THRESHOLD = 0.7
REQUIRED_SECONDS = 3  # segundos que debe mantenerse la detección

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Detección con YOLO ---
    results = model(frame, verbose=False)
    detections = results[0].boxes

    phone_detected = False
    phone_box = None
    phone_conf = 0

    # --- Buscar detección de 'phone' ---
    for box in detections:
        cls = int(box.cls[0])        # clase detectada
        conf = float(box.conf[0])    # confianza
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 1 and conf > DETECTION_THRESHOLD:
            phone_detected = True
            phone_box = (x1, y1, x2, y2)
            phone_conf = conf
            break  # solo necesitamos una detección confiable

    # --- Control de tiempo ---
    current_time = time.time()

    if phone_detected:
        if phone_detected_since is None:
            phone_detected_since = current_time  # primera detección
        elif current_time - phone_detected_since >= REQUIRED_SECONDS:
            # Detección mantenida el tiempo suficiente → mostrar
            if phone_box:
                x1, y1, x2, y2 = phone_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Phone {phone_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    else:
        # Si no hay detección, reiniciar el contador
        phone_detected_since = None

    # --- Mostrar resultado ---
    cv2.imshow("Detección de Celular", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
