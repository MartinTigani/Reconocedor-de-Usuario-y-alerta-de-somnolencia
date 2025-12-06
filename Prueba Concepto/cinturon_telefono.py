import cv2
import os
from ultralytics import YOLO

# --- Obtener ruta absoluta del script actual ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CELULAR_MODEL_PATH = os.path.join(SCRIPT_DIR, "celular.pt")
CINTURON_MODEL_PATH = os.path.join(SCRIPT_DIR, "cinturon.pt")

# --- Verificar modelos ---
for path in [CELULAR_MODEL_PATH, CINTURON_MODEL_PATH]:
    if not os.path.exists(path):
        print(f"❌ No se encontró el modelo en: {path}")
        exit()

# --- Cargar modelos ---
model_celular = YOLO(CELULAR_MODEL_PATH)
model_cinturon = YOLO(CINTURON_MODEL_PATH)

# --- Abrir cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara iniciada. Presioná 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Detección con ambos modelos ---
    results_celular = model_celular(frame, verbose=False)
    results_cinturon = model_cinturon(frame, verbose=False)

    mensaje = ""

    # --- Procesar detecciones de celular ---
    for box in results_celular[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Si el modelo tiene solo una clase 'phone' (cls==0 o cls==1 según el YAML)
        if cls == 1 and conf > 0.7:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            mensaje = "📱 Teléfono detectado"

    # --- Procesar detecciones de cinturón ---
    for box in results_cinturon[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # El modelo tiene solo una clase 'phone' cls==0
        if cls == 0 and conf > 0.5:  # cls==0 → cinturón
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Seatbelt {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            mensaje = "✅ Cinturón de seguridad detectado"

    # --- Mostrar mensaje en pantalla ---
    if mensaje:
        cv2.putText(frame, mensaje, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # --- Mostrar resultado ---
    cv2.imshow("Detección Celular y Cinturón", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
