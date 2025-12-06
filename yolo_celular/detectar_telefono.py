import cv2
import os
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
cap = cv2.VideoCapture(0)  # 0 = cámara predeterminada

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara iniciada. Presioná 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Detección con YOLO ---
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # --- Dibujar solo si se detecta 'phone' ---
    for box in detections:
        cls = int(box.cls[0])        # clase detectada
        conf = float(box.conf[0])    # confianza
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordenadas del bounding box

        # Si el modelo tiene solo una clase ('phone'), su id es 0.
        # Si tu YAML tenía 'head:0' y 'phone:1', cambiá a cls == 1.
        if cls == 1 and conf > 0.8:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Phone {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # --- Mostrar resultado ---
    cv2.imshow("Detección de Celular", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
