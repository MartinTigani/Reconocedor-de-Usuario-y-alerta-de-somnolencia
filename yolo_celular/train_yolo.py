import os
from ultralytics import YOLO

# --- Rutas absolutas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, 'data.yaml')

# --- Cargar modelo base (preentrenado y liviano) ---
model = YOLO('yolov8s.pt')

# --- Entrenamiento ---
model.train(
    data=DATA_YAML,
    epochs=60,
    imgsz=640,
    batch=2,
    workers=0,
    device='cpu',
    name='phone_detector',
    project='runs/train',
    freeze=10,
    patience=30,
    augment=True
)

# --- Evaluación final ---
metrics = model.val(data=DATA_YAML)

print("\n✅ Entrenamiento completado (modo CPU).")
print("Resultados guardados en: 'runs/train/phone_detector'")
print(f"Precisión promedio (mAP50): {metrics.box.map50:.3f}")
