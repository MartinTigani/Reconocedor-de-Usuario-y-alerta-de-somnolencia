import cv2
import os
import time

# -----------------------
# Configuración
# -----------------------
OUTPUT_DIR = "owner_images"      # carpeta relativa donde se guardan las fotos
WEBCAM_INDEX = 0                 # índice de la cámara (0 por defecto)
SAVE_FULL_FRAME = False          # False -> guarda el recorte de la cara; True -> guarda el frame completo
IMAGE_PREFIX = "owner"           # prefijo para los archivos guardados
MIN_FACE_SIZE = (80, 80)         # tamaño mínimo de la cara para considerar (ancho, alto) en px

# Asegurarse que exista la carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargamos el clasificador Haar frontal (viene con OpenCV)
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    raise RuntimeError(f"No se pudo cargar el Haar cascade desde: {haar_path}")

# Abrimos la webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam. Verificá el índice o que no esté siendo usada por otra app.")

print("Captura iniciada. Presioná 'k' para guardar la cara detectada, 'q' para salir.")

saved_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame de la cámara.")
            break

        # trabajamos con una versión en escala de grises para la detección (más rápida)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectamos caras (devuelve rects: x,y,w,h)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # dibujamos rectángulos alrededor de las caras detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        # mostramos instrucciones en pantalla
        cv2.putText(frame, "Presiona 'k' para guardar la cara detectada, 'q' para salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # mostrar el frame resultante
        cv2.imshow("Capture Faces - presiona k para guardar, q para salir", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # salir
            print("Saliendo...")
            break
        elif key == ord('k'):
            # al presionar k guardamos la primera cara detectada (si existe)
            if len(faces) == 0:
                print("[!] No se detectó ninguna cara. Alineate frente a la cámara y probá de nuevo.")
                continue

            # tomamos la primer cara detectada (x,y,w,h)
            x, y, w, h = faces[0]

            # opcional: ampliar un poco el recorte para incluir contexto (margen)
            MARGIN = 0.2  # 20% de margen alrededor
            x1 = max(0, int(x - w * MARGIN))
            y1 = max(0, int(y - h * MARGIN))
            x2 = min(frame.shape[1], int(x + w + w * MARGIN))
            y2 = min(frame.shape[0], int(y + h + h * MARGIN))

            if SAVE_FULL_FRAME:
                save_img = frame.copy()
            else:
                save_img = frame[y1:y2, x1:x2].copy()

            # crear nombre de archivo con timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{IMAGE_PREFIX}_{timestamp}_{saved_count+1}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # guardamos la imagen en disco
            success = cv2.imwrite(filepath, save_img)
            if success:
                saved_count += 1
                print(f"[+] Guardada imagen #{saved_count}: {filepath}")
            else:
                print("[ERROR] No se pudo guardar la imagen. Verificá permisos y espacio en disco.")

finally:
    # limpieza: liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
    print("Proceso terminado. Se guardaron", saved_count, "imagenes.")