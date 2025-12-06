import os
import cv2
import face_recognition
import pickle
import time

# -----------------------
# Configuración y rutas
# -----------------------

# Carpeta base del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OWNER_IMAGES_DIR = os.path.join(BASE_DIR, "owner_images")      # donde están las fotos
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")          # donde se guardan embeddings
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "owner_embeddings.pkl")

# Asegurarse que existan las carpetas
os.makedirs(OWNER_IMAGES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Nombre del propietario
OWNER_NAME = "Agustin"

# Parámetros de reconocimiento
TOLERANCE = 0.5         # tolerancia de distancia para reconocimiento de rostro
TIME_THRESHOLD = 5       # segundos que debe mantenerse el rostro antes de mostrar mensaje

# -----------------------
# Funciones auxiliares
# -----------------------

def generate_embeddings():
    """
    Genera embeddings para todas las imágenes jpg en owner_images/ y los guarda
    en embeddings/owner_embeddings.pkl
    """
    embeddings = []
    image_files = []
    for root, dirs, files in os.walk(OWNER_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_files.append(os.path.join(root, file))

    if len(image_files) == 0:
        print(f"[ERROR] No se encontraron imágenes .jpg en {OWNER_IMAGES_DIR}")
        return None

    for img_file in image_files:
        img_path = os.path.join(OWNER_IMAGES_DIR, img_file)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"[WARN] No se detectó rostro en {img_file}, se saltea.")
            continue

        face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        embeddings.append(face_encoding)
        print(f"[INFO] Procesada imagen: {img_file}")

    if len(embeddings) == 0:
        print("[ERROR] No se generó ningún embedding válido.")
        return None

    # Guardar embeddings en disco
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[INFO] Embeddings guardados en {EMBEDDINGS_FILE}")

    return embeddings

def load_embeddings():
    """
    Carga embeddings desde el archivo, si existe. Sino devuelve None
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        return None

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    print(f"[INFO] Embeddings cargados desde {EMBEDDINGS_FILE}")
    return embeddings

# -----------------------
# Menú principal
# -----------------------

def main():
    while True:
        print("\n=== MENU ===")
        print("1) Enrolar propietario")
        print("2) Verificar propietario en cámara")
        print("3) Salir")

        choice = input("Elegí una opción: ").strip()
        if choice == "1":
            # ENROLAR
            if os.path.exists(EMBEDDINGS_FILE):
                overwrite = input("Ya existen embeddings. Desea sobrescribirlos? (s/n): ").lower()
                if overwrite != "s":
                    continue
            embeddings = generate_embeddings()
            if embeddings:
                print("[INFO] Enrolamiento completado.")
            else:
                print("[ERROR] No se pudo generar embeddings.")
        elif choice == "2":
            # VERIFICAR
            embeddings = load_embeddings()
            if embeddings is None:
                print("[ERROR] No hay embeddings. Primero enrolá el propietario.")
                continue

            # Abrimos webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] No se pudo abrir la cámara.")
                continue

            recognized_start = None   # tiempo de inicio del reconocimiento continuo
            last_status = None        # último estado mostrado
            font = cv2.FONT_HERSHEY_SIMPLEX

            print("[INFO] Presioná 'q' para salir del modo verificación.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] No se pudo leer frame de la cámara.")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                status = "Intruso Detectado"  # por defecto

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(embeddings, face_encoding, TOLERANCE)
                    if True in matches:
                        status = f"Usuario Autorizado. Bienvenido al Monitor de Seguridad"
                        break  # basta con uno

                # Manejo de tiempo continuo para 5 segundos
                if status == last_status:
                    if recognized_start is None:
                        recognized_start = time.time()
                else:
                    recognized_start = time.time()
                    last_status = status

                elapsed = time.time() - recognized_start

                if elapsed >= TIME_THRESHOLD:
                    # Mostramos mensaje grande en pantalla
                    cv2.putText(frame, status, (50, 50), font, 1.2, (0, 255, 0) if "Autorizado" in status else (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Verificando...", (50, 50), font, 1.0, (255, 255, 255), 2)

                # Dibujar rectángulo alrededor de caras detectadas
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

                cv2.imshow("Verificación de rostro", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif choice == "3":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida, intentá nuevamente.")

# -----------------------
# Inicio del script
# -----------------------
if __name__ == "__main__":
    main()