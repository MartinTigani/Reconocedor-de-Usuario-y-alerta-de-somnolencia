import os
import cv2
import pickle
import time
import numpy as np
import face_recognition

# =========================
# CONFIGURACIONES GLOBALES
# =========================

# Ruta base (la del script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta de imágenes y embeddings (rutas relativas)
OWNER_IMAGES_DIR = os.path.join(BASE_DIR, "owner_images")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "owner_embeddings.pkl")

# Parámetros
THRESHOLD_COSINE = 0.42       # Umbral de similitud (más bajo = más estricto)
SECONDS_CONFIRM = 5           # Tiempo continuo necesario
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================
# FUNCIONES AUXILIARES
# =========================

def generate_embeddings_by_user():
    """
    Recorre owner_images/<usuario>/*.jpg y genera embeddings por usuario.
    Guarda el resultado en embeddings/owner_embeddings.pkl
    """
    data = {}

    if not os.path.exists(OWNER_IMAGES_DIR):
        print(f"[ERROR] No existe la carpeta {OWNER_IMAGES_DIR}")
        return None

    # Recorremos subcarpetas (cada una representa un usuario)
    for user_name in os.listdir(OWNER_IMAGES_DIR):
        user_folder = os.path.join(OWNER_IMAGES_DIR, user_name)
        if not os.path.isdir(user_folder):
            continue

        print(f"[INFO] Procesando imágenes de {user_name}...")
        embeddings_user = []

        for fname in os.listdir(user_folder):
            if not fname.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(user_folder, fname)
            try:
                img = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(img)
                if len(boxes) == 0:
                    print(f"[WARN] No se detectó rostro en {fname}, omitida.")
                    continue
                enc = face_recognition.face_encodings(img, known_face_locations=[boxes[0]])
                if enc:
                    embeddings_user.append(enc[0])
                    print(f"[OK] {fname} procesada.")
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

        if embeddings_user:
            data[user_name] = embeddings_user
            print(f"[INFO] Usuario '{user_name}' -> {len(embeddings_user)} embeddings.")
        else:
            print(f"[WARN] No se generaron embeddings para {user_name}.")

    # Guardamos embeddings si hay al menos un usuario
    if not data:
        print("[ERROR] No se generaron embeddings para ningún usuario.")
        return None

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Embeddings guardados en {EMBEDDINGS_FILE}")
    return data


def load_embeddings_by_user():
    """Carga embeddings desde disco si existen"""
    if not os.path.exists(EMBEDDINGS_FILE):
        print("[WARN] No existen embeddings. Debes enrolar usuarios primero.")
        return None
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Embeddings cargados. Usuarios detectados: {list(data.keys())}")
    return data


def recognize_face_from_embeddings(embeddings_dict, probe_encoding, threshold=THRESHOLD_COSINE):
    """
    Compara el rostro detectado (probe_encoding) con los embeddings de todos los usuarios.
    Retorna (user_name, dist) si hay coincidencia, o (None, None) si no hay match.
    """
    best_user = None
    best_dist = 1.0

    for user, enc_list in embeddings_dict.items():
        for enc in enc_list:
            # Distancia coseno
            a = probe_encoding / np.linalg.norm(probe_encoding)
            b = enc / np.linalg.norm(enc)
            dist = 1.0 - np.dot(a, b)
            if dist < best_dist:
                best_dist = dist
                best_user = user

    if best_user is None or best_dist > threshold:
        return (None, None)
    return (best_user, best_dist)


# =========================
# LOOP DE VERIFICACIÓN
# =========================

def verify_user():
    """Verifica usuarios en tiempo real con cámara."""
    embeddings_dict = load_embeddings_by_user()
    if embeddings_dict is None:
        print("[ERROR] No se encontraron embeddings.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara.")
        return

    authorized_user = None
    last_change_time = time.time()
    current_state = None

    print("[INFO] Iniciando verificación facial...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducir tamaño y convertir a RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detección de rostro
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_user = None

        for face_encoding in face_encodings:
            user, dist = recognize_face_from_embeddings(embeddings_dict, face_encoding)
            if user:
                detected_user = user
                break

        new_state = "authorized" if detected_user else "unauthorized"

        # Si el estado cambia, reiniciamos el cronómetro
        if new_state != current_state:
            current_state = new_state
            last_change_time = time.time()

        elapsed = time.time() - last_change_time

        # Mostrar mensaje solo si mantiene el estado por más de 5 segundos
        if elapsed >= SECONDS_CONFIRM:
            if current_state == "authorized":
                message = f"Usuario Autorizado ({detected_user}). Bienvenido al Monitor de Seguridad"
                color = (0, 255, 0)
            else:
                message = "Intruso Detectado"
                color = (0, 0, 255)
        else:
            if current_state == "authorized":
                message = f"Reconociendo a {detected_user}..."
                color = (0, 255, 255)
            else:
                message = "Verificando identidad..."
                color = (255, 255, 0)

        # Mostrar mensaje grande en pantalla
        cv2.putText(frame, message, (30, 60), FONT, 0.8, color, 2, cv2.LINE_AA)
        cv2.imshow("Verificación Facial", frame)

        # Salir con tecla q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# MENÚ PRINCIPAL
# =========================

def main():
    print("\n===== SISTEMA DE RECONOCIMIENTO FACIAL =====")
    print("1. Enrolar usuarios (generar embeddings)")
    print("2. Verificar identidad en vivo")
    print("3. Salir")

    opcion = input("Selecciona una opción: ")

    if opcion == "1":
        generate_embeddings_by_user()
    elif opcion == "2":
        verify_user()
    else:
        print("Saliendo...")


if __name__ == "__main__":
    main()
