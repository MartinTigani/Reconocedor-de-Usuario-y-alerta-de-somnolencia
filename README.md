¡Me encanta esa combinación! Podés armar un **“Driver Safety Monitor”** con tres módulos en tiempo real y una fusión de riesgos común. Acá tenés un plan aterrizado para que lo puedas construir y evaluar sin perderte.

# Arquitectura (modular y en tiempo real)

* **Entrada:** cámara interior (webcam/USB).
* **Módulo A – Somnolencia:** Face & landmarks → EAR/MAR + pose de cabeza → PERCLOS/episodios.
* **Módulo B – Cinturón:** Detección de cinturón en ROI del torso (guiado por hombros/pecho).
* **Módulo C – Celular al volante:** Detección de “cell phone” + proximidad a mano/cabeza.
* **Fusión:** puntaje de riesgo (verde/amarillo/rojo) + alertas (visual/sonido/log).

# Módulo A — Detección de somnolencia (rápido y explicable)

* **Landmarks:** MediaPipe Face Mesh / Face Detection.
* **Métricas por frame:**

  * **EAR (Eye Aspect Ratio)** → ojos cerrados si EAR < 0.21–0.25 por ≥ n frames seguidos (ej. n=6 a 30 fps ≈ 200 ms).
  * **MAR (Mouth Aspect Ratio)** → bostezo si MAR > 0.6–0.7 por ≥ m frames (ej. 1–2 s).
  * **Head pose:** pitch (cabeceo) > umbral durante t segundos.
* **Temporales:**

  * **PERCLOS (60 s):** % de tiempo con ojos cerrados (riesgo si > 0.20).
  * Contador de parpadeos/min (picos de EAR).
* **Salida A:** `score_somnolencia ∈ [0,100]` + clase {OK, leve, moderada, alta}.

# Módulo B — Detector de uso de cinturón

* **ROI guiado por pose:** tomá hombros (landmarks) y definí un **crop torácico**.
* **Two-step eficiente:**

  1. **Detector ligero** (YOLO-tiny / EfficientDet-lite/0) entrenado con dos clases: {cinturón, sin-cinturón} dentro del ROI.
  2. (Opcional) Verificación geométrica: línea oblicua que cruza clavícula-cadera (Hough + color/contraste) para falsos positivos.
* **Dataset:** fotos propias en distintas ropas/ángulos + aumentación (rotación, brillo, recorte). 300–600 imágenes suelen alcanzar con transfer learning.
* **Salida B:** `score_cinturon` (100 si se ve cinturón de forma estable ≥ 1 s, 0 si no).

# Módulo C — Control de celular al volante

* **Detección de objetos:** el modelo ya trae **“cell phone”** (COCO).
* **Lógica espacial:**

  * Detectar **celular** y **mano** (puede ser con keypoints de MediaPipe Hands o solo “cell phone” + cercanía a cabeza/volante).
  * **Evento riesgo** si celular está a < d píxeles de la cara/mano o si permanece en ROI “cabeza/volante” ≥ 2 s.
* **Robustez:** suprimir detecciones inestables con media móvil/EMA.
* **Salida C:** `score_celular` (sube con persistencia y proximidad).

# Fusión de riesgos (simple y transparente)

* **Score global:**

  ```
  score_global = wA*score_somnolencia + wB*(100 - score_cinturon) + wC*score_celular
  ```

  con pesos iniciales: `wA=0.5`, `wB=0.25`, `wC=0.25`.
* **Estados:**

  * Verde: < 25
  * Amarillo: 25–50
  * Rojo: > 50 (activar buzzer/overlay rojo “¡Deténgase!”).
* **Histeresis:** pedí 2–3 s sostenidos para subir/bajar de nivel y evitar “flicker”.

# Pipeline y detalles de implementación

1. **Captura & preproc:** 640×360 @ 30 fps; convertir a RGB; opcional ecualización en Y (YCrCb) si hay poca luz.
2. **Landmarks/pose:** una sola inferencia de MediaPipe por frame → EAR/MAR + hombros/cabeza (para ROIs de B y C).
3. **Detecciones ligeras:**

   * Cinturón: correr detector solo en ROI del torso (menor costo).
   * Celular: detector en ROI cabeza-volante; si la CPU sufre, ejecutá cada 2–3 frames.
4. **Temporal:** buffers circulares (60 s para PERCLOS, 5–10 s para cinturón/celular).
5. **Overlay:** rectángulos, marcadores y **barra de riesgo** (verde→rojo).
6. **Logs:** CSV con timestamp y métricas (EAR/MAR/PERCLOS, estados, FPS).

# Dataset y validación

* **Somnolencia:** podés **validar** con tus propios videos (no hace falta entrenar).
* **Cinturón & celular:** hacé un mini-dataset propio (mismo auto, varios sujetos, día/noche).

  * 70/15/15 split, aumentación (±20° rotación, ±25% brillo/contraste, recortes).
* **Métricas:**

  * Módulos B/C: mAP@0.5, precisión/recall por clase, latencia (ms).
  * Módulo A: ACC/F1 para “ojos cerrados/bostezo”, AUC para episodios; PERCLOS absoluto.
  * **Global:** tasa de falsas alarmas por hora y tiempo a la detección (TtD).

# Rendimiento y deploy

* **Tiempo real en CPU:** 640×360, skip-frame (p.ej. correr detecciones B/C cada 2 frames), batch=1.
* **Aceleración opcional:** ONNX Runtime / TFLite, modelos “nano/lite”.
* **Consumo:** logueá FPS y uso de CPU para el informe.

# Entregables (sugeridos)

* `safety_monitor.py` (CLI con flags: `--camera`, `--weights_belt`, `--weights_phone`, `--roi_margins`, `--weights_fusion`…).
* Video demo (2–3 min) con casos positivos/negativos.
* Informe (8–12 págs): diseño, datasets, umbrales, resultados, discusión de errores (gafas, contraluz, ropa oscura, reflejos).
* Carpeta `/experiments` con CSVs y gráficos (PERCLOS, score global vs. tiempo).

# Riesgos y mitigaciones

* **Poca luz / reflejos:** LED tenue, ISO automático, ecualización de histograma en Y.
* **Gafas de sol / barbijo:** fiarte más de **pose de cabeza** y parpadeo relativo que de MAR.
* **Ropa negra / cinturón oscuro:** recalibrar detector con ejemplos “difíciles”; usar ROI preciso guiado por hombros; (opcional) componente de textura/diagonalidad.
* **Falsas alarmas de celular (GPS en soporte):** exigir **movimiento de mano** o proximidad a la cabeza para subir score.

---

Si querés, te preparo un **starter kit** en Python con:

* EAR/MAR + PERCLOS ya calculados,
* ROIs de torso/cabeza a partir de landmarks,
* hooks para que enchufes tus dos detectores (cinturón/celular),
* overlay con barra de riesgo y logging a CSV.