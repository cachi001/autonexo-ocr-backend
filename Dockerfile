FROM python:3.12-slim

# libgl1/libglib2.0-0: opencv-python-headless igual necesita estas libs del
# sistema en tiempo de ejecución (no solo en build) para decodificar imágenes.
# xvfb: display virtual para correr Chromium en modo headful real (ver
# dnrpa_informe.py) -el modo headless, aunque se le parchee el user-agent y
# el navigator.webdriver, sigue renderizando distinto por dentro (GPU/canvas)
# y reCAPTCHA Enterprise lo puede detectar igual; probado en vivo sin mejora.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Chromium para el flujo de validación DNRPA (dnrpa_informe.py) -no se
# levanta en el arranque del proceso, solo cuando una agencia pide validar
# un dominio, así no queda residente en memoria compitiendo con el OCR
# todo el tiempo-. --with-deps instala las librerías del sistema que
# Chromium necesita (nss, atk, cups, fuentes, etc.).
RUN playwright install --with-deps chromium

COPY backend/ backend/
COPY frontend/ frontend/

# Los modelos de OCR se descargan la primera vez que se usan, no en el
# build -> el primer request después de un deploy o reinicio tarda más
# (~20-30s) mientras los baja.
ENV PYTHONUNBUFFERED=1
# Render free tiene 0.1 CPU: procesar una imagen de 1600px (el default local,
# con CPU de sobra) tarda demasiado acá. 1000px sigue siendo legible para el
# texto de la cédula (ver MIN_OCR_SHORT_SIDE en vision.py) pero reduce bastante
# el tiempo de detección + OCR en una CPU débil.
ENV CEDULA_MAX_DIM=1000
# El cuello de botella real medido en producción no es la detección sino el
# OCR en sí (ver vision.py): bajar esto de 700 a 550 solo en Render reduce
# los píxeles que entran al reconocimiento de texto, validado localmente sin
# pérdida de calidad en los campos clave (marca, modelo, chasis, titular).
ENV MIN_OCR_SHORT_SIDE=550

EXPOSE 8000
WORKDIR /app/backend
# xvfb-run levanta un display virtual (:99 por defecto) y lo cierra solo al
# terminar el proceso -Chromium corre headful de verdad contra ese display.
CMD ["sh", "-c", "xvfb-run -a uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
