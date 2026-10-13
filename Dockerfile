FROM python:3.12-slim

# libgl1/libglib2.0-0: opencv-python-headless igual necesita estas libs del
# sistema en tiempo de ejecución (no solo en build) para decodificar imágenes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

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

EXPOSE 8000
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
