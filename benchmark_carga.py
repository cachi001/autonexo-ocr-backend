"""Mide tiempos de carga de OpenCV y PaddleOCR, y prueba el enderezado
(detect_and_warp) sobre una imagen torcida real para ver si OpenCV conviene
seguir usando tal cual, o hace falta ajustar algo."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

t0 = time.perf_counter()
import cv2  # noqa: E402
t_cv2_import = time.perf_counter() - t0
print(f"Import cv2 (OpenCV): {t_cv2_import:.3f}s  (versión {cv2.__version__})")

t0 = time.perf_counter()
import numpy as np  # noqa: E402
import vision  # noqa: E402
t_vision_import = time.perf_counter() - t0
print(f"Import numpy + módulo vision propio: {t_vision_import:.3f}s")

# --- Test de enderezado sobre la imagen torcida real ---
img_path = Path(__file__).parent.parent / "Captura de pantalla 2026-10-10 172711.png"
print(f"\nImagen de prueba: {img_path.name} ({img_path.stat().st_size / 1024:.0f} KB)")

image_bytes = img_path.read_bytes()

t0 = time.perf_counter()
resultado = vision.run_pipeline(image_bytes)
t_pipeline = time.perf_counter() - t0
print(f"Pipeline OpenCV completo (detectar + enderezar + mejorar): {t_pipeline:.3f}s")
print(f"  documento_detectado: {resultado['documento_detectado']}")

out_dir = Path(__file__).parent / "test_images"
out_dir.mkdir(exist_ok=True)

warped_bgr = cv2.cvtColor(resultado["enhanced_gray"], cv2.COLOR_GRAY2BGR)
cv2.imwrite(str(out_dir / "benchmark_warp_original.jpg"),
            cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR))
cv2.imwrite(str(out_dir / "benchmark_warp_resultado.jpg"), warped_bgr)
print(f"  Guardado: test_images/benchmark_warp_resultado.jpg (comparar visualmente)")

# --- Carga de PaddleOCR ---
print("\nCargando motor PaddleOCR (primera vez, incluye carga de modelos)...")
t0 = time.perf_counter()
from paddleocr import PaddleOCR  # noqa: E402

engine = PaddleOCR(
    lang="es",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    enable_mkldnn=False,
)
t_ocr_load = time.perf_counter() - t0
print(f"Carga del motor PaddleOCR: {t_ocr_load:.3f}s")

print("\nCorriendo OCR sobre la imagen enderezada...")
t0 = time.perf_counter()
textos = []
result = engine.predict(warped_bgr)
for page in result:
    textos.extend(page.get("rec_texts", []))
t_ocr_run = time.perf_counter() - t0
print(f"Tiempo de OCR (inferencia): {t_ocr_run:.3f}s")
print(f"Líneas de texto detectadas: {len(textos)}")
for t in textos[:15]:
    print(f"    - {t!r}")

print("\n--- Resumen ---")
print(f"Import OpenCV:         {t_cv2_import:.3f}s")
print(f"Import numpy+vision:   {t_vision_import:.3f}s")
print(f"Pipeline OpenCV (warp+mejora): {t_pipeline:.3f}s")
print(f"Carga modelo PaddleOCR (una sola vez, al primer request): {t_ocr_load:.3f}s")
print(f"Inferencia OCR (por imagen, cada request): {t_ocr_run:.3f}s")
print(f"TOTAL primer request (cold start): {t_cv2_import + t_vision_import + t_pipeline + t_ocr_load + t_ocr_run:.3f}s")
print(f"TOTAL requests siguientes (motor ya cargado): {t_pipeline + t_ocr_run:.3f}s")
