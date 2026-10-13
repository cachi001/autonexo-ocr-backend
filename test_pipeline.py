import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import cv2
import vision

image_bytes = Path("test_images/cedula_test.jpg").read_bytes()
resultado = vision.run_pipeline(image_bytes)

print("Documento detectado:", resultado["documento_detectado"])
cv2.imwrite("test_images/cedula_processed.jpg", cv2.cvtColor(resultado["enhanced_gray"], cv2.COLOR_GRAY2BGR))

import easyocr

reader = easyocr.Reader(["es", "en"], gpu=False)
ocr_raw = reader.readtext(resultado["enhanced_gray"], detail=0, paragraph=False)
texto_crudo = [t for t in ocr_raw if t.strip()]

print("\n--- Texto OCR crudo ---")
for line in texto_crudo:
    print(repr(line))

campos = vision.map_fields(texto_crudo)
print("\n--- Campos mapeados ---")
for k, v in campos.items():
    print(f"{k}: {v}")
