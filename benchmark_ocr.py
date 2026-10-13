"""
Banco de pruebas: variantes de preprocesado de OpenCV x motores de OCR,
para elegir la mejor combinación sobre un caso real (foto de celular de
una cédula verde de moto, con holograma de fondo).

Mide cada combinación contra los valores reales conocidos de esa cédula
(ground truth) con fuzzy matching, y imprime un ranking.
"""
from __future__ import annotations

import difflib
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "backend"))
import vision  # noqa: E402

IMG_PATH = "test_images/cedula_real_moto.jpg"

GROUND_TRUTH = {
    "dominio": "A208SDV",
    "dominio_sticker": "AXE78773",
    "marca": "ZANELLA",
    "modelo": "SAPUCAI 150F",
    "tipo": "MOTOCICLETA",
    "uso": "PRIVADO",
    "cuadro": "8A6MAST53NS002671",
    "motor": "ZS162FMJ2N105860",
    "cilindrada": "150CM3",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(ch for ch in s.upper() if ch.isalnum())


def best_match_ratio(haystack_norm: str, needle_norm: str) -> float:
    L = len(needle_norm)
    if not haystack_norm or L == 0:
        return 0.0
    best = 0.0
    for i in range(0, len(haystack_norm)):
        window = haystack_norm[i : i + L + 3]
        if not window:
            break
        r = difflib.SequenceMatcher(None, window, needle_norm).ratio()
        if r > best:
            best = r
            if best > 0.995:
                break
    return best


def score_text(full_text: str) -> tuple[float, dict]:
    h = _norm(full_text)
    detalle = {}
    total = 0.0
    for campo, val in GROUND_TRUTH.items():
        r = best_match_ratio(h, _norm(val))
        detalle[campo] = round(r, 2)
        total += r
    return total / len(GROUND_TRUTH), detalle


# ---------------------------------------------------------------------------
# Variantes de preprocesado de OpenCV, todas partiendo del mismo recorte
# ya enderezado (minAreaRect) para aislar el efecto de cada técnica
# ---------------------------------------------------------------------------

def build_variants(warped_bgr: np.ndarray) -> dict[str, np.ndarray]:
    variants = {}

    big = vision._upscale_if_small(warped_bgr)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    variants["A_color_upscale"] = big
    variants["B_gray_upscale"] = gray

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    norm = cv2.divide(gray, bg, scale=255)
    variants["C_illum_norm"] = norm

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(norm)
    variants["D_illum_clahe"] = contrasted

    denoised = cv2.bilateralFilter(contrasted, d=7, sigmaColor=50, sigmaSpace=50)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(denoised, 1.2, blurred, -0.2, 0)
    variants["E_full_pipeline_actual"] = sharpened

    otsu = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants["F_otsu_binaria"] = otsu

    adapt = cv2.adaptiveThreshold(
        contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    variants["G_adaptive_binaria"] = adapt

    # upscale mas agresivo (2x sobre lo que ya subio _upscale_if_small)
    h, w = contrasted.shape[:2]
    extra_big = cv2.resize(contrasted, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants["H_illum_clahe_2x"] = extra_big

    # version color con CLAHE en canal L (LAB), sin pasar a gris
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    color_clahe = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    variants["I_color_clahe_lab"] = color_clahe

    return variants


# ---------------------------------------------------------------------------
# Motores de OCR
# ---------------------------------------------------------------------------

def run_easyocr(reader, img) -> str:
    lines = reader.readtext(img, detail=0, paragraph=False)
    return " ".join(lines)


def run_tesseract(img) -> str:
    import pytesseract

    return pytesseract.image_to_string(img, lang="eng", config="--psm 6")


def run_paddleocr(engine, img) -> str:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = engine.predict(img)
    texts = []
    for page in result:
        texts.extend(page.get("rec_texts", []))
    return " ".join(texts)


def main():
    image_bytes = Path(IMG_PATH).read_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = vision._resize_max(img)
    warped, detected = vision.detect_and_warp(img)
    print("documento detectado:", detected, "shape:", warped.shape)

    variants = build_variants(warped)
    for name, im in variants.items():
        cv2.imwrite(f"test_images/variant_{name}.jpg", im)

    print("\nCargando motores de OCR (puede tardar la primera vez)...")
    import easyocr

    easy_reader = easyocr.Reader(["es", "en"], gpu=False)

    from paddleocr import PaddleOCR

    paddle_engine = PaddleOCR(lang="es", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    results = []

    for name, im in variants.items():
        for engine_name, fn in [
            ("EasyOCR", lambda im=im: run_easyocr(easy_reader, im)),
            ("Tesseract", lambda im=im: run_tesseract(im)),
            ("PaddleOCR", lambda im=im: run_paddleocr(paddle_engine, im)),
        ]:
            t0 = time.time()
            try:
                text = fn()
            except Exception as e:
                text = ""
                print(f"  ERROR {engine_name}/{name}: {e}")
            dt = time.time() - t0
            score, detalle = score_text(text)
            results.append((score, engine_name, name, dt, detalle, text))
            print(f"{engine_name:10s} | {name:25s} | score={score:.3f} | {dt:5.1f}s")

    results.sort(key=lambda r: -r[0])
    print("\n\n=== RANKING (mejor a peor) ===")
    for score, engine_name, name, dt, detalle, text in results[:10]:
        print(f"\n#{results.index((score, engine_name, name, dt, detalle, text))+1} {engine_name} / {name}  score={score:.3f}  tiempo={dt:.1f}s")
        print("  detalle:", detalle)
        print("  texto:", text[:200].replace("\n", " | "))


if __name__ == "__main__":
    main()
