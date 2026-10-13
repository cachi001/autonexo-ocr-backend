"""
Genera las imágenes de GUÍA que usa la UI (estilo Uber: "así tenés que encuadrar
el documento"). Mantiene solo lo esencial del diseño real para orientar el
encuadre (franja celeste horizontal abajo, QR, orden de los campos) — SIN
marca de agua de fondo, SIN franja lateral de patente repetida (eso no sale en
el documento real) y SIN ningún logo/nombre de organismo oficial: el
prototipo no valida contra el Estado, solo lee los datos de la foto.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CARD_W, CARD_H = 900, 560
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_PATH_BOLD = "C:/Windows/Fonts/arialbd.ttf"

AZUL = (30, 105, 175)
AZUL_CLARO = (200, 226, 248)
GRIS = (120, 120, 125)
GRIS_CLARO = (225, 225, 228)
NEGRO = (25, 25, 28)
ROJO = (190, 40, 40)

OUT_DIR = Path(__file__).parent / "frontend" / "assets"

FRANJA_H = 86  # franja celeste horizontal solida, de ancho completo, abajo


def _base_card() -> Image.Image:
    img = Image.new("RGBA", (CARD_W, CARD_H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([4, 4, CARD_W - 5, CARD_H - 5], radius=22, outline=(190, 190, 195), width=3)
    return img


def _franja_inferior(img: Image.Image):
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([4, CARD_H - FRANJA_H, CARD_W - 5, CARD_H - 5], radius=10, fill=AZUL_CLARO)


def _placeholder_qr(img: Image.Image, x: int, y: int, size: int = 95):
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + size, y + size], fill=(255, 255, 255))
    draw.rectangle([x, y, x + size, y + size], outline=NEGRO, width=1)
    cell = size // 8
    import random

    random.seed(7)
    for i in range(8):
        for j in range(8):
            if random.random() > 0.52:
                draw.rectangle([x + i * cell, y + j * cell, x + (i + 1) * cell, y + (j + 1) * cell], fill=NEGRO)
    for ex, ey in [(x + 2, y + 2), (x + size - 20, y + 2), (x + 2, y + size - 20)]:
        draw.rectangle([ex, ey, ex + 18, ey + 18], fill=(255, 255, 255))
        draw.rectangle([ex, ey, ex + 18, ey + 18], outline=NEGRO, width=2)
        draw.rectangle([ex + 5, ey + 5, ex + 13, ey + 13], fill=NEGRO)


def _circulo(img: Image.Image, cx: int, cy: int, r: int, color=AZUL):
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3, fill=(255, 255, 255))
    draw.ellipse([cx - r + 10, cy - r + 10, cx + r - 10, cy + r - 10], outline=color, width=1)


def _marca_ejemplo(img: Image.Image, x: int, y: int):
    big = ImageFont.truetype(FONT_PATH_BOLD, 14)
    small = ImageFont.truetype(FONT_PATH, 12)
    d = ImageDraw.Draw(img)
    d.text((x, y), "EJEMPLO ILUSTRATIVO", font=big, fill=ROJO)
    d.text((x, y + 18), "no es un documento real", font=small, fill=ROJO)


def generar_frente() -> Image.Image:
    img = _base_card()
    _franja_inferior(img)

    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(FONT_PATH_BOLD, 24)
    subtitle_font = ImageFont.truetype(FONT_PATH, 13)
    label_font = ImageFont.truetype(FONT_PATH_BOLD, 18)
    value_font = ImageFont.truetype(FONT_PATH, 18)
    dominio_font = ImageFont.truetype(FONT_PATH_BOLD, 21)
    footer_font = ImageFont.truetype(FONT_PATH, 11)

    x0 = 44
    _circulo(img, x0 + 24, 60, 26)
    draw.text((x0 + 60, 32), "DOCUMENTO DEL VEHÍCULO  (EJEMPLO)", font=subtitle_font, fill=GRIS)
    draw.text((x0 + 60, 48), "FICHA DE IDENTIFICACIÓN DEL VEHÍCULO", font=title_font, fill=NEGRO)
    draw.text((x0 + 60, 80), "Formato ilustrativo para encuadrar la foto", font=subtitle_font, fill=GRIS)
    draw.text((x0 + 60, 95), "no corresponde a ningún organismo oficial", font=subtitle_font, fill=GRIS)

    _placeholder_qr(img, CARD_W - 148, 30, size=92)
    draw = ImageDraw.Draw(img)

    draw.text((x0, 124), "Dominio:  AA 000 AA", font=dominio_font, fill=NEGRO)

    campos = [
        ("MARCA", "MARCA EJEMPLO"),
        ("MODELO", "MODELO EJEMPLO"),
        ("TIPO", "TIPO DE VEHÍCULO"),
        ("USO", "PRIVADO"),
        ("CHASIS", "XXXXXXXXXXXXXXXXX"),
        ("MOTOR", "XXXXXXXXXXXX"),
        ("VENCE", "SIN VENCIMIENTO"),
    ]
    y = 160
    for label, value in campos:
        draw.text((x0, y), f"{label}:", font=label_font, fill=NEGRO)
        draw.text((x0 + 140, y), value, font=value_font, fill=(55, 55, 60))
        y += 33

    draw.text((x0, CARD_H - FRANJA_H // 2 - 7), "Esta ficha es solo una guía de encuadre, no un documento emitido por el Estado", font=footer_font, fill=(60, 60, 65))

    _marca_ejemplo(img, CARD_W - 210, 134)
    return img.convert("RGB")


def generar_dorso() -> Image.Image:
    img = _base_card()
    _franja_inferior(img)

    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(FONT_PATH_BOLD, 20)
    subtitle_font = ImageFont.truetype(FONT_PATH, 12)
    label_font = ImageFont.truetype(FONT_PATH_BOLD, 17)
    value_font = ImageFont.truetype(FONT_PATH_BOLD, 17)
    legal_font = ImageFont.truetype(FONT_PATH, 10)

    x0 = 40
    draw.text((x0, 26), "AUTOMOTOR", font=title_font, fill=NEGRO)
    draw.text((x0 + 180, 26), "IDENTIFICACIÓN", font=title_font, fill=NEGRO)
    draw.text((x0 + 400, 26), "Original (ejemplo)", font=subtitle_font, fill=GRIS)

    y = 76
    draw.text((x0, y), "TITULAR:", font=label_font, fill=NEGRO)
    draw.text((x0 + 90, y), "NOMBRE APELLIDO (EJEMPLO)", font=value_font, fill=(40, 40, 45))
    y += 34
    draw.text((x0, y), "DOMICILIO:", font=label_font, fill=NEGRO)
    draw.text((x0 + 110, y), "CALLE N° 000, PISO: -- DEPTO: --, CIUDAD", font=value_font, fill=(40, 40, 45))
    y += 34
    draw.text((x0, y), "DOC. D.N.I.:", font=label_font, fill=NEGRO)
    draw.text((x0 + 128, y), "00.000.000", font=value_font, fill=(40, 40, 45))

    _circulo(img, CARD_W - 150, 165, 46, color=(140, 140, 145))
    draw.line([(CARD_W - 230, 210), (CARD_W - 60, 210)], fill=GRIS, width=1)
    draw.text((CARD_W - 150, 216), "FIRMA", font=subtitle_font, fill=GRIS)

    y_legal = 268
    draw.line([(x0, y_legal), (CARD_W - x0, y_legal)], fill=GRIS_CLARO, width=1)
    draw.text((x0, y_legal + 10), "ESTA FICHA ES SOLO UNA GUÍA DE ENCUADRE PARA LA FOTO", font=legal_font, fill=GRIS)
    draw.text((x0, y_legal + 24), "NO ES UN DOCUMENTO EMITIDO POR NINGÚN ORGANISMO OFICIAL", font=legal_font, fill=GRIS)

    _marca_ejemplo(img, CARD_W - 210, 60)

    draw.text((x0 + 10, CARD_H - FRANJA_H // 2 - 7), "Ficha ilustrativa — ejemplo de encuadre", font=legal_font, fill=(50, 50, 55))

    return img.convert("RGB")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generar_frente().save(OUT_DIR / "guia-frente.jpg", quality=92)
    generar_dorso().save(OUT_DIR / "guia-dorso.jpg", quality=92)
    print("Guías generadas en", OUT_DIR)
