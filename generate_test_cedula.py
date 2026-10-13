"""
Genera una imagen sintética de una 'cédula verde' (layout aproximado) y le aplica
una perspectiva torcida + un gradiente de luz (glare) + ruido, para poder probar
el pipeline de OpenCV + OCR con un caso parecido a una foto de celular real,
sin necesitar un documento real.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARD_W, CARD_H = 900, 560
FONT_PATH = "C:/Windows/Fonts/arial.ttf"

CAMPOS = [
    ("DOMINIO", "AB123CD"),
    ("MARCA", "VOLKSWAGEN"),
    ("TIPO", "SEDAN 4 PUERTAS"),
    ("MODELO", "GOL TREND"),
    ("ANIO MODELO", "2019"),
    ("MOTOR", "CLZ123456"),
    ("CHASIS", "9BWAB05X1KP654321"),
    ("TITULAR", "PEREZ JUAN CARLOS"),
]


GREEN = (30, 95, 45)
GREEN_LIGHT = (225, 240, 228)
OCHRE = (180, 130, 40)


def build_flat_card() -> Image.Image:
    # cartulina blanca con líneas verdes y detalles ocre, como describe la
    # documentación oficial de la cédula verde argentina
    img = Image.new("RGB", (CARD_W, CARD_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH, 30)
    subtitle_font = ImageFont.truetype(FONT_PATH, 18)
    label_font = ImageFont.truetype(FONT_PATH, 20)
    value_font = ImageFont.truetype(FONT_PATH, 25)

    # guilloche simplificado: líneas finas verdes de fondo en toda la cartulina
    for i in range(-CARD_H, CARD_W, 14):
        draw.line([(i, 0), (i + CARD_H, CARD_H)], fill=GREEN_LIGHT, width=1)

    draw.rectangle([10, 10, CARD_W - 11, CARD_H - 11], outline=GREEN, width=5)
    draw.rectangle([18, 18, CARD_W - 19, CARD_H - 19], outline=OCHRE, width=1)

    draw.text((42, 28), "REPUBLICA ARGENTINA", font=subtitle_font, fill=OCHRE)
    draw.text((42, 52), "CEDULA DE IDENTIFICACION DEL AUTOMOTOR", font=title_font, fill=GREEN)
    draw.line([(42, 94), (CARD_W - 42, 94)], fill=OCHRE, width=2)

    # sello circular simple arriba a la derecha (simula el sello/logo del registro)
    draw.ellipse([CARD_W - 120, 24, CARD_W - 50, 94], outline=GREEN, width=3)
    draw.text((CARD_W - 110, 50), "DNRPA", font=subtitle_font, fill=GREEN)

    y = 130
    for label, value in CAMPOS:
        draw.text((48, y), f"{label}:", font=label_font, fill=(90, 90, 90))
        draw.text((320, y - 2), value, font=value_font, fill=(15, 15, 15))
        y += 48

    return img


def apply_photo_distortion(flat_img: Image.Image) -> np.ndarray:
    img = cv2.cvtColor(np.array(flat_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # lienzo más grande para simular que la cédula está sobre una mesa, no a borde de cuadro
    canvas = np.full((h + 300, w + 300, 3), (40, 40, 45), dtype=np.uint8)
    canvas[150:150 + h, 150:150 + w] = img

    # perspectiva: estiramos las 4 esquinas de forma desigual (como una foto en ángulo)
    src = np.float32([[150, 150], [150 + w, 150], [150 + w, 150 + h], [150, 150 + h]])
    dst = np.float32([
        [180, 190],
        [150 + w - 40, 120],
        [150 + w - 10, 150 + h - 30],
        [210, 150 + h + 40],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, M, (w + 300, h + 300), borderValue=(40, 40, 45))

    # glare: un gradiente de luz radial en una esquina, simulando el reflejo del plástico
    glare = np.zeros_like(warped, dtype=np.float32)
    yy, xx = np.mgrid[0:warped.shape[0], 0:warped.shape[1]]
    cx, cy = warped.shape[1] * 0.75, warped.shape[0] * 0.25
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glare_strength = np.clip(1 - dist / (warped.shape[1] * 0.5), 0, 1) * 160
    for c in range(3):
        glare[:, :, c] = glare_strength
    warped = np.clip(warped.astype(np.float32) + glare, 0, 255).astype(np.uint8)

    # un poco de ruido + blur leve para simular foto de celular
    noise = np.random.normal(0, 4, warped.shape).astype(np.float32)
    warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    warped = cv2.GaussianBlur(warped, (3, 3), 0)

    return warped


if __name__ == "__main__":
    flat = build_flat_card()
    photo = apply_photo_distortion(flat)
    out_path = "test_images/cedula_test.jpg"
    cv2.imwrite(out_path, photo)
    print("Generada:", out_path, photo.shape)
