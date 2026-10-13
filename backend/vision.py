"""
Pipeline de visión para la cédula verde:
1) OpenCV detecta el documento dentro de la foto (contorno + warp de perspectiva).
2) OpenCV mejora la imagen (ilumina parejo para sacar reflejos, sube contraste, enfoca).
3) EasyOCR lee el texto de la imagen ya corregida.
4) Un parser por etiquetas conocidas mapea el texto crudo a los campos de la cédula.
"""
from __future__ import annotations

import base64
import difflib
import re
import unicodedata

import os

import cv2
import numpy as np

# 1600 está validado en el prototipo local (CPU de sobra). En el deploy real
# (Render free, 0.1 CPU) cada segundo de proceso cuenta -> se puede bajar sin
# tocar el valor local vía la variable de entorno CEDULA_MAX_DIM (seteada
# solo en el Dockerfile de deploy).
MAX_DIM = int(os.environ.get("CEDULA_MAX_DIM", "1600"))


def _to_b64_jpg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("No se pudo codificar la imagen")
    return base64.b64encode(buf).decode("ascii")


def _resize_max(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _order_points(pts: np.ndarray) -> np.ndarray:
    # devuelve los 4 puntos ordenados: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left: suma mínima
    rect[2] = pts[np.argmax(s)]       # bottom-right: suma máxima
    rect[1] = pts[np.argmin(diff)]    # top-right: x-y mínima
    rect[3] = pts[np.argmax(diff)]    # bottom-left: x-y máxima
    return rect


def _contornos_por_bordes_multiescala(gray: np.ndarray) -> list[np.ndarray]:
    """Intento 1: Canny + dilatación, probado con VARIOS pares de umbrales en
    vez de uno solo fijo.

    Con fondo blanco y poco contraste, el borde que detecta Canny es tenue y
    discontinuo -> con dilatación chica (5x5, 1 iteración) quedaban varios
    fragmentos de contorno sueltos en vez de uno solo que cierre todo el
    documento, y el más grande de esos fragmentos no llegaba al 20% del área
    de la imagen (bug encontrado probando con cedula_auto_frente.jpg: el
    contorno más grande medía ~10% en vez de ~70%). Un kernel más grande y una
    iteración más cierran esos huecos. Un único par de umbrales (50,150)
    igual puede fallar según la iluminación de cada foto -> se prueban tres
    pares (más sensible, intermedio, más exigente) y se devuelven todos los
    contornos candidatos; el llamador elige el mejor entre todos los métodos."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    candidatos = []
    for umbral_bajo, umbral_alto in ((30, 100), (50, 150), (75, 200)):
        edges = cv2.Canny(blurred, umbral_bajo, umbral_alto)
        edges = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            candidatos.append(max(contours, key=cv2.contourArea))
    return candidatos


def _contorno_por_umbral_global(gray: np.ndarray) -> np.ndarray | None:
    """Intento 2: cuando el fondo es blanco y de bajo contraste, Canny no
    encuentra un borde fuerte entre el documento y el fondo (bug encontrado
    probando con cedula_auto_frente.jpg). En vez de bordes, separa por brillo:
    Otsu binariza automáticamente claro/oscuro, que sí distingue el plástico
    de la cédula del fondo aunque el contraste de bordes sea bajo. Funciona
    bien con iluminación pareja en toda la foto; si la luz es dispareja
    (sombra de un lado), lo resuelve el método siguiente."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _contorno_por_umbral_local(gray: np.ndarray) -> np.ndarray | None:
    """Intento 3: umbral ADAPTATIVO (no global como Otsu) — en vez de un solo
    valor de corte para toda la imagen, calcula un umbral propio para cada
    zona. Sirve para fotos con luz dispareja (una sombra de un lado de la
    cédula, un reflejo de celular en una esquina) donde Otsu binariza mal
    porque intenta ajustar un único corte a toda la foto."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binaria = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


_ASPECTO_CEDULA = 85.6 / 54.0  # proporción real de una cédula tipo tarjeta (ISO/IEC 7810 ID-1), ~1.585


def _contorno_por_grabcut(img_bgr: np.ndarray) -> np.ndarray | None:
    """Último recurso, mucho más lento (2-6s) que los métodos anteriores
    (~0,1-0,3s en total) -> por eso NO se corre siempre, solo cuando los
    métodos rápidos no alcanzan.

    Bug encontrado probando con una foto real del usuario: cédula apoyada
    sobre OTRO papel con texto impreso y letra manuscrita de fondo. Ni los
    bordes (Canny) ni el brillo (Otsu/adaptativo) separan bien ahí, porque el
    papel de fondo también tiene bordes y zonas claras/oscuras por todos
    lados -> el "documento" que se elegía terminaba siendo básicamente toda
    la foto. La diferencia real entre la cédula y ese fondo es el COLOR (la
    cédula es una tarjeta plástica celeste/turquesa; el fondo es papel mate
    blanco/crema con texto), no el brillo ni los bordes -> por eso hace falta
    un método que sí use color y textura juntos, no solo uno de los dos.
    GrabCut hace exactamente eso: separa primer plano de fondo por color y
    textura, sembrado con un rectángulo que asume el documento ocupa la zona
    central de la foto (un recorte del 8% en cada borde).

    El costo de GrabCut escala con la cantidad de píxeles, no con el
    contenido -> los "5,9s" de la nota anterior se midieron en la máquina de
    desarrollo (CPU de sobra); en el 0.1 CPU de Render esa misma cuenta tarda
    muchísimo más (caso real medido: ~200s totales en una foto que dispara
    este método). Por eso corre sobre una copia achicada a 400px de lado
    mayor -el resultado de "dónde está el documento" no necesita resolución
    completa- y al final se escalan las coordenadas del contorno de vuelta al
    tamaño real para que el warp posterior siga siendo preciso."""
    h, w = img_bgr.shape[:2]
    escala = min(1.0, 400 / max(h, w))
    chica = (
        cv2.resize(img_bgr, (max(1, int(w * escala)), max(1, int(h * escala))), interpolation=cv2.INTER_AREA)
        if escala < 1
        else img_bgr
    )
    hc, wc = chica.shape[:2]
    mask = np.zeros((hc, wc), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (int(wc * 0.08), int(hc * 0.08), int(wc * 0.84), int(hc * 0.84))
    try:
        cv2.grabCut(chica, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    mask_fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    mejor = max(contours, key=cv2.contourArea)
    if escala < 1:
        mejor = (mejor.astype(np.float32) / escala).astype(np.int32)
    return mejor


def _distancia_a_proporcion_cedula(c: np.ndarray) -> float:
    _, (w, h), _ = cv2.minAreaRect(c)
    lado_mayor, lado_menor = max(w, h), min(w, h)
    proporcion = lado_mayor / lado_menor if lado_menor else 0
    return abs(proporcion - _ASPECTO_CEDULA)


# si el mejor candidato rápido ocupa más de esto del área de la foto, es
# sospechoso de ser "el fondo entero" en vez del documento -> ahí se prueba
# GrabCut para ver si encuentra algo más ajustado.
_AREA_SOSPECHOSA = 0.55


def _buscar_contorno_dominante(
    gray: np.ndarray, img_area: int, img_bgr: np.ndarray | None = None
) -> np.ndarray | None:
    """Junta los candidatos de los métodos rápidos (bordes en 3 escalas,
    umbral global, umbral local) y elige el más parecido a una cédula real.

    Bug encontrado probando con una foto real del usuario (cédula sobre un
    papel con texto de fondo): el contorno "más grande" terminaba siendo el
    fondo entero, no la cédula -> antes se elegía por área nomás, así que ese
    contorno espurio le ganaba al que sí era la cédula real.

    Primer intento de arreglo: medir el rectángulo alineado a los ejes
    (bounding box) y descartar el que cubriera casi toda la foto. Eso rompió
    el caso de la foto rotada 20°: ahí el rectángulo alineado a los ejes de
    la cédula real TAMBIÉN cubre casi toda la foto, simplemente porque al
    estar inclinada sus esquinas tocan los bordes del lienzo.

    Arreglo: usar minAreaRect (el rectángulo ROTADO que mejor envuelve el
    contorno) en vez del bounding box -> distingue bien rotación real de
    contorno espurio. Pero en la foto con el papel de fondo, el mejor
    candidato de los métodos rápidos seguía siendo demasiado grande (78% del
    área) aunque pasara el filtro de proporción -> se agrega un último
    recurso (GrabCut, ver arriba) para esos casos sospechosos."""
    candidatos = list(_contornos_por_bordes_multiescala(gray))
    for metodo in (_contorno_por_umbral_global, _contorno_por_umbral_local):
        c = metodo(gray)
        if c is not None:
            candidatos.append(c)

    def _candidato_valido(c: np.ndarray) -> bool:
        _, (w, h), _ = cv2.minAreaRect(c)
        return 0.2 * img_area <= w * h <= 0.93 * img_area

    validos = [c for c in candidatos if _candidato_valido(c)]
    mejor = min(validos, key=_distancia_a_proporcion_cedula) if validos else None

    area_mejor = 0.0
    if mejor is not None:
        _, (w, h), _ = cv2.minAreaRect(mejor)
        area_mejor = (w * h) / img_area

    if (mejor is None or area_mejor > _AREA_SOSPECHOSA) and img_bgr is not None:
        c_grabcut = _contorno_por_grabcut(img_bgr)
        if c_grabcut is not None and _candidato_valido(c_grabcut):
            _, (w, h), _ = cv2.minAreaRect(c_grabcut)
            area_grabcut = (w * h) / img_area
            if mejor is None or area_grabcut < area_mejor:
                mejor = c_grabcut

    return mejor


def detect_and_warp(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Busca el rectángulo de la cédula en la foto y la endereza (warp de perspectiva).
    Devuelve (imagen_resultado, se_detecto_documento).

    Usa minAreaRect en vez de approxPolyDP sobre el contorno crudo: con esquinas
    redondeadas, texturas de fondo (madera, mesa) y hologramas, Canny no siempre
    cierra el contorno hasta las 4 esquinas exactas, y approxPolyDP termina
    "cortando" una esquina -> eso generaba un enderezado con corte/sesgo visible.
    minAreaRect ajusta el rectángulo rotado mínimo que contiene a TODO el
    contorno detectado, así que es mucho más tolerante a ese tipo de ruido."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = img.shape[0] * img.shape[1]
    c = _buscar_contorno_dominante(gray, img_area, img_bgr=img)
    if c is None:
        return img, False

    # Bug encontrado con una captura de pantalla ya recortada al documento
    # (sin fondo alrededor): los métodos de contorno encontraron un
    # rectángulo INTERNO de la tarjeta (ej. el bloque de texto del domicilio)
    # en vez del borde real de la cédula, y lo tomaban como válido porque por
    # casualidad tenía una proporción parecida a 1.585 -> el resultado quedaba
    # enfocado/recortado en una porción irrelevante, perdiendo campos como
    # TITULAR. Si la imagen COMPLETA ya se parece a una cédula (proporción
    # cercana a la real) más de lo que se parece el candidato recortado, es
    # señal de que la imagen ya viene encuadrada al documento -> se usa
    # entera en vez de recortar.
    h_img, w_img = img.shape[:2]
    distancia_imagen_completa = abs(max(w_img, h_img) / min(w_img, h_img) - _ASPECTO_CEDULA)
    if distancia_imagen_completa <= _distancia_a_proporcion_cedula(c):
        return img, True

    box = cv2.boxPoints(cv2.minAreaRect(c))
    doc_contour = box

    rect = _order_points(doc_contour.astype("float32"))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 50 or max_height < 50:
        return img, False

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped, True


# Por debajo de esto, EasyOCR lee mal el texto chico de la cédula (validado
# en 700 con CPU de sobra). En Render (0.1 CPU) esto domina el tiempo total:
# logs reales mostraron ~60-72s solo en reconocimiento por imagen con 700px
# de lado corto -> bajable solo en el deploy vía MIN_OCR_SHORT_SIDE, sin
# tocar el valor local ya validado.
MIN_OCR_SHORT_SIDE = int(os.environ.get("MIN_OCR_SHORT_SIDE", "700"))


def _upscale_if_small(img: np.ndarray) -> np.ndarray:
    """Las fotos de celular suelen traer el documento recortado en baja resolución
    (ej. una cédula de 400x260px). El texto de la cédula es chico, así que si el
    recorte queda por debajo de un umbral lo agrandamos con interpolación cúbica
    antes de pasarlo al OCR."""
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side >= MIN_OCR_SHORT_SIDE:
        return img
    scale = MIN_OCR_SHORT_SIDE / short_side
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """Agranda la imagen ya enderezada si quedó chica, y la pasa a gris.

    Antes esta función también normalizaba iluminación (dividir por un blur
    grande), subía contraste con CLAHE, reducía ruido y afilaba con unsharp
    mask -> se sacó todo eso (sesión 2026-10-12) después de medirlo en serio:
    comparando el mismo OCR CON y SIN ese procesado en 5 cédulas reales
    distintas, nunca dio un resultado mejor que dejar la imagen como la
    entrega el warp -- en el caso más difícil probado (cédula con fondo de
    papel con texto) directamente empeoraba la lectura (marca, modelo y
    chasis mal leídos, motor con texto basura), y en un caso inventó un
    campo "dominio" que no existía, leyendo ruido del dorso. La hipótesis es
    que CLAHE + bilateral + unsharp mask amplifican textura de fondo (marca
    de agua DNRPA, bordes de otro papel) tanto o más que el texto real de la
    cédula -> PaddleOCR ya trae su propio preprocesado interno y le alcanza
    con la imagen enderezada sin agregarle más proceso encima."""
    img = _upscale_if_small(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run_pipeline(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen enviada")

    img = _resize_max(img)
    original_b64 = _to_b64_jpg(img)

    warped, detectado = detect_and_warp(img)
    enhanced_gray = enhance_for_ocr(warped)
    processed_b64 = _to_b64_jpg(cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR))

    return {
        "original_b64": original_b64,
        "processed_b64": processed_b64,
        "enhanced_gray": enhanced_gray,
        "documento_detectado": detectado,
    }


# ---------------------------------------------------------------------------
# Mapeo de campos a partir del texto crudo que devuelve el OCR
# ---------------------------------------------------------------------------

_LABELS = {
    # campos del FRENTE (datos del vehiculo)
    "dominio": ["dominio", "patente"],
    "marca": ["marca"],
    "tipo": ["tipo"],
    "modelo": ["modelo"],
    "motor": ["motor"],
    "chasis": ["chasis", "cuadro"],  # las cedulas de moto usan "cuadro" en vez de "chasis"
    # el "ano modelo" casi nunca aparece en el formato nuevo de cedula (lo que
    # sí trae a veces es "VENCE", que es el vencimiento de LA CEDULA, no del
    # auto) -> lo dejamos para no perder el caso si aparece, pero no es un
    # campo confiable para comparar contra el stock
    "anio": ["ano modelo", "anio modelo", "ano fabricacion", "anio fabricacion"],
    # campos del DORSO (datos del titular) - ojo: "domicilio" es MUY parecido a
    # "dominio" en fuzzy matching, por eso tiene que estar en esta misma tabla
    # (para que el comparador elija entre los dos) y nunca evaluarse aislado
    "titular": ["titular", "propietario"],
    "domicilio": ["domicilio"],
    "doc_dni": ["doc d n i", "documento"],
}

_PATENTE_RE = re.compile(r"\b([A-Z]{2}\s?\d{3}\s?[A-Z]{2}|[A-Z]{3}\s?\d{3})\b")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    return _strip_accents(s).lower().strip()


_LABEL_RATIO_MIN = 0.65  # el OCR en documentos con fondo de seguridad suele corromper letras


def _best_label_match(norm_line: str, usados: set[str]) -> tuple[str, str, float] | None:
    """Encuentra la (campo, etiqueta) más parecida al PRINCIPIO de la línea,
    tolerando que el OCR haya corrompido algunas letras de la etiqueta.

    OJO: el ancho del prefijo tiene que ser casi exactamente len(label), sin
    margen extra. Con margen, un label corto como "motor" matcheaba por
    similitud contra "AUTOMOTOR" (que lo contiene como substring) aunque no
    esté al principio de la palabra -> falso positivo. Comparar strings de
    largo parecido evita eso."""
    best: tuple[str, str, float] | None = None
    for campo, labels in _LABELS.items():
        if campo in usados:
            continue
        for label in labels:
            if len(label) < 4:
                continue  # labels muy cortos (ej. "dni") generan falsos positivos por azar
            prefix = norm_line[: len(label) + 1]
            ratio = difflib.SequenceMatcher(None, prefix, label).ratio()
            if best is None or ratio > best[2]:
                best = (campo, label, ratio)
    return best


def map_fields(ocr_lines: list[str]) -> dict:
    """Para cada línea leída por el OCR, busca a qué etiqueta de campo se parece
    más (tolerando errores de OCR en la propia etiqueta). Si la línea trae la
    etiqueta y el valor juntos ("MARCA ZANELLA") separa por palabras; si la
    línea es solo la etiqueta, el valor se busca en la línea siguiente."""
    campos: dict[str, str] = {}
    usados: set[str] = set()

    for i, raw_line in enumerate(ocr_lines):
        norm_line = _norm(raw_line)
        match = _best_label_match(norm_line, usados)
        if match is None or match[2] < _LABEL_RATIO_MIN:
            continue
        campo, label, _ratio = match

        # cortamos por CANTIDAD DE CARACTERES (no por palabras): el OCR a veces
        # pega la etiqueta y el valor sin espacio ("MODELO:SAPUCAI 150 F"), y
        # cortar por palabras se comía la primera palabra del valor
        resto = raw_line[len(label):].lstrip(" :.-").strip()
        if not resto and i + 1 < len(ocr_lines):
            resto = ocr_lines[i + 1].strip(" :.-")

        if resto:
            campos[campo] = resto
            usados.add(campo)

    # la patente suele reconocerse mejor por patrón que por etiqueta, cuando el OCR no la corrompe
    full_text = " ".join(ocr_lines).upper()
    m = _PATENTE_RE.search(full_text)
    if m and not campos.get("dominio"):
        campos["dominio"] = m.group(1).replace(" ", "")

    return campos


# ---------------------------------------------------------------------------
# Filtro de ruido: la cédula tiene la palabra "DNRPA" repetida en diagonal
# como marca de agua de seguridad en TODO el fondo. El OCR la detecta de
# verdad (no la inventa) pero como fragmentos sueltos y solapados
# ("PADNARAON", "DNRPADN", "RPADNP", "R", "A"...) que no aportan nada y
# ensucian el texto que se le muestra al usuario.
# ---------------------------------------------------------------------------

_MARCA_AGUA_REPETIDA = ("DNRPA" * 6)  # "DNRPADNRPADNRPA..." para comparar fragmentos sueltos


def _es_ruido_marca_agua(line: str) -> bool:
    letras = "".join(ch for ch in _norm(line).upper() if ch.isalpha())
    if not letras:
        return False
    if any(ch.isdigit() for ch in line):
        return False  # un campo real (dominio, chasis, etc.) siempre trae números
    if len(letras) <= 2:
        return True  # 1-2 letras sueltas: nunca es un valor de campo real
    if len(letras) > 12:
        return False  # las etiquetas/valores reales (CEDULA, VEHICULOS, PRIVADO...) son mas largas

    # la marca de agua está armada SOLO con las letras D,N,R,P,A repetidas.
    # Una palabra real corta (ej. "FIRMA") puede compartir un par de esas
    # letras y pegar un ratio alto por azar -> exigimos además que la mayoría
    # de las letras de la línea pertenezcan a ese alfabeto reducido.
    dominancia = sum(1 for ch in letras if ch in "DNRPA") / len(letras)
    if dominancia < 0.65:
        return False

    umbral = 0.4 if len(letras) <= 5 else 0.55
    ratio = best_match_ratio_en_texto(_MARCA_AGUA_REPETIDA, letras)
    return ratio >= umbral


def filtrar_ruido(ocr_lines: list[str]) -> list[str]:
    """Saca del texto los fragmentos sueltos de la marca de agua DNRPA, para
    que lo que se le muestra al usuario (y lo que entra al mapeo de campos)
    sea solo texto con contenido real."""
    return [l for l in ocr_lines if l.strip() and not _es_ruido_marca_agua(l)]


# ---------------------------------------------------------------------------
# Clasificación de tipo de documento: distinguir una cédula verde real (de
# TITULAR) de una cédula azul (de AUTORIZADO A CONDUCIR) u otro papel que
# muestre datos de auto parecidos pero no acredite titularidad.
# ---------------------------------------------------------------------------

_MARCADORES_VERDE = [
    "identificacion de vehiculos",
    "identificacion del automotor",
    "cedula de identificacion",
]
_MARCADORES_AZUL = [
    "autorizado a conducir",
    "autorizacion para conducir",
    "conductor autorizado",
    "cedula azul",
]


def _mejor_marcador(norm_text: str, marcadores: list[str]) -> float:
    best = 0.0
    for m in marcadores:
        r = best_match_ratio_en_texto(norm_text, m)
        if r > best:
            best = r
    return best


def best_match_ratio_en_texto(haystack: str, needle: str) -> float:
    """Busca la mejor coincidencia aproximada de `needle` dentro de `haystack`,
    deslizando una ventana del mismo largo (para tolerar ruido de OCR)."""
    L = len(needle)
    if not haystack or L == 0:
        return 0.0
    best = 0.0
    for i in range(0, len(haystack)):
        window = haystack[i : i + L + 3]
        if not window:
            break
        r = difflib.SequenceMatcher(None, window, needle).ratio()
        if r > best:
            best = r
            if best > 0.995:
                break
    return best


def classify_document(ocr_lines: list[str]) -> dict:
    """Clasifica el documento como cedula_verde / cedula_azul / desconocido,
    buscando frases marcadoras del título en todo el texto leído (tolerante a
    errores de OCR). Devuelve también el ratio de confianza de cada lado para
    que el llamador decida qué tan estricto ser."""
    norm_text = _norm(" ".join(ocr_lines))

    score_verde = _mejor_marcador(norm_text, _MARCADORES_VERDE)
    score_azul = _mejor_marcador(norm_text, _MARCADORES_AZUL)

    if score_azul >= 0.75 and score_azul >= score_verde:
        tipo = "cedula_azul"
    elif score_verde >= 0.65:
        tipo = "cedula_verde"
    else:
        tipo = "desconocido"

    return {
        "tipo": tipo,
        "score_verde": round(score_verde, 2),
        "score_azul": round(score_azul, 2),
    }
