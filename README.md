# Prototipo: validación de cédula verde con OpenCV + OCR

Prueba de concepto técnica. **No carga stock** — el vehículo ya está cargado
por la agencia; esta pantalla valida, a partir de una foto de la cédula verde,
que los datos coinciden con lo publicado, y muestra un sello de
"Documentación validada" pensado para generar confianza en el comprador.

## Pipeline

1. **OpenCV** detecta el borde del documento dentro de la foto (contorno +
   `approxPolyDP`) y lo endereza con un warp de perspectiva
   (`getPerspectiveTransform` + `warpPerspective`).
2. **OpenCV** mejora la imagen para el OCR: normaliza la iluminación dividiendo
   por un blur grande (esto es lo que saca reflejos/sombras del plástico de la
   cédula), sube contraste con CLAHE, reduce ruido y enfoca (unsharp mask).
3. **EasyOCR** lee el texto de la imagen ya corregida.
4. Un parser por etiquetas conocidas (`DOMINIO`, `MARCA`, `MODELO`, etc.)
   mapea el texto crudo a campos estructurados.
5. El backend compara esos campos contra los datos que la agencia ya cargó
   en el stock (patente/marca/modelo/año) y devuelve un sello de validación.

Todo el código del pipeline está en `backend/vision.py` (preprocesado +
mapeo) y `backend/main.py` (API + comparación + sello).

## Cómo correrlo

Requiere Python 3.12 (EasyOCR/Torch todavía no tienen wheels para 3.14,
por eso el venv se creó con `py -3.12`).

```bash
cd prototipo-cedula-ocr
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8731
```

Abrir `http://127.0.0.1:8731/`. La primera vez que se procesa una imagen,
EasyOCR descarga sus modelos (~130 MB) — tarda más esa primera vez.

## Imagen de prueba sintética

Como no había una cédula verde real a mano, `generate_test_cedula.py` genera
una cédula sintética con los campos típicos y le aplica una distorsión de
perspectiva + un gradiente de luz (glare) + ruido, para simular una foto de
celular mal tomada:

```bash
./.venv/Scripts/python.exe generate_test_cedula.py
```

Genera `test_images/cedula_test.jpg`. Con esa imagen se validó el pipeline
completo end-to-end (ver `test_images/cedula_processed.jpg` para ver el
resultado del preprocesado, y `test_images/resultado_pagina.png` /
`resultado_mismatch.png` para ver la página funcionando en ambos casos:
datos que coinciden y datos que no coinciden).

`test_pipeline.py` corre el pipeline sin levantar el servidor, útil para
iterar rápido sobre `vision.py`.

## Qué falta para producción (deliberadamente fuera de este prototipo)

- El parser de campos es heurístico (busca etiquetas conocidas) — una cédula
  verde real tiene variaciones de formato entre versiones/provincias que
  conviene testear con fotos reales.
- No hay detección de que la imagen sea realmente una cédula verde (podría
  aceptar cualquier documento rectangular).
- El umbral de "documentación validada" (3 de 4 campos clave) es arbitrario,
  habría que ajustarlo con casos reales.
- No se graba nada en una base de datos ni se conecta a una ficha de stock
  real — eso es lo próximo si el pipeline se valida con fotos reales.
