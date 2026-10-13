"""
Prototipo: validar la ficha de un vehículo ya cargado en el stock contra
su cédula verde, usando OpenCV (preprocesado de imagen) + PaddleOCR (lectura
de texto) + un mapeo a campos conocidos.

Por qué PaddleOCR y no EasyOCR: se corrió un benchmark (benchmark_ocr.py) con
una foto real de una cédula de moto, comparando EasyOCR, Tesseract y PaddleOCR
sobre las mismas variantes de preprocesado. PaddleOCR (modelo PP-OCRv6) ganó
por lejos: leyó practicamente perfecto todos los campos clave, mientras que
EasyOCR confundía sistemáticamente 0/O, 1/I, 2/Z y corrompía las etiquetas.

Pide FRENTE y DORSO por separado (como el flujo de Uber): el frente trae los
datos del vehículo (dominio, marca, modelo, chasis, motor), el dorso trae los
datos del titular (nombre, domicilio, DNI). El dorso es opcional, pero sin él
no se puede confirmar quién es el titular registral del vehículo.

También clasifica el documento (cedula_verde / cedula_azul / desconocido) para
evitar validar con una cédula azul (autorizado a conducir, NO acredita
titularidad) que a primera vista muestra datos muy parecidos.

Esto NO es "cargar stock desde la cédula": el flujo es carga normal del
vehículo -> foto de la cédula -> el sistema valida que coincide -> sello
de "documentación validada" para el comprador.
"""
from __future__ import annotations

import asyncio
import difflib
import unicodedata
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import leads
import stock
import vision

app = FastAPI(title="Prototipo validación cédula verde")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ocr_engine = None  # pesado de inicializar: se carga una sola vez, en el primer request

# PaddleOCR "de verdad" (el framework completo de paddlepaddle) hace OOM kill
# en el plan free de Render (límite duro de 512MB) incluso con los modelos
# "mobile" más chicos -el framework en sí ya usa varios cientos de MB solo al
# importarse-. RapidOCR corre los MISMOS pesos de modelo PP-OCR (no es un
# motor distinto, es el mismo entrenado por PaddleOCR) pero vía ONNX Runtime
# en vez del framework completo de Paddle -> mucho más liviano en memoria,
# entra cómodo en 512MB. La calidad de lectura debería ser equivalente a la
# medida en el benchmark original (PaddleOCR le ganó a EasyOCR/Tesseract),
# porque son los mismos pesos; no se volvió a medir cabeza a cabeza con
# RapidOCR específicamente.


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _run_ocr(engine, img) -> list[str]:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result, _ = engine(img)
    if not result:
        return []
    return [texto for _box, texto, _score in result if texto.strip()]


class CampoComparado(BaseModel):
    declarado: str | None
    extraido: str | None
    coincide: bool


class TitularInfo(BaseModel):
    nombre: str | None
    domicilio: str | None
    dni: str | None


class DiagnosticoTecnico(BaseModel):
    """Evidencia de que el pipeline de OpenCV+OCR corrió -imágenes y texto
    crudo- separada del resultado funcional. No es lo que una integración
    real necesitaría consumir; es solo para verificar el prototipo."""

    frente_documento_detectado: bool
    frente_original_b64: str
    frente_procesada_b64: str
    frente_texto_ocr: list[str]
    dorso_documento_detectado: bool = False
    dorso_original_b64: str | None = None
    dorso_procesada_b64: str | None = None
    dorso_texto_ocr: list[str] = []


class ScanResponse(BaseModel):
    tipo_documento: str
    sello: str
    sello_ok: bool
    alertas: list[str]
    vehiculo: dict[str, CampoComparado]
    titular: TitularInfo
    diagnostico: DiagnosticoTecnico


class _LadoResultado(BaseModel):
    """Uso interno nada más: lo que arma _procesar_lado, no es lo que la API devuelve."""

    presente: bool
    documento_detectado: bool = False
    original_b64: str | None = None
    processed_b64: str | None = None
    texto_crudo: list[str] = []
    campos_extraidos: dict[str, str] = {}


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.upper().strip()


def _coincide(declarado: str | None, extraido: str | None, umbral: float = 0.75) -> bool:
    d, e = _norm(declarado), _norm(extraido)
    if not d or not e:
        return False
    if d == e:
        return True
    return difflib.SequenceMatcher(None, d, e).ratio() >= umbral


def _procesar_lado(image_bytes: bytes, engine) -> _LadoResultado:
    resultado = vision.run_pipeline(image_bytes)
    texto_ocr = _run_ocr(engine, resultado["enhanced_gray"])
    texto_crudo = vision.filtrar_ruido(texto_ocr)
    campos = vision.map_fields(texto_crudo)
    return _LadoResultado(
        presente=True,
        documento_detectado=resultado["documento_detectado"],
        original_b64=resultado["original_b64"],
        processed_b64=resultado["processed_b64"],
        texto_crudo=texto_crudo,
        campos_extraidos=campos,
    )


@app.post("/scan-cedula-verde", response_model=ScanResponse)
async def scan_cedula_verde(
    frente: UploadFile = File(...),
    dorso: UploadFile | None = File(None),
    patente: str = Form(""),
    marca: str = Form(""),
    modelo: str = Form(""),
    chasis: str = Form(""),
):
    engine = _get_ocr_engine()

    # Probado correr frente y dorso en paralelo (dos hilos): midió MÁS lento
    # (34,8s vs 28,7s secuencial) — PaddleOCR ya usa varios hilos de CPU para
    # UNA sola lectura, así que dos lecturas a la vez compiten por el mismo
    # CPU en vez de repartirse el trabajo. Se vuelve a secuencial a propósito.
    frente_bytes = await frente.read()
    dorso_bytes = await dorso.read() if dorso is not None else None

    frente_res = await asyncio.to_thread(_procesar_lado, frente_bytes, engine)
    dorso_res = (
        await asyncio.to_thread(_procesar_lado, dorso_bytes, engine)
        if dorso_bytes is not None
        else _LadoResultado(presente=False)
    )

    todo_el_texto = list(frente_res.texto_crudo) + list(dorso_res.texto_crudo)
    tipo_doc = vision.classify_document(todo_el_texto)

    alertas: list[str] = []
    if tipo_doc["tipo"] == "cedula_azul":
        alertas.append(
            "Esto parece una CÉDULA AZUL (autorizado a conducir), no una cédula verde. "
            "La cédula azul NO acredita titularidad del vehículo — pedí la cédula verde del titular."
        )
    elif tipo_doc["tipo"] == "desconocido":
        alertas.append(
            "No se pudo confirmar que la imagen sea una cédula de identificación de vehículos. Revisar manualmente."
        )
    if not dorso_res.presente:
        alertas.append("Falta la foto del dorso: no se pudo verificar el titular registral.")

    declarados = {"dominio": patente, "marca": marca, "modelo": modelo, "chasis": chasis}
    campos_frente = frente_res.campos_extraidos
    vehiculo = {
        campo: CampoComparado(
            declarado=declarados.get(campo) or None,
            extraido=campos_frente.get(campo),
            coincide=_coincide(declarados.get(campo), campos_frente.get(campo)),
        )
        for campo in ("dominio", "marca", "modelo", "chasis")
    }

    titular = TitularInfo(
        nombre=dorso_res.campos_extraidos.get("titular"),
        domicilio=dorso_res.campos_extraidos.get("domicilio"),
        dni=dorso_res.campos_extraidos.get("doc_dni"),
    )

    campos_clave_ok = sum(1 for c in vehiculo.values() if c.coincide)
    sello_ok = campos_clave_ok >= 3 and tipo_doc["tipo"] == "cedula_verde" and dorso_res.presente
    if sello_ok:
        sello = "Documentación validada"
    elif tipo_doc["tipo"] == "cedula_azul":
        sello = "Documento incorrecto (cédula azul)"
    else:
        sello = "Revisar manualmente"

    diagnostico = DiagnosticoTecnico(
        frente_documento_detectado=frente_res.documento_detectado,
        frente_original_b64=frente_res.original_b64,
        frente_procesada_b64=frente_res.processed_b64,
        frente_texto_ocr=frente_res.texto_crudo,
        dorso_documento_detectado=dorso_res.documento_detectado,
        dorso_original_b64=dorso_res.original_b64,
        dorso_procesada_b64=dorso_res.processed_b64,
        dorso_texto_ocr=dorso_res.texto_crudo,
    )

    return ScanResponse(
        tipo_documento=tipo_doc["tipo"],
        sello=sello,
        sello_ok=sello_ok,
        alertas=alertas,
        vehiculo=vehiculo,
        titular=titular,
        diagnostico=diagnostico,
    )


_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_frontend_dir / "index.html"))


@app.get("/stock")
async def stock_page():
    return FileResponse(str(_frontend_dir / "stock.html"))


@app.post("/stock/cargar-por-cedula")
async def stock_cargar_por_cedula(
    frente: UploadFile = File(...),
    dorso: UploadFile | None = File(None),
):
    """Carga de UN vehículo sin pasar por Excel/CSV: la cédula verde se usa acá
    para completar el formulario de alta, no para validar un dato ya cargado
    (ese es el otro endpoint, /scan-cedula-verde). Mapeo: dominio/marca/modelo/
    anio salen directo del frente; precio/km/color/tipo/descripcion quedan
    vacíos porque no están en la cédula — los completa la agencia antes de
    confirmar. Pasa por la misma validación que una fila de CSV (stock._validar_fila)
    para que ambos caminos de carga (1 a 1 o en masa) converjan en el mismo chequeo."""
    engine = _get_ocr_engine()
    frente_res = _procesar_lado(await frente.read(), engine)
    dorso_res = (
        _procesar_lado(await dorso.read(), engine)
        if dorso is not None
        else _LadoResultado(presente=False)
    )
    campos = frente_res.campos_extraidos

    todo_el_texto = list(frente_res.texto_crudo) + list(dorso_res.texto_crudo)
    tipo_doc = vision.classify_document(todo_el_texto)

    fila_cruda = {
        "dominio": campos.get("dominio", ""),
        "marca": campos.get("marca", ""),
        "modelo": campos.get("modelo", ""),
        "anio": campos.get("anio", ""),
        "tipo": "",
        "precio": "",
        "km": "",
        "color": "",
        "descripcion": "",
    }
    fila_validada = stock._validar_fila(fila_cruda, numero=1)

    advertencias: list[str] = []
    if not frente_res.documento_detectado:
        advertencias.append("No se detectó con claridad el borde del documento en la foto — revisá los datos leídos con más cuidado que de costumbre.")
    if tipo_doc["tipo"] != "cedula_verde":
        advertencias.append(f"El documento no parece una cédula verde (detectado: {tipo_doc['tipo']}). Los datos pueden no ser confiables.")
    if not campos.get("dominio"):
        advertencias.append("No se pudo leer el dominio — completalo a mano antes de guardar.")

    return {
        "fila": fila_validada.model_dump(),
        "chasis_detectado": campos.get("chasis"),
        "tipo_documento": tipo_doc["tipo"],
        "advertencias": advertencias,
        "diagnostico": {
            "frente_documento_detectado": frente_res.documento_detectado,
            "frente_original_b64": frente_res.original_b64,
            "frente_procesada_b64": frente_res.processed_b64,
            "dorso_documento_detectado": dorso_res.documento_detectado,
            "dorso_original_b64": dorso_res.original_b64,
            "dorso_procesada_b64": dorso_res.processed_b64,
        },
    }


@app.get("/stock/plantilla.csv")
async def stock_plantilla():
    from fastapi.responses import Response

    return Response(
        content=stock.generar_plantilla_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=plantilla_stock.csv"},
    )


@app.get("/stock/datos")
async def stock_datos():
    return {"vehiculos": stock.cargar_stock_con_metricas()}


@app.post("/stock/validar")
async def stock_validar(file: UploadFile = File(...)):
    contenido = await file.read()
    try:
        resultado = stock.validar_csv(contenido)
    except ValueError as e:
        return {"error": str(e)}
    return resultado


class ConfirmarPayload(BaseModel):
    filas: list[stock.FilaValidada]


@app.post("/stock/confirmar")
async def stock_confirmar(payload: ConfirmarPayload):
    agregados = stock.guardar_filas_validas(payload.filas)
    return {"agregados": agregados, "cantidad": len(agregados)}


class ValidarDocumentacionPayload(BaseModel):
    validada: bool


@app.post("/stock/{dominio}/documentacion")
async def stock_set_documentacion(dominio: str, payload: ValidarDocumentacionPayload):
    vehiculo = stock.set_documentacion_validada(dominio, payload.validada)
    if vehiculo is None:
        return {"error": f"No se encontró el dominio '{dominio}' en el stock"}
    return vehiculo


# ---------------------------------------------------------------------------
# Lado comprador: catálogo público + consultas (leads)
# ---------------------------------------------------------------------------

@app.get("/catalogo")
async def catalogo_page():
    return FileResponse(str(_frontend_dir / "catalogo.html"))


@app.get("/catalogo/stock")
async def catalogo_stock():
    """Mismo stock que el panel de agencia, pero sin las métricas internas
    (vistas_panel_ejemplo, consultas_whatsapp_ejemplo) — esas son para la
    agencia, no para que el comprador las vea."""
    vehiculos = stock.cargar_stock_con_metricas()
    publico = [
        {k: v for k, v in veh.items() if k not in ("vistas_panel_ejemplo", "consultas_whatsapp_ejemplo")}
        for veh in vehiculos
    ]
    return {"vehiculos": publico}


@app.post("/catalogo/{dominio}/vista")
async def catalogo_vista(dominio: str):
    vehiculo = stock.registrar_vista(dominio)
    if vehiculo is None:
        return {"error": f"No se encontró el dominio '{dominio}' en el stock"}
    return {"dominio": dominio, "vistas_reales": vehiculo["vistas_reales"]}


class ConsultaPayload(BaseModel):
    nombre: str
    mensaje: str
    telefono: str | None = None


@app.post("/catalogo/{dominio}/consulta")
async def catalogo_consulta(dominio: str, payload: ConsultaPayload):
    """El comprador consulta por un vehículo. Esto NUNCA confirma ni promete
    nada — solo registra el interés como pendiente. La agencia es quien
    responde y cierra, siempre a mano (ver leads.py)."""
    if not stock.cargar_stock():
        return {"error": "No hay stock cargado"}
    if dominio not in {v["dominio"] for v in stock.cargar_stock()}:
        return {"error": f"No se encontró el dominio '{dominio}' en el stock"}
    lead = leads.crear_lead(dominio, payload.nombre, payload.mensaje, payload.telefono)
    return {"ok": True, "lead": lead}


# ---------------------------------------------------------------------------
# Lado agencia: panel de interesados (leads)
# ---------------------------------------------------------------------------

@app.get("/leads")
async def leads_listar():
    vehiculos = {v["dominio"]: v for v in stock.cargar_stock()}
    resultado = []
    for lead in leads.listar_leads():
        veh = vehiculos.get(lead["dominio"], {})
        resultado.append({
            **lead,
            "vehiculo_titulo": f"{veh.get('marca', '?')} {veh.get('modelo', '?')}" if veh else "(vehículo eliminado)",
        })
    return {"leads": resultado}


class EstadoLeadPayload(BaseModel):
    estado: str


@app.post("/leads/{lead_id}/estado")
async def leads_set_estado(lead_id: str, payload: EstadoLeadPayload):
    try:
        lead = leads.actualizar_estado(lead_id, payload.estado)
    except ValueError as e:
        return {"error": str(e)}
    if lead is None:
        return {"error": f"No se encontró el lead '{lead_id}'"}
    return lead
