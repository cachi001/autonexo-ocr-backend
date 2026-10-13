"""
Prototipo de carga de stock masiva: plantilla CSV descargable -> la agencia la
llena y la sube -> se valida fila por fila -> vista previa con errores ->
confirmar guarda en el "stock" (acá, un JSON en disco; en producción sería la
base de datos real).

Formato elegido para este prototipo: CSV simple (no Excel) porque no requiere
dependencias extra (pandas/openpyxl) y cualquier agencia puede abrirlo y
editarlo con Excel/Google Sheets igual. Si el prototipo valida que el flujo
sirve, pasar a .xlsx es un cambio menor.
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

STOCK_FILE = Path(__file__).parent.parent / "stock_data.json"

COLUMNAS = ["dominio", "marca", "modelo", "anio", "tipo", "precio", "km", "color", "descripcion"]
COLUMNAS_OBLIGATORIAS = {"dominio", "marca", "modelo", "anio", "precio"}

# Vieja (ABC123), Mercosur auto (AB123CD), Mercosur moto (A123BCD) —
# probar con una cédula real de moto (sesión 2026-10-11) mostró que faltaba
# este último formato: el dominio se leía bien pero la validación lo rechazaba.
_PATENTE_RE = re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$|^[A-Z]{3}\d{3}$|^[A-Z]\d{3}[A-Z]{3}$")

FILA_EJEMPLO = {
    "dominio": "AB123CD",
    "marca": "VOLKSWAGEN",
    "modelo": "GOL TREND",
    "anio": "2019",
    "tipo": "SEDAN",
    "precio": "8500000",
    "km": "65000",
    "color": "BLANCO",
    "descripcion": "Único dueño, service oficial",
}


def generar_plantilla_csv() -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNAS)
    writer.writeheader()
    writer.writerow(FILA_EJEMPLO)
    return buf.getvalue()


class FilaValidada(BaseModel):
    numero_fila: int
    datos: dict
    errores: list[str]
    valida: bool


class ResultadoValidacion(BaseModel):
    filas: list[FilaValidada]
    total: int
    validas: int
    con_error: int


def _validar_fila(fila: dict, numero: int) -> FilaValidada:
    errores: list[str] = []
    datos = {k: (fila.get(k) or "").strip() for k in COLUMNAS}

    for campo in COLUMNAS_OBLIGATORIAS:
        if not datos.get(campo):
            errores.append(f"falta '{campo}'")

    dominio = datos["dominio"].upper().replace(" ", "")
    datos["dominio"] = dominio
    if dominio and not _PATENTE_RE.match(dominio):
        errores.append(f"dominio '{dominio}' no tiene formato de patente válido (ej. AB123CD o ABC123)")

    if datos["anio"]:
        try:
            anio = int(datos["anio"])
            if not (1960 <= anio <= datetime.now().year + 1):
                errores.append(f"año '{anio}' fuera de rango razonable")
        except ValueError:
            errores.append(f"año '{datos['anio']}' no es un número")

    if datos["precio"]:
        try:
            precio = float(datos["precio"].replace(".", "").replace(",", "."))
            if precio <= 0:
                errores.append("precio debe ser mayor a 0")
        except ValueError:
            errores.append(f"precio '{datos['precio']}' no es un número válido")

    if datos["km"]:
        try:
            float(datos["km"])
        except ValueError:
            errores.append(f"km '{datos['km']}' no es un número válido")

    return FilaValidada(numero_fila=numero, datos=datos, errores=errores, valida=not errores)


def validar_csv(contenido: bytes) -> ResultadoValidacion:
    texto = contenido.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(texto))

    faltantes = COLUMNAS_OBLIGATORIAS - set(c.strip().lower() for c in (reader.fieldnames or []))
    if faltantes:
        raise ValueError(f"Al archivo le faltan columnas obligatorias: {', '.join(sorted(faltantes))}")

    filas = [_validar_fila(row, i + 2) for i, row in enumerate(reader)]  # +2: fila 1 es el header
    validas = sum(1 for f in filas if f.valida)
    return ResultadoValidacion(filas=filas, total=len(filas), validas=validas, con_error=len(filas) - validas)


def cargar_stock() -> list[dict]:
    if not STOCK_FILE.exists():
        return []
    return json.loads(STOCK_FILE.read_text(encoding="utf-8"))


def cargar_stock_con_metricas() -> list[dict]:
    """Métricas de ejemplo (consultas/clics) para mostrar cómo se vería un
    panel de 'qué vehículos consultan más' -NO son datos reales todavía, no
    hay tracking conectado-, son deterministas por dominio solo para que la
    demo se vea consistente entre recargas."""
    import hashlib

    import leads as _leads  # import acá adentro: evita import circular con leads.py

    conteo_real = _leads.contar_por_dominio()
    stock = cargar_stock()
    for v in stock:
        h = int(hashlib.md5(v["dominio"].encode()).hexdigest(), 16)
        v["consultas_whatsapp_ejemplo"] = h % 40
        v["vistas_panel_ejemplo"] = (h // 7) % 120
        v.setdefault("documentacion_validada", False)
        v.setdefault("documentacion_validada_fecha", None)
        v.setdefault("vistas_reales", 0)
        v["consultas_reales"] = conteo_real.get(v["dominio"], 0)
    return stock


def guardar_filas_validas(filas: list[FilaValidada]) -> list[dict]:
    stock = cargar_stock()
    existentes = {v["dominio"] for v in stock}
    agregados = []
    for f in filas:
        if not f.valida:
            continue
        if f.datos["dominio"] in existentes:
            continue  # evita duplicados por dominio en este prototipo
        fila = dict(f.datos)
        fila["documentacion_validada"] = False
        fila["documentacion_validada_fecha"] = None
        stock.append(fila)
        existentes.add(fila["dominio"])
        agregados.append(fila)
    STOCK_FILE.write_text(json.dumps(stock, ensure_ascii=False, indent=2), encoding="utf-8")
    return agregados


def registrar_vista(dominio: str) -> dict | None:
    """Cuenta real de vistas desde el catálogo público (lado comprador) —
    separada a propósito de vistas_panel_ejemplo, que sigue siendo el dato
    simulado del panel interno. Esta es la primera métrica real del prototipo."""
    stock = cargar_stock()
    for v in stock:
        if v["dominio"] == dominio:
            v["vistas_reales"] = v.get("vistas_reales", 0) + 1
            STOCK_FILE.write_text(json.dumps(stock, ensure_ascii=False, indent=2), encoding="utf-8")
            return v
    return None


def set_documentacion_validada(dominio: str, validada: bool) -> dict | None:
    """Prototipo: activa/desactiva a mano el sello de 'papeles validados' en un
    vehículo del stock. En el sistema real esto lo dispararía el resultado de
    /scan-cedula-verde (sello_ok), no un botón manual — acá se simula para
    poder mostrar cómo se ve la card sin tener que escanear una cédula real
    por cada fila de stock."""
    stock = cargar_stock()
    for v in stock:
        if v["dominio"] == dominio:
            v["documentacion_validada"] = validada
            v["documentacion_validada_fecha"] = datetime.now().strftime("%Y-%m-%d") if validada else None
            STOCK_FILE.write_text(json.dumps(stock, ensure_ascii=False, indent=2), encoding="utf-8")
            return v
    return None
