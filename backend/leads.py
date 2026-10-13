"""Prototipo: registro de interesados (leads) por vehículo.

Simula el único punto de contacto entre comprador y agencia que no pasa por
WhatsApp directo: el comprador consulta desde el catálogo público, el sistema
registra el interés (nunca confirma ni promete nada por sí solo) y la agencia
ve la consulta en su panel y la marca a mano como respondida/cerrada. Mismo
principio que el roadmap de Autoclick: "el bot pide, la persona confirma" —
acá ni siquiera hay bot todavía, pero el patrón de NUNCA cerrar nada
automáticamente ya se respeta desde el prototipo.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

LEADS_FILE = Path(__file__).parent.parent / "leads_data.json"

ESTADOS_VALIDOS = {"pendiente", "respondido", "cerrado"}


class Lead(BaseModel):
    id: str
    dominio: str
    nombre: str
    telefono: str | None = None
    mensaje: str
    fecha: str
    estado: str = "pendiente"


def _cargar() -> list[dict]:
    if not LEADS_FILE.exists():
        return []
    return json.loads(LEADS_FILE.read_text(encoding="utf-8"))


def _guardar(leads: list[dict]) -> None:
    LEADS_FILE.write_text(json.dumps(leads, ensure_ascii=False, indent=2), encoding="utf-8")


def crear_lead(dominio: str, nombre: str, mensaje: str, telefono: str | None = None) -> dict:
    leads = _cargar()
    lead = Lead(
        id=uuid.uuid4().hex[:10],
        dominio=dominio,
        nombre=nombre.strip() or "Interesado sin nombre",
        telefono=(telefono or "").strip() or None,
        mensaje=mensaje.strip(),
        fecha=datetime.now().strftime("%Y-%m-%d %H:%M"),
        estado="pendiente",
    ).model_dump()
    leads.append(lead)
    _guardar(leads)
    return lead


def listar_leads() -> list[dict]:
    # más nuevos primero: es lo que la agencia quiere ver arriba
    return sorted(_cargar(), key=lambda l: l["fecha"], reverse=True)


def actualizar_estado(lead_id: str, nuevo_estado: str) -> dict | None:
    if nuevo_estado not in ESTADOS_VALIDOS:
        raise ValueError(f"Estado inválido: {nuevo_estado!r} (válidos: {sorted(ESTADOS_VALIDOS)})")
    leads = _cargar()
    for lead in leads:
        if lead["id"] == lead_id:
            lead["estado"] = nuevo_estado
            _guardar(leads)
            return lead
    return None


def contar_por_dominio() -> dict[str, int]:
    """Cuántas consultas reales tiene cada vehículo — para mostrar en la card
    de stock al lado de las métricas de ejemplo (que siguen siendo simuladas)."""
    conteo: dict[str, int] = {}
    for lead in _cargar():
        conteo[lead["dominio"]] = conteo.get(lead["dominio"], 0) + 1
    return conteo
