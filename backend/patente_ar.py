"""
Trae marca/modelo/año/radicación de un dominio consultando la ficha pública
de patente.ar (https://patente.ar/{DOMINIO}) -un agregador de datos
vehiculares (incluye fuente DNRPA) que expone esos campos gratis en una
página pensada para SEO/IA, sin login ni pago. Confirmado en
https://patente.ar/robots.txt: "Indexación abierta para todos los bots
(buscadores + IA)", Allow: / -scrapear esta página puntual está
explícitamente permitido por ellos mismos, no es una zona gris.

Por qué hace falta un navegador (y no un fetch simple como el proxy de
Radicación por Patente): el sitio está detrás de Cloudflare, que le
devuelve a un fetch de servidor sin navegador una página de verificación
("Just a moment...") en vez del contenido -probado en vivo, 403 efectivo-.
Un Chromium real (headless, sin necesidad de Xvfb/headful: acá no hay
reCAPTCHA Enterprise como en la DNRPA, es un desafío de Cloudflare más
liviano) la pasa sin mostrar ningún desafío visible -probado en vivo
también-. A diferencia del flujo de dnrpa_informe.py, acá NO hace falta
ningún humano: es una sola carga de página, totalmente automática.

Los datos vienen de un bloque <script type="application/ld+json"> con
schema.org que el sitio ya genera para buscadores -mucho más estable para
parsear que texto libre del HTML visible.
"""
from __future__ import annotations

import json
import re

from playwright.async_api import async_playwright

SITE_URL = "https://patente.ar/{dominio}"


async def consultar(dominio: str) -> dict:
    dominio = dominio.strip().upper()
    playwright = None
    browser = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        contexto = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = await contexto.new_page()
        page.set_default_timeout(30000)
        await page.goto(SITE_URL.format(dominio=dominio), wait_until="networkidle")

        bloque = await page.query_selector('script[type="application/ld+json"]')
        if bloque is None:
            titulo = await page.title()
            return {
                "ok": False,
                "error": f"patente.ar no devolvió datos estructurados para ese dominio (título de la página: '{titulo}').",
            }
        contenido = await bloque.inner_text()
        datos = json.loads(contenido)

        propiedades: dict[str, str] = {}
        for grafo in datos:
            for nodo in grafo.get("@graph", []):
                for prop in nodo.get("additionalProperty", []):
                    propiedades[prop.get("propertyID", "")] = prop.get("value", "")

        marca = propiedades.get("brand", "")
        modelo = propiedades.get("model", "")
        anio = propiedades.get("vehicleModelDate", "")
        if not marca or not modelo:
            return {"ok": False, "error": f"patente.ar no tiene información cargada para el dominio {dominio}."}

        return {
            "ok": True,
            "dominio": dominio,
            "marca": marca.upper(),
            "modelo": modelo.upper(),
            "anio": anio,
            "provincia": propiedades.get("province", ""),
        }
    except Exception as e:
        return {"ok": False, "error": f"No se pudo consultar patente.ar: {e}"}
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass
