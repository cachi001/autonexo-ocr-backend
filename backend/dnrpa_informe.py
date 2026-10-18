"""
Automatiza el flujo real "Obtener Informe Web" de la DNRPA
(https://www2.jus.gov.ar/dnrpa-site/#!/solicitante) para traer marca/modelo/
año reales de un dominio ya cargado en el stock, gratis -sin el costo de un
informe formal-. Es la validación que "Radicación por Patente" (el otro
proxy, en main /api/dnrpa de autonexo-demo) NO puede dar: esa solo devuelve
tipo de vehículo y registro, nunca marca/modelo/año.

Por qué hace falta un navegador real (Playwright) y no un simple fetch como
el otro proxy: este flujo pide resolver un reCAPTCHA v2 con desafío visual
("seleccioná las imágenes con..."). Ese desafío genera un token
(RecaptchaResponse) que Google valida server-side contra el navegador real
que lo resolvió -no se puede fabricar ni reutilizar con un POST directo a la
API interna de la DNRPA (ObtenerVehiculo), confirmado inspeccionando el
tráfico real: el body de ese POST lleva un RecaptchaResponse de ~2000
caracteres atado a esa sesión-. El único camino honesto es que una persona
(el empleado de la agencia) resuelva el captcha de verdad, como lo haría
si entrara a la DNRPA a mano; este módulo solo le muestra el desafío dentro
de AutoNexo en vez de mandarlo al sitio de la DNRPA.

Memoria: Render free tiene 512MB totales y el motor de OCR (RapidOCR) ya
usa una parte importante apenas se carga -por eso ya se abandonó
paddlepaddle completo en este mismo backend, ver main.py-. Un Chromium
headless corriendo full-time sumaría 150-300MB residentes encima de eso.
Para no competir con el OCR todo el tiempo: el navegador se abre recién
cuando se inicia una consulta y se cierra apenas termina (éxito, error o
timeout) -nunca queda cacheado como sí se hace con _ocr_engine-. Esto paga
~1-2s extra de arranque por consulta a cambio de no tener Chromium residente
en memoria cuando nadie lo está usando. Además se permite una sola sesión
de captcha a la vez (lock global): con 512MB compartidos no alcanza para
dos Chromium concurrentes.
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid

from playwright.async_api import async_playwright, Browser, Page, Playwright
from playwright_stealth import Stealth

SITE_URL = "https://www2.jus.gov.ar/dnrpa-site/#!/solicitante"
SESSION_TIMEOUT_SECONDS = 300  # una IP de Render nueva/poco usada hace que reCAPTCHA pida más rondas de lo normal (riesgo visto en vivo: 9+ rondas) -3 min quedaba corto

_lock = asyncio.Lock()
_sesiones: dict[str, "_Sesion"] = {}
_lock_tomado_en: float | None = None  # para poder autoliberar un lock que quedó pegado


class _Sesion:
    def __init__(self, session_id: str, playwright: Playwright, browser: Browser, page: Page):
        self.id = session_id
        self.playwright = playwright
        self.browser = browser
        self.page = page
        self.created_at = time.time()

    async def cerrar(self):
        global _lock_tomado_en
        try:
            await self.browser.close()
        except Exception:
            pass
        try:
            await self.playwright.stop()
        except Exception:
            pass
        _sesiones.pop(self.id, None)
        if _lock.locked():
            _lock.release()
            _lock_tomado_en = None


async def _limpiar_si_vencida(sesion: "_Sesion") -> bool:
    if time.time() - sesion.created_at > SESSION_TIMEOUT_SECONDS:
        await sesion.cerrar()
        return True
    return False


def _bframe(page: Page):
    for frame in page.frames:
        if "recaptcha/api2/bframe" in frame.url:
            return frame
    return None


async def _leer_desafio(frame) -> dict | None:
    """Si hay un desafío de imágenes visible, devuelve el enunciado + una
    captura de la grilla de 9 imágenes. Si no hay desafío (ya se resolvió,
    o reCAPTCHA no lo pidió), devuelve None."""
    try:
        # En headful/Xvfb el panel tarda más en pintar que en headless -4s se
        # quedaba corto y tiraba "no se pudo leer el captcha" con el desafío
        # todavía cargando (probado en vivo).
        prompt_el = await frame.wait_for_selector("div.rc-imageselect-desc-no-canonical, div.rc-imageselect-desc", timeout=12000)
    except Exception:
        return None
    prompt = (await prompt_el.inner_text()).replace("\n", " ").strip()
    tabla_33 = await frame.query_selector("table.rc-imageselect-table-33")
    tabla = tabla_33 or await frame.query_selector("table.rc-imageselect-table-44")
    if tabla is None:
        return None
    # El panel hace un fade-in: si se captura apenas aparece, la foto sale
    # mezclada con lo que había debajo a medio transicionar.
    await frame.page.wait_for_timeout(700)
    captura = await tabla.screenshot(type="png")
    return {
        "prompt": prompt,
        "imagen_b64": base64.b64encode(captura).decode("ascii"),
        # 3x3 (9 fotos distintas, se elige una vez) vs. 4x4 (1 foto grande
        # cuadriculada, "seleccioná todas las que haya" con reintentos).
        "celdas": 9 if tabla_33 is not None else 16,
    }


async def _resultado_o_error(page: Page) -> dict | None:
    """Después de resolver el captcha, la propia SPA llama a ObtenerVehiculo
    y pinta el resultado (o un error) en el DOM -no hace falta replicar esa
    llamada a mano, alcanza con leer lo que ya renderizó."""
    try:
        detalle = await page.wait_for_selector("text=Detalles del Vehículo", timeout=6000)
    except Exception:
        detalle = None
    if detalle is not None:
        filas = await page.query_selector_all("table tr")
        datos: dict[str, str] = {}
        for fila in filas:
            celdas = await fila.query_selector_all("td")
            if len(celdas) == 2:
                clave = (await celdas[0].inner_text()).strip()
                valor = (await celdas[1].inner_text()).strip()
                datos[clave] = valor
        if "Marca" in datos:
            return {
                "ok": True,
                "dominio": datos.get("Dominio", ""),
                "marca": datos.get("Marca", ""),
                "modelo": datos.get("Modelo", ""),
                "anio": datos.get("Año", ""),
                "registro": datos.get("Registro", ""),
            }
    # Errores típicos que la SPA muestra como alert si el dominio no existe
    # o el captcha salió mal de verdad (no solo "challenge nuevo").
    alerta = await page.query_selector("div.alert-danger, div.ui-pnotify-text")
    if alerta is not None:
        texto = (await alerta.inner_text()).strip()
        if texto:
            return {"ok": False, "error": texto}
    return None


async def iniciar(cuit: str, dominio: str, alta: dict | None = None) -> dict:
    """Abre un Chromium nuevo, completa CUIT + dominio y deja el captcha
    listo para mostrar. `alta` (apellido/nombre/email/celular) solo se usa
    si ese CUIT TODAVÍA no está registrado como solicitante en la DNRPA -se
    completa una sola vez por agencia, igual que se hizo a mano para la
    agencia de ejemplo."""
    global _lock_tomado_en
    if _lock.locked():
        # Si el lock lleva más tiempo tomado que el límite de sesión, algo
        # quedó pegado a mitad de camino (un request que nunca llegó a
        # crear la sesión, o una pestaña que se cerró sin volver a llamar
        # al endpoint de captcha) -se libera solo en vez de bloquear para
        # siempre a la agencia.
        if _lock_tomado_en is not None and time.time() - _lock_tomado_en > SESSION_TIMEOUT_SECONDS:
            for sesion_vieja in list(_sesiones.values()):
                await sesion_vieja.cerrar()
            if _lock.locked():
                _lock.release()
            _lock_tomado_en = None
        else:
            return {"error": "Ya hay otra consulta DNRPA en curso. Esperá a que termine y probá de nuevo."}
    await _lock.acquire()
    _lock_tomado_en = time.time()
    playwright = None
    browser = None
    try:
        playwright = await async_playwright().start()
        # headless=False (headful real, contra el display virtual Xvfb que
        # arranca xvfb-run en el Dockerfile): el modo headless, aunque se le
        # parchee user-agent/navigator.webdriver/CDP (ver stealth abajo),
        # sigue renderizando distinto por dentro -probado en vivo: con todo
        # eso aplicado, la cantidad de rondas de captcha no bajó nada. Un
        # Chromium headful de verdad renderiza igual que el de cualquier
        # usuario, sin esa diferencia de fondo.
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        # El humano (empleado de la agencia) sigue resolviendo el captcha de
        # verdad en cada ronda -esto no lo reemplaza, solo deja de delatar
        # que el navegador está controlado por automatización (CDP), algo
        # que no tiene nada que ver con si hay una persona real del otro
        # lado resolviendo la imagen.
        contexto = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="es-AR",
        )
        stealth = Stealth(navigator_user_agent=False)  # el UA ya lo seteamos arriba, no pisarlo
        await stealth.apply_stealth_async(contexto)
        page = await contexto.new_page()
        page.set_default_timeout(45000)
        await page.goto(SITE_URL, wait_until="domcontentloaded")
        await page.get_by_role("link", name="Informes online").click()
        await page.get_by_role("spinbutton", name="CUIL / CUIT").fill(cuit)
        await page.get_by_role("button", name="Buscar Solicitante").click()

        # El resultado de la búsqueda (encontrado o no) tarda un instante en
        # volver -hay que esperarlo antes de mirar el DOM, si no esta
        # comprobación corre siempre antes de que la respuesta llegue.
        try:
            await page.wait_for_selector("text=Solicitante no encontrado", timeout=5000)
            no_encontrado = True
        except Exception:
            no_encontrado = False
        if no_encontrado:
            if not alta or not all(alta.get(k) for k in ("apellido", "nombre", "email", "celular_codigo_area", "celular_numero", "celular_operador")):
                await browser.close()
                await playwright.stop()
                _lock.release()
                _lock_tomado_en = None
                return {
                    "error": (
                        f"El CUIT {cuit} todavía no está registrado como solicitante en la DNRPA "
                        "y faltan los datos de contacto de la agencia para darlo de alta. "
                        "Completá Nombre/Apellido/Email/Celular en Datos de la agencia."
                    )
                }
            await page.get_by_role("textbox", name="Apellido").fill(alta["apellido"])
            await page.get_by_role("textbox", name="Nombre").fill(alta["nombre"])
            await page.get_by_role("textbox", name="Email", exact=True).fill(alta["email"])
            await page.get_by_role("textbox", name="Repetir Email").fill(alta["email"])
            await page.locator("#codigoAreaCelular").fill(alta["celular_codigo_area"])
            await page.locator("#telefonoCelular").fill(alta["celular_numero"])
            await page.locator("#operadorCelular").select_option(alta["celular_operador"])

        await page.get_by_role("button", name="Continuar").click()
        await page.get_by_role("textbox", name="Patente").fill(dominio.upper())
        await page.get_by_role("button", name="Validar Vehículo").click()

        session_id = str(uuid.uuid4())
        sesion = _Sesion(session_id, playwright, browser, page)
        _sesiones[session_id] = sesion

        frame = await _esperar_bframe(page)
        if frame is None:
            resultado = await _resultado_o_error(page)
            await sesion.cerrar()
            if resultado and resultado.get("ok"):
                return {"session_id": None, "resultado": resultado}
            return {"error": (resultado or {}).get("error", "La DNRPA no devolvió un resultado. Probá de nuevo.")}

        desafio = await _leer_desafio(frame)
        if desafio is None:
            resultado = await _resultado_o_error(page)
            await sesion.cerrar()
            if resultado and resultado.get("ok"):
                return {"session_id": None, "resultado": resultado}
            return {"error": (resultado or {}).get("error", "No se pudo leer el captcha. Probá de nuevo.")}

        return {"session_id": session_id, "desafio": desafio}
    except Exception as e:
        # Si falló antes de guardar la sesión, nadie más va a cerrar el
        # navegador -hacerlo acá, si no queda un Chromium zombie consumiendo
        # RAM (y compitiendo con el próximo intento) para siempre.
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
        _lock.release()
        _lock_tomado_en = None
        return {"error": f"No se pudo iniciar la consulta a la DNRPA: {e}"}


async def _esperar_bframe(page: Page, timeout_s: float = 15.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        frame = _bframe(page)
        if frame is not None:
            return frame
        await asyncio.sleep(0.3)
    return None


def _comprobar_sesion(session_id: str) -> tuple["_Sesion | None", dict | None]:
    sesion = _sesiones.get(session_id)
    if sesion is None:
        return None, {"error": "Esa consulta ya expiró o no existe. Empezá de nuevo."}
    return sesion, None


async def marcar_celda(session_id: str, indice: int) -> dict:
    """Clickea UNA celda, sin tocar 'Verificar' -igual que el captcha real de
    Google: tocar una imagen correcta la reemplaza al toque en esa celda
    puntual, sin perder las demás ni mandar nada todavía. Devuelve la grilla
    actualizada (con el reemplazo, si lo hubo) para que se vea en vivo."""
    sesion, err = _comprobar_sesion(session_id)
    if err:
        return err
    if await _limpiar_si_vencida(sesion):
        minutos = SESSION_TIMEOUT_SECONDS // 60
        return {"error": f"Pasaron más de {minutos} minutos sin resolver el captcha. Empezá de nuevo."}

    frame = _bframe(sesion.page)
    if frame is None:
        await sesion.cerrar()
        return {"error": "Se perdió el captcha. Empezá de nuevo."}

    celdas = await frame.query_selector_all("table.rc-imageselect-table-33 td, table.rc-imageselect-table-44 td")
    if 0 <= indice < len(celdas):
        await celdas[indice].click()
    await asyncio.sleep(0.5)

    desafio = await _leer_desafio(frame)
    if desafio is None:
        return {"error": "Se perdió el captcha. Probá de nuevo."}
    return {"session_id": session_id, "desafio": desafio}


async def verificar(session_id: str) -> dict:
    """Recién acá se manda lo marcado -clickea el botón real 'Verificar' de
    reCAPTCHA. Puede devolver una ronda nueva (si Google pide más) o el
    resultado final."""
    sesion, err = _comprobar_sesion(session_id)
    if err:
        return err
    if await _limpiar_si_vencida(sesion):
        minutos = SESSION_TIMEOUT_SECONDS // 60
        return {"error": f"Pasaron más de {minutos} minutos sin resolver el captcha. Empezá de nuevo."}

    page = sesion.page
    frame = _bframe(page)
    if frame is None:
        await sesion.cerrar()
        return {"error": "Se perdió el captcha. Empezá de nuevo."}

    verificar_btn = await frame.query_selector("#recaptcha-verify-button")
    if verificar_btn is not None:
        await verificar_btn.click()
    await asyncio.sleep(1.5)

    nuevo_desafio = await _leer_desafio(frame) if _bframe(page) else None
    if nuevo_desafio is not None:
        return {"session_id": session_id, "desafio": nuevo_desafio}

    resultado = await _resultado_o_error(page)
    await sesion.cerrar()
    if resultado is None:
        return {"error": "La DNRPA no respondió a tiempo. Probá de nuevo."}
    if not resultado.get("ok"):
        return {"error": resultado.get("error", "La DNRPA devolvió un error.")}
    return {"resultado": resultado}
