# -*- coding: utf-8 -*-
"""
sheets_sync.py
--------------
Guarda y lee las respuestas del cuestionario en una Google Sheet, para que
los datos sean persistentes incluso si Streamlit Cloud reinicia el servidor
(el CSV local NO es confiable en la nube gratuita).

Si no hay credenciales configuradas en st.secrets, todas las funciones
regresan valores "vacíos" sin tronar, y la app cae automáticamente al CSV
local (ver core.py) — así en tu compu, sin configurar nada, sigue
funcionando exactamente igual que antes.
"""
import streamlit as st
import pandas as pd

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
NOMBRE_HOJA = "respuestas"


def is_configured():
    try:
        return "gcp_service_account" in st.secrets and "sheet_id" in st.secrets
    except Exception:
        return False


def _get_worksheet():
    import gspread
    from google.oauth2.service_account import Credentials

    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    sh = client.open_by_key(st.secrets["sheet_id"])
    try:
        ws = sh.worksheet(NOMBRE_HOJA)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=NOMBRE_HOJA, rows=2000, cols=80)
    return ws


def append_row(fila: dict) -> bool:
    """Agrega una respuesta nueva a la hoja. Regresa True si se guardó bien."""
    if not is_configured():
        return False
    try:
        ws = _get_worksheet()
        primera_celda = ws.acell("A1").value
        # La primera columna de toda fila SIEMPRE se llama "id" (ver core.py).
        # Si A1 no dice "id", significa que falta el encabezado (o está roto),
        # así que lo insertamos arriba de todo, sin importar qué había antes.
        if primera_celda != "id":
            ws.insert_row(list(fila.keys()), index=1)
        ws.append_row([str(v) if v is not None else "" for v in fila.values()])
        return True
    except Exception as e:
        st.warning(f"No se pudo guardar en Google Sheets (se usará respaldo local): {e}")
        return False


def read_all():
    """Regresa un DataFrame con todas las respuestas, o None si falla/no está configurado."""
    if not is_configured():
        return None
    try:
        ws = _get_worksheet()
        valores = ws.get_all_values()
        if not valores or len(valores) < 2:
            return pd.DataFrame()
        headers = valores[0]
        filas = valores[1:]
        headers_unicos = []
        conteo = {}
        for h in headers:
            h = h if h else "col_vacia"
            if h in conteo:
                conteo[h] += 1
                headers_unicos.append(f"{h}_{conteo[h]}")
            else:
                conteo[h] = 0
                headers_unicos.append(h)
        return pd.DataFrame(filas, columns=headers_unicos)
    except Exception as e:
        st.warning(f"No se pudo leer Google Sheets (se usará respaldo local): {e}")
        return None


def reemplazar_todo(df: pd.DataFrame) -> bool:
    """Sobrescribe toda la hoja con un DataFrame nuevo (usado al cargar un CSV externo)."""
    if not is_configured():
        return False
    try:
        ws = _get_worksheet()
        ws.clear()
        ws.append_row(list(df.columns))
        for _, row in df.iterrows():
            ws.append_row([str(v) if pd.notna(v) else "" for v in row.tolist()])
        return True
    except Exception as e:
        st.warning(f"No se pudo sobrescribir Google Sheets: {e}")
        return False