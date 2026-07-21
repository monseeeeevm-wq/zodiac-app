# -*- coding: utf-8 -*-
"""
core.py
-------
Lógica de negocio: transformar respuestas del cuestionario en datos
cuantitativos/cualitativos, y estadística descriptiva implementada
"a mano" (algoritmos propios, sin depender de pandas.describe()).
"""
import os
import json
import math
import statistics
from datetime import datetime

import pandas as pd

from questions import (
    QUESTIONS, SIGNOS, ELEMENTOS, ELEMENTO_POR_SIGNO,
    MODALIDADES, MODALIDAD_POR_SIGNO,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(DATA_DIR, "respuestas.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1) TRANSFORMAR RESPUESTAS -> VECTOR (aquí se aplica el peso aritmético)
# ---------------------------------------------------------------------
def respuestas_a_vector(respuestas):
    """
    respuestas: dict {q_id: opcion_index (0-3)}
    Devuelve: (vector_dict {signo: score}, etiquetas_dict {q_id: texto_opcion})
    """
    vector = {s: 0.0 for s in SIGNOS}
    etiquetas = {}
    for q in QUESTIONS:
        idx = respuestas.get(q["id"])
        if idx is None:
            continue
        opcion = q["options"][idx]
        etiquetas[q["id"]] = opcion["text"]
        for s, w in opcion["weights"].items():
            vector[s] += w
    return vector, etiquetas


def vector_a_elemento_score(vector):
    """Agrega el score por signo al score por elemento (Fuego/Tierra/Aire/Agua)."""
    elem_score = {e: 0.0 for e in ELEMENTOS}
    for signo, score in vector.items():
        elem_score[ELEMENTO_POR_SIGNO[signo]] += score
    return elem_score


def vector_a_modalidad_score(vector):
    """Agrega el score por signo al score por modalidad (Cardinal/Fijo/Mutable)."""
    mod_score = {m: 0.0 for m in MODALIDADES}
    for signo, score in vector.items():
        mod_score[MODALIDAD_POR_SIGNO[signo]] += score
    return mod_score


def nueva_fila(nombre, edad, genero, signo_real, respuestas):
    """Construye una fila completa lista para agregar al dataset."""
    vector, etiquetas = respuestas_a_vector(respuestas)
    elem_score = vector_a_elemento_score(vector)
    mod_score = vector_a_modalidad_score(vector)

    signo_predominante = max(vector, key=vector.get)
    elemento_predominante = max(elem_score, key=elem_score.get)
    modalidad_predominante = max(mod_score, key=mod_score.get)

    fila = {
        "id": "resp_" + datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nombre": nombre or "Anónimo",
        "edad": edad,
        "genero": genero,
        "signo_real": signo_real,
    }
    # Cuantitativo: score numérico por signo
    for s in SIGNOS:
        fila[f"score_{s}"] = round(vector[s], 4)
    for e in ELEMENTOS:
        fila[f"score_elem_{e}"] = round(elem_score[e], 4)
    for m in MODALIDADES:
        fila[f"score_mod_{m}"] = round(mod_score[m], 4)
    # Cualitativo: etiquetas
    fila["signo_predominante"] = signo_predominante
    fila["elemento_predominante"] = elemento_predominante
    fila["modalidad_predominante"] = modalidad_predominante
    fila["coincide_signo_real"] = (signo_real == signo_predominante) if signo_real else None
    for q in QUESTIONS:
        fila[f"resp_{q['id']}"] = etiquetas.get(q["id"], "")
    return fila


# ---------------------------------------------------------------------
# 2) PERSISTENCIA DEL DATASET
# ---------------------------------------------------------------------
def cargar_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, dtype={"id": str})
        df["id"] = "resp_" + df["id"].astype(str).str.replace("resp_", "", regex=False)
        return df
    return pd.DataFrame()


def guardar_fila(fila):
    df = cargar_dataset()
    nueva = pd.DataFrame([fila])
    df = pd.concat([df, nueva], ignore_index=True) if not df.empty else nueva
    df.to_csv(DATASET_PATH, index=False)
    return df


def reemplazar_dataset(df_nuevo):
    df_nuevo.to_csv(DATASET_PATH, index=False)


def score_cols():
    return [f"score_{s}" for s in SIGNOS]


def elem_score_cols():
    return [f"score_elem_{e}" for e in ELEMENTOS]


def mod_score_cols():
    return [f"score_mod_{m}" for m in MODALIDADES]


# ---------------------------------------------------------------------
# 3) ESTADÍSTICA "PROPIA" -- implementada manualmente (no solo .describe())
# ---------------------------------------------------------------------
def media_manual(valores):
    n = len(valores)
    return sum(valores) / n if n else 0.0


def varianza_manual(valores):
    n = len(valores)
    if n < 2:
        return 0.0
    m = media_manual(valores)
    return sum((x - m) ** 2 for x in valores) / (n - 1)


def desviacion_manual(valores):
    return math.sqrt(varianza_manual(valores))


def moda_manual(valores):
    try:
        return statistics.mode(valores)
    except statistics.StatisticsError:
        return valores[0] if valores else None


def mediana_manual(valores):
    v = sorted(valores)
    n = len(v)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (v[mid - 1] + v[mid]) / 2
    return v[mid]


def frecuencia_manual(valores):
    """Tabla de frecuencias absolutas y relativas para una columna categórica."""
    n = len(valores)
    conteo = {}
    for v in valores:
        conteo[v] = conteo.get(v, 0) + 1
    return {k: {"frecuencia": v, "relativa": round(v / n, 4) if n else 0}
            for k, v in sorted(conteo.items(), key=lambda x: -x[1])}


def resumen_estadistico(df):
    """
    Genera un reporte estadístico propio (no usa df.describe()):
    - Distribución de signo y elemento (frecuencias)
    - Media, mediana, moda, varianza, desviación estándar por cada score numérico
    """
    reporte = {"n_muestras": len(df), "generado": datetime.now().isoformat()}

    if df.empty:
        return reporte

    reporte["frecuencia_signo"] = frecuencia_manual(df["signo_predominante"].tolist())
    reporte["frecuencia_elemento"] = frecuencia_manual(df["elemento_predominante"].tolist())
    reporte["frecuencia_modalidad"] = frecuencia_manual(df["modalidad_predominante"].tolist())

    if "coincide_signo_real" in df.columns:
        validos = df["coincide_signo_real"].dropna()
        if len(validos) > 0:
            reporte["pct_coincidencia_signo_real"] = round(float(validos.mean()) * 100, 1)
            reporte["n_con_signo_real"] = int(len(validos))

    numericas = {}
    for col in score_cols() + elem_score_cols() + mod_score_cols():
        if col in df.columns:
            valores = df[col].astype(float).tolist()
            numericas[col] = {
                "media": round(media_manual(valores), 4),
                "mediana": round(mediana_manual(valores), 4),
                "moda": moda_manual(valores),
                "varianza": round(varianza_manual(valores), 4),
                "desviacion_std": round(desviacion_manual(valores), 4),
                "min": round(min(valores), 4) if valores else 0,
                "max": round(max(valores), 4) if valores else 0,
            }
    reporte["estadistica_numerica"] = numericas
    return reporte


def reporte_a_texto(reporte):
    """Convierte el reporte a texto plano descargable."""
    lines = []
    lines.append("REPORTE ESTADÍSTICO - PERFIL ZODIACAL")
    lines.append(f"Generado: {reporte.get('generado')}")
    lines.append(f"Número de muestras: {reporte.get('n_muestras')}")
    lines.append("")
    lines.append("--- Frecuencia por Signo ---")
    for k, v in reporte.get("frecuencia_signo", {}).items():
        lines.append(f"  {k}: {v['frecuencia']} ({v['relativa']*100:.1f}%)")
    lines.append("")
    lines.append("--- Frecuencia por Elemento ---")
    for k, v in reporte.get("frecuencia_elemento", {}).items():
        lines.append(f"  {k}: {v['frecuencia']} ({v['relativa']*100:.1f}%)")
    lines.append("")
    lines.append("--- Frecuencia por Modalidad ---")
    for k, v in reporte.get("frecuencia_modalidad", {}).items():
        lines.append(f"  {k}: {v['frecuencia']} ({v['relativa']*100:.1f}%)")
    lines.append("")
    lines.append("--- Estadística numérica por variable ---")
    for col, stats_ in reporte.get("estadistica_numerica", {}).items():
        lines.append(f"  {col}:")
        for k, v in stats_.items():
            lines.append(f"      {k}: {v}")
    return "\n".join(lines)
