# -*- coding: utf-8 -*-
"""
clustering.py
-------------
Entrenamiento de algoritmos NO supervisados sobre los vectores de
características (scores por signo), evaluación de resultados y
guardado/carga del modelo con metadatos.

Algoritmos disponibles (justificación en el reporte de la Actividad 2):
    - K-Means          -> particional, rápido, fácil de interpretar
    - Jerárquico        -> genera dendrograma, muestra el PROCESO de
                            agrupamiento paso a paso (ideal para exponer)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

from core import MODELS_DIR, score_cols, elem_score_cols, mod_score_cols


ALGORITMOS = {
    "kmeans": "K-Means",
    "jerarquico": "Clusterización Jerárquica (Agglomerative)",
}


def preparar_features(df, modo="signo"):
    """
    modo='signo'     -> usa los 12 scores por signo (multi-clase, 12 categorías)
    modo='elemento'  -> usa los 4 scores por elemento (Fuego/Tierra/Aire/Agua)
    modo='modalidad' -> usa los 3 scores por modalidad (Cardinal/Fijo/Mutable)
    """
    if modo == "signo":
        cols = score_cols()
    elif modo == "elemento":
        cols = elem_score_cols()
    elif modo == "modalidad":
        cols = mod_score_cols()
    else:
        raise ValueError(f"Modo desconocido: {modo}")
    X = df[cols].astype(float).values
    return X, cols


def entrenar(df, algoritmo="kmeans", modo="signo", n_clusters=None):
    """
    Entrena el algoritmo elegido sobre el dataset y devuelve un dict
    con el modelo, las etiquetas de cluster, métricas y datos para graficar.
    """
    X, cols = preparar_features(df, modo)
    n_samples = X.shape[0]

    if n_clusters is None:
        n_clusters = {"signo": 12, "elemento": 4, "modalidad": 3}.get(modo, 2)
    n_clusters = max(2, min(n_clusters, max(2, n_samples - 1)))  # mínimo binario

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    proceso = {}

    if algoritmo == "kmeans":
        modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = modelo.fit_predict(X_scaled)
        proceso["inertia"] = float(modelo.inertia_)
        proceso["n_iter"] = int(modelo.n_iter_)
        proceso["centros"] = modelo.cluster_centers_.tolist()
    elif algoritmo == "jerarquico":
        modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = modelo.fit_predict(X_scaled)
        # matriz de enlace para dendrograma
        Z = linkage(X_scaled, method="ward")
        proceso["linkage_matrix"] = Z.tolist()
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    # Métrica de evaluación (calidad del clustering)
    try:
        sil = float(silhouette_score(X_scaled, labels)) if n_samples > n_clusters else None
    except Exception:
        sil = None

    # Pureza: comparar cluster vs etiqueta real (signo/elemento predominante),
    # aunque el algoritmo es no supervisado, esto sirve para EVALUAR qué tan
    # bien el agrupamiento natural coincide con las categorías reales.
    etiqueta_real_col = {
        "signo": "signo_predominante",
        "elemento": "elemento_predominante",
        "modalidad": "modalidad_predominante",
    }[modo]
    etiquetas_reales = df[etiqueta_real_col].tolist()
    pureza = calcular_pureza(labels, etiquetas_reales)

    # Reducción de dimensionalidad para visualizar en 2D (PCA)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X_scaled)
    varianza_explicada = pca.explained_variance_ratio_.tolist()

    resultado = {
        "algoritmo": algoritmo,
        "modo": modo,
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "labels": labels.tolist(),
        "silhouette_score": round(sil, 4) if sil is not None else None,
        "pureza": round(pureza, 4),
        "coords_2d": coords_2d.tolist(),
        "varianza_explicada_pca": [round(v, 4) for v in varianza_explicada],
        "proceso": proceso,
        "feature_cols": cols,
        "etiquetas_reales": etiquetas_reales,
    }
    return modelo, scaler, resultado


def calcular_pureza(labels, etiquetas_reales):
    """
    Pureza = para cada cluster, tomamos la etiqueta real mayoritaria y
    contamos cuántos puntos coinciden / total. Es la forma estándar y
    sencilla de "traducir" un resultado no supervisado a una métrica
    de acierto comparable.
    """
    import collections
    labels = np.array(labels)
    etiquetas_reales = np.array(etiquetas_reales)
    total = len(labels)
    if total == 0:
        return 0.0
    correctos = 0
    for c in set(labels):
        idx = labels == c
        etiquetas_c = etiquetas_reales[idx]
        if len(etiquetas_c) == 0:
            continue
        mayoritaria = collections.Counter(etiquetas_c).most_common(1)[0][1]
        correctos += mayoritaria
    return correctos / total


def dendrograma_dict(resultado):
    """Extrae la info de dendrograma en formato serializable (para Plotly)."""
    Z = np.array(resultado["proceso"]["linkage_matrix"])
    dendro = dendrogram(Z, no_plot=True)
    return {
        "icoord": dendro["icoord"],
        "dcoord": dendro["dcoord"],
        "ivl": dendro["ivl"],
    }


# ---------------------------------------------------------------------
# GUARDAR / CARGAR MODELO CON METADATOS
# ---------------------------------------------------------------------
def guardar_modelo(modelo, scaler, resultado, nombre_dataset="respuestas.csv"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{resultado['algoritmo']}_{resultado['modo']}_{ts}"
    model_path = os.path.join(MODELS_DIR, f"{slug}.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{slug}_meta.json")

    joblib.dump({"modelo": modelo, "scaler": scaler}, model_path)

    metadata = {
        "nombre_modelo": slug,
        "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algoritmo": ALGORITMOS.get(resultado["algoritmo"], resultado["algoritmo"]),
        "algoritmo_id": resultado["algoritmo"],
        "modo_clasificacion": resultado["modo"],
        "n_clusters": resultado["n_clusters"],
        "n_muestras_entrenamiento": resultado["n_samples"],
        "dataset_origen": nombre_dataset,
        "features_usadas": resultado["feature_cols"],
        "silhouette_score": resultado["silhouette_score"],
        "pureza": resultado["pureza"],
        "varianza_explicada_pca": resultado["varianza_explicada_pca"],
        "archivo_modelo": os.path.basename(model_path),
        "descripcion": (
            f"Modelo {ALGORITMOS.get(resultado['algoritmo'])} entrenado sobre "
            f"{resultado['n_samples']} perfiles, agrupando en {resultado['n_clusters']} "
            f"clusters usando modo '{resultado['modo']}'. Silhouette={resultado['silhouette_score']}, "
            f"Pureza={resultado['pureza']}."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return model_path, meta_path, metadata


def listar_modelos_guardados():
    modelos = []
    if not os.path.exists(MODELS_DIR):
        return modelos
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith("_meta.json"):
            with open(os.path.join(MODELS_DIR, fname), encoding="utf-8") as f:
                modelos.append(json.load(f))
    modelos.sort(key=lambda m: m["fecha_generacion"], reverse=True)
    return modelos


def cargar_modelo_por_nombre(nombre_modelo):
    model_path = os.path.join(MODELS_DIR, f"{nombre_modelo}.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{nombre_modelo}_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    bundle = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return bundle, metadata
