# -*- coding: utf-8 -*-
"""
app.py
------
Aplicativo de recolección de datos, análisis estadístico y clasificación
no supervisada de perfiles de personalidad / signo zodiacal.

DOS MODOS:
  - Modo Encuestado (público, el que se comparte por link):
        solo ve y contesta el cuestionario.
  - Modo Administrador (protegido con contraseña vía st.secrets):
        ve Datos, Estadística, Entrenamiento/Resultados y Modelos Guardados.
        Se activa agregando ?admin=1 a la URL, o desde el botón en la
        barra lateral.

Ejecutar con:
    streamlit run app.py

Configura tu contraseña de administrador en .streamlit/secrets.toml
(este archivo NO se sube a GitHub, ver secrets.toml.example):
    admin_password = "tu-clave-aqui"
"""
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from questions import QUESTIONS, SIGNOS, ELEMENTOS, MODALIDADES, DATO_CURIOSO_POR_SIGNO
from core import (
    nueva_fila, cargar_dataset, guardar_fila, reemplazar_dataset,
    score_cols, elem_score_cols, mod_score_cols,
    resumen_estadistico, reporte_a_texto,
)
from clustering import (
    entrenar, guardar_modelo, listar_modelos_guardados,
    cargar_modelo_por_nombre, dendrograma_dict, ALGORITMOS, MODELS_DIR,
)

st.set_page_config(page_title="Perfil Zodiacal - No Supervisado", page_icon="✨", layout="wide")


def inyectar_estilos():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,500;0,600;0,700;0,800;1,500&display=swap');

    :root {
        --bg-page: #FFF9F5;
        --bg-surface: #FFFFFF;
        --bg-surface-2: #F8ECFB;
        --lavender: #C9A7EB;
        --lavender-deep: #A97FDB;
        --pink: #F7A9C4;
        --peach: #FFC79A;
        --text-primary: #4A3B5C;
        --text-muted: #8B7A9E;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #FFF9F5 0%, #FCEFFA 45%, #FFF3EA 100%);
    }
    [data-testid="stHeader"] { background: rgba(255,249,245,0); }

    html, body, [class*="css"] { color: var(--text-primary); font-family: 'Montserrat', sans-serif; }

    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
    }
    h1 {
        background: linear-gradient(90deg, var(--lavender-deep), var(--pink) 55%, var(--peach));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        padding-bottom: 4px;
    }
    [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface-2);
        border-right: 1px solid rgba(169,127,219,0.15);
    }

    /* Tarjeta del formulario */
    [data-testid="stForm"] {
        background: var(--bg-surface);
        border: 1px solid rgba(201,167,235,0.35);
        border-radius: 24px;
        padding: 30px 32px;
        box-shadow: 0 10px 30px rgba(169,127,219,0.15);
    }

    /* Radios como tarjetitas seleccionables */
    div[role="radiogroup"] { gap: 4px; }
    div[role="radiogroup"] label {
        background: #FDF8FF;
        border: 1.5px solid rgba(201,167,235,0.3);
        border-radius: 14px;
        padding: 9px 16px !important;
        margin-bottom: 5px !important;
        transition: all 0.15s ease;
    }
    div[role="radiogroup"] label:hover {
        background: linear-gradient(90deg, rgba(201,167,235,0.15), rgba(247,169,196,0.15));
        border-color: var(--pink);
    }

    /* Botones */
    .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, var(--lavender-deep), var(--pink));
        color: #FFFFFF;
        border: none;
        border-radius: 16px;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 6px 16px rgba(169,127,219,0.35);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 20px rgba(247,169,196,0.5);
        color: #FFFFFF;
    }

    /* Inputs */
    input, textarea, [data-baseweb="select"] > div { border-radius: 12px !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(201,167,235,0.25); }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px 12px 0 0;
        color: var(--text-muted);
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: var(--lavender-deep) !important;
        border-bottom: 3px solid var(--pink) !important;
    }

    /* Métricas, tablas, dataframes */
    [data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: var(--bg-surface);
        border-radius: 16px;
        border: 1px solid rgba(201,167,235,0.25);
        padding: 8px;
    }
    [data-testid="stMetricValue"] { color: var(--lavender-deep) !important; font-weight: 800; }

    /* Alerts */
    [data-testid="stAlert"] { border-radius: 16px; }

    /* Expander */
    details {
        background: var(--bg-surface);
        border-radius: 14px;
        border: 1px solid rgba(201,167,235,0.3);
    }

    /* Divider */
    hr { border-color: rgba(201,167,235,0.3) !important; }

    /* Transiciones suaves generales */
    button, [data-baseweb="select"], [data-baseweb="input"], .stSlider, [data-testid="stCheckbox"] {
        transition: all 0.15s ease;
    }

    /* Sliders (pestaña Entrenamiento) */
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--lavender-deep) !important;
        box-shadow: 0 2px 8px rgba(169,127,219,0.4);
    }
    [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, var(--lavender-deep), var(--pink)) !important;
    }

    /* Checkboxes */
    [data-testid="stCheckbox"] label span {
        border-radius: 6px !important;
    }
    [data-testid="stCheckbox"]:hover { opacity: 0.85; }

    /* Selects / dropdowns */
    [data-baseweb="select"] > div {
        border-color: rgba(201,167,235,0.4) !important;
        background: #FDF8FF !important;
    }
    [data-baseweb="select"] > div:hover {
        border-color: var(--pink) !important;
    }
    [data-baseweb="popover"] li:hover {
        background: rgba(247,169,196,0.15) !important;
    }

    /* Cargador de archivos (pestaña Datos) */
    [data-testid="stFileUploaderDropzone"] {
        background: #FDF8FF;
        border: 2px dashed rgba(201,167,235,0.5);
        border-radius: 16px;
        transition: border-color 0.15s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--pink);
    }

    /* Bloques JSON (Modelos Guardados) */
    [data-testid="stJson"] {
        background: #FDF8FF !important;
        border-radius: 14px;
        border: 1px solid rgba(201,167,235,0.3);
        padding: 10px;
    }

    /* Radio buttons fuera del cuestionario (ej. selector de algoritmo) más "pill" */
    div[role="radiogroup"] label[data-baseweb="radio"] {
        cursor: pointer;
    }

    /* Contenedor principal con más aire, look más "pro" */
    .block-container { padding-top: 2.2rem; max-width: 1200px; }

    /* Scroll suave y bonito en tablas */
    [data-testid="stDataFrame"] * { scrollbar-width: thin; }

    /* Progress / spinner con el mismo acento */
    .stSpinner > div { border-top-color: var(--lavender-deep) !important; }

    /* ============ RESPONSIVE: MÓVIL ============ */
    @media (max-width: 640px) {
        .block-container { padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.7rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.05rem !important; }
        [data-testid="stForm"] { padding: 18px 16px; border-radius: 18px; }
        div[role="radiogroup"] label {
            padding: 12px 14px !important;
            font-size: 0.95rem;
            min-height: 44px;
        }
        .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
            width: 100%;
            padding: 0.75rem 1rem;
            font-size: 1rem;
        }
        [data-testid="column"] { min-width: 100% !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.8rem; padding: 6px 8px; }
    }
    </style>
    """, unsafe_allow_html=True)


inyectar_estilos()

# ---------------------------------------------------------------------
# ESTADO
# ---------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = cargar_dataset()
if "pagina" not in st.session_state:
    st.session_state.pagina = 1
if "resultado_entrenamiento" not in st.session_state:
    st.session_state.resultado_entrenamiento = None
if "modelo_entrenado" not in st.session_state:
    st.session_state.modelo_entrenado = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


# ---------------------------------------------------------------------
# CONTROL DE ACCESO ADMIN
# ---------------------------------------------------------------------
def clave_admin_configurada():
    try:
        return st.secrets["admin_password"]
    except Exception:
        return None


def render_login_admin():
    st.sidebar.markdown("### 🔒 Acceso administrador")
    clave_real = clave_admin_configurada()
    if clave_real is None:
        st.sidebar.warning(
            "No hay contraseña configurada todavía. Crea `.streamlit/secrets.toml` "
            "con `admin_password = \"tu-clave\"` (ver `secrets.toml.example`)."
        )
        return
    intento = st.sidebar.text_input("Contraseña", type="password", key="clave_intento")
    if st.sidebar.button("Entrar"):
        if intento == clave_real:
            st.session_state.is_admin = True
            st.rerun()
        else:
            st.sidebar.error("Contraseña incorrecta.")


# ¿Se pidió modo admin por URL? (?admin=1)
query_params = st.query_params
pidio_admin_por_url = query_params.get("admin") == "1"

if not st.session_state.is_admin:
    if pidio_admin_por_url:
        render_login_admin()
    else:
        with st.sidebar.expander("¿Eres el administrador?"):
            render_login_admin()
else:
    st.sidebar.success("Sesión de administrador activa ✅")
    if st.sidebar.button("Cerrar sesión admin"):
        st.session_state.is_admin = False
        st.rerun()


# =======================================================================
# FUNCIÓN: RENDERIZAR EL CUESTIONARIO (se usa en ambos modos)
# =======================================================================
def render_cuestionario():
    st.subheader("Cuestionario de perfil")
    st.write(
        f"Responde las **{len(QUESTIONS)} preguntas**. Cada respuesta suma puntos "
        "a uno o varios signos zodiacales según su asociación astrológica."
    )

    with st.form("form_cuestionario", clear_on_submit=True):
        c_nom, c_edad = st.columns([2, 1])
        with c_nom:
            nombre = st.text_input("Nombre")
        with c_edad:
            edad = st.number_input("Edad", min_value=0, max_value=120, step=1, value=None, placeholder="Edad")

        c_genero, c_signo = st.columns(2)
        with c_genero:
            genero = st.selectbox("Género", ["Masculino", "Femenino", "Prefiero no decir"], index=None, placeholder="Selecciona...")
        with c_signo:
            signo_real = st.selectbox("¿Cuál es tu signo zodiacal real?", SIGNOS, index=None, placeholder="Selecciona...")

        st.divider()
        respuestas = {}
        for q in QUESTIONS:
            opciones_texto = [opt["text"] for opt in q["options"]]
            seleccion = st.radio(q["text"], opciones_texto, key=f"radio_{q['id']}", index=None)
            if seleccion is not None:
                respuestas[q["id"]] = opciones_texto.index(seleccion)
        enviado = st.form_submit_button("Enviar respuestas", type="primary")

    if enviado:
        faltantes = [q["text"] for q in QUESTIONS if q["id"] not in respuestas]
        if not nombre or not nombre.strip():
            st.error("Escribe tu nombre para poder enviar tus respuestas.")
        elif faltantes:
            st.error(f"Te faltó responder {len(faltantes)} pregunta(s). Completa todas antes de enviar.")
        else:
            fila = nueva_fila(nombre, edad, genero, signo_real, respuestas)
            st.session_state.df = guardar_fila(fila)

            st.success(
                f"¡Listo, {nombre}! Según tus respuestas, tu perfil es **{fila['signo_predominante']}** "
                f"(elemento: **{fila['elemento_predominante']}**, "
                f"modalidad: **{fila['modalidad_predominante']}**)."
            )
            if signo_real:
                if fila["coincide_signo_real"]:
                    st.info(
                        f"✨ ¡Coincide con tu signo real (**{signo_real}**)! Tu forma de pensar y actuar "
                        f"encaja con los rasgos clásicos de tu signo."
                    )
                else:
                    st.info(
                        f"🤔 Tu signo real es **{signo_real}**, pero tus respuestas se parecen más a "
                        f"**{fila['signo_predominante']}**. No pasa nada — el cuestionario mide rasgos de "
                        f"personalidad (cómo decides, te comunicas, reaccionas al estrés, etc.), no tu fecha "
                        f"de nacimiento. Muchos signos comparten elemento o modalidad, así que es normal que "
                        f"tu forma de ser se parezca más a otro signo del zodiaco."
                    )
            dato = DATO_CURIOSO_POR_SIGNO.get(fila["signo_predominante"])
            if dato:
                st.markdown(
                    f"""<div style="background:linear-gradient(90deg, rgba(201,167,235,0.18), rgba(247,169,196,0.18));
                    border:1px solid rgba(201,167,235,0.4); border-radius:16px; padding:16px 20px; margin-top:8px;">
                    <b>💫 {fila['signo_predominante']} en tendencia:</b><br>{dato}
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.balloons()


# =======================================================================
# MODO ENCUESTADO (público) — solo cuestionario, sin más
# =======================================================================
if not st.session_state.is_admin:
    st.title("✨ Descubre tu Perfil Zodiacal")
    st.caption("Responde el cuestionario. Tus respuestas ayudan a un proyecto de análisis de datos.")
    render_cuestionario()
    st.stop()


# =======================================================================
# MODO ADMINISTRADOR — app completa
# =======================================================================
st.title("✨ Perfil Zodiacal — Análisis No Supervisado (Admin)")
st.caption(
    "Extracción de Conocimiento en Base de Datos · Unidad IV · "
    "Recolección de datos, estadística propia y clustering (K-Means / Jerárquico)"
)

tabs = st.tabs([
    "📋 Cuestionario",
    "🗂️ Datos",
    "📊 Estadística",
    "🤖 Entrenamiento y Resultados",
    "💾 Modelos Guardados",
])

# ---- TAB 1: Cuestionario (el admin también puede probarlo) ----
with tabs[0]:
    render_cuestionario()

# ---- TAB 2: Datos (carga, visualización paginada, filtros, modo, descarga) ----
with tabs[1]:
    st.subheader("Explorador de datos")

    col_up, col_info = st.columns([1, 2])
    with col_up:
        archivo = st.file_uploader("Cargar dataset (.csv)", type=["csv"])
        if archivo is not None:
            df_cargado = pd.read_csv(archivo)
            st.session_state.df = df_cargado
            reemplazar_dataset(df_cargado)
            st.success(f"Dataset cargado: {len(df_cargado)} filas.")

    df = st.session_state.df
    with col_info:
        st.metric("Total de registros", len(df))

    if df.empty:
        st.info("Aún no hay datos. Comparte el link del cuestionario o carga un CSV.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            filtro_tipo = st.selectbox("Filtrar por", ["Todos", "Signo", "Elemento", "Modalidad"])
        with c2:
            if filtro_tipo == "Signo":
                valor_filtro = st.selectbox("Signo", ["Todos"] + SIGNOS)
            elif filtro_tipo == "Elemento":
                valor_filtro = st.selectbox("Elemento", ["Todos"] + ELEMENTOS)
            elif filtro_tipo == "Modalidad":
                valor_filtro = st.selectbox("Modalidad", ["Todos"] + MODALIDADES)
            else:
                valor_filtro = "Todos"
        with c3:
            modo_vista = st.radio("Modo de vista", ["Cualitativo", "Cuantitativo"], horizontal=True)
        with c4:
            por_pagina = st.selectbox("Filas por página", [20, 50, 100], index=0)

        df_filtrado = df.copy()
        col_por_filtro = {
            "Signo": "signo_predominante",
            "Elemento": "elemento_predominante",
            "Modalidad": "modalidad_predominante",
        }
        if filtro_tipo in col_por_filtro and valor_filtro != "Todos":
            df_filtrado = df_filtrado[df_filtrado[col_por_filtro[filtro_tipo]] == valor_filtro]

        if modo_vista == "Cualitativo":
            cols_mostrar = ["id", "fecha", "nombre", "edad", "genero", "signo_real",
                             "signo_predominante", "elemento_predominante",
                             "modalidad_predominante", "coincide_signo_real"]
            cols_mostrar += [c for c in df_filtrado.columns if c.startswith("resp_")]
        else:
            cols_mostrar = ["id", "fecha", "nombre", "edad", "genero", "signo_real"] + score_cols() + elem_score_cols() + mod_score_cols()
        cols_mostrar = [c for c in cols_mostrar if c in df_filtrado.columns]

        total_filas = len(df_filtrado)
        total_paginas = max(1, -(-total_filas // por_pagina))
        st.session_state.pagina = min(st.session_state.pagina, total_paginas)

        pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
        with pcol1:
            if st.button("◀ Anterior") and st.session_state.pagina > 1:
                st.session_state.pagina -= 1
        with pcol3:
            if st.button("Siguiente ▶") and st.session_state.pagina < total_paginas:
                st.session_state.pagina += 1
        with pcol2:
            st.write(f"Página **{st.session_state.pagina}** de **{total_paginas}** "
                      f"({total_filas} registros filtrados)")

        ini = (st.session_state.pagina - 1) * por_pagina
        fin = ini + por_pagina
        st.dataframe(df_filtrado[cols_mostrar].iloc[ini:fin], use_container_width=True)

        st.download_button(
            "⬇️ Descargar datos filtrados (CSV)",
            data=df_filtrado[cols_mostrar].to_csv(index=False).encode("utf-8"),
            file_name="datos_filtrados.csv",
            mime="text/csv",
        )

# ---- TAB 3: Estadística ----
with tabs[2]:
    st.subheader("Estadística básica (algoritmos propios)")
    df = st.session_state.df

    if df.empty:
        st.info("No hay datos aún para generar estadística.")
    else:
        reporte = resumen_estadistico(df)

        if "pct_coincidencia_signo_real" in reporte:
            st.metric(
                "Coincidencia signo real vs. calculado por el cuestionario",
                f"{reporte['pct_coincidencia_signo_real']}%",
                help=f"Basado en {reporte['n_con_signo_real']} personas que indicaron su signo real.",
            )

        mostrar_frecuencias = st.checkbox("Mostrar frecuencias (signo / elemento / modalidad)", value=True)
        mostrar_graficas = st.checkbox("Mostrar gráficas (histograma / pastel)", value=True)
        mostrar_numerica = st.checkbox("Mostrar estadística numérica detallada", value=True)

        if mostrar_frecuencias:
            st.markdown("#### Distribución por categoría")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.write("**Frecuencia por Signo**")
                st.table(pd.DataFrame(reporte["frecuencia_signo"]).T)
            with fc2:
                st.write("**Frecuencia por Elemento**")
                st.table(pd.DataFrame(reporte["frecuencia_elemento"]).T)
            with fc3:
                st.write("**Frecuencia por Modalidad**")
                st.table(pd.DataFrame(reporte["frecuencia_modalidad"]).T)

        if mostrar_graficas:
            st.markdown("#### Gráficas")
            g1, g2, g3 = st.columns(3)
            with g1:
                fig_hist = px.histogram(
                    df, x="signo_predominante", color="signo_predominante",
                    title="Histograma — Signo predominante",
                    category_orders={"signo_predominante": SIGNOS},
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with g2:
                fig_pie = px.pie(
                    df, names="elemento_predominante",
                    title="Distribución por Elemento",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with g3:
                fig_pie_mod = px.pie(
                    df, names="modalidad_predominante",
                    title="Distribución por Modalidad",
                )
                st.plotly_chart(fig_pie_mod, use_container_width=True)

        if mostrar_numerica:
            st.markdown("#### Estadística numérica por variable (media, mediana, moda, varianza, desv. estándar)")
            st.dataframe(pd.DataFrame(reporte["estadistica_numerica"]).T, use_container_width=True)

        st.download_button(
            "⬇️ Descargar reporte estadístico (TXT)",
            data=reporte_a_texto(reporte).encode("utf-8"),
            file_name="reporte_estadistico.txt",
            mime="text/plain",
        )
        st.download_button(
            "⬇️ Descargar reporte estadístico (JSON)",
            data=json.dumps(reporte, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="reporte_estadistico.json",
            mime="application/json",
        )

# ---- TAB 4: Entrenamiento y Resultados ----
with tabs[3]:
    st.subheader("Entrenamiento del algoritmo no supervisado")
    df = st.session_state.df

    MIN_MUESTRAS = 6
    if len(df) < MIN_MUESTRAS:
        st.warning(f"Necesitas al menos {MIN_MUESTRAS} registros para entrenar (tienes {len(df)}).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            algoritmo = st.radio(
                "Elige el algoritmo",
                options=list(ALGORITMOS.keys()),
                format_func=lambda k: ALGORITMOS[k],
            )
        with c2:
            modo = st.radio(
                "Modo de clasificación",
                options=["signo", "elemento", "modalidad"],
                format_func=lambda m: {
                    "signo": "Por Signo (12 categorías)",
                    "elemento": "Por Elemento (4 categorías)",
                    "modalidad": "Por Modalidad (3 categorías)",
                }[m],
            )
        with c3:
            default_k = {"signo": 12, "elemento": 4, "modalidad": 3}[modo]
            n_clusters = st.slider("Número de clusters (mínimo 2 = binario)", 2, 12, default_k)

        if st.button("🚀 Entrenar modelo", type="primary"):
            with st.spinner("Entrenando..."):
                modelo, scaler, resultado = entrenar(df, algoritmo=algoritmo, modo=modo, n_clusters=n_clusters)
                st.session_state.resultado_entrenamiento = resultado
                st.session_state.modelo_entrenado = (modelo, scaler)
            st.success("¡Entrenamiento completo!")

    resultado = st.session_state.resultado_entrenamiento
    if resultado is not None:
        st.markdown("---")
        st.markdown(f"### Resultados — {ALGORITMOS[resultado['algoritmo']]}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters generados", resultado["n_clusters"])
        m2.metric("Silhouette score", resultado["silhouette_score"])
        m3.metric("Pureza vs. etiqueta real", f"{resultado['pureza']*100:.1f}%")
        m4.metric("Muestras usadas", resultado["n_samples"])

        st.markdown("#### Proceso del algoritmo")
        if resultado["algoritmo"] == "kmeans":
            st.write(f"- Inercia final: `{resultado['proceso']['inertia']:.3f}`")
            st.write(f"- Iteraciones hasta converger: `{resultado['proceso']['n_iter']}`")
            st.write("- K-Means inicializa centroides, asigna cada punto al centroide "
                      "más cercano, recalcula centroides y repite hasta converger.")
        else:
            st.write("- Clusterización jerárquica aglomerativa (enlace `ward`): cada punto "
                      "empieza como su propio cluster y se van fusionando los más cercanos "
                      "hasta llegar al número de clusters definido. Dendrograma abajo ⬇️")
            dendro = dendrograma_dict(resultado)
            fig_dendro = go.Figure()
            for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
                fig_dendro.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="royalblue"), showlegend=False))
            fig_dendro.update_layout(title="Dendrograma", xaxis_title="Muestras", yaxis_title="Distancia")
            st.plotly_chart(fig_dendro, use_container_width=True)

        st.markdown("#### Visualización 2D (reducción de dimensionalidad con PCA)")
        st.caption(f"Varianza explicada por los 2 componentes: {resultado['varianza_explicada_pca']}")
        df_plot = pd.DataFrame(resultado["coords_2d"], columns=["PC1", "PC2"])
        df_plot["cluster"] = [str(c) for c in resultado["labels"]]
        df_plot["etiqueta_real"] = resultado["etiquetas_reales"]
        fig_pca = px.scatter(
            df_plot, x="PC1", y="PC2", color="cluster", symbol="etiqueta_real",
            title="Clusters encontrados por el algoritmo (PCA 2D)", hover_data=["etiqueta_real"],
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        df_resultados = df.copy().reset_index(drop=True)
        df_resultados["cluster_asignado"] = resultado["labels"]
        st.markdown("#### Tabla de resultados (cada registro con su cluster asignado)")
        st.dataframe(
            df_resultados[["id", "nombre", "edad", "signo_predominante", "elemento_predominante",
                            "modalidad_predominante", "cluster_asignado"]],
            use_container_width=True,
        )

        colD1, colD2 = st.columns(2)
        with colD1:
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=df_resultados.to_csv(index=False).encode("utf-8"),
                file_name="resultados_clustering.csv",
                mime="text/csv",
            )
        with colD2:
            resumen_json = {k: v for k, v in resultado.items() if k not in ("labels",)}
            st.download_button(
                "⬇️ Descargar resumen del proceso (JSON)",
                data=json.dumps(resumen_json, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
                file_name="proceso_algoritmo.json",
                mime="application/json",
            )

        st.markdown("---")
        st.markdown("#### Guardar modelo entrenado")
        if st.button("💾 Guardar modelo con metadatos"):
            modelo, scaler = st.session_state.modelo_entrenado
            model_path, meta_path, metadata = guardar_modelo(modelo, scaler, resultado)
            st.success(f"Modelo guardado como `{metadata['nombre_modelo']}`")
            st.json(metadata)

# ---- TAB 5: Modelos Guardados (acceso vía metadatos) ----
with tabs[4]:
    st.subheader("Modelos guardados")
    modelos = listar_modelos_guardados()

    if not modelos:
        st.info("Aún no hay modelos guardados. Entrena y guarda uno en la pestaña anterior.")
    else:
        nombres = [m["nombre_modelo"] for m in modelos]
        seleccionado = st.selectbox("Selecciona un modelo (por metadato)", nombres)
        meta = next(m for m in modelos if m["nombre_modelo"] == seleccionado)

        st.markdown("#### Metadatos del modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("Algoritmo", meta["algoritmo"])
        c2.metric("Silhouette", meta["silhouette_score"])
        c3.metric("Pureza", f"{meta['pureza']*100:.1f}%")
        st.write(f"**Fecha de generación:** {meta['fecha_generacion']}")
        st.write(f"**Modo de clasificación:** {meta['modo_clasificacion']}")
        st.write(f"**N° de clusters:** {meta['n_clusters']}")
        st.write(f"**Muestras de entrenamiento:** {meta['n_muestras_entrenamiento']}")
        st.write(f"**Descripción:** {meta['descripcion']}")
        st.json(meta)

        model_file = os.path.join(MODELS_DIR, meta["archivo_modelo"])
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                st.download_button(
                    "⬇️ Descargar archivo del modelo (.pkl)",
                    data=f.read(),
                    file_name=meta["archivo_modelo"],
                    mime="application/octet-stream",
                )
        st.download_button(
            "⬇️ Descargar metadatos (JSON)",
            data=json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{meta['nombre_modelo']}_meta.json",
            mime="application/json",
        )
