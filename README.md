# ✨ Perfil Zodiacal — Análisis No Supervisado

App de recolección de datos, estadística propia y clustering (K-Means /
Jerárquico) para perfilar personalidad ↔ signo zodiacal.
Proyecto: Extracción de Conocimiento en Base de Datos — Unidad IV.

## 1. Instalación

```bash
cd zodiac_app
pip install -r requirements.txt
```

## 2. Ejecutar la app

```bash
streamlit run app.py
```

Se abrirá en tu navegador (normalmente `http://localhost:8501`).

## 3. Modo Encuestado vs. Modo Administrador

La misma app tiene dos caras:

- **Modo Encuestado** (lo que ve la gente al abrir el link): solo el
  cuestionario. Nada de estadística ni entrenamiento.
- **Modo Administrador** (solo tú): las 5 pestañas completas.

Para entrar como admin:
1. Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml`
   y pon tu contraseña real (este archivo **no se sube a GitHub**, está en
   `.gitignore`).
2. Abre la app y en la barra lateral usa el expander **"¿Eres el
   administrador?"**, o entra directo con `tuapp.streamlit.app/?admin=1`.

## 4. Publicar la app (Streamlit Community Cloud, gratis)

1. Crea un repo en tu GitHub y sube esta carpeta (el `.gitignore` ya evita
   subir datos, modelos y tu contraseña).
2. Ve a https://share.streamlit.io → "New app" → conecta tu repo →
   selecciona `app.py` como archivo principal → Deploy.
3. En el dashboard de esa app, ve a **Settings → Secrets** y pega:
   ```
   admin_password = "tu-clave-real"
   ```
4. Comparte el link normal (`tuapp.streamlit.app`) con la gente para que
   conteste el cuestionario. Tú entras con `tuapp.streamlit.app/?admin=1`.

⚠️ Nota: en Streamlit Cloud el sistema de archivos no es 100% persistente
entre reinicios del contenedor. Para la entrega/exposición esto no es
problema (vas a estar recolectando y analizando en la misma sesión), pero
si quieres persistencia a largo plazo lo ideal a futuro sería mover
`data/respuestas.csv` a una base de datos real (ej. Google Sheets API o
una BD tipo Supabase) — lo dejamos como posible mejora, no es necesario
para la entrega.

## 5. Estructura del proyecto

```
zodiac_app/
├── app.py            # App Streamlit (UI, 5 pestañas)
├── questions.py       # Banco de 18 preguntas + metodología de pesos
├── core.py            # Vectores de respuesta + estadística "propia"
├── clustering.py       # K-Means / Jerárquico + guardado de modelo con metadatos
├── test_flow.py         # Script de prueba end-to-end (QA, opcional)
├── data/               # Aquí se guarda respuestas.csv (se crea solo)
├── models/             # Aquí se guardan los modelos .pkl + *_meta.json
└── requirements.txt
```

## 6. Flujo de uso (para tu exposición)

0. **Los 12 signos** se pueden agrupar de 3 formas distintas:
   - Por **Signo** (12 categorías)
   - Por **Elemento**: Fuego / Tierra / Aire / Agua (4 categorías)
   - Por **Modalidad**: Cardinal / Fijo / Mutable (3 categorías) — esta es
     la agrupación "extra interesante" que quizás no todos conocen.
1. **Cuestionario** (modo Encuestado) → varias personas contestan las 18
   preguntas usando el link público.
2. **Datos** (modo Admin) → se pueden ver paginados (20/50/100), filtrar
   por signo, elemento o modalidad, cambiar entre vista
   Cualitativa/Cuantitativa, cargar un CSV externo o descargar lo filtrado.
3. **Estadística** (modo Admin) → frecuencia por signo/elemento/modalidad,
   histograma, gráficas de pastel, y estadística numérica (media, mediana,
   moda, varianza, desviación estándar) calculada **a mano** (sin
   `df.describe()`) en `core.py`.
4. **Entrenamiento y Resultados** (modo Admin) → eliges algoritmo (K-Means
   o Jerárquico), modo (12 signos / 4 elementos / 3 modalidades) y número
   de clusters (mínimo 2 = binario). Al entrenar ves: silhouette score,
   pureza, el "proceso" del algoritmo (inercia/iteraciones de K-Means, o
   dendrograma del jerárquico), y una proyección PCA 2D de los clusters.
   Todo descargable (CSV/JSON).
5. **Guardar modelo** → genera `.pkl` + `_meta.json` con fecha, algoritmo,
   n° de muestras, métricas, etc.
6. **Modelos Guardados** → localizas cualquier modelo por su metadato y lo
   descargas (modelo + metadata).

## 5. La metodología de pesos (parte medular)

Ver el docstring completo en `questions.py`. Resumen: cada opción de
respuesta se asocia a "k" signos (1 a 3). El peso que suma a cada signo
asociado es `1/k` (reparto uniforme), de forma que **cada pregunta aporta
exactamente 1.0 punto en total**, sin importar a cuántos signos esté
asociada su opción elegida. Esto asegura que ninguna pregunta pese más
que otra en el resultado final.

## 6. Algoritmos elegidos (justificación breve — detalle en Actividad 2)

- **K-Means**: partición rápida, fácil de justificar matemáticamente
  (distancia euclidiana a centroides), ideal como primera aproximación.
- **Jerárquico (Agglomerative, enlace ward)**: genera un dendrograma que
  **muestra el proceso paso a paso** de cómo se van fusionando los perfiles
  más parecidos — perfecto para explicar en la exposición.
- DBSCAN y GMM se investigan y documentan en el reporte de la Actividad 2,
  pero no se implementan en la app por ser menos adecuados para este caso
  (pocas muestras por clase, sin ruido/densidad variable real).
