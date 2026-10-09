"""
Demostración visual e interactiva de Active Learning (Aprendizaje Activo).

Interfaz en español pensada para una exposición universitaria: se controla
con botones, sin mostrar ni explicar el código.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from active_learning import Demostracion, frontera_decision, preparar_datos

# --------------------------------------------------------------------------- #
# Configuración general y estilos
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Active Learning — Demostración",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem;
                 color: #111827; }
    .subtitle  { font-size: 1.15rem; color: #4b5563; margin-top: 0; }
    .panel-title { font-size: 1.6rem; font-weight: 700; text-align: center;
                   color: #111827; }
    /* Las cajas tienen fondo claro fijo: forzamos texto oscuro para que sean
       legibles también cuando Streamlit usa el tema oscuro. */
    .metric-box {
        background: #f5f7fa; border-radius: 14px; padding: 0.6rem 1rem;
        text-align: center; color: #111827;
    }
    .metric-box b { color: #111827; }
    .explicacion {
        font-size: 1.15rem; line-height: 1.5; background: #eef4ff;
        border-left: 6px solid #3b82f6; padding: 1rem 1.2rem; border-radius: 10px;
        color: #1e293b;
    }
    .explicacion b, .explicacion i { color: #1e293b; }
    .analogia {
        font-size: 1.1rem; font-style: italic; background: #fff7ed;
        border-left: 6px solid #f59e0b; padding: 0.9rem 1.1rem; border-radius: 10px;
        color: #7c2d12;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colores por categoría (consistentes en toda la app)
COLOR_CLASE = {0: "#ef4444", 1: "#2563eb"}  # rojo / azul
COLOR_GRIS = "#b8bfc9"
COLOR_BORDE_SELECCION = "#111827"


# --------------------------------------------------------------------------- #
# Estado de la sesión
# --------------------------------------------------------------------------- #
def iniciar_demo():
    datos = preparar_datos(
        n_muestras=400,
        semilla=42,
        superposicion=2.6,
        tam_test=0.4,
        n_inicial_por_clase=3,
    )
    # Cantidad de ejemplos a etiquetar por ronda (controlada por el usuario).
    n_por_ronda = int(st.session_state.get("n_por_ronda", 6))
    st.session_state.demo = Demostracion(
        datos=datos, n_por_ronda=n_por_ronda, semilla=42
    )
    st.session_state.fase = "inicio"  # inicio | seleccion | revelacion | comparacion


if "demo" not in st.session_state:
    iniciar_demo()

demo: Demostracion = st.session_state.demo


# --------------------------------------------------------------------------- #
# Gráfico de un panel (una estrategia)
# --------------------------------------------------------------------------- #
def figura_panel(estrategia, mostrar_seleccion: bool) -> go.Figure:
    datos = demo.datos
    fig = go.Figure()

    # 1) Frontera de decisión como mapa de probabilidad suave.
    xx, yy, zz = frontera_decision(estrategia.modelo, datos, paso=120)
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            showscale=False,
            colorscale=[[0.0, "rgba(239,68,68,0.18)"],
                        [0.5, "rgba(255,255,255,0.0)"],
                        [1.0, "rgba(37,99,235,0.18)"]],
            contours=dict(start=0, end=1, size=0.5),
            line=dict(width=0),
            hoverinfo="skip",
        )
    )
    # Línea de frontera (prob = 0.5)
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            contours=dict(start=0.5, end=0.5, size=0.5, coloring="lines"),
            line=dict(width=3, color="#374151", dash="dash"),
            hoverinfo="skip",
        )
    )

    etiquetados = np.array(estrategia.etiquetados, dtype=int)
    disponibles = estrategia.indices_disponibles
    seleccion = estrategia.seleccion_actual

    # 2) Ejemplos sin etiquetar (gris). Excluimos los seleccionados si se muestran.
    if mostrar_seleccion and len(seleccion) > 0:
        set_sel = set(seleccion.tolist())
        disponibles_vis = np.array([i for i in disponibles if i not in set_sel], dtype=int)
    else:
        disponibles_vis = disponibles

    if len(disponibles_vis) > 0:
        fig.add_trace(
            go.Scatter(
                x=datos.X_pool[disponibles_vis, 0],
                y=datos.X_pool[disponibles_vis, 1],
                mode="markers",
                marker=dict(color=COLOR_GRIS, size=7, opacity=0.55),
                name="Sin etiquetar",
                hoverinfo="skip",
            )
        )

    # 3) Ejemplos etiquetados, coloreados por categoría.
    for clase, color in COLOR_CLASE.items():
        idx = etiquetados[datos.y_pool[etiquetados] == clase]
        if len(idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=datos.X_pool[idx, 0],
                    y=datos.X_pool[idx, 1],
                    mode="markers",
                    marker=dict(color=color, size=12,
                                line=dict(color="white", width=1.5)),
                    name=f"Etiquetado · categoría {clase}",
                    hoverinfo="skip",
                )
            )

    # 4) Ejemplos propuestos para la próxima ronda (borde llamativo, aún en gris).
    if mostrar_seleccion and len(seleccion) > 0:
        fig.add_trace(
            go.Scatter(
                x=datos.X_pool[seleccion, 0],
                y=datos.X_pool[seleccion, 1],
                mode="markers",
                marker=dict(
                    color=COLOR_GRIS,
                    size=16,
                    line=dict(color=COLOR_BORDE_SELECCION, width=3.5),
                    symbol="circle",
                ),
                name="Propuesto para etiquetar",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        xaxis=dict(showticklabels=False, zeroline=False,
                   range=[datos.x_min - 0.6, datos.x_max + 0.6]),
        yaxis=dict(showticklabels=False, zeroline=False,
                   range=[datos.y_min - 0.6, datos.y_max + 0.6]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=12)),
        plot_bgcolor="white",
    )
    return fig


def figura_precision() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=demo.aleatoria.historial_n_etiquetas,
            y=[p * 100 for p in demo.aleatoria.historial_precision],
            mode="lines+markers",
            name="Selección aleatoria",
            line=dict(color="#9ca3af", width=3),
            marker=dict(size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=demo.incertidumbre.historial_n_etiquetas,
            y=[p * 100 for p in demo.incertidumbre.historial_precision],
            mode="lines+markers",
            name="Active learning (incertidumbre)",
            line=dict(color="#16a34a", width=3),
            marker=dict(size=9),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        xaxis_title="Cantidad de ejemplos etiquetados",
        yaxis_title="Precisión en prueba (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        plot_bgcolor="white",
    )
    return fig


def figura_muestra_datos(datos, n_muestra: int = 120) -> go.Figure:
    """Muestra una porción del conjunto para entender la naturaleza de los datos.

    Dibuja una muestra aleatoria (reproducible) de los puntos del pool,
    coloreados por su categoría real, para que se vea la superposición entre
    las dos clases. No muestra todos los puntos a propósito.
    """
    rng = np.random.default_rng(0)
    n = min(n_muestra, len(datos.X_pool))
    idx = rng.choice(len(datos.X_pool), size=n, replace=False)

    fig = go.Figure()
    for clase, color in COLOR_CLASE.items():
        sel = idx[datos.y_pool[idx] == clase]
        fig.add_trace(
            go.Scatter(
                x=datos.X_pool[sel, 0],
                y=datos.X_pool[sel, 1],
                mode="markers",
                marker=dict(color=color, size=10, opacity=0.75,
                            line=dict(color="white", width=1)),
                name=f"Categoría {clase}",
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
        xaxis=dict(title="Coordenada X", zeroline=False),
        yaxis=dict(title="Coordenada Y", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        plot_bgcolor="white",
    )
    return fig


# --------------------------------------------------------------------------- #
# Textos de explicación contextual
# --------------------------------------------------------------------------- #
def texto_explicacion() -> str:
    fase = st.session_state.fase
    if fase == "inicio":
        return (
            "<b>Inicio.</b> Ambos modelos arrancan con los mismos pocos ejemplos "
            "etiquetados (en color) y con muchos ejemplos sin etiquetar (en gris). "
            "La línea punteada es la <b>frontera de decisión</b>: cómo separa cada "
            "modelo las dos categorías por ahora."
        )
    if fase == "seleccion":
        return (
            "<b>Selección de ejemplos.</b> Cada estrategia propone qué ejemplos pedir "
            "al oráculo en la próxima ronda (resaltados con borde negro). La "
            "<b>aleatoria</b> elige al azar; <b>active learning</b> elige los puntos "
            "más confusos, normalmente cerca de la frontera."
        )
    if fase == "revelacion":
        return (
            "<b>Revelación de etiquetas.</b> El oráculo respondió: los ejemplos "
            "elegidos ya tienen color y los modelos se reentrenaron. Observa cómo se "
            "movió la frontera de decisión en cada panel."
        )
    return (
        "<b>Comparación de resultados.</b> Con la <i>misma cantidad</i> de ejemplos "
        "etiquetados, compara la precisión de ambos modelos en el conjunto de prueba. "
        "Es el resultado experimental de <i>esta</i> simulación."
    )


# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.markdown('<div class="big-title">🎯 Aprendizaje Activo (Active Learning)</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">¿Aprende mejor un modelo si elige <b>qué</b> ejemplos '
    'etiquetar, en lugar de etiquetar al azar?</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="analogia">“El modelo es un estudiante que, en vez de pedir que le '
    'revisen ejercicios al azar, le muestra al profesor aquellos que más lo '
    'confunden.”</div>',
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------------------------------- #
# Sección 1: ¿De qué va este ejercicio?
# --------------------------------------------------------------------------- #
with st.expander("ℹ️ ¿De qué va este ejercicio? (léeme primero)", expanded=True):
    izq, der = st.columns(2)
    with izq:
        st.markdown(
            """
#### 🧩 El problema
Etiquetar datos (que una persona experta diga la respuesta correcta de cada
caso) es **caro y lento**. El *aprendizaje activo* busca una idea sencilla:
si el modelo pudiera **elegir** qué ejemplos pedir que le etiqueten, ¿aprendería
más rápido que pidiéndolos al azar?

#### 🔬 Qué estamos implementando
Una simulación que compara dos modelos que parten **igual** y, en cada ronda,
etiquetan la **misma cantidad** de ejemplos nuevos:

- **Selección aleatoria:** elige los ejemplos al azar.
- **Active learning (incertidumbre):** elige los ejemplos donde el modelo está
  **menos seguro** (menor probabilidad máxima predicha), que suelen estar cerca
  de la frontera entre las dos categorías.

Ambos usan la misma clasificación (regresión logística) y se evalúan en un
conjunto de prueba separado que nunca se usa para entrenar.
            """
        )
    with der:
        st.markdown(
            """
#### 🕹️ Cómo usar la demo
1. Mira el estado inicial: pocos ejemplos de color (etiquetados) y muchos en
   gris (sin etiquetar), con la frontera de decisión punteada.
2. Pulsa **“👉 Ver ejemplos propuestos”** para ver, con borde negro, qué
   ejemplos propone cada estrategia para la próxima ronda.
3. Pulsa **“🏷️ Etiquetar siguiente ronda”**: el *oráculo* responde, esos
   ejemplos se colorean y ambos modelos se reentrenan.
4. Pulsa **“📊 Ver comparación de precisión”** y repite varias rondas para ver
   cómo evoluciona la precisión con la misma cantidad de etiquetas.
5. **“🔄 Reiniciar demostración”** vuelve al estado inicial.

> La gráfica de precisión es el resultado de **esta** simulación. No implica que
> el aprendizaje activo gane siempre: depende de los datos, el modelo y el azar.
            """
        )

# --------------------------------------------------------------------------- #
# Sección 2: Naturaleza de los datos
# --------------------------------------------------------------------------- #
with st.expander("🔎 Conoce los datos (una muestra)", expanded=True):
    st.markdown(
        "Usamos un conjunto **sintético en 2D** con **dos categorías que se "
        "superponen** a propósito, para que existan casos dudosos visibles en el "
        "plano. Aquí ves una muestra: no están todos los puntos, solo los "
        "suficientes para entender su naturaleza."
    )
    cg, ct = st.columns([1.3, 1])
    with cg:
        st.plotly_chart(figura_muestra_datos(demo.datos, n_muestra=120),
                        width='stretch', key="muestra_datos")
    with ct:
        datos = demo.datos
        n_total = len(datos.X_pool) + len(datos.X_test)
        n_c0 = int((datos.y_pool == 0).sum() + (datos.y_test == 0).sum())
        n_c1 = int((datos.y_pool == 1).sum() + (datos.y_test == 1).sum())
        st.markdown(
            f"""
**Resumen del conjunto**

- Total de puntos: **{n_total}**
- Categoría 0 (🔴): **{n_c0}** · Categoría 1 (🔵): **{n_c1}**
- Dimensiones por punto: **2** (coordenadas X e Y)
- Pool de entrenamiento: **{len(datos.X_pool)}** · Prueba (separado):
  **{len(datos.X_test)}**
- Etiquetados al inicio: **{len(datos.indices_iniciales)}**

Cada punto tiene dos coordenadas y pertenece a una de las dos categorías. En la
zona central las nubes se mezclan: ahí viven los casos difíciles.
            """
        )
        st.markdown("**Primeras filas (coordenadas y categoría real):**")
        tabla = {
            "X": [round(float(v), 2) for v in datos.X_pool[:6, 0]],
            "Y": [round(float(v), 2) for v in datos.X_pool[:6, 1]],
            "Categoría": [int(v) for v in datos.y_pool[:6]],
        }
        st.dataframe(tabla, width='stretch', hide_index=True)

st.divider()

# --------------------------------------------------------------------------- #
# Sección 3: Paradigma y criterio de selección (antes del laboratorio)
# --------------------------------------------------------------------------- #
with st.expander("🧭 Antes de empezar: ¿qué paradigma de Active Learning usamos?",
                 expanded=True):
    pa, pb = st.columns(2)
    with pa:
        st.markdown(
            """
#### 📦 Paradigma: muestreo basado en conjunto (*pool-based*)
Trabajamos con el escenario de **aprendizaje activo basado en conjunto**
(*pool-based active learning*): existe un **conjunto fijo** de ejemplos ya
disponibles pero **sin etiquetar** (el *pool* gris). En cada ronda, la
estrategia **mira todo ese conjunto**, puntúa cada ejemplo y elige los que
pedirá etiquetar al oráculo.

Es el escenario más común en la práctica (lo contrario a generar ejemplos
nuevos o decidir sobre un flujo que llega uno a uno).
            """
        )
    with pb:
        st.markdown(
            """
#### 🎯 Criterio: muestreo por incertidumbre (*menor confianza*)
Para la estrategia de active learning usamos **muestreo por incertidumbre**
(*uncertainty sampling*) con el criterio de **menor confianza**
(*least confident*):

1. El modelo predice, para cada ejemplo sin etiquetar, la probabilidad de cada
   categoría.
2. Se mira la **probabilidad máxima** (la de la clase más probable): si es alta,
   el modelo está seguro; si es baja, está dudando.
3. Se eligen los ejemplos con la **probabilidad máxima más baja**, es decir,
   donde el modelo está **menos seguro**.

En 2D, esos puntos suelen caer **cerca de la frontera de decisión**, en la zona
donde las dos categorías se mezclan.
            """
        )
    st.markdown(
        r"""
**En fórmula**, se elige el ejemplo $x$ que maximiza la incertidumbre de menor
confianza:

$$x^{*} = \arg\max_{x}\;\Big(1 - \max_{c}\,P(y=c \mid x)\Big)$$

equivalente a elegir el $x$ con **menor** $\max_{c} P(y=c\mid x)$. Cuando se
etiqueta más de uno por ronda, se toman los de menor confianza en orden.

La estrategia **aleatoria** no usa ningún criterio: escoge del mismo conjunto
*al azar*. Sirve como línea base para comparar.
        """
    )

st.divider()

# --------------------------------------------------------------------------- #
# Control manual: ejemplos a etiquetar por ronda
# --------------------------------------------------------------------------- #
# El máximo razonable es la cantidad de ejemplos que aún quedan disponibles.
disponibles_ahora = min(
    len(demo.aleatoria.indices_disponibles),
    len(demo.incertidumbre.indices_disponibles),
)
tope = max(1, disponibles_ahora)

st.markdown("##### 🎚️ ¿Cuántos ejemplos etiquetar por ronda?")
nuevo_n = st.slider(
    "Elige cuántas etiquetas aprobará el oráculo en la próxima ronda. "
    "Se aplica por igual a ambas estrategias.",
    min_value=1,
    max_value=min(30, tope),
    value=min(int(demo.n_por_ronda), min(30, tope)),
    step=1,
    key="n_por_ronda",
    disabled=not demo.hay_disponibles,
)
# Si el usuario cambió el valor, lo aplicamos y re-proponemos la selección
# para que la vista previa (puntos con borde) refleje de inmediato el nuevo n.
if demo.hay_disponibles and nuevo_n != demo.n_por_ronda:
    demo.fijar_n_por_ronda(nuevo_n)
    st.session_state.fase = "seleccion"
    st.rerun()

st.caption(
    f"En la próxima ronda, cada estrategia etiquetará **{demo.n_por_ronda}** "
    "ejemplo(s). Cambiar este valor no afecta lo ya etiquetado; se aplica a las "
    "rondas siguientes."
)

st.write("")

# --------------------------------------------------------------------------- #
# Controles
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1])

with c1:
    if st.button(f"🏷️ Etiquetar siguiente ronda ({demo.n_por_ronda})",
                 type="primary",
                 width='stretch', disabled=not demo.hay_disponibles):
        # Avanza una ronda: revela las etiquetas propuestas, reentrena y
        # vuelve a proponer la siguiente selección.
        demo.avanzar_ronda()
        st.session_state.fase = "revelacion"
        st.rerun()

with c2:
    if st.button("🔄 Reiniciar demostración", width='stretch'):
        iniciar_demo()
        st.rerun()

with c3:
    st.markdown(
        f'<div class="metric-box"><b>Ejemplos etiquetados</b><br>'
        f'<span style="font-size:1.8rem">{demo.aleatoria.n_etiquetados}</span></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div class="metric-box"><b>Ronda</b><br>'
        f'<span style="font-size:1.8rem">{demo.ronda}</span></div>',
        unsafe_allow_html=True,
    )

if not demo.hay_disponibles:
    st.info("Ya no quedan ejemplos sin etiquetar: se etiquetó todo el conjunto "
            "disponible. Reinicia para volver a empezar.")

# Navegación de la narración: permite recorrer las fases didácticas sin
# cambiar el estado de los modelos. Útil para explicar paso a paso.
n1, n2 = st.columns(2)
with n1:
    if st.button("👉 Ver ejemplos propuestos", width='stretch'):
        st.session_state.fase = "seleccion"
        st.rerun()
with n2:
    if st.button("📊 Ver comparación de precisión", width='stretch'):
        st.session_state.fase = "comparacion"
        st.rerun()

st.write("")

# --------------------------------------------------------------------------- #
# Explicación contextual
# --------------------------------------------------------------------------- #
st.markdown(f'<div class="explicacion">{texto_explicacion()}</div>',
            unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------- #
# Paneles comparables
# --------------------------------------------------------------------------- #
col_izq, col_der = st.columns(2)

with col_izq:
    st.markdown('<div class="panel-title">🎲 Selección aleatoria</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-box">Precisión en prueba: '
        f'<b style="font-size:1.5rem">'
        f'{demo.aleatoria.historial_precision[-1]*100:.1f}%</b></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(figura_panel(demo.aleatoria, mostrar_seleccion=True),
                    width='stretch', key="panel_aleatoria")

with col_der:
    st.markdown('<div class="panel-title">🧠 Active learning</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-box">Precisión en prueba: '
        f'<b style="font-size:1.5rem">'
        f'{demo.incertidumbre.historial_precision[-1]*100:.1f}%</b></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(figura_panel(demo.incertidumbre, mostrar_seleccion=True),
                    width='stretch', key="panel_incertidumbre")

# --------------------------------------------------------------------------- #
# Gráfico comparativo de precisión
# --------------------------------------------------------------------------- #
st.write("")
st.markdown("### 📈 Precisión vs. cantidad de ejemplos etiquetados")
st.plotly_chart(figura_precision(), width='stretch', key="precision")
st.caption(
    "Comparación experimental de esta simulación. No implica que el aprendizaje "
    "activo supere siempre a la selección aleatoria: el resultado depende de los "
    "datos, el modelo y el azar."
)

# --------------------------------------------------------------------------- #
# Glosario y notas
# --------------------------------------------------------------------------- #
with st.expander("📚 ¿Qué significan los términos? (glosario)"):
    st.markdown(
        """
- **Ejemplos etiquetados:** los casos cuya respuesta correcta ya conocemos.
- **Ejemplos sin etiquetar:** los casos disponibles que todavía no han sido revisados.
- **Oráculo:** la persona experta que responde la consulta del modelo.
- **Incertidumbre:** qué tan poco seguro está el modelo de su predicción. Aquí se mide
  como la probabilidad máxima predicha: si la mayor probabilidad entre las dos
  categorías es baja, el modelo está dudando.

**Objetivo del aprendizaje activo:** llegar a buena precisión con *menos* etiquetas.
Ten en cuenta que el costo real no es solo el número de etiquetas: también influye el
tiempo y la dificultad de etiquetar cada caso.
        """
    )

with st.expander("🔬 ¿Cómo funciona esta simulación?"):
    st.markdown(
        """
- Datos sintéticos 2D de **dos categorías con superposición**, para que existan casos
  dudosos visibles en el plano.
- El **conjunto de datos**, el **conjunto de prueba** y el **grupo inicial etiquetado**
  son fijos e idénticos para ambas estrategias (comparación justa).
- En cada ronda ambas estrategias agregan la **misma cantidad** de ejemplos.
- **Incertidumbre:** se eligen los ejemplos con **menor probabilidad máxima** predicha.
- **Aleatoria:** se eligen ejemplos **al azar** del mismo conjunto disponible.
- Las etiquetas verdaderas de los datos sintéticos actúan como el **oráculo**. El modelo
  no las usa hasta que los ejemplos son seleccionados y etiquetados.
- La **precisión** se mide en un conjunto de prueba separado, que nunca se usa para
  entrenar.
        """
    )

with st.expander("📖 Referencias"):
    st.markdown(
        """
Referencias conceptuales principales:

- Burr Settles. *Active Learning Literature Survey*. Computer Sciences Technical
  Report 1648, University of Wisconsin–Madison, 2009. (Secciones sobre aprendizaje
  activo basado en conjuntos — *pool-based* — y muestreo por incertidumbre.)
- Burr Settles. *From Theories to Queries: Active Learning in Practice*. JMLR
  Workshop and Conference Proceedings, 2011. (Introducción y desafíos prácticos.)

Esta demostración se inspira conceptualmente en esos trabajos. Los resultados
numéricos mostrados aquí provienen únicamente de esta simulación y **no** se atribuyen
a dichos artículos.
        """
    )
