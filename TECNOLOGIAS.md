# Reporte de tecnologías del proyecto

Este documento describe las librerías de Python que usamos en la demostración de
**Active Learning** (aprendizaje activo), por qué las elegimos y cómo se usan en
el proyecto. Para cada una se incluye una introducción breve y un pequeño
ejemplo tomado o inspirado en nuestro propio código.

> Resumen: `scikit-learn` hace el aprendizaje (modelo y datos), `NumPy` hace los
> cálculos numéricos (incluida la selección por incertidumbre), `Plotly` dibuja
> las visualizaciones y `Streamlit` arma la interfaz interactiva. `pytest`
> verifica que la lógica central funcione.

## Versiones usadas

| Librería | Versión | Rol en el proyecto |
|---|---|---|
| scikit-learn | 1.9.1 | Datos sintéticos, modelo y evaluación |
| NumPy | 2.5.3 | Cálculo numérico y selección de instancias |
| Plotly | 5.24.1 | Gráficos interactivos (puntos, frontera, precisión) |
| Streamlit | 1.65.0 | Interfaz web y controles de la demo |
| pytest | 9.1.1 | Pruebas de la lógica de active learning |

Las dependencias de ejecución están declaradas en `requirements.txt`.

---

## scikit-learn

**Qué es.** Es la librería estándar de *machine learning* clásico en Python.
Ofrece modelos, utilidades para preparar datos y métricas de evaluación con una
interfaz uniforme (`fit`, `predict`, `predict_proba`, `score`).

**Por qué la usamos.** Necesitábamos tres cosas y las tres vienen en
scikit-learn: generar un conjunto de datos sintético controlable, entrenar un
clasificador sencillo e interpretable, y poder leer la **confianza** de sus
predicciones (clave para el muestreo por incertidumbre).

**Cómo se usa en el proyecto.** Todo el uso de scikit-learn vive en
`active_learning.py` (la lógica, separada de la interfaz):

- `make_blobs` → en `generar_datos()`. Crea las dos nubes de puntos 2D
  (`centers=[(-2,-2),(2,2)]`) con la superposición controlada por
  `cluster_std=superposicion`, para que haya casos dudosos visibles.
- `train_test_split` → en `preparar_datos()`. Separa el **pool** del
  **conjunto de prueba** con `stratify=y` (mantiene la proporción de clases) y
  `random_state` fijo, para que el test sea idéntico en ambas estrategias y
  nunca se use para entrenar.
- `LogisticRegression(max_iter=1000)` → en `EstadoEstrategia.entrenar()`. Es el
  clasificador; se reentrena desde cero en cada ronda solo con los ejemplos ya
  etiquetados.
- `modelo.predict_proba(...)` → se usa en dos lugares:
  `EstadoEstrategia.proponer_siguiente()` (para medir la incertidumbre de cada
  ejemplo) y `frontera_decision()` (para dibujar la frontera como contorno de
  probabilidad 0.5).
- `modelo.score(X_test, y_test)` → en `EstadoEstrategia.precision_test()`. Es la
  precisión que alimenta el gráfico comparativo.

```python
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1) Datos sintéticos 2D con dos categorías que se superponen
X, y = make_blobs(n_samples=400, centers=[(-2, -2), (2, 2)],
                  cluster_std=2.6, random_state=42)

# 2) Conjunto de prueba separado (no se usa para entrenar)
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)

# 3) Entrenar y evaluar
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_pool[:6], y_pool[:6])         # solo con lo ya etiquetado
precision = modelo.score(X_test, y_test)   # precisión en prueba

# 4) Confianza de la predicción (base del muestreo por incertidumbre)
probas = modelo.predict_proba(X_pool)      # P(clase) por ejemplo
```

---

## NumPy

**Qué es.** La librería base de cómputo numérico en Python: arreglos
(`ndarray`) y operaciones vectorizadas rápidas. Casi todo el ecosistema
científico de Python se apoya en ella.

**Por qué la usamos.** El corazón del active learning por incertidumbre es un
cálculo numérico: tomar las probabilidades, quedarse con la máxima de cada
ejemplo y ordenar para elegir las menos confiables. NumPy lo hace de forma
directa y eficiente. También la usamos para la aleatoriedad reproducible.

**Cómo se usa en el proyecto.** NumPy aparece en `active_learning.py` (cálculo)
y en `app.py` (preparación de datos para graficar):

- **Selección por incertidumbre** → en `EstadoEstrategia.proponer_siguiente()`:
  `probas.max(axis=1)` obtiene la confianza (probabilidad máxima) de cada
  ejemplo y `np.argsort(...)` los ordena de menos a más seguro para quedarse con
  los `n` más dudosos. Este es, literalmente, el corazón del active learning.
- **Aleatoriedad reproducible** → `np.random.default_rng(semilla)` se usa en
  `preparar_datos()` (elegir el grupo inicial etiquetado) y en la estrategia
  `aleatoria` dentro de `proponer_siguiente()` (`rng.choice(...)`).
- **Índices y conjuntos** → en toda la clase `EstadoEstrategia` se manejan los
  ejemplos como arreglos de índices (`np.array`, filtrado booleano como
  `datos.y_pool[etiquetados] == clase`).
- **Malla de la frontera** → en `frontera_decision()`: `np.linspace` +
  `np.meshgrid` + `np.c_[...]` arman la cuadrícula de puntos sobre la que el
  modelo predice para dibujar la frontera.

```python
import numpy as np

# Elegir los ejemplos de MENOR confianza (muestreo por incertidumbre)
probas = modelo.predict_proba(X_disponibles)
confianza_max = probas.max(axis=1)         # confianza de cada ejemplo
orden = np.argsort(confianza_max)          # de menos a más seguro
seleccion = X_disponibles[orden[:n]]       # los n más inciertos

# Selección aleatoria reproducible (línea base)
rng = np.random.default_rng(42)
aleatoria = rng.choice(len(X_disponibles), size=n, replace=False)
```

---

## Plotly

**Qué es.** Una librería de visualización interactiva. Produce gráficos con
zoom, desplazamiento y leyendas que se pueden activar o desactivar, ideales para
mostrar algo en vivo frente a una audiencia.

**Por qué la usamos.** La demo necesita gráficos claros y manipulables: ver los
puntos por categoría, la frontera de decisión y los ejemplos propuestos
resaltados. Plotly se integra directamente con Streamlit.

**Cómo se usa en el proyecto.** Todas las figuras se construyen en `app.py`, en
tres funciones:

- `figura_panel(estrategia, ...)` → es el gráfico principal de cada panel. Usa
  dos `go.Contour` (uno para el mapa de probabilidad suave y otro para la línea
  de **frontera de decisión** en probabilidad 0.5) y varios `go.Scatter`: gris
  para los **sin etiquetar**, de color por **categoría** para los etiquetados, y
  uno con `marker.line` grueso para resaltar los ejemplos **propuestos** para la
  próxima ronda.
- `figura_precision()` → un `go.Scatter` en modo `lines+markers` por estrategia
  (gris para la aleatoria, verde para active learning) que compara la
  **precisión** acumulada ronda a ronda.
- `figura_muestra_datos(datos, ...)` → la vista inicial de la naturaleza de los
  datos: una muestra de puntos coloreados por categoría.
- El `fig.update_layout(...)` fija alto, leyenda horizontal y fondo blanco para
  una lectura clara en proyección.

```python
import plotly.graph_objects as go

fig = go.Figure()
# Puntos etiquetados coloreados por categoría
fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                         marker=dict(color="#2563eb", size=12)))
# Frontera de decisión: contorno en probabilidad 0.5
fig.add_trace(go.Contour(x=gx, y=gy, z=probabilidades,
                         contours=dict(start=0.5, end=0.5, size=0.5,
                                       coloring="lines")))
```

En Streamlit, la figura se muestra con `st.plotly_chart(fig, width='stretch')`.

---

## Streamlit

**Qué es.** Un framework para crear aplicaciones web de datos escribiendo solo
Python, sin HTML ni JavaScript. Cada interacción vuelve a ejecutar el script de
arriba a abajo y el estado se guarda en `st.session_state`.

**Por qué la usamos.** Permite presentar la demo con botones y controles, sin
mostrar código, y ponerla en marcha con un solo comando. Es el requisito central
del proyecto: una interfaz limpia controlable por quien expone.

**Cómo se usa en el proyecto.** `app.py` es toda la interfaz:

- `st.session_state` → guarda la simulación (`st.session_state.demo`) y la fase
  didáctica actual (`st.session_state.fase`) entre clics. La función
  `iniciar_demo()` crea el objeto `Demostracion` y lo deja en el estado.
- `st.button` → los controles de la demo: “🏷️ Etiquetar siguiente ronda”
  (llama a `demo.avanzar_ronda()`), “🔄 Reiniciar demostración”, y los botones
  de narración “👉 Ver ejemplos propuestos” / “📊 Ver comparación de precisión”.
  Tras cada clic se llama a `st.rerun()`.
- `st.slider` (con `key="n_por_ronda"`) → permite elegir manualmente cuántos
  ejemplos etiqueta cada estrategia por ronda; al cambiarlo se llama a
  `demo.fijar_n_por_ronda(...)`.
- `st.columns` → arma los dos paneles comparables lado a lado y las secciones a
  dos columnas.
- `st.expander` → las secciones introductorias (de qué va, los datos, el
  paradigma) y el glosario/referencias.
- `st.plotly_chart(fig, width='stretch')` → inserta las figuras de Plotly.
- `st.markdown(..., unsafe_allow_html=True)` + un bloque `<style>` → aplican los
  estilos propios (títulos grandes, cajas de métrica y de explicación).
- `st.set_page_config(...)` → título, ícono y diseño ancho de la página.

```python
import streamlit as st

# Guardar estado entre interacciones
if "contador" not in st.session_state:
    st.session_state.contador = 0

# Control manual del tamaño de ronda
n = st.slider("Ejemplos por ronda", min_value=1, max_value=30, value=6)

if st.button("Etiquetar siguiente ronda"):
    st.session_state.contador += n
    st.rerun()  # vuelve a ejecutar el script con el nuevo estado
```

Para ejecutar la app: `streamlit run app.py`.

---

## pytest

**Qué es.** El framework de pruebas más usado en Python. Descubre funciones que
empiezan con `test_` y reporta cuáles pasan o fallan.

**Por qué la usamos.** El active learning tiene reglas que deben cumplirse para
que la comparación sea justa (mismos datos iniciales, misma cantidad por ronda,
selección realmente por menor confianza, y no usar etiquetas antes de tiempo).
Las pruebas blindan esas reglas.

**Cómo se usa en el proyecto.** El archivo `test_active_learning.py` contiene 14
pruebas que blindan la lógica de `active_learning.py`. Entre otras, verifican
que:

- los datos (pool, test y grupo inicial) son **idénticos** con la misma semilla
  (comparación justa);
- cada ronda agrega la **misma cantidad** de ejemplos en ambas estrategias;
- la estrategia de incertidumbre elige **exactamente** los de menor
  probabilidad máxima;
- las etiquetas **no se usan** antes de que los ejemplos sean seleccionados;
- el control manual `fijar_n_por_ronda(...)` cambia el tamaño sin tocar lo ya
  etiquetado.

```python
def test_seleccion_por_incertidumbre_elige_menor_confianza():
    demo = _demo()
    # La selección debe coincidir con los de menor probabilidad máxima
    ...
    assert set(demo.incertidumbre.seleccion_actual.tolist()) == esperados
```

Para ejecutarlas: `pytest -q`.

---

## Cómo encajan todas juntas

1. **scikit-learn** genera los datos y entrena el modelo; devuelve
   probabilidades por clase.
2. **NumPy** toma esas probabilidades y decide qué instancia etiquetar
   (muestreo por incertidumbre) o elige al azar (línea base).
3. **scikit-learn** reentrena con los nuevos ejemplos y mide la precisión en el
   conjunto de prueba.
4. **Plotly** visualiza puntos, frontera y la evolución de la precisión.
5. **Streamlit** envuelve todo en una interfaz con botones y controles.
6. **pytest** verifica que las reglas de la simulación se cumplan.

## Enlaces útiles

- scikit-learn: <https://scikit-learn.org/stable/>
- NumPy: <https://numpy.org/doc/>
- Plotly (Python): <https://plotly.com/python/>
- Streamlit: <https://docs.streamlit.io/>
- pytest: <https://docs.pytest.org/>
