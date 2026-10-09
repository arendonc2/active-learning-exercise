# Demostración de Active Learning (Aprendizaje Activo)

Demostración visual e interactiva, en español, para una exposición
universitaria. Compara dos formas de elegir qué ejemplos etiquetar:

- **Selección aleatoria:** elige al azar los ejemplos nuevos a etiquetar.
- **Active learning por incertidumbre:** elige los ejemplos sobre los que el
  modelo está menos seguro (menor probabilidad máxima predicha).

Ambos modelos parten del mismo grupo inicial de ejemplos etiquetados y, en cada
ronda, agregan la misma cantidad de ejemplos nuevos. Así la comparación es justa.

La interfaz se controla con botones y no requiere mostrar ni explicar el código.

## Qué se ve en pantalla

- Dos paneles comparables: *Selección aleatoria* y *Active learning*.
- En cada panel, un gráfico de puntos con:
  - ejemplos etiquetados coloreados por categoría (rojo / azul),
  - ejemplos sin etiquetar en gris,
  - la frontera de decisión actual del modelo (línea punteada),
  - los ejemplos propuestos para la próxima ronda, resaltados con borde negro.
- Botón **“Etiquetar siguiente ronda”**: consulta al oráculo, revela las
  etiquetas de los ejemplos elegidos y reentrena ambos modelos.
- Botón **“Reiniciar demostración”**: vuelve al estado inicial.
- Gráfico de **precisión vs. cantidad de ejemplos etiquetados** para ambos modelos.
- Contador de ejemplos etiquetados, número de ronda e indicador de precisión.
- Una explicación corta que cambia según la fase: inicio, selección de ejemplos,
  revelación de etiquetas y comparación de resultados.
- Glosario, nota metodológica y referencias.

## Requisitos

- Python 3.9 o superior (probado con 3.12).

## Instalación

Desde la carpeta del proyecto:

```bash
# 1) Crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

# 2) Instalar dependencias
pip install -r requirements.txt
```

## Ejecutar la demo

```bash
streamlit run app.py
```

Se abrirá en el navegador (normalmente en http://localhost:8501). Para la
presentación: pulsa **“Etiquetar siguiente ronda”** varias veces y explica cómo
cambian la frontera y la precisión en cada panel.

## Ejecutar las pruebas

```bash
pip install pytest
pytest -q
```

Las pruebas verifican lo esencial: los datos son fijos y la comparación es
justa, cada ronda agrega la misma cantidad de ejemplos, la estrategia de
incertidumbre elige realmente los puntos de menor confianza y las etiquetas no
se usan antes de ser consultadas.

## Estructura

- `app.py` — interfaz Streamlit + visualización Plotly.
- `active_learning.py` — lógica de datos, estrategias y evaluación (sin UI).
- `test_active_learning.py` — pruebas de la lógica central.
- `requirements.txt` — dependencias.

## Nota metodológica

El gráfico comparativo es el resultado experimental de **esta** simulación. No
implica que el aprendizaje activo supere siempre a la selección aleatoria: el
resultado depende de los datos, del modelo y del azar. El objetivo del
aprendizaje activo es alcanzar buena precisión con *menos* etiquetas; en la
práctica, el costo real también depende del tiempo y la dificultad de etiquetar.

## Referencias

Referencias conceptuales principales:

- Burr Settles. *Active Learning Literature Survey*. Computer Sciences Technical
  Report 1648, University of Wisconsin–Madison, 2009.
- Burr Settles. *From Theories to Queries: Active Learning in Practice*. JMLR
  Workshop and Conference Proceedings, 2011.

Esta demostración se inspira conceptualmente en esos trabajos. Los resultados
numéricos aquí mostrados provienen únicamente de la simulación y no se atribuyen
a dichos artículos.
