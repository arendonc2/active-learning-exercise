"""
Lógica central de la demostración de Active Learning.

Este módulo contiene toda la "mecánica" de la simulación, separada de la
interfaz de Streamlit para que pueda probarse de forma aislada.

Idea general
------------
- Tenemos un conjunto de datos sintético 2D con dos categorías que se
  superponen un poco (para que existan casos dudosos visibles en el plano).
- Dividimos los datos en: conjunto de entrenamiento (pool) y conjunto de
  prueba (test). El test NUNCA se usa para entrenar.
- Dentro del pool, al inicio solo un pequeño grupo está "etiquetado".
- Dos estrategias compiten partiendo del MISMO estado inicial:
    * aleatoria: elige al azar qué ejemplos etiquetar en cada ronda.
    * incertidumbre: elige los ejemplos donde el modelo está menos seguro
      (menor probabilidad máxima predicha).
- En cada ronda ambas estrategias etiquetan la misma cantidad de ejemplos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Datos sintéticos
# --------------------------------------------------------------------------- #
def generar_datos(
    n_muestras: int = 400,
    semilla: int = 42,
    superposicion: float = 2.6,
):
    """Genera un conjunto 2D de dos categorías con cierta superposición.

    `superposicion` controla la desviación estándar de los grupos: valores
    más altos acercan y mezclan las dos nubes de puntos, creando casos dudosos.
    """
    X, y = make_blobs(
        n_samples=n_muestras,
        centers=[(-2.0, -2.0), (2.0, 2.0)],
        cluster_std=superposicion,
        random_state=semilla,
    )
    return X.astype(float), y.astype(int)


@dataclass
class DatosDemo:
    """Contenedor con los datos fijos de la demostración.

    Todo lo que debe permanecer constante entre estrategias y entre
    reinicios "justos" vive aquí: el pool, el test y el grupo inicial.
    """

    X_pool: np.ndarray
    y_pool: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    indices_iniciales: np.ndarray  # índices (sobre X_pool) etiquetados al inicio

    @property
    def x_min(self) -> float:
        return float(min(self.X_pool[:, 0].min(), self.X_test[:, 0].min()))

    @property
    def x_max(self) -> float:
        return float(max(self.X_pool[:, 0].max(), self.X_test[:, 0].max()))

    @property
    def y_min(self) -> float:
        return float(min(self.X_pool[:, 1].min(), self.X_test[:, 1].min()))

    @property
    def y_max(self) -> float:
        return float(max(self.X_pool[:, 1].max(), self.X_test[:, 1].max()))


def preparar_datos(
    n_muestras: int = 400,
    semilla: int = 42,
    superposicion: float = 2.6,
    tam_test: float = 0.4,
    n_inicial_por_clase: int = 3,
) -> DatosDemo:
    """Prepara los datos fijos: pool, test y grupo inicial etiquetado.

    Se usa una semilla fija para que el conjunto de datos, el conjunto de
    prueba y el grupo inicial sean idénticos en ambas estrategias y entre
    ejecuciones. Así la comparación es justa.
    """
    X, y = generar_datos(n_muestras, semilla, superposicion)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=tam_test, random_state=semilla, stratify=y
    )

    rng = np.random.default_rng(semilla)
    indices_iniciales = []
    for clase in np.unique(y_pool):
        idx_clase = np.where(y_pool == clase)[0]
        elegidos = rng.choice(idx_clase, size=n_inicial_por_clase, replace=False)
        indices_iniciales.extend(elegidos.tolist())
    indices_iniciales = np.array(sorted(indices_iniciales), dtype=int)

    return DatosDemo(
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        indices_iniciales=indices_iniciales,
    )


# --------------------------------------------------------------------------- #
# Estrategias
# --------------------------------------------------------------------------- #
@dataclass
class EstadoEstrategia:
    """Estado mutable de una estrategia a lo largo de las rondas."""

    nombre: str
    datos: DatosDemo
    etiquetados: List[int] = field(default_factory=list)   # índices etiquetados
    historial_precision: List[float] = field(default_factory=list)
    historial_n_etiquetas: List[int] = field(default_factory=list)
    modelo: LogisticRegression | None = None
    seleccion_actual: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def __post_init__(self):
        self.etiquetados = list(self.datos.indices_iniciales)

    # -- conjuntos derivados ------------------------------------------------- #
    @property
    def indices_disponibles(self) -> np.ndarray:
        """Índices del pool que todavía NO están etiquetados."""
        todos = set(range(len(self.datos.X_pool)))
        return np.array(sorted(todos - set(self.etiquetados)), dtype=int)

    @property
    def n_etiquetados(self) -> int:
        return len(self.etiquetados)

    # -- entrenamiento y evaluación ----------------------------------------- #
    def entrenar(self) -> None:
        """Entrena el modelo solo con los ejemplos ya etiquetados."""
        idx = np.array(self.etiquetados, dtype=int)
        X = self.datos.X_pool[idx]
        y = self.datos.y_pool[idx]
        # max_iter alto y solver estable para que converja con pocos datos.
        self.modelo = LogisticRegression(max_iter=1000)
        self.modelo.fit(X, y)

    def precision_test(self) -> float:
        """Precisión sobre el conjunto de prueba separado."""
        if self.modelo is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía.")
        return float(self.modelo.score(self.datos.X_test, self.datos.y_test))

    def registrar_precision(self) -> None:
        self.historial_precision.append(self.precision_test())
        self.historial_n_etiquetas.append(self.n_etiquetados)

    # -- selección de la próxima ronda -------------------------------------- #
    def proponer_siguiente(self, n: int, semilla: int) -> np.ndarray:
        """Devuelve los índices que esta estrategia propone etiquetar.

        No modifica el estado: solo propone. La revelación de etiquetas ocurre
        en `etiquetar_seleccion`. Así respetamos que el modelo no vea las
        etiquetas verdaderas antes de que los ejemplos sean seleccionados.
        """
        disponibles = self.indices_disponibles
        if len(disponibles) == 0:
            self.seleccion_actual = np.array([], dtype=int)
            return self.seleccion_actual

        n = min(n, len(disponibles))

        if self.nombre == "aleatoria":
            rng = np.random.default_rng(semilla)
            seleccion = rng.choice(disponibles, size=n, replace=False)
        elif self.nombre == "incertidumbre":
            # Menor probabilidad máxima predicha = mayor incertidumbre.
            probas = self.modelo.predict_proba(self.datos.X_pool[disponibles])
            confianza_max = probas.max(axis=1)
            orden = np.argsort(confianza_max)  # de menos a más seguro
            seleccion = disponibles[orden[:n]]
        else:
            raise ValueError(f"Estrategia desconocida: {self.nombre}")

        self.seleccion_actual = np.array(sorted(seleccion.tolist()), dtype=int)
        return self.seleccion_actual

    def etiquetar_seleccion(self) -> None:
        """"Revela" las etiquetas de la selección actual y las incorpora.

        Simula la consulta al oráculo: ahora conocemos la respuesta correcta
        (la etiqueta verdadera del dato sintético) de los ejemplos elegidos.
        """
        for idx in self.seleccion_actual:
            if idx not in self.etiquetados:
                self.etiquetados.append(int(idx))
        self.seleccion_actual = np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# Orquestador de la demostración
# --------------------------------------------------------------------------- #
@dataclass
class Demostracion:
    """Coordina las dos estrategias sobre los mismos datos."""

    datos: DatosDemo
    n_por_ronda: int = 6
    semilla: int = 42
    ronda: int = 0
    aleatoria: EstadoEstrategia = field(init=False)
    incertidumbre: EstadoEstrategia = field(init=False)

    def __post_init__(self):
        self.aleatoria = EstadoEstrategia("aleatoria", self.datos)
        self.incertidumbre = EstadoEstrategia("incertidumbre", self.datos)
        # Entrenamiento inicial con el mismo grupo etiquetado.
        self.aleatoria.entrenar()
        self.incertidumbre.entrenar()
        self.aleatoria.registrar_precision()
        self.incertidumbre.registrar_precision()
        # Proponer la primera selección de cada estrategia.
        self._proponer_todas()

    def _proponer_todas(self) -> None:
        # Semilla distinta por ronda para que la aleatoria varíe entre rondas,
        # pero reproducible.
        self.aleatoria.proponer_siguiente(self.n_por_ronda, self.semilla + self.ronda)
        self.incertidumbre.proponer_siguiente(self.n_por_ronda, self.semilla + self.ronda)

    def fijar_n_por_ronda(self, n: int) -> None:
        """Cambia cuántos ejemplos se etiquetan por ronda y re-propone.

        No avanza la ronda ni toca lo ya etiquetado: solo ajusta el tamaño de
        la próxima tanda y vuelve a calcular la selección propuesta para que la
        vista previa (los puntos con borde) refleje el nuevo valor de inmediato.
        El cambio se aplica por igual a ambas estrategias, manteniendo la
        comparación justa.
        """
        self.n_por_ronda = max(1, int(n))
        self._proponer_todas()

    @property
    def hay_disponibles(self) -> bool:
        return (
            len(self.aleatoria.indices_disponibles) > 0
            and len(self.incertidumbre.indices_disponibles) > 0
        )

    def avanzar_ronda(self) -> None:
        """Etiqueta la selección propuesta, reentrena y propone la siguiente."""
        if not self.hay_disponibles:
            return
        self.aleatoria.etiquetar_seleccion()
        self.incertidumbre.etiquetar_seleccion()

        self.aleatoria.entrenar()
        self.incertidumbre.entrenar()

        self.aleatoria.registrar_precision()
        self.incertidumbre.registrar_precision()

        self.ronda += 1
        self._proponer_todas()


def frontera_decision(modelo: LogisticRegression, datos: DatosDemo, paso: int = 200):
    """Devuelve la malla y las probabilidades para dibujar la frontera.

    Retorna (xx, yy, zz) donde zz es la probabilidad de la clase 1 en cada
    punto de la malla. La frontera de decisión es el contorno zz = 0.5.
    """
    margen = 0.5
    xs = np.linspace(datos.x_min - margen, datos.x_max + margen, paso)
    ys = np.linspace(datos.y_min - margen, datos.y_max + margen, paso)
    xx, yy = np.meshgrid(xs, ys)
    malla = np.c_[xx.ravel(), yy.ravel()]
    zz = modelo.predict_proba(malla)[:, 1].reshape(xx.shape)
    return xx, yy, zz
