"""Pruebas de la lógica central de la demostración de Active Learning."""

import numpy as np

from active_learning import (
    Demostracion,
    EstadoEstrategia,
    frontera_decision,
    preparar_datos,
)


def _demo():
    datos = preparar_datos(semilla=42)
    return Demostracion(datos=datos, n_por_ronda=6, semilla=42)


def test_datos_son_fijos_y_justos():
    """El pool, el test y el grupo inicial deben ser idénticos con la misma semilla."""
    d1 = preparar_datos(semilla=42)
    d2 = preparar_datos(semilla=42)
    assert np.array_equal(d1.X_pool, d2.X_pool)
    assert np.array_equal(d1.X_test, d2.X_test)
    assert np.array_equal(d1.indices_iniciales, d2.indices_iniciales)


def test_test_no_se_solapa_con_pool_por_tamano():
    datos = preparar_datos(n_muestras=400, tam_test=0.4)
    # 40% de 400 = 160 en test, 240 en pool.
    assert len(datos.X_test) == 160
    assert len(datos.X_pool) == 240


def test_ambas_estrategias_arrancan_igual():
    demo = _demo()
    assert demo.aleatoria.etiquetados == demo.incertidumbre.etiquetados
    assert demo.aleatoria.n_etiquetados == len(demo.datos.indices_iniciales)


def test_cada_ronda_agrega_misma_cantidad():
    demo = _demo()
    n0 = demo.aleatoria.n_etiquetados
    demo.avanzar_ronda()
    assert demo.aleatoria.n_etiquetados == n0 + demo.n_por_ronda
    assert demo.incertidumbre.n_etiquetados == n0 + demo.n_por_ronda


def test_seleccion_por_incertidumbre_elige_menor_confianza():
    """La estrategia de incertidumbre debe elegir los puntos de menor prob. máxima."""
    demo = _demo()
    est = demo.incertidumbre
    disponibles = est.indices_disponibles
    probas = est.modelo.predict_proba(demo.datos.X_pool[disponibles])
    confianza = probas.max(axis=1)
    orden = np.argsort(confianza)
    esperados = set(disponibles[orden[: demo.n_por_ronda]].tolist())
    assert set(est.seleccion_actual.tolist()) == esperados


def test_modelo_no_usa_etiquetas_antes_de_etiquetar():
    """Los índices seleccionados no deben estar entre los etiquetados hasta avanzar."""
    demo = _demo()
    seleccion = set(demo.incertidumbre.seleccion_actual.tolist())
    etiquetados = set(demo.incertidumbre.etiquetados)
    assert seleccion.isdisjoint(etiquetados)
    demo.avanzar_ronda()
    assert seleccion.issubset(set(demo.incertidumbre.etiquetados))


def test_precision_entre_0_y_1():
    demo = _demo()
    p = demo.aleatoria.precision_test()
    assert 0.0 <= p <= 1.0


def test_historial_crece_con_rondas():
    demo = _demo()
    assert len(demo.aleatoria.historial_precision) == 1  # registro inicial
    demo.avanzar_ronda()
    demo.avanzar_ronda()
    assert len(demo.aleatoria.historial_precision) == 3
    assert demo.aleatoria.historial_n_etiquetas[-1] == demo.aleatoria.n_etiquetados


def test_frontera_decision_tiene_forma_correcta():
    demo = _demo()
    xx, yy, zz = frontera_decision(demo.aleatoria.modelo, demo.datos, paso=50)
    assert xx.shape == (50, 50)
    assert zz.shape == (50, 50)
    assert np.all((zz >= 0.0) & (zz <= 1.0))


def test_no_quedan_disponibles_al_final():
    datos = preparar_datos(semilla=42)
    demo = Demostracion(datos=datos, n_por_ronda=50, semilla=42)
    for _ in range(20):
        if not demo.hay_disponibles:
            break
        demo.avanzar_ronda()
    # Debe detenerse sin error aunque se pida más de lo disponible.
    assert demo.aleatoria.n_etiquetados <= len(datos.X_pool)


def test_fijar_n_por_ronda_cambia_tamano_de_seleccion():
    """Cambiar el n por ronda debe reflejarse en la cantidad propuesta."""
    demo = _demo()
    demo.fijar_n_por_ronda(10)
    assert demo.n_por_ronda == 10
    assert len(demo.aleatoria.seleccion_actual) == 10
    assert len(demo.incertidumbre.seleccion_actual) == 10


def test_fijar_n_por_ronda_no_toca_lo_ya_etiquetado():
    """Ajustar el tamaño no debe alterar los ejemplos ya etiquetados."""
    demo = _demo()
    etiquetados_antes = list(demo.aleatoria.etiquetados)
    demo.fijar_n_por_ronda(3)
    assert demo.aleatoria.etiquetados == etiquetados_antes


def test_fijar_n_por_ronda_minimo_uno():
    demo = _demo()
    demo.fijar_n_por_ronda(0)
    assert demo.n_por_ronda == 1


def test_avanzar_respeta_n_elegido_por_usuario():
    demo = _demo()
    n0 = demo.aleatoria.n_etiquetados
    demo.fijar_n_por_ronda(8)
    demo.avanzar_ronda()
    assert demo.aleatoria.n_etiquetados == n0 + 8
    assert demo.incertidumbre.n_etiquetados == n0 + 8
