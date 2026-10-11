"""Pruebas del algoritmo k-medias, del modelo final y de la consulta por consola."""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from agrupar import K_ELEGIDO, NOMBRES_GRUPOS, ajustar_kmedias, ordenar_grupos
from consultar import ErrorEntrada, buscar_estacion, cargar, main
from datos import COLUMNAS_PERFIL
from kmedias import distancia_euclidiana, inercia, kmedias, silueta_de_un_punto


# --- Teoría del capítulo 16 ------------------------------------------------------
def test_distancia_euclidiana_conocida():
    assert distancia_euclidiana([0, 0], [3, 4]) == pytest.approx(5.0)
    assert distancia_euclidiana([1, 2, 3], [1, 2, 3]) == 0


def test_kmedias_propio_en_ejemplo_de_libro():
    """Seis puntos en dos nubes evidentes: el algoritmo debe separarlas."""
    X = np.array([[1, 1], [1.5, 2], [1, 0.5], [8, 8], [9, 8.5], [8.5, 9]])
    r = kmedias(X, centroides_iniciales=X[[0, 1]])
    assert len(set(r.etiquetas[:3])) == 1 and len(set(r.etiquetas[3:])) == 1
    assert r.etiquetas[0] != r.etiquetas[3]
    assert np.allclose(sorted(r.centroides[:, 0]), [7 / 6, 8.5])


def test_la_inercia_nunca_aumenta():
    X = np.random.default_rng(0).normal(size=(200, 5))
    r = kmedias(X, X[:4])
    assert all(b <= a + 1e-9 for a, b in zip(r.historial_inercia, r.historial_inercia[1:]))


def test_kmedias_propio_igual_a_sklearn(perfiles_guardados):
    X = perfiles_guardados[COLUMNAS_PERFIL].to_numpy(dtype=float)
    ini = X[[0, 50, 100]]
    propio = kmedias(X, ini)
    sk = KMeans(n_clusters=3, init=ini, n_init=1, algorithm="lloyd", tol=0).fit(X)
    assert adjusted_rand_score(propio.etiquetas, sk.labels_) == 1.0
    assert propio.inercia == pytest.approx(sk.inertia_, rel=1e-9)


def test_silueta_manual_igual_a_sklearn(perfiles_guardados, resultados):
    X = perfiles_guardados[COLUMNAS_PERFIL].to_numpy(dtype=float)
    a = resultados["asignacion"].set_index("codigo_estacion").loc[
        perfiles_guardados["codigo_estacion"]]
    etiquetas = a["grupo"].to_numpy()
    sk = silhouette_samples(X, etiquetas)
    for i in (0, 40, 151):
        assert silueta_de_un_punto(X, etiquetas, i) == pytest.approx(sk[i], abs=1e-9)


# --- Modelo final ---------------------------------------------------------------------
def test_tres_grupos_y_todas_las_estaciones_asignadas(resultados):
    a = resultados["asignacion"]
    assert len(a) == 152 and set(a["grupo"]) == {0, 1, 2}
    assert a["grupo"].value_counts().min() >= 20          # ningún grupo marginal


def test_cada_estacion_esta_en_el_grupo_del_centroide_mas_cercano(perfiles_guardados, resultados):
    X = perfiles_guardados[COLUMNAS_PERFIL].to_numpy(dtype=float)
    C = resultados["centroides"][COLUMNAS_PERFIL].to_numpy(dtype=float)
    a = resultados["asignacion"].set_index("codigo_estacion").loc[
        perfiles_guardados["codigo_estacion"]]
    mas_cercano = np.sqrt(((X[:, None] - C[None]) ** 2).sum(axis=2)).argmin(axis=1)
    assert (mas_cercano == a["grupo"].to_numpy()).all()


def test_centroide_es_la_media_de_su_grupo(perfiles_guardados, resultados):
    X = perfiles_guardados[COLUMNAS_PERFIL].to_numpy(dtype=float)
    a = resultados["asignacion"].set_index("codigo_estacion").loc[
        perfiles_guardados["codigo_estacion"]]
    C = resultados["centroides"][COLUMNAS_PERFIL].to_numpy(dtype=float)
    for g in range(K_ELEGIDO):
        assert np.allclose(X[a["grupo"].to_numpy() == g].mean(axis=0), C[g], atol=1e-3)


def test_calidad_minima_del_agrupamiento(resultados):
    m = resultados["metricas"]
    assert m["silueta_media"] > 0.3
    assert (resultados["asignacion"]["silueta"] > 0).all()   # ninguna estación mal ubicada


def test_estable_frente_a_la_semilla(perfiles_guardados):
    X = perfiles_guardados[COLUMNAS_PERFIL].to_numpy(dtype=float)
    base = ajustar_kmedias(X, semilla=0)[1]
    for s in (1, 7, 123):
        assert (ajustar_kmedias(X, semilla=s)[1] == base).all()


def test_estable_entre_quincenas(resultados):
    for q in resultados["metricas"]["estabilidad_quincenas"].values():
        assert q["porcentaje"] >= 95


def test_ward_coincide_con_kmedias_en_la_mayoria(resultados):
    assert resultados["metricas"]["ward"]["coincidencia_pct"] >= 80


def test_ordenar_grupos_es_determinista():
    """El grupo 0 es siempre el de mayor diferencia (punta mañana − punta tarde)."""
    C = np.zeros((3, len(COLUMNAS_PERFIL)))
    iam = [COLUMNAS_PERFIL.index(f"H0{h}") for h in (6, 7, 8)]
    ipm = [COLUMNAS_PERFIL.index(f"H{h}") for h in (16, 17, 18, 19)]
    C[0, ipm] = 10           # grupo de la tarde
    C[1, iam] = 10           # grupo de la mañana
    assert ordenar_grupos(C).tolist() == [2, 0, 1]


# --- Interpretación (casos de sentido común) -------------------------------------------
@pytest.mark.parametrize("estacion, grupo", [
    ("Calle 72", 0), ("Calle 100 - Marketmedios", 0), ("Museo del Oro", 0),
    ("Portal Américas", 2), ("Portal Suba", 2), ("Cabecera Usme", 2),
])
def test_estaciones_conocidas_en_el_grupo_esperado(resultados, estacion, grupo):
    a = resultados["asignacion"].set_index("estacion")
    assert a.loc[estacion, "grupo"] == grupo
    assert a.loc[estacion, "nombre_grupo"] == NOMBRES_GRUPOS[grupo]


def test_los_grupos_tienen_el_perfil_que_dice_su_nombre(resultados):
    g = {r["grupo"]: r for r in resultados["metricas"]["grupos"]}
    assert g[0]["hora_pico_habil"] in (6, 7, 8) and g[0]["pct_punta_manana"] > g[0]["pct_punta_tarde"]
    assert g[2]["hora_pico_habil"] in (17, 18, 19) and g[2]["pct_punta_tarde"] > g[2]["pct_punta_manana"]
    assert abs(g[1]["pct_punta_manana"] - g[1]["pct_punta_tarde"]) < 5


# --- Consulta por consola --------------------------------------------------------------
@pytest.mark.parametrize("consulta, nombre", [
    ("09122", "Calle 72"), ("calle 72", "Calle 72"), ("portal norte", "Cabecera Autopista Norte"),
    ("portal americas", "Portal Américas"), ("Ricuarte", "Ricaurte"),
])
def test_buscar_estacion(consulta, nombre):
    datos, _ = cargar()
    assert datos.loc[buscar_estacion(datos, consulta), "estacion"] == nombre


@pytest.mark.parametrize("consulta", ["calle", "xyz123", "  "])
def test_buscar_estacion_invalida(consulta):
    datos, _ = cargar()
    with pytest.raises(ErrorEntrada):
        buscar_estacion(datos, consulta)


def test_cli_codigos_de_salida(capsys):
    assert main(["--estacion", "Calle 72"]) == 0
    assert "Destino (pico de la mañana)" in capsys.readouterr().out
    assert main(["--resumen"]) == 0
    assert main(["--comparar", "Calle 72", "Portal Americas"]) == 0
    assert main(["--grupo", "7"]) == 1
    assert main([]) == 1
