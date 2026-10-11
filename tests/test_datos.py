"""Pruebas de la limpieza de datos y de la construcción de los perfiles."""
import numpy as np
import pandas as pd
import pytest

from datos import (COLUMNAS_FINDE, COLUMNAS_HABIL, COLUMNAS_PERFIL, DIAS_DEL_MES, HORAS,
                   a_entero, columnas_dia, construir_perfiles, limpiar, normalizar,
                   separar_codigo)


def test_separador_de_miles():
    assert a_entero(pd.Series(["0", "45", "1.026", "12.345"])).tolist() == [0, 45, 1026, 12345]


def test_separar_codigo_y_nombre():
    assert separar_codigo("(09122) Calle 72") == ("09122", "Calle 72")


def test_normalizar_quita_tildes_y_signos():
    assert normalizar("Héroes - Gel hada") == "heroes gel hada"


def test_limpiar_elimina_vacias_corrales_y_valida_total():
    dias = columnas_dia()
    base = {"Línea": "(11) Zona K Calle 26", "Acceso de Estación": "(01) Acceso", "MES": "ABRIL",
            "INTERVALO": "07:15", **{d: "1" for d in dias}, "Total general": str(DIAS_DEL_MES)}
    filas = [dict(base, **{"Estación": "(06001) Modelia"}),
             dict(base, **{"Estación": "(50008) Corral Portal Dorado"}),
             {k: np.nan for k in base} | {"Estación": np.nan}]
    assert limpiar(pd.DataFrame(filas))["estacion"].tolist() == ["Modelia"]
    with pytest.raises(ValueError):
        limpiar(pd.DataFrame([dict(base, **{"Estación": "(06001) Modelia",
                                            "Total general": "999"})]))


def test_limpieza_conserva_las_salidas(limpio, crudo):
    validas = crudo.dropna(subset=["Línea"])
    excluidas = validas["Estación"].str.contains("Corral|Bicicletero", case=False)
    esperado = a_entero(validas["Total general"]).sum() - a_entero(
        validas.loc[excluidas, "Total general"]).sum()
    assert limpio[columnas_dia()].to_numpy().sum() == esperado


def test_dataset_por_hora_completo(por_hora):
    n = por_hora["codigo_estacion"].nunique()
    assert len(por_hora) == n * DIAS_DEL_MES * len(HORAS)
    assert not por_hora.duplicated(["codigo_estacion", "dia", "hora"]).any()


def test_una_fila_por_estacion_y_36_variables(perfiles):
    assert len(perfiles) == 152
    assert perfiles["codigo_estacion"].is_unique
    assert len(COLUMNAS_PERFIL) == 36
    assert not perfiles[COLUMNAS_PERFIL].isna().any().any()


def test_cada_perfil_suma_100(perfiles):
    assert np.allclose(perfiles[COLUMNAS_HABIL].sum(axis=1), 100, atol=0.01)
    assert np.allclose(perfiles[COLUMNAS_FINDE].sum(axis=1), 100, atol=0.01)


def test_perfil_calculado_a_mano():
    """Estación ficticia: 2 días hábiles y 1 sábado con valores conocidos."""
    filas = []
    for dia, tipo, por_hora in [(1, 0, {7: 30, 18: 10}), (2, 0, {7: 10, 18: 10}),
                                (6, 1, {7: 5, 18: 15})]:
        for h in HORAS:
            filas.append({"fecha": f"2024-04-{dia:02d}", "dia": dia, "dia_semana": 0,
                          "tipo_dia": tipo, "hora": h, "codigo_zona": "1", "zona": "Z",
                          "codigo_estacion": "X", "estacion": "Prueba", "num_accesos": 1,
                          "es_portal": 0, "salidas": por_hora.get(h, 0)})
    p = construir_perfiles(pd.DataFrame(filas)).iloc[0]
    # Hábil: promedio 7h = 20, 18h = 10 -> 66,67 % y 33,33 %
    assert p["H07"] == pytest.approx(66.6667, abs=1e-3)
    assert p["H18"] == pytest.approx(33.3333, abs=1e-3)
    # Fin de semana: 7h = 5, 18h = 15 -> 25 % y 75 %
    assert p["F07"] == pytest.approx(25.0) and p["F18"] == pytest.approx(75.0)
    assert p["relacion_finde_habil"] == pytest.approx(20 / 30, abs=1e-3)


def test_archivo_de_perfiles_es_reproducible(perfiles, perfiles_guardados):
    pd.testing.assert_frame_equal(
        perfiles[COLUMNAS_PERFIL].reset_index(drop=True),
        perfiles_guardados[COLUMNAS_PERFIL].reset_index(drop=True), check_exact=False, atol=1e-4)
