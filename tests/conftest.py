import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datos import RUTA_PERFILES, RUTA_POR_HORA, RUTA_RESULTADOS  # noqa: E402


@pytest.fixture(scope="session")
def crudo():
    from datos import cargar_crudo
    return cargar_crudo()


@pytest.fixture(scope="session")
def limpio(crudo):
    from datos import limpiar
    return limpiar(crudo)


@pytest.fixture(scope="session")
def por_hora(limpio):
    from datos import agregar_por_hora
    return agregar_por_hora(limpio)


@pytest.fixture(scope="session")
def perfiles(por_hora):
    from datos import construir_perfiles
    return construir_perfiles(por_hora)


@pytest.fixture(scope="session")
def perfiles_guardados():
    if not RUTA_PERFILES.exists():
        pytest.skip("Ejecute primero src/preparar_datos.py")
    return pd.read_csv(RUTA_PERFILES, dtype={"codigo_estacion": str, "codigo_zona": str})


@pytest.fixture(scope="session")
def resultados():
    ruta = RUTA_RESULTADOS / "asignacion_estaciones.csv"
    if not ruta.exists():
        pytest.skip("Ejecute primero src/agrupar.py")
    return {
        "asignacion": pd.read_csv(ruta, dtype={"codigo_estacion": str}),
        "centroides": pd.read_csv(RUTA_RESULTADOS / "centroides.csv"),
        "metricas": json.loads((RUTA_RESULTADOS / "metricas.json").read_text(encoding="utf-8")),
    }
