"""
Carga, limpieza y transformación de los datos de TransMilenio.

Fuente: "Consolidado de Salidas Sistema Troncal por Franja Horaria",
TRANSMILENIO S.A., Datos Abiertos Bogotá (licencia CC BY 4.0), abril de 2024.

Pasos:
  1. limpiar()            filas vacías, separador de miles, corrales, validación de totales
  2. agregar_por_hora()   una fila por estación, día y hora
  3. construir_perfiles() una fila por estación con su perfil horario de demanda
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas y parámetros
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_CRUDO = RAIZ / "data" / "raw" / "salidas_troncal_abril_2024.csv"
RUTA_POR_HORA = RAIZ / "data" / "processed" / "salidas_por_hora.csv"
RUTA_PERFILES = RAIZ / "data" / "processed" / "perfiles_estaciones.csv"
RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"

ANIO, MES, DIAS_DEL_MES = 2024, 4, 30
HORA_INICIO, HORA_FIN = 5, 22
HORAS = list(range(HORA_INICIO, HORA_FIN + 1))          # 18 horas de operación
PATRON_EXCLUIR = re.compile(r"corral|bicicletero", re.IGNORECASE)

# Prefijos de las columnas del perfil: H = día hábil, F = fin de semana (sábado y domingo)
COLUMNAS_HABIL = [f"H{h:02d}" for h in HORAS]
COLUMNAS_FINDE = [f"F{h:02d}" for h in HORAS]
COLUMNAS_PERFIL = COLUMNAS_HABIL + COLUMNAS_FINDE         # 36 variables del agrupamiento


# ---------------------------------------------------------------------------
# 1. Limpieza
# ---------------------------------------------------------------------------
def columnas_dia() -> list[str]:
    return [f"DÍA {d:02d}" for d in range(1, DIAS_DEL_MES + 1)]


def a_entero(serie: pd.Series) -> pd.Series:
    """'1.026' -> 1026: el archivo usa punto como separador de miles."""
    return pd.to_numeric(serie.astype(str).str.strip().str.replace(".", "", regex=False),
                         errors="raise").astype(int)


def separar_codigo(texto: str) -> tuple[str, str]:
    """'(09122) Calle 72' -> ('09122', 'Calle 72')."""
    m = re.match(r"^\((\w+)\)\s*(.+)$", str(texto).strip())
    return (m.group(1), m.group(2).strip()) if m else ("", str(texto).strip())


def cargar_crudo(ruta: Path = RUTA_CRUDO) -> pd.DataFrame:
    """CSV original: separador ';', codificación Windows-1252."""
    return pd.read_csv(ruta, sep=";", encoding="cp1252", dtype=str)


def limpiar(crudo: pd.DataFrame) -> pd.DataFrame:
    df = crudo.dropna(subset=["Línea", "Estación"]).copy()
    df = df[~df["Estación"].str.contains(PATRON_EXCLUIR)]
    dias = columnas_dia()
    for c in dias:
        df[c] = a_entero(df[c])
    if not (df[dias].sum(axis=1) == a_entero(df["Total general"])).all():
        raise ValueError("La suma de los días no coincide con 'Total general'")
    df["codigo_estacion"], df["estacion"] = zip(*df["Estación"].map(separar_codigo))
    df["codigo_zona"], df["zona"] = zip(*df["Línea"].map(separar_codigo))
    df["acceso"] = df["Acceso de Estación"].str.strip()
    df["hora"] = df["INTERVALO"].str[:2].astype(int)
    return df[["codigo_zona", "zona", "codigo_estacion", "estacion", "acceso",
               "INTERVALO", "hora"] + dias]


# ---------------------------------------------------------------------------
# 2. Agregación por estación, día y hora
# ---------------------------------------------------------------------------
def tipo_dia(d: date) -> int:
    """0 = hábil, 1 = sábado, 2 = domingo (abril de 2024 no tuvo festivos)."""
    return 2 if d.weekday() == 6 else (1 if d.weekday() == 5 else 0)


def agregar_por_hora(limpio: pd.DataFrame) -> pd.DataFrame:
    largo = limpio.melt(id_vars=["codigo_zona", "zona", "codigo_estacion", "estacion", "hora"],
                        value_vars=columnas_dia(), var_name="columna_dia", value_name="salidas")
    largo["dia"] = largo["columna_dia"].str[-2:].astype(int)
    claves = ["codigo_zona", "zona", "codigo_estacion", "estacion", "dia", "hora"]
    por_hora = largo.groupby(claves, as_index=False)["salidas"].sum()

    estaciones = por_hora[claves[:4]].drop_duplicates()
    rejilla = (estaciones.merge(pd.DataFrame({"dia": range(1, DIAS_DEL_MES + 1)}), how="cross")
               .merge(pd.DataFrame({"hora": HORAS}), how="cross"))
    por_hora = rejilla.merge(por_hora, how="left", on=claves)
    por_hora["salidas"] = por_hora["salidas"].fillna(0).astype(int)

    fechas = por_hora["dia"].map(lambda d: date(ANIO, MES, d))
    por_hora["fecha"] = fechas.map(date.isoformat)
    por_hora["dia_semana"] = fechas.map(lambda f: f.weekday())
    por_hora["tipo_dia"] = fechas.map(tipo_dia)
    accesos = limpio.groupby("codigo_estacion")["acceso"].nunique()
    por_hora["num_accesos"] = por_hora["codigo_estacion"].map(accesos).astype(int)
    por_hora["es_portal"] = por_hora["estacion"].str.contains(r"portal|cabecera",
                                                              case=False).astype(int)
    columnas = ["fecha", "dia", "dia_semana", "tipo_dia", "hora", "codigo_zona", "zona",
                "codigo_estacion", "estacion", "num_accesos", "es_portal", "salidas"]
    return por_hora[columnas].sort_values(["codigo_estacion", "dia", "hora"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Perfil horario de cada estación (la entrada del agrupamiento)
# ---------------------------------------------------------------------------
def construir_perfiles(por_hora: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada estación calcula:
      - H05..H22: % de las salidas de un día hábil promedio que ocurre en cada hora
      - F05..F22: % de las salidas de un día de fin de semana promedio en cada hora
    Cada bloque de 18 columnas suma 100. Así se compara la FORMA de la demanda
    y no su tamaño: una estación pequeña y un portal pueden quedar en el mismo grupo.

    Columnas descriptivas (no entran al agrupamiento): demanda diaria, relación
    fin de semana / hábil, % de salidas en la punta de la mañana y de la tarde.
    """
    df = por_hora.copy()
    df["bloque"] = np.where(df["tipo_dia"] == 0, "H", "F")
    media = (df.groupby(["codigo_estacion", "bloque", "hora"])["salidas"].mean()
             .unstack(["bloque", "hora"]))
    habil, finde = media["H"][HORAS], media["F"][HORAS]
    total_h, total_f = habil.sum(axis=1), finde.sum(axis=1)

    perfiles = pd.concat([
        habil.div(total_h, axis=0).mul(100).set_axis(COLUMNAS_HABIL, axis=1),
        finde.div(total_f, axis=0).mul(100).set_axis(COLUMNAS_FINDE, axis=1),
    ], axis=1)

    info = (df.drop_duplicates("codigo_estacion")
            .set_index("codigo_estacion")[["estacion", "codigo_zona", "zona",
                                           "num_accesos", "es_portal"]])
    perfiles = info.join(perfiles)
    perfiles["salidas_dia_habil"] = total_h.round(0).astype(int)
    perfiles["salidas_dia_finde"] = total_f.round(0).astype(int)
    perfiles["relacion_finde_habil"] = (total_f / total_h).round(3)
    perfiles["pct_punta_manana"] = perfiles[[f"H{h:02d}" for h in (6, 7, 8)]].sum(axis=1).round(2)
    perfiles["pct_punta_tarde"] = perfiles[[f"H{h:02d}" for h in (16, 17, 18, 19)]].sum(axis=1).round(2)
    perfiles[COLUMNAS_PERFIL] = perfiles[COLUMNAS_PERFIL].round(4)
    return perfiles.reset_index().sort_values("codigo_estacion").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Búsqueda de estaciones por nombre
# ---------------------------------------------------------------------------
ALIAS_ESTACIONES = {
    "portal norte": "Cabecera Autopista Norte",
    "portal usme": "Cabecera Usme",
    "portal 80": "Cabecera Calle 80",
    "portal calle 80": "Cabecera Calle 80",
    "portal sur": "Portal Sur JFK Coop. Financiera",
    "heroes": "Heroes - Gel hada",
    "calle 100": "Calle 100 - Marketmedios",
    "jimenez": "Avenida Jimenez",
    "av jimenez": "Avenida Jimenez",
}


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni signos: 'Héroes' -> 'heroes'."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
