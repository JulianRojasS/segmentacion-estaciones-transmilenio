"""
Paso 3: consultar los resultados del agrupamiento por consola.

Ejemplos:
    python src/consultar.py --resumen
    python src/consultar.py --estacion "Calle 72"
    python src/consultar.py --estacion "portal americas"
    python src/consultar.py --grupo 1
    python src/consultar.py --comparar "Calle 72" "Portal Americas"
"""
from __future__ import annotations

import argparse
import difflib
import sys

import numpy as np
import pandas as pd

from datos import (ALIAS_ESTACIONES, COLUMNAS_FINDE, COLUMNAS_HABIL, COLUMNAS_PERFIL, HORAS,
                   RUTA_PERFILES, RUTA_RESULTADOS, normalizar)
from kmedias import distancia_euclidiana

RUTA_ASIGNACION = RUTA_RESULTADOS / "asignacion_estaciones.csv"
RUTA_CENTROIDES = RUTA_RESULTADOS / "centroides.csv"


class ErrorEntrada(ValueError):
    """Error en lo que escribió el usuario."""


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    for ruta in (RUTA_PERFILES, RUTA_ASIGNACION, RUTA_CENTROIDES):
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}. Ejecute primero "
                                    "python src/preparar_datos.py y python src/agrupar.py")
    perfiles = pd.read_csv(RUTA_PERFILES, dtype={"codigo_estacion": str, "codigo_zona": str})
    asignacion = pd.read_csv(RUTA_ASIGNACION, dtype={"codigo_estacion": str})
    datos = perfiles.merge(asignacion[["codigo_estacion", "grupo", "nombre_grupo",
                                       "distancia_centroide", "silueta"]], on="codigo_estacion")
    centroides = pd.read_csv(RUTA_CENTROIDES)
    return datos.set_index("codigo_estacion"), centroides


def buscar_estacion(datos: pd.DataFrame, consulta: str) -> str:
    """Devuelve el código de la estación: acepta código, nombre exacto, aproximado o alias."""
    consulta = consulta.strip()
    if consulta in datos.index:
        return consulta
    q = normalizar(consulta)
    if not q:
        raise ErrorEntrada("Debe indicar el nombre de una estación.")
    q = normalizar(ALIAS_ESTACIONES.get(q, q))
    nombres = {c: normalizar(n) for c, n in datos["estacion"].items()}
    exactas = [c for c, n in nombres.items() if n == q]
    if len(exactas) == 1:
        return exactas[0]
    contienen = [c for c, n in nombres.items() if q in n]
    if len(contienen) == 1:
        return contienen[0]
    if len(contienen) > 1:
        opciones = ", ".join(sorted(datos.loc[contienen, "estacion"]))
        raise ErrorEntrada(f"'{consulta}' es ambiguo. Coincide con: {opciones}")
    parecida = difflib.get_close_matches(q, list(nombres.values()), n=1, cutoff=0.75)
    if parecida:
        return next(c for c, n in nombres.items() if n == parecida[0])
    raise ErrorEntrada(f"No se encontró la estación '{consulta}'. "
                       "Use --listar para ver los nombres disponibles.")


def mas_parecidas(datos: pd.DataFrame, codigo: str, n: int = 5) -> pd.Series:
    X = datos[COLUMNAS_PERFIL].to_numpy(dtype=float)
    x = datos.loc[codigo, COLUMNAS_PERFIL].to_numpy(dtype=float)
    d = pd.Series(np.sqrt(((X - x) ** 2).sum(axis=1)), index=datos.index).drop(codigo)
    return d.nsmallest(n)


def distancias_a_centroides(datos, centroides, codigo) -> list[float]:
    x = datos.loc[codigo, COLUMNAS_PERFIL].to_numpy(dtype=float)
    return [distancia_euclidiana(x, centroides.loc[g, COLUMNAS_PERFIL].to_numpy(dtype=float))
            for g in range(len(centroides))]


def hora_pico(fila, columnas) -> int:
    return HORAS[int(np.argmax(fila[columnas].to_numpy(dtype=float)))]


# ---------------------------------------------------------------------------
def mostrar_estacion(datos, centroides, codigo):
    e = datos.loc[codigo]
    g = int(e["grupo"])
    print(f"Estación: {e['estacion']} ({codigo}) – {e['zona']}")
    print(f"Grupo:    {g}. {e['nombre_grupo']}\n")
    print("Perfil de la estación:")
    print(f"  Salidas en un día hábil promedio:  {int(e['salidas_dia_habil']):,}")
    print(f"  Hora pico día hábil / fin de semana: {hora_pico(e, COLUMNAS_HABIL)}:00 / "
          f"{hora_pico(e, COLUMNAS_FINDE)}:00")
    print(f"  % de salidas en la punta de la mañana (6-9 h): {e['pct_punta_manana']:.1f} %")
    print(f"  % de salidas en la punta de la tarde (16-20 h): {e['pct_punta_tarde']:.1f} %")
    print(f"  Fin de semana / día hábil: {e['relacion_finde_habil']:.2f}\n")
    print("Distancia euclidiana a cada centroide (el grupo es el más cercano):")
    for i, d in enumerate(distancias_a_centroides(datos, centroides, codigo)):
        marca = "  <- asignado" if i == g else ""
        print(f"  Grupo {i} ({centroides.loc[i, 'nombre']}): {d:6.2f}{marca}")
    print(f"\nCoeficiente de silueta: {e['silueta']:.3f} "
          f"({'bien ubicada' if e['silueta'] >= 0.25 else 'en la frontera entre grupos'})")
    print("\nLas 5 estaciones con el perfil más parecido:")
    for c, d in mas_parecidas(datos, codigo).items():
        print(f"  {datos.loc[c, 'estacion']:<40} grupo {int(datos.loc[c, 'grupo'])}  "
              f"distancia {d:5.2f}")


def mostrar_grupo(datos, centroides, g):
    if g not in range(len(centroides)):
        raise ErrorEntrada(f"El grupo debe estar entre 0 y {len(centroides) - 1}.")
    m = datos[datos["grupo"] == g].sort_values("distancia_centroide")
    print(f"Grupo {g}. {centroides.loc[g, 'nombre']}: {len(m)} estaciones "
          "(de la más típica a la menos típica)\n")
    print(f"{'Estación':<40} {'Zona':<24} {'Dist.':>6} {'Silueta':>8} {'Salidas/día':>12}")
    for _, e in m.iterrows():
        print(f"{e['estacion']:<40} {e['zona']:<24} {e['distancia_centroide']:6.2f} "
              f"{e['silueta']:8.3f} {int(e['salidas_dia_habil']):>12,}")


def mostrar_resumen(datos, centroides):
    print(f"{len(datos)} estaciones agrupadas en {len(centroides)} grupos con k-medias\n")
    print(f"{'Grupo':<44} {'Est.':>5} {'Pico hábil':>11} {'Punta AM':>9} {'Punta PM':>9} "
          f"{'Finde/hábil':>12}")
    for g in range(len(centroides)):
        m = datos[datos["grupo"] == g]
        pico = HORAS[int(np.argmax(centroides.loc[g, COLUMNAS_HABIL].to_numpy(dtype=float)))]
        print(f"{g}. {centroides.loc[g, 'nombre']:<41} {len(m):>5} {pico:>9}:00 "
              f"{m['pct_punta_manana'].mean():>8.1f}% {m['pct_punta_tarde'].mean():>8.1f}% "
              f"{m['relacion_finde_habil'].mean():>12.2f}")


def comparar(datos, a, b):
    ea, eb = datos.loc[a], datos.loc[b]
    xa = ea[COLUMNAS_PERFIL].to_numpy(dtype=float)
    xb = eb[COLUMNAS_PERFIL].to_numpy(dtype=float)
    dif = xa - xb
    print(f"{ea['estacion']} (grupo {int(ea['grupo'])})  vs  {eb['estacion']} (grupo {int(eb['grupo'])})\n")
    print("Horas con mayor diferencia en el perfil (puntos porcentuales):")
    for i in np.argsort(-np.abs(dif))[:5]:
        print(f"  {COLUMNAS_PERFIL[i]}: {xa[i]:5.1f} % vs {xb[i]:5.1f} %  (diferencia {dif[i]:+.1f})")
    print(f"\nDistancia euclidiana = raíz(suma de las 36 diferencias al cuadrado) = "
          f"{distancia_euclidiana(xa, xb):.2f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Consulta los grupos de estaciones de TransMilenio.")
    p.add_argument("--estacion", help="Nombre o código de una estación")
    p.add_argument("--grupo", type=int, help="Número de grupo a listar")
    p.add_argument("--resumen", action="store_true", help="Resumen de los grupos")
    p.add_argument("--comparar", nargs=2, metavar=("A", "B"), help="Compara dos estaciones")
    p.add_argument("--listar", action="store_true", help="Lista las estaciones disponibles")
    args = p.parse_args(argv)
    try:
        datos, centroides = cargar()
        if args.listar:
            for c, e in datos.sort_values(["zona", "estacion"]).iterrows():
                print(f"{c}  {e['estacion']:<45} {e['zona']}")
        elif args.resumen:
            mostrar_resumen(datos, centroides)
        elif args.estacion:
            mostrar_estacion(datos, centroides, buscar_estacion(datos, args.estacion))
        elif args.grupo is not None:
            mostrar_grupo(datos, centroides, args.grupo)
        elif args.comparar:
            comparar(datos, *(buscar_estacion(datos, x) for x in args.comparar))
        else:
            raise ErrorEntrada("Indique --resumen, --estacion, --grupo, --comparar o --listar.")
    except (ErrorEntrada, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
