"""
Paso 1: preparación de los datos.

Lee el CSV original de TransMilenio, lo limpia, genera el dataset por hora y
el dataset de perfiles horarios por estación (entrada del agrupamiento).

Uso:
    python src/preparar_datos.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datos import (COLUMNAS_FINDE, COLUMNAS_HABIL, HORAS, RUTA_CRUDO, RUTA_FIGURAS,
                   RUTA_PERFILES, RUTA_POR_HORA, RUTA_RESULTADOS, agregar_por_hora,
                   cargar_crudo, columnas_dia, construir_perfiles, limpiar)
from graficas import COLORES_SERIES, TEXTO_SECUNDARIO, estilo_base

# Un ejemplo de cada tipo de estación, con el color que luego tendrá su grupo
EJEMPLOS = {"02300": ("Calle 100", COLORES_SERIES[0]),
            "04104": ("Avenida 68", COLORES_SERIES[2]),
            "05000": ("Portal Américas", COLORES_SERIES[1])}


def grafica_ejemplos(perfiles):
    """Perfil de tres estaciones con comportamientos distintos."""
    estilo_base()
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for eje, cols, titulo in [(ejes[0], COLUMNAS_HABIL, "Día hábil"),
                              (ejes[1], COLUMNAS_FINDE, "Fin de semana")]:
        for codigo, (nombre, color) in EJEMPLOS.items():
            fila = perfiles.loc[perfiles["codigo_estacion"] == codigo, cols].iloc[0]
            eje.plot(HORAS, fila.values, color=color, lw=2, marker="o", ms=3,
                     label=nombre)
        eje.set_title(titulo, loc="left")
        eje.set_xticks(HORAS[::2])
        eje.set_xlabel("Hora del día")
    ejes[0].set_ylabel("% de las salidas del día")
    ejes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Perfil horario de tres estaciones (cada curva suma 100 %)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    ruta = RUTA_FIGURAS / "perfiles_ejemplo.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def main():
    print("Leyendo", RUTA_CRUDO)
    crudo = cargar_crudo()
    limpio = limpiar(crudo)
    por_hora = agregar_por_hora(limpio)
    perfiles = construir_perfiles(por_hora)

    for ruta in (RUTA_POR_HORA.parent, RUTA_FIGURAS):
        ruta.mkdir(parents=True, exist_ok=True)
    por_hora.to_csv(RUTA_POR_HORA, index=False, encoding="utf-8")
    perfiles.to_csv(RUTA_PERFILES, index=False, encoding="utf-8")

    resumen = {
        "filas_originales": int(len(crudo)),
        "filas_vacias_eliminadas": int(crudo["Línea"].isna().sum()),
        "filas_limpias": int(len(limpio)),
        "estaciones": int(limpio["codigo_estacion"].nunique()),
        "zonas_troncales": int(limpio["codigo_zona"].nunique()),
        "accesos": int(limpio["acceso"].nunique()),
        "salidas_totales_mes": int(limpio[columnas_dia()].to_numpy().sum()),
        "filas_por_hora": int(len(por_hora)),
        "filas_perfiles": int(len(perfiles)),
        "variables_perfil": len(COLUMNAS_HABIL) + len(COLUMNAS_FINDE),
    }
    with open(RUTA_RESULTADOS / "resumen_datos.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    fig = grafica_ejemplos(perfiles)

    print(f"Filas originales:        {resumen['filas_originales']:,}")
    print(f"Filas vacías eliminadas: {resumen['filas_vacias_eliminadas']:,}")
    print(f"Salidas totales del mes: {resumen['salidas_totales_mes']:,}")
    print(f"Dataset por hora:        {resumen['filas_por_hora']:,} filas -> {RUTA_POR_HORA.name}")
    print(f"Perfiles:                {resumen['filas_perfiles']} estaciones × "
          f"{resumen['variables_perfil']} variables -> {RUTA_PERFILES.name}")
    print(f"Gráfica:                 {fig}")


if __name__ == "__main__":
    main()
