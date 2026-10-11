"""
Paso 2: agrupamiento no supervisado de las estaciones troncales.

Cada estación se describe con su perfil horario de demanda (36 variables:
% de salidas por hora en día hábil y en fin de semana). Sin usar ninguna
etiqueta, el algoritmo busca grupos de estaciones con perfiles parecidos.

  1. Selección del número de grupos k: método del codo, coeficiente de silueta
     e índice de Davies-Bouldin para k = 2..8.
  2. K-medias (scikit-learn) con k elegido; comprobación con la implementación
     propia de src/kmedias.py.
  3. Agrupamiento jerárquico aglomerativo (método de Ward) y dendrograma.
  4. Validación: estabilidad frente a la semilla y frente al periodo
     (primera vs. segunda quincena de abril).
  5. Interpretación: perfil promedio de cada grupo, PCA en 2D, composición por zona.

Uso:
    python src/agrupar.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, silhouette_samples,
                             silhouette_score)

from datos import (COLUMNAS_FINDE, COLUMNAS_HABIL, COLUMNAS_PERFIL, HORAS, RUTA_FIGURAS,
                   RUTA_PERFILES, RUTA_POR_HORA, RUTA_RESULTADOS, construir_perfiles)
from graficas import COLORES_SERIES, REJILLA, TEXTO_PRIMARIO, TEXTO_SECUNDARIO, estilo_base
from kmedias import inercia, kmedias, silueta_de_un_punto

SEMILLA = 42
K_ELEGIDO = 3
RANGO_K = range(2, 9)

# Los grupos se ordenan por (punta de la mañana − punta de la tarde) del centroide,
# de mayor a menor, para que el número de cada grupo sea siempre el mismo.
NOMBRES_GRUPOS = {
    0: "Destino (pico de la mañana)",
    1: "Mixta",
    2: "Origen residencial (pico de la tarde)",
}
COLOR_GRUPO = {0: COLORES_SERIES[0], 1: COLORES_SERIES[2], 2: COLORES_SERIES[1]}


# ---------------------------------------------------------------------------
def cargar_perfiles() -> pd.DataFrame:
    return pd.read_csv(RUTA_PERFILES, dtype={"codigo_estacion": str, "codigo_zona": str})


def matriz(perfiles: pd.DataFrame) -> np.ndarray:
    """Las 36 variables están en la misma unidad (% del día), así que no se escalan."""
    return perfiles[COLUMNAS_PERFIL].to_numpy(dtype=float)


def evaluar_k(X) -> pd.DataFrame:
    filas = []
    for k in [1, *RANGO_K]:
        km = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit(X)
        fila = {"k": k, "inercia": round(float(km.inertia_), 1)}
        if k > 1:
            fila["silueta"] = round(float(silhouette_score(X, km.labels_)), 4)
            fila["davies_bouldin"] = round(float(davies_bouldin_score(X, km.labels_)), 4)
            fila["tamanos"] = sorted(np.bincount(km.labels_).tolist(), reverse=True)
        filas.append(fila)
    return pd.DataFrame(filas)


def ordenar_grupos(centroides: np.ndarray) -> np.ndarray:
    """Devuelve el nuevo número de cada grupo según (punta mañana − punta tarde)."""
    idx = {c: i for i, c in enumerate(COLUMNAS_PERFIL)}
    manana = centroides[:, [idx[f"H{h:02d}"] for h in (6, 7, 8)]].sum(axis=1)
    tarde = centroides[:, [idx[f"H{h:02d}"] for h in (16, 17, 18, 19)]].sum(axis=1)
    orden = np.argsort(-(manana - tarde))          # orden[i] = grupo original en la posición i
    nuevo = np.empty_like(orden)
    nuevo[orden] = np.arange(len(orden))
    return nuevo


def ajustar_kmedias(X, k=K_ELEGIDO, semilla=SEMILLA):
    km = KMeans(n_clusters=k, n_init=50, random_state=semilla).fit(X)
    mapa = ordenar_grupos(km.cluster_centers_)
    etiquetas = mapa[km.labels_]
    centroides = np.empty_like(km.cluster_centers_)
    centroides[mapa] = km.cluster_centers_
    return km, etiquetas, centroides


def asignar(X, centroides) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - centroides[None]) ** 2).sum(axis=2)).argmin(axis=1)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def grafica_seleccion_k(tabla, ruta):
    estilo_base()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(tabla["k"], tabla["inercia"], color=COLORES_SERIES[0], lw=2, marker="o", ms=5)
    a1.set_title("Método del codo", loc="left")
    a1.set_xlabel("Número de grupos k")
    a1.set_ylabel("Inercia (suma de distancias²)")
    t = tabla.dropna(subset=["silueta"])
    a2.plot(t["k"], t["silueta"], color=COLORES_SERIES[0], lw=2, marker="o", ms=5)
    a2.set_title("Coeficiente de silueta (más alto = mejor)", loc="left")
    a2.set_xlabel("Número de grupos k")
    a2.set_ylabel("Silueta media")
    for eje in (a1, a2):
        eje.axvline(K_ELEGIDO, color=TEXTO_SECUNDARIO, lw=1, ls="--")
    a1.set_xticks(tabla["k"])
    a2.set_xticks(t["k"])
    a1.annotate("k elegido = 3", (K_ELEGIDO, tabla.loc[tabla.k == K_ELEGIDO, "inercia"].iloc[0]),
                xytext=(12, 18), textcoords="offset points", color=TEXTO_SECUNDARIO, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_perfiles_grupos(centroides, tamanos, ruta):
    estilo_base()
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    nh = len(COLUMNAS_HABIL)
    for eje, sl, titulo in [(ejes[0], slice(0, nh), "Día hábil"),
                            (ejes[1], slice(nh, 2 * nh), "Fin de semana")]:
        for g in range(len(centroides)):
            eje.plot(HORAS, centroides[g, sl], color=COLOR_GRUPO[g], lw=2.2, marker="o", ms=3,
                     label=f"{g}. {NOMBRES_GRUPOS[g]} ({tamanos[g]} est.)")
        eje.set_title(titulo, loc="left")
        eje.set_xticks(HORAS[::2])
        eje.set_xlabel("Hora del día")
    ejes[0].set_ylabel("% de las salidas del día (centroide)")
    manejadores, textos = ejes[0].get_legend_handles_labels()
    fig.legend(manejadores, textos, frameon=False, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Perfil promedio de cada grupo (centroides de k-medias)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_pca(X, etiquetas, perfiles, ruta):
    estilo_base()
    pca = PCA(n_components=2, random_state=SEMILLA)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 6))
    for g in sorted(np.unique(etiquetas)):
        m = etiquetas == g
        ax.scatter(Z[m, 0], Z[m, 1], s=42, color=COLOR_GRUPO[g], edgecolor="white", lw=0.8,
                   label=f"{g}. {NOMBRES_GRUPOS[g]}", zorder=3)
    # (nombre en los datos, texto, desplazamiento de la etiqueta en puntos)
    destacadas = [("Calle 100 - Marketmedios", "Calle 100", (6, 4)),
                  ("Calle 72", "Calle 72", (6, 4)),
                  ("Ricaurte", "Ricaurte", (6, -2)),
                  ("Avenida Jimenez", "Av. Jiménez", (6, 2)),
                  ("Recinto Ferial", "Recinto Ferial", (6, -10)),
                  ("Cabecera Autopista Norte", "Portal Norte", (6, 6)),
                  ("Portal Américas", "Portal Américas", (-20, 14)),
                  ("Portal Suba", "Portal Suba", (8, -14)),
                  ("Avenida 68", "Av. 68", (6, 4))]
    for nombre, texto, desplaz in destacadas:
        i = perfiles.index[perfiles["estacion"] == nombre]
        if len(i):
            i = i[0]
            ax.annotate(texto, (Z[i, 0], Z[i, 1]), xytext=desplaz, textcoords="offset points",
                        fontsize=8.5, color=TEXTO_PRIMARIO,
                        arrowprops=dict(arrowstyle="-", color=TEXTO_SECUNDARIO, lw=0.6))
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"Componente principal 1 ({var[0]:.0%} de la varianza)")
    ax.set_ylabel(f"Componente principal 2 ({var[1]:.0%} de la varianza)")
    ax.set_title("Estaciones proyectadas en 2D (PCA), coloreadas por grupo", loc="left")
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return pca


def grafica_dendrograma(Z_link, grupos_hojas, ruta, k):
    """Colorea cada rama con el color del grupo si todas sus hojas son de ese grupo."""
    estilo_base()
    n = len(grupos_hojas)
    grupo_nodo = {i: int(g) for i, g in enumerate(grupos_hojas)}
    for j, (a, b, _, _) in enumerate(Z_link):
        ga, gb = grupo_nodo[int(a)], grupo_nodo[int(b)]
        grupo_nodo[n + j] = ga if ga == gb else -1
    color = lambda nodo: COLOR_GRUPO.get(grupo_nodo[nodo], TEXTO_SECUNDARIO)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    corte = (Z_link[-k, 2] + Z_link[-k + 1, 2]) / 2
    dendrogram(Z_link, ax=ax, no_labels=True, link_color_func=color)
    ax.axhline(corte, color=TEXTO_SECUNDARIO, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], corte, f"corte en {k} grupos", va="bottom", ha="right",
            color=TEXTO_SECUNDARIO, fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=COLOR_GRUPO[g], lw=2,
                              label=f"{g}. {NOMBRES_GRUPOS[g]}") for g in range(k)],
              frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Dendrograma del agrupamiento jerárquico (método de Ward) – 152 estaciones",
                 loc="left")
    ax.set_ylabel("Distancia de fusión")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    perfiles = cargar_perfiles()
    X = matriz(perfiles)
    print(f"Matriz de entrada: {X.shape[0]} estaciones × {X.shape[1]} variables")

    # 1. Selección de k
    tabla_k = evaluar_k(X)
    grafica_seleccion_k(tabla_k, RUTA_FIGURAS / "seleccion_k.png")
    print("\nSelección del número de grupos:")
    print(tabla_k.to_string(index=False))

    # 2. K-medias con k elegido
    km, etiquetas, centroides = ajustar_kmedias(X)
    tamanos = np.bincount(etiquetas).tolist()
    sil_media = float(silhouette_score(X, etiquetas))
    sil_individual = silhouette_samples(X, etiquetas)

    # 2b. Comprobación con la implementación propia (mismos centroides iniciales)
    rng = np.random.default_rng(SEMILLA)
    iniciales = X[rng.choice(len(X), size=K_ELEGIDO, replace=False)]
    propio = kmedias(X, iniciales)
    sk = KMeans(n_clusters=K_ELEGIDO, init=iniciales, n_init=1, algorithm="lloyd",
                tol=0).fit(X)
    i_ej = int(np.where(perfiles["estacion"] == "Calle 72")[0][0])
    verificacion = {
        "iteraciones_propia": propio.iteraciones,
        "etiquetas_identicas_a_sklearn": bool(adjusted_rand_score(propio.etiquetas, sk.labels_) == 1.0),
        "diferencia_maxima_centroides": float(np.abs(
            np.sort(propio.centroides, axis=0) - np.sort(sk.cluster_centers_, axis=0)).max()),
        "inercia_propia": round(propio.inercia, 4),
        "inercia_sklearn": round(float(sk.inertia_), 4),
        "inercia_modelo_final_manual": round(inercia(X, etiquetas, centroides), 4),
        "inercia_modelo_final_sklearn": round(float(km.inertia_), 4),
        "silueta_calle72_manual": round(silueta_de_un_punto(X, etiquetas, i_ej), 4),
        "silueta_calle72_sklearn": round(float(sil_individual[i_ej]), 4),
        "historial_inercia_propia": [round(v, 1) for v in propio.historial_inercia],
    }

    # 3. Jerárquico de Ward
    Z_link = linkage(X, method="ward")
    ward = fcluster(Z_link, t=K_ELEGIDO, criterion="maxclust") - 1
    ari_ward = float(adjusted_rand_score(etiquetas, ward))
    # Se renumeran los grupos de Ward con el mismo criterio para comparar
    cent_ward = np.vstack([X[ward == g].mean(axis=0) for g in range(K_ELEGIDO)])
    ward = ordenar_grupos(cent_ward)[ward]
    cruce = pd.crosstab(pd.Series(etiquetas, name="kmedias"), pd.Series(ward, name="ward"))
    grafica_dendrograma(Z_link, ward, RUTA_FIGURAS / "dendrograma.png", K_ELEGIDO)

    # 4a. Estabilidad frente a la semilla
    ari_semillas = [float(adjusted_rand_score(etiquetas, ajustar_kmedias(X, semilla=s)[1]))
                    for s in range(20)]

    # 4b. Estabilidad temporal: perfiles con cada quincena, asignados a los centroides
    por_hora = pd.read_csv(RUTA_POR_HORA, dtype={"codigo_estacion": str, "codigo_zona": str})
    estabilidad = {}
    for nombre, dias in [("quincena_1", range(1, 16)), ("quincena_2", range(16, 31))]:
        p = construir_perfiles(por_hora[por_hora["dia"].isin(dias)])
        p = p.set_index("codigo_estacion").loc[perfiles["codigo_estacion"]]
        asign = asignar(p[COLUMNAS_PERFIL].to_numpy(dtype=float), centroides)
        estabilidad[nombre] = {"coinciden": int((asign == etiquetas).sum()),
                               "porcentaje": round(float((asign == etiquetas).mean() * 100), 1),
                               "cambian": perfiles.loc[asign != etiquetas, "estacion"].tolist()}

    # 5. Interpretación
    perfiles["grupo"] = etiquetas
    perfiles["nombre_grupo"] = perfiles["grupo"].map(NOMBRES_GRUPOS)
    perfiles["grupo_ward"] = ward
    dist = np.sqrt(((X[:, None, :] - centroides[None]) ** 2).sum(axis=2))
    perfiles["distancia_centroide"] = dist[np.arange(len(X)), etiquetas].round(3)
    perfiles["silueta"] = sil_individual.round(4)
    grafica_perfiles_grupos(centroides, tamanos, RUTA_FIGURAS / "perfiles_grupos.png")
    pca = grafica_pca(X, etiquetas, perfiles, RUTA_FIGURAS / "pca_grupos.png")

    resumen_grupos = []
    for g in range(K_ELEGIDO):
        m = perfiles[perfiles["grupo"] == g]
        tipicas = m.nsmallest(5, "distancia_centroide")["estacion"].tolist()
        mayores = m.nlargest(5, "salidas_dia_habil")["estacion"].tolist()
        resumen_grupos.append({
            "grupo": g, "nombre": NOMBRES_GRUPOS[g], "estaciones": int(len(m)),
            "pct_punta_manana": round(float(m["pct_punta_manana"].mean()), 1),
            "pct_punta_tarde": round(float(m["pct_punta_tarde"].mean()), 1),
            "relacion_finde_habil": round(float(m["relacion_finde_habil"].mean()), 3),
            "salidas_dia_habil_media": int(m["salidas_dia_habil"].mean()),
            "salidas_dia_habil_total": int(m["salidas_dia_habil"].sum()),
            "portales": int(m["es_portal"].sum()),
            "silueta_media": round(float(m["silueta"].mean()), 4),
            "estaciones_mas_tipicas": tipicas,
            "estaciones_mas_grandes": mayores,
            "hora_pico_habil": int(HORAS[int(np.argmax(centroides[g, :len(HORAS)]))]),
            "hora_pico_finde": int(HORAS[int(np.argmax(centroides[g, len(HORAS):]))]),
        })
    zonas = pd.crosstab(perfiles["zona"], perfiles["grupo"]).rename(columns=NOMBRES_GRUPOS)

    # Guardar
    columnas_salida = ["codigo_estacion", "estacion", "zona", "grupo", "nombre_grupo",
                       "grupo_ward", "distancia_centroide", "silueta", "salidas_dia_habil",
                       "relacion_finde_habil", "pct_punta_manana", "pct_punta_tarde",
                       "es_portal", "num_accesos"]
    perfiles.sort_values(["grupo", "distancia_centroide"])[columnas_salida].to_csv(
        RUTA_RESULTADOS / "asignacion_estaciones.csv", index=False, encoding="utf-8")
    pd.DataFrame(centroides, columns=COLUMNAS_PERFIL).assign(
        grupo=range(K_ELEGIDO), nombre=[NOMBRES_GRUPOS[g] for g in range(K_ELEGIDO)]
    )[["grupo", "nombre", *COLUMNAS_PERFIL]].round(4).to_csv(
        RUTA_RESULTADOS / "centroides.csv", index=False, encoding="utf-8")
    zonas.to_csv(RUTA_RESULTADOS / "grupos_por_zona.csv", encoding="utf-8")
    tabla_k.to_csv(RUTA_RESULTADOS / "seleccion_k.csv", index=False, encoding="utf-8")

    metricas = {
        "k_elegido": K_ELEGIDO,
        "silueta_media": round(sil_media, 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, etiquetas)), 4),
        "inercia": round(float(km.inertia_), 1),
        "tamanos": tamanos,
        "verificacion_kmedias_propio": verificacion,
        "ward": {"ari_vs_kmedias": round(ari_ward, 4),
                 "coincidencia_pct": round(float((ward == etiquetas).mean() * 100), 1),
                 "tabla_cruzada": cruce.to_dict()},
        "estabilidad_semillas": {"ari_minimo": round(min(ari_semillas), 4),
                                 "ari_medio": round(float(np.mean(ari_semillas)), 4)},
        "estabilidad_quincenas": estabilidad,
        "pca_varianza_explicada": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "grupos": resumen_grupos,
    }
    with open(RUTA_RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    # Consola
    print(f"\nK-medias con k = {K_ELEGIDO}: silueta media = {sil_media:.3f}, "
          f"Davies-Bouldin = {metricas['davies_bouldin']:.3f}")
    for r in resumen_grupos:
        print(f"\n  Grupo {r['grupo']} – {r['nombre']}: {r['estaciones']} estaciones")
        print(f"    Punta mañana {r['pct_punta_manana']} % | punta tarde {r['pct_punta_tarde']} % | "
              f"fin de semana/hábil {r['relacion_finde_habil']:.2f} | hora pico hábil {r['hora_pico_habil']}:00")
        print(f"    Más típicas: {', '.join(r['estaciones_mas_tipicas'])}")
    print("\nComprobación de la implementación propia de k-medias:")
    print(f"  Etiquetas idénticas a scikit-learn: {verificacion['etiquetas_identicas_a_sklearn']} "
          f"({verificacion['iteraciones_propia']} iteraciones)")
    print(f"  Inercia propia {verificacion['inercia_propia']} vs scikit-learn {verificacion['inercia_sklearn']}")
    print(f"  Silueta de Calle 72: manual {verificacion['silueta_calle72_manual']} vs "
          f"scikit-learn {verificacion['silueta_calle72_sklearn']}")
    print(f"\nJerárquico de Ward: coincide con k-medias en el "
          f"{metricas['ward']['coincidencia_pct']} % de las estaciones (ARI = {ari_ward:.3f})")
    print(f"Estabilidad con 20 semillas: ARI mínimo = {min(ari_semillas):.3f}")
    for q, v in estabilidad.items():
        print(f"Estabilidad {q}: {v['coinciden']}/{len(X)} estaciones en el mismo grupo ({v['porcentaje']} %)")
    print(f"\nResultados en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
