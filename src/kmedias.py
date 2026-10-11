"""
Implementación propia de k-medias (algoritmo de Lloyd), tal como se describe en
el capítulo 16 de Palma Méndez (2008). Sirve para comprobar que el resultado de
scikit-learn corresponde exactamente al algoritmo teórico.

Algoritmo:
  1. Elegir k centroides iniciales.
  2. Asignación: cada ejemplo va al grupo del centroide más cercano
     (distancia euclidiana).
  3. Actualización: cada centroide pasa a ser la media de los ejemplos de su grupo.
  4. Repetir 2 y 3 hasta que ninguna asignación cambie.

Función objetivo (inercia o suma de errores cuadráticos, SSE):
  J = Σ_k Σ_{x ∈ C_k} ||x − μ_k||²
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def distancia_euclidiana(a, b) -> float:
    """d(a, b) = raíz( Σ (a_i − b_i)² )."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def matriz_distancias(X, centroides) -> np.ndarray:
    """Distancia de cada ejemplo (filas) a cada centroide (columnas)."""
    X, C = np.asarray(X, dtype=float), np.asarray(centroides, dtype=float)
    return np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))


def inercia(X, etiquetas, centroides) -> float:
    """Suma de las distancias al cuadrado de cada ejemplo a su centroide."""
    X, C = np.asarray(X, dtype=float), np.asarray(centroides, dtype=float)
    return float(((X - C[etiquetas]) ** 2).sum())


@dataclass
class ResultadoKMedias:
    etiquetas: np.ndarray
    centroides: np.ndarray
    inercia: float
    iteraciones: int
    historial_inercia: list


def kmedias(X, centroides_iniciales, max_iter: int = 300) -> ResultadoKMedias:
    X = np.asarray(X, dtype=float)
    C = np.array(centroides_iniciales, dtype=float)
    etiquetas = None
    historial = []
    for iteracion in range(1, max_iter + 1):
        # Paso de asignación
        nuevas = matriz_distancias(X, C).argmin(axis=1)
        # Paso de actualización
        for k in range(len(C)):
            miembros = X[nuevas == k]
            if len(miembros):            # un grupo vacío conserva su centroide
                C[k] = miembros.mean(axis=0)
        historial.append(inercia(X, nuevas, C))
        if etiquetas is not None and np.array_equal(nuevas, etiquetas):
            break
        etiquetas = nuevas
    return ResultadoKMedias(etiquetas, C, historial[-1], iteracion, historial)


def silueta_de_un_punto(X, etiquetas, i) -> float:
    """
    s(i) = (b − a) / max(a, b)
      a = distancia media de i a los demás puntos de su grupo
      b = menor distancia media de i a los puntos de otro grupo
    """
    X = np.asarray(X, dtype=float)
    d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    propio = etiquetas == etiquetas[i]
    propio[i] = False
    if propio.sum() == 0:
        return 0.0
    a = d[propio].mean()
    b = min(d[etiquetas == g].mean() for g in np.unique(etiquetas) if g != etiquetas[i])
    return float((b - a) / max(a, b))
