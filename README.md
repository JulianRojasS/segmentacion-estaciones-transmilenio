# Segmentación de estaciones de TransMilenio – Aprendizaje no supervisado

Agrupamiento de las **152 estaciones troncales de TransMilenio** según su perfil horario de demanda, con **k-medias** y **agrupamiento jerárquico**, a partir de **datos abiertos reales** de TRANSMILENIO S.A. (abril de 2024).

Este proyecto continúa el sistema de rutas (actividad 3) y el modelo de congestión (actividad 5). En lugar de predecir una etiqueta conocida, aquí el algoritmo **descubre por sí solo qué tipos de estaciones existen**.

**Curso:** Inteligencia Artificial – Actividad 6: métodos no supervisados
**Referencia teórica:** Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*, cap. 16 "Técnicas de agrupamiento". McGraw-Hill.

## Resultado principal

K-medias con k = 3 encontró tres tipos de estaciones:

| Grupo | Estaciones | Hora pico (día hábil) | Ejemplos |
|---|---:|---:|---|
| 0. Destino (pico de la mañana) | 56 | 7:00 | Calle 100, Calle 72, Marly, Av. Jiménez |
| 1. Mixta | 49 | 7:00 y 18:00 | Avenida 68, Normandía, Portal Norte, Portal El Dorado |
| 2. Origen residencial (pico de la tarde) | 47 | 18:00 | Portal Américas, Portal Suba, Portal Sur, Portal Usme |

![Perfiles de los grupos](resultados/figuras/perfiles_grupos.png)

- Silueta media 0,376, sin ninguna estación con silueta negativa.
- Los mismos grupos con 20 semillas distintas y el 99,3 % de las estaciones en el mismo grupo al usar solo la primera o solo la segunda quincena del mes.
- Coincidencia del 86,8 % con el agrupamiento jerárquico de Ward.
- La implementación propia de k-medias (`src/kmedias.py`) reproduce exactamente el resultado de scikit-learn.

## Estructura del repositorio

```
segmentacion-estaciones-transmilenio/
├── data/
│   ├── raw/salidas_troncal_abril_2024.csv    ← fuente original de TransMilenio (sin modificar)
│   └── processed/
│       ├── salidas_por_hora.csv              ← estación × día × hora
│       └── perfiles_estaciones.csv           ← entrada del agrupamiento: 152 estaciones × 36 variables
├── src/
│   ├── datos.py            ← limpieza, transformación y perfiles
│   ├── kmedias.py          ← k-medias, distancia euclidiana, inercia y silueta implementados a mano
│   ├── graficas.py         ← estilo de las gráficas
│   ├── preparar_datos.py   ← paso 1
│   ├── agrupar.py          ← paso 2: selección de k, k-medias, Ward, estabilidad y gráficas
│   └── consultar.py        ← paso 3: consultas por consola
├── tests/                  ← 39 pruebas automáticas (pytest)
├── resultados/             ← asignación de estaciones, centroides, métricas y figuras
└── docs/
    ├── DESCRIPCION_DATOS.md
    └── PRUEBAS.md
```

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone <URL-del-repositorio>
cd segmentacion-estaciones-transmilenio
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Preparar los datos
python src/preparar_datos.py

# 2. Agrupar las estaciones (unos 5 segundos)
python src/agrupar.py

# 3. Consultar los resultados
python src/consultar.py --resumen
python src/consultar.py --estacion "Calle 72"
python src/consultar.py --estacion "portal americas"
python src/consultar.py --comparar "Calle 72" "Portal Americas"
python src/consultar.py --grupo 1
python src/consultar.py --listar

# 4. Pruebas
python -m pytest -v
```

Ejemplo:

```
Estación: Calle 72 (09122) – Zona A Caracas
Grupo:    0. Destino (pico de la mañana)

Distancia euclidiana a cada centroide (el grupo es el más cercano):
  Grupo 0 (Destino (pico de la mañana)):   3.13  <- asignado
  Grupo 1 (Mixta):  14.98
  Grupo 2 (Origen residencial (pico de la tarde)):  32.89
```

## Cómo funciona

1. **Datos.** *Consolidado de Salidas Sistema Troncal por Franja Horaria*: salidas por acceso cada 15 minutos. Se limpian (filas vacías, separador de miles, corrales) y se suman por estación y hora. Ver [docs/DESCRIPCION_DATOS.md](docs/DESCRIPCION_DATOS.md).
2. **Perfil de cada estación.** El porcentaje de las salidas del día que ocurre en cada hora, de 5:00 a 22:00, en día hábil y en fin de semana: 36 variables. Se usan porcentajes para agrupar por la **forma** de la demanda y no por el tamaño de la estación.
3. **Número de grupos.** Método del codo, coeficiente de silueta e índice de Davies-Bouldin para k = 2 a 8.
4. **K-medias.** Cada estación se asigna al centroide más cercano (distancia euclidiana) y cada centroide se recalcula como la media de su grupo, hasta que nada cambia.
5. **Jerárquico.** Agrupamiento aglomerativo de Ward y dendrograma, para comparar.
6. **Validación.** Estabilidad frente a la semilla y al periodo, comparación con Ward y verificación de la implementación propia. Ver [docs/PRUEBAS.md](docs/PRUEBAS.md).

## Fuente de los datos

**TRANSMILENIO S.A.**, *Consolidado de Salidas Sistema Troncal por Franja Horaria – abril de 2024*, [Datos Abiertos Bogotá](https://datosabiertos.bogota.gov.co/dataset/consolidado-de-salidas-sistema-troncal-por-franja-horaria), licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Autor

_Nombre del estudiante_
