"""
Algoritmo Genetico para Hyperparameter Optimization (Optimizacion de
Hiperparametros)
==========================================================================
Dataset: Wine (sklearn) - clasificacion de 3 tipos de vino
Modelo base: RandomForestClassifier
Objetivo: encontrar la mejor combinacion de hiperparametros
(n_estimators, max_depth, min_samples_split, max_features)
que maximiza el accuracy promedio en validacion cruzada.
"""

import random
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

random.seed(7)
np.random.seed(7)

# -----------------------------------------------------------------------
# 0. DATOS
# -----------------------------------------------------------------------
X, y = load_wine(return_X_y=True)

# -----------------------------------------------------------------------
# ESPACIO DE BUSQUEDA DE HIPERPARAMETROS (genes)
# -----------------------------------------------------------------------
RANGO_N_ESTIMATORS = (10, 200)     # entero
RANGO_MAX_DEPTH = (1, 20)          # entero (0 se interpreta como None)
RANGO_MIN_SAMPLES_SPLIT = (2, 10)  # entero
OPCIONES_MAX_FEATURES = ["sqrt", "log2", None]  # categorico

TAM_POBLACION = 16
N_GENERACIONES = 20
PROB_CRUCE = 0.8
PROB_MUTACION = 0.2
TAM_TORNEO = 3


# -----------------------------------------------------------------------
# 1. REPRESENTACION / CROMOSOMA
# -----------------------------------------------------------------------
# Cada individuo (cromosoma) es una lista de 4 genes:
# [n_estimators, max_depth, min_samples_split, indice_max_features]
def crear_individuo():
    return [
        random.randint(*RANGO_N_ESTIMATORS),
        random.randint(*RANGO_MAX_DEPTH),
        random.randint(*RANGO_MIN_SAMPLES_SPLIT),
        random.randint(0, len(OPCIONES_MAX_FEATURES) - 1),
    ]


# -----------------------------------------------------------------------
# 2. INICIALIZACION DE LA POBLACION
# -----------------------------------------------------------------------
def inicializar_poblacion(tam):
    return [crear_individuo() for _ in range(tam)]


def decodificar(individuo):
    n_estimators, max_depth, min_samples_split, idx_mf = individuo
    return {
        "n_estimators": n_estimators,
        "max_depth": None if max_depth == 0 else max_depth,
        "min_samples_split": min_samples_split,
        "max_features": OPCIONES_MAX_FEATURES[idx_mf],
    }


# -----------------------------------------------------------------------
# 3. FUNCION DE APTITUD (FITNESS)
# -----------------------------------------------------------------------
def calcular_fitness(individuo):
    params = decodificar(individuo)
    modelo = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    accuracy = cross_val_score(modelo, X, y, cv=5, scoring="accuracy").mean()
    return accuracy


# -----------------------------------------------------------------------
# 4. SELECCION (Torneo)
# -----------------------------------------------------------------------
def seleccion_torneo(poblacion, fitnesses, k=TAM_TORNEO):
    participantes = random.sample(list(zip(poblacion, fitnesses)), k)
    ganador = max(participantes, key=lambda x: x[1])
    return ganador[0][:]


# -----------------------------------------------------------------------
# 5. CRUZAMIENTO (Uniforme: cada gen se hereda de un padre al azar)
# -----------------------------------------------------------------------
def cruzamiento(padre1, padre2):
    if random.random() > PROB_CRUCE:
        return padre1[:], padre2[:]
    hijo1, hijo2 = [], []
    for g1, g2 in zip(padre1, padre2):
        if random.random() < 0.5:
            hijo1.append(g1); hijo2.append(g2)
        else:
            hijo1.append(g2); hijo2.append(g1)
    return hijo1, hijo2


# -----------------------------------------------------------------------
# 6. MUTACION (reinicio aleatorio de un gen)
# -----------------------------------------------------------------------
def mutacion(individuo):
    rangos = [RANGO_N_ESTIMATORS, RANGO_MAX_DEPTH, RANGO_MIN_SAMPLES_SPLIT, None]
    for i in range(len(individuo)):
        if random.random() < PROB_MUTACION:
            if i == 3:
                individuo[i] = random.randint(0, len(OPCIONES_MAX_FEATURES) - 1)
            else:
                individuo[i] = random.randint(*rangos[i])
    return individuo


# -----------------------------------------------------------------------
# 7. CICLO PRINCIPAL DEL ALGORITMO GENETICO (con criterio de TERMINACION)
# -----------------------------------------------------------------------
def algoritmo_genetico():
    poblacion = inicializar_poblacion(TAM_POBLACION)
    mejor_individuo, mejor_fitness = None, -np.inf
    historial = []

    for gen in range(N_GENERACIONES):
        fitnesses = [calcular_fitness(ind) for ind in poblacion]

        idx_mejor = int(np.argmax(fitnesses))
        if fitnesses[idx_mejor] > mejor_fitness:
            mejor_fitness = fitnesses[idx_mejor]
            mejor_individuo = poblacion[idx_mejor][:]

        historial.append(mejor_fitness)
        print(f"Gen {gen+1:02d}/{N_GENERACIONES} | Mejor accuracy historico: {mejor_fitness:.4f} "
              f"| Params: {decodificar(mejor_individuo)}")

        nueva_poblacion = [mejor_individuo[:]]  # elitismo
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_torneo(poblacion, fitnesses)
            padre2 = seleccion_torneo(poblacion, fitnesses)
            hijo1, hijo2 = cruzamiento(padre1, padre2)
            hijo1 = mutacion(hijo1)
            hijo2 = mutacion(hijo2)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < TAM_POBLACION:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    return mejor_individuo, mejor_fitness, historial


if __name__ == "__main__":
    mejor, fit, historial = algoritmo_genetico()
    print("\n================ RESULTADO FINAL ================")
    print(f"Mejor accuracy (CV=5): {fit:.4f}")
    print(f"Mejores hiperparametros encontrados: {decodificar(mejor)}")
