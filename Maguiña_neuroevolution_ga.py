"""
Algoritmo Genetico para Neuroevolution (Evolucion de Arquitecturas de
Redes Neuronales)
==========================================================================
Dataset: Digits (sklearn) - reconocimiento de digitos manuscritos (0-9)
Modelo base: MLPClassifier (red neuronal totalmente conectada)
Objetivo: evolucionar la ARQUITECTURA de la red (numero de capas ocultas,
neuronas por capa y funcion de activacion) que maximiza el accuracy.
Los pesos de cada red se siguen entrenando con backpropagation (via
MLPClassifier); el AG solo busca la mejor "forma" de la red.
"""

import random
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
random.seed(1)
np.random.seed(1)

# -----------------------------------------------------------------------
# 0. DATOS
# -----------------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

# -----------------------------------------------------------------------
# ESPACIO DE BUSQUEDA DE LA ARQUITECTURA (genes)
# -----------------------------------------------------------------------
MAX_CAPAS = 3
RANGO_NEURONAS = (4, 64)           # neuronas por capa oculta
OPCIONES_ACTIVACION = ["relu", "tanh", "logistic"]

TAM_POBLACION = 10
N_GENERACIONES = 8
PROB_CRUCE = 0.7
PROB_MUTACION = 0.3
TAM_TORNEO = 3


# -----------------------------------------------------------------------
# 1. REPRESENTACION / CROMOSOMA
# -----------------------------------------------------------------------
# El cromosoma es un diccionario:
#   n_capas: cuantas capas ocultas tiene la red (1 a MAX_CAPAS)
#   neuronas: lista de tamanos de cada capa (solo se usan los primeros n_capas)
#   activacion: funcion de activacion de las capas ocultas
def crear_individuo():
    n_capas = random.randint(1, MAX_CAPAS)
    neuronas = [random.randint(*RANGO_NEURONAS) for _ in range(MAX_CAPAS)]
    activacion = random.choice(OPCIONES_ACTIVACION)
    return {"n_capas": n_capas, "neuronas": neuronas, "activacion": activacion}


# -----------------------------------------------------------------------
# 2. INICIALIZACION DE LA POBLACION
# -----------------------------------------------------------------------
def inicializar_poblacion(tam):
    return [crear_individuo() for _ in range(tam)]


def hidden_layer_sizes(individuo):
    return tuple(individuo["neuronas"][: individuo["n_capas"]])


# -----------------------------------------------------------------------
# 3. FUNCION DE APTITUD (FITNESS)
# -----------------------------------------------------------------------
def calcular_fitness(individuo):
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes(individuo),
        activation=individuo["activacion"],
        max_iter=200,
        random_state=0,
    )
    accuracy = cross_val_score(modelo, X, y, cv=3, scoring="accuracy").mean()
    return accuracy


# -----------------------------------------------------------------------
# 4. SELECCION (Torneo)
# -----------------------------------------------------------------------
def seleccion_torneo(poblacion, fitnesses, k=TAM_TORNEO):
    participantes = random.sample(list(zip(poblacion, fitnesses)), k)
    ganador = max(participantes, key=lambda x: x[1])
    return dict(ganador[0])  # copia


# -----------------------------------------------------------------------
# 5. CRUZAMIENTO (intercambia capas / activacion entre dos padres)
# -----------------------------------------------------------------------
def cruzamiento(padre1, padre2):
    if random.random() > PROB_CRUCE:
        return dict(padre1), dict(padre2)

    hijo1 = {
        "n_capas": random.choice([padre1["n_capas"], padre2["n_capas"]]),
        "neuronas": [random.choice([a, b]) for a, b in zip(padre1["neuronas"], padre2["neuronas"])],
        "activacion": random.choice([padre1["activacion"], padre2["activacion"]]),
    }
    hijo2 = {
        "n_capas": random.choice([padre1["n_capas"], padre2["n_capas"]]),
        "neuronas": [random.choice([a, b]) for a, b in zip(padre1["neuronas"], padre2["neuronas"])],
        "activacion": random.choice([padre1["activacion"], padre2["activacion"]]),
    }
    return hijo1, hijo2


# -----------------------------------------------------------------------
# 6. MUTACION (cambia n_capas, una capa de neuronas o la activacion)
# -----------------------------------------------------------------------
def mutacion(individuo):
    if random.random() < PROB_MUTACION:
        individuo["n_capas"] = random.randint(1, MAX_CAPAS)
    for i in range(len(individuo["neuronas"])):
        if random.random() < PROB_MUTACION:
            individuo["neuronas"][i] = random.randint(*RANGO_NEURONAS)
    if random.random() < PROB_MUTACION:
        individuo["activacion"] = random.choice(OPCIONES_ACTIVACION)
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
            mejor_individuo = dict(poblacion[idx_mejor])

        historial.append(mejor_fitness)
        print(f"Gen {gen+1:02d}/{N_GENERACIONES} | Mejor accuracy historico: {mejor_fitness:.4f} "
              f"| Arquitectura: capas={hidden_layer_sizes(mejor_individuo)} "
              f"activacion={mejor_individuo['activacion']}")

        nueva_poblacion = [dict(mejor_individuo)]  # elitismo
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
    print(f"Mejor accuracy (CV=3): {fit:.4f}")
    print(f"Mejor arquitectura: capas ocultas = {hidden_layer_sizes(mejor)}, "
          f"activacion = {mejor['activacion']}")
