# Actividad 02 — Algoritmos Genéticos aplicados a Machine Learning

**Curso:** Aprendizaje de Máquina
**Objetivo:** Comprender la aplicación de los algoritmos genéticos (AG) en el aprendizaje de máquina.
**Equipo:** GRUBER COAQUIRA, Gian Carlos 1, FLORES MAMANI, Adams Fredy 2, MAGUIÑA CUTIPA, Luis Antonio 3, | 
**Fecha de presentación:** Miércoles 16/09/2026

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Gruber_feature_selection_ga.ipynb` | AG para **selección de características** (Feature Selection) sobre el dataset Breast Cancer Wisconsin, usando Regresión Logística. |
| `Adams_hyperparameter_optimization_ga.ipynb` | AG para **optimización de hiperparámetros** de un `RandomForestClassifier` sobre el dataset Wine. |
| `Maguiña_neuroevolution_ga.py` | AG para **Neuroevolution**: evolución de la arquitectura de una red neuronal (`MLPClassifier`) sobre el dataset Digits. |
| `Gruber_feature_selection_ga.ipynb` / `Adams_hyperparameter_optimization_ga.py` / `Maguiña_neuroevolution_ga.py` | Versiones en script `.py` de los mismos ejemplos, para ejecutar directamente en VS Code o terminal. |
| `requirements.txt` | Librerías necesarias para ejecutar los notebooks/scripts. |

## Cómo ejecutar los ejemplos

### Opción A: Google Colab
1. Sube cada archivo `.ipynb` a [Google Colab](https://colab.research.google.com/) (`Archivo > Subir cuaderno`), o ábrelo directamente desde GitHub con `Archivo > Abrir cuaderno > GitHub` pegando la URL de este repositorio.
2. Ejecuta las celdas en orden (`Entorno de ejecución > Ejecutar todas`).
3. Todas las librerías usadas (`numpy`, `scikit-learn`, `matplotlib`) ya vienen preinstaladas en Colab.

### Opción B: Jupyter Notebook local
```bash
git clone <URL-de-este-repositorio>
cd <carpeta-del-repositorio>
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Luego abre cualquiera de los 3 archivos `.ipynb` y ejecuta todas las celdas.

### Opción C: Visual Studio Code / terminal
```bash
pip install -r requirements.txt
python 01_feature_selection_ga.py
python 02_hyperparameter_optimization_ga.py
python 03_neuroevolution_ga.py
```

## Estructura común de los 3 ejemplos (ciclo del Algoritmo Genético)

Cada notebook desarrolla, de forma explícita y documentada, las siguientes etapas:

1. **Representación (cromosoma):** cómo se codifica una solución candidata (subconjunto de características, combinación de hiperparámetros o arquitectura de red).
2. **Inicialización:** generación de la población inicial de individuos aleatorios.
3. **Función de aptitud (fitness):** métrica que se optimiza — accuracy de validación cruzada (con una pequeña penalización en el caso de feature selection, para preferir modelos más simples).
4. **Selección:** selección por torneo (se eligen `k` individuos al azar y gana el de mayor fitness).
5. **Cruzamiento:** combinación de dos padres para producir nuevos individuos (un punto, uniforme, o intercambio de capas según el ejemplo).
6. **Mutación:** alteración aleatoria de genes para mantener diversidad genética y evitar óptimos locales.
7. **Terminación:** número fijo de generaciones, conservando siempre al mejor individuo encontrado (elitismo).

## Resumen de cada ejemplo

### 1. Feature Selection
- **Dataset:** Breast Cancer Wisconsin (30 características, clasificación binaria).
- **Cromosoma:** vector binario de 30 bits (1 = característica incluida).
- **Modelo evaluado:** Regresión Logística.
- **Resultado típico:** reduce de 30 a ~9-15 características manteniendo o mejorando el accuracy de validación cruzada.

### 2. Hyperparameter Optimization
- **Dataset:** Wine (13 características, 3 clases).
- **Cromosoma:** `[n_estimators, max_depth, min_samples_split, índice_max_features]`.
- **Modelo evaluado:** `RandomForestClassifier`.
- **Resultado típico:** ~97-98% de accuracy (CV=5) con hiperparámetros ajustados automáticamente.

### 3. Neuroevolution
- **Dataset:** Digits (dígitos manuscritos 0-9).
- **Cromosoma:** `{n_capas, neuronas por capa, función de activación}`.
- **Modelo evaluado:** `MLPClassifier` (los pesos se entrenan con backpropagation; el AG evoluciona solo la topología).
- **Resultado típico:** ~94-95% de accuracy (CV=3) con una arquitectura de 2 capas ocultas encontrada automáticamente.

## Notas
- Todos los ejemplos usan `random.seed()` / `np.random.seed()` fijos para que los resultados sean reproducibles.
- Los tres notebooks fueron ejecutados de principio a fin (`Run All`) antes de subirse, y se conservan sus salidas y gráficas de convergencia como evidencia de que funcionan correctamente.
- El resumen ejecutivo en PDF entregado en el aula virtual contiene el enlace a este repositorio.
