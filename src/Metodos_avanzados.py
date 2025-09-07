import pandas as pd
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Cargar los datos desde archivos
X_train = pd.read_csv("../results/X_train.csv")
y_train = pd.read_csv("../results/y_train.csv").values.ravel()
X_test = pd.read_csv("../results/X_test.csv")
y_test = pd.read_csv("../results/y_test.csv").values.ravel()

# Aplicamos SMOTE para equilibrar las clases
print(
    "Aplicando SMOTE para balancear las clases en el conjunto de entrenamiento...")
smote = SMOTE(random_state=16)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Nueva distribución de clases tras SMOTE:\n",
      pd.Series(y_train).value_counts())


# Función auxiliar para calcular sensibilidad y especificidad
def obtener_metricas(y_real, y_pred):
    matriz = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = matriz.ravel()
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensibilidad, especificidad, matriz


modelos = {
    "Regresión Logística L1": LogisticRegression(penalty='l1',
                                                 solver='liblinear',
                                                 max_iter=1000,
                                                 class_weight='balanced',
                                                 random_state=16),
    "Regresión Logística L2": LogisticRegression(penalty='l2', solver='lbfgs',
                                                 max_iter=1000,
                                                 class_weight='balanced',
                                                 random_state=16),
    "SVM RBF": SVC(kernel='rbf', class_weight='balanced', random_state=16),
    "SVM Lineal": SVC(kernel='linear', class_weight='balanced',
                      random_state=16)
}

resultados = {}
for nombre, modelo in modelos.items():
    print(f"\n🔹 Entrenando {nombre}...")
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio
    predicciones = modelo.predict(X_test)
    acc = balanced_accuracy_score(y_test, predicciones)
    sensibilidad, especificidad, matriz = obtener_metricas(y_test,
                                                           predicciones)

    resultados[nombre] = {
        "Balanced Accuracy": acc,
        "Sensibilidad": sensibilidad,
        "Especificidad": especificidad,
        "Tiempo de entrenamiento": tiempo,
        "Reporte": classification_report(y_test, predicciones)
    }

    print(f"✔️ Tiempo de entrenamiento: {tiempo:.2f} segundos")
    print(f"✔️ Balanced Accuracy: {acc:.4f}")
    print(
        f"✔️ Sensibilidad: {sensibilidad:.4f}, Especificidad: {especificidad:.4f}")
    print("\n🔍 Reporte de clasificación:\n",
          classification_report(y_test, predicciones))

# Optimización de hiperparámetros
print("\n🔎 Buscando los mejores hiperparámetros para cada modelo...")
parametros = {
    "Regresión Logística L1": {"C": [0.1, 1, 5, 10, 20, 50]},
    "Regresión Logística L2": {"C": [0.1, 1, 5, 10, 20, 50]},
    "SVM RBF": {"C": [0.01, 0.1, 1, 5, 10],
                "gamma": [0.0001, 0.001, 0.01, 0.1]},
    "SVM Lineal": {"C": [0.01, 0.1, 1, 5, 10]}
}

modelos_optimizados = {}
for nombre, param_grid in parametros.items():
    print(f"\n🔍 Optimizando {nombre}...")
    grid_search = GridSearchCV(modelos[nombre], param_grid,
                               scoring="balanced_accuracy", cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    modelos_optimizados[nombre] = grid_search.best_estimator_
    print(
        f"📈 Mejores hiperparámetros para {nombre}: {grid_search.best_params_}")


# Guardar modelos optimizados
def save_models(models, path="../results/"):
    print("\n💾 Guardando modelos optimizados...")
    os.makedirs(path, exist_ok=True)  # Crear la carpeta si no existe

    for name, model in models.items():
        filename = os.path.join(path, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, filename)
        print(f"✅ Modelo {name} guardado en {filename}")


save_models(modelos_optimizados)

# Comparación de modelos base vs optimizados
print("\n📊 Comparación de modelos antes y después de la optimización:")
modelos_lista = list(modelos.keys())
acc_base = [resultados[m]["Balanced Accuracy"] for m in modelos_lista]
acc_opt = [
    balanced_accuracy_score(y_test, modelos_optimizados[m].predict(X_test)) for
    m in modelos_lista]

for i in range(len(modelos_lista)):
    print(
        f"{modelos_lista[i]} - Base: {acc_base[i]:.4f}, Optimizado: {acc_opt[i]:.4f}")

# Gráfico de comparación
plt.figure(figsize=(10, 6))
plt.bar(modelos_lista, acc_base, label="Base", alpha=0.6, color="blue")
plt.bar(modelos_lista, acc_opt, label="Optimizado", alpha=0.6, color="green")
plt.ylabel("Balanced Accuracy")
plt.title("Comparación de Modelos")
plt.legend()
plt.show()
