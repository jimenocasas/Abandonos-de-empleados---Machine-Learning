import pandas as pd
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# 📂 Cargar los datos de entrenamiento y prueba
X_train = pd.read_csv("../results/X_train.csv")
y_train = pd.read_csv("../results/y_train.csv").values.ravel()
X_test = pd.read_csv("../results/X_test.csv")
y_test = pd.read_csv("../results/y_test.csv").values.ravel()

# ⚖️ Aplicamos SMOTE para balancear las clases
print("Aplicando SMOTE para mejorar el balance de clases en el conjunto de entrenamiento...")
smote = SMOTE(random_state=16)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Distribución de clases tras SMOTE:\n", pd.Series(y_train).value_counts())

# 📏 Función auxiliar para métricas de evaluación
def obtener_metricas(y_real, y_pred):
    matriz = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = matriz.ravel()
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensibilidad, especificidad, matriz

# 🌲 Entrenamos y evaluamos un modelo Random Forest con hiperparámetros por defecto
print("\nEntrenando Random Forest con configuración estándar...")
inicio = time.time()
modelo_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=16)
modelo_rf.fit(X_train, y_train)
tiempo = time.time() - inicio

# 🔍 Evaluación del modelo base
pred_rf = modelo_rf.predict(X_test)
acc_rf = balanced_accuracy_score(y_test, pred_rf)
sens_rf, esp_rf, matriz_rf = obtener_metricas(y_test, pred_rf)

print(f"Tiempo de entrenamiento: {tiempo:.2f} segundos")
print(f" Balanced Accuracy: {acc_rf:.4f}")
print(f"Sensibilidad: {sens_rf:.4f}, Especificidad: {esp_rf:.4f}")
print("\nReporte de clasificación:\n", classification_report(y_test, pred_rf))

# 🔎 Optimización de hiperparámetros para Random Forest
print("\nBuscando los mejores hiperparámetros para Random Forest...")

param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=16),
    param_grid_rf, scoring="balanced_accuracy", cv=5, n_jobs=-1
)
grid_rf.fit(X_train, y_train)
modelo_rf_opt = grid_rf.best_estimator_
print(f"\nMejores hiperparámetros para Random Forest: {grid_rf.best_params_}")

# Guardar el modelo optimizado
joblib.dump(modelo_rf_opt, "../results/random_forest_optimized.pkl")
print("\n✅ Modelo optimizado guardado en '../results/random_forest_optimized.pkl'")

# 📊 Comparación entre modelo base y optimizado
acc_rf_opt = balanced_accuracy_score(y_test, modelo_rf_opt.predict(X_test))

print("\nComparación de Resultados para Random Forest:")
print(f"Balanced Accuracy - Base: {acc_rf:.4f}, Optimizado: {acc_rf_opt:.4f}")

# 📈 Visualización de comparación
plt.figure(figsize=(6, 4))
plt.bar(["Base", "Optimizado"], [acc_rf, acc_rf_opt], color=["blue", "green"], alpha=0.7)
plt.ylabel("Balanced Accuracy")
plt.title("Comparación de Random Forest antes y después de optimización")
plt.show()