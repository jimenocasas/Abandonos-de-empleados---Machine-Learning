import pandas as pd
import numpy as np
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from itertools import product

# 📂 Paths
X_TRAIN_PATH = "../results/X_train.csv"
Y_TRAIN_PATH = "../results/y_train.csv"

# 📥 Cargar datos
def load_data():
    print("📥 Cargando datos de entrenamiento...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    print(f"✅ Datos cargados: {X_train.shape[0]} muestras, {X_train.shape[1]} características.")
    return X_train, y_train

# 🔧 Definir métodos de escalado e imputación
def get_scalers():
    return {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler(),
        "Robust": RobustScaler()
    }

def get_imputers():
    return {
        "Mean": SimpleImputer(strategy='mean'),
        "Median": SimpleImputer(strategy='median')
    }

# 🔍 Evaluar combinaciones de escalado e imputación
def evaluate_combinations(X_train, y_train, scalers, imputers):
    print("🚀 Evaluando combinaciones de escalado e imputación...")
    results = []

    for (scaler_name, scaler), (imputer_name, imputer) in product(scalers.items(), imputers.items()):
        print(f"⚙️ Probando: Escalado={scaler_name}, Imputación={imputer_name}...")

        pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('knn', KNeighborsClassifier())
        ])

        start_time = time.time()
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy')
        mean_score = scores.mean()
        elapsed_time = time.time() - start_time

        results.append((scaler_name, imputer_name, mean_score, elapsed_time))
        print(f"✅ Resultado: Balanced Accuracy={mean_score:.4f}, Tiempo={elapsed_time:.2f}s")

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    print("📄 Resumen de Resultados:")
    for scaler, imputer, score, elapsed_time in sorted_results:
        print(f"🔹 Escalado: {scaler}, Imputación: {imputer}, Balanced Accuracy: {score:.4f}, Tiempo: {elapsed_time:.2f}s")

    return sorted_results

# 🚀 Función principal
def main():
    print("🎯 Iniciando evaluación de KNN con distintas combinaciones...")

    # 📥 Cargar datos
    X_train, y_train = load_data()

    # 🔧 Obtener métodos de escalado e imputación
    scalers = get_scalers()
    imputers = get_imputers()

    # 🔍 Evaluar combinaciones
    evaluate_combinations(X_train, y_train, scalers, imputers)

    print("🏁 Proceso finalizado.")

# 🔥 Ejecutar script
if __name__ == "__main__":
    main()