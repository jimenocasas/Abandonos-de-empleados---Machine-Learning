import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# 📌 Cargar los datos procesados
def load_data():
    print("\n📥 Cargando datos de entrenamiento y prueba...")
    FILE_PATH_XTRAIN = "../results/X_train.csv"
    FILE_PATH_YTRAIN = "../results/y_train.csv"
    FILE_PATH_XTEST = "../results/X_test.csv"
    FILE_PATH_YTEST = "../results/y_test.csv"

    X_train = pd.read_csv(FILE_PATH_XTRAIN)
    y_train = pd.read_csv(FILE_PATH_YTRAIN).values.ravel()
    X_test = pd.read_csv(FILE_PATH_XTEST)
    y_test = pd.read_csv(FILE_PATH_YTEST).values.ravel()

    print("✅ Datos cargados correctamente.")
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data()


# 📌 Evaluación inicial con hiperparámetros por defecto
def evaluate_models(models, X_train, y_train, X_test, y_test):
    print("\n🔹 Evaluando modelos con hiperparámetros por defecto...")

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        print(
            f"✅ {name} evaluado - Tiempo de entrenamiento: {training_time:.4f}s - Balanced Accuracy: {balanced_acc:.4f}")


models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}
evaluate_models(models, X_train, y_train, X_test, y_test)


# 📌 Optimización de hiperparámetros con GridSearchCV
def optimize_hyperparameters(X_train, y_train):
    print("\n🔍 Optimizando hiperparámetros para KNN y Árbol de Decisión...")

    param_grid_knn = {"n_neighbors": range(1, 21),
                      "weights": ["uniform", "distance"]}
    param_grid_tree = {"max_depth": range(1, 21),
                       "min_samples_split": [2, 5, 10]}

    best_models = {}

    start_time = time.time()
    knn_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5,
                              scoring="balanced_accuracy", n_jobs=-1)
    knn_search.fit(X_train, y_train)
    knn_time = time.time() - start_time
    best_models["KNN"] = knn_search.best_estimator_

    start_time = time.time()
    tree_search = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, cv=5,
                               scoring="balanced_accuracy", n_jobs=-1)
    tree_search.fit(X_train, y_train)
    tree_time = time.time() - start_time
    best_models["Decision Tree"] = tree_search.best_estimator_

    print(
        f"✅ KNN optimizado: {knn_search.best_params_} - Tiempo: {knn_time:.4f}s")
    print(
        f"✅ Árbol de Decisión optimizado: {tree_search.best_params_} - Tiempo: {tree_time:.4f}s")

    return best_models, knn_search.best_params_, tree_search.best_params_


best_models, best_params_knn, best_params_tree = optimize_hyperparameters(
    X_train, y_train)


# 📌 Guardar los modelos optimizados
def save_models(models, path="../results/"):
    print("\n💾 Guardando modelos optimizados...")
    os.makedirs(path, exist_ok=True)  # Crear la carpeta si no existe

    for name, model in models.items():
        filename = os.path.join(path, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, filename)
        print(f"✅ Modelo {name} guardado en {filename}")


save_models(best_models)


# 📌 Evaluación final con modelos optimizados
def final_evaluation(models, X_train, y_train, X_test, y_test):
    print("\n📊 Evaluando modelos optimizados...")

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        print(
            f"✅ {name} optimizado - Tiempo de entrenamiento: {training_time:.4f}s - Balanced Accuracy: {balanced_acc:.4f}")


final_evaluation(best_models, X_train, y_train, X_test, y_test)


# 📌 Visualización del impacto de hiperparámetros
def plot_hyperparameter_effects(param_grid_knn, param_grid_tree, X_train,
                                y_train):
    print("\n📈 Generando gráficos de impacto de hiperparámetros...")

    plt.figure(figsize=(12, 5))

    # KNN: Número de vecinos
    knn_neighbors = list(param_grid_knn["n_neighbors"])
    knn_scores = [
        cross_val_score(KNeighborsClassifier(n_neighbors=n), X_train, y_train,
                        cv=5, scoring='balanced_accuracy').mean() for n in
        knn_neighbors]

    plt.subplot(1, 2, 1)
    plt.plot(knn_neighbors, knn_scores, marker='o')
    plt.xlabel("Número de vecinos (K)")
    plt.ylabel("Balanced Accuracy")
    plt.title("Efecto del número de vecinos en KNN")

    # Árboles: Profundidad máxima
    tree_depths = list(param_grid_tree["max_depth"])
    tree_scores = [
        cross_val_score(DecisionTreeClassifier(max_depth=d), X_train, y_train,
                        cv=5, scoring='balanced_accuracy').mean() for d in
        tree_depths]

    plt.subplot(1, 2, 2)
    plt.plot(tree_depths, tree_scores, marker='o')
    plt.xlabel("Profundidad máxima del árbol")
    plt.ylabel("Balanced Accuracy")
    plt.title("Efecto de max_depth en Árboles de Decisión")

    plt.tight_layout()
    plt.show()


plot_hyperparameter_effects({"n_neighbors": range(1, 21)},
                            {"max_depth": range(1, 21)}, X_train, y_train)


# 📌 Matriz de confusión
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()


for name, model in best_models.items():
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, name)

print("\n✅ Evaluación finalizada.")
