import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report


# 🚀 4. Entrenar el modelo definitivo (SVM RBF con hiperparámetros óptimos)
modelo_final = SVC(kernel='rbf', class_weight='balanced', C=5, gamma=0.1)

# Entrenar el modelo con los datos procesados
X_train = pd.read_csv("../results/X_train.csv")
y_train = pd.read_csv("../results/y_train.csv").values.ravel()
modelo_final.fit(X_train, y_train)

#Cargamos el predictor del modelo final
X_test = pd.read_csv("../results/X_test.csv") #El X_test ya está procesado

y_predict = modelo_final.predict(X_test)

#Hacemos la comparación de las predicciones con los valores reales
y_test = pd.read_csv("../results/y_test.csv").values.ravel()
balanced_accuracy = balanced_accuracy_score(y_test, y_predict)
matriz_conf = confusion_matrix(y_test, y_predict)

# Calcular Sensibilidad (TPR) y Especificidad (TNR)
tn, fp, fn, tp = matriz_conf.ravel()
tpr = tp / (tp + fn)  # Sensibilidad
tnr = tn / (tn + fp)  # Especificidad

# Imprimir métricas
print(f"\n🔍 Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"✔️ Sensibilidad (TPR): {tpr:.4f}")
print(f"✔️ Especificidad (TNR): {tnr:.4f}")
print("\n🔍 Reporte de clasificación:\n", classification_report(y_test,
                                                               y_predict))

# 🔥 Visualización de métricas
plt.figure(figsize=(12, 5))

# Matriz de confusión
plt.subplot(1, 2, 1)
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")

# Balanced Accuracy, TPR y TNR
plt.subplot(1, 2, 2)
metrics = [balanced_accuracy, tpr, tnr]
names = ["Balanced Accuracy", "TPR (Sensibilidad)", "TNR (Especificidad)"]
plt.barh(names, metrics, color=["blue", "green", "red"])
plt.xlim(0, 1)
plt.xlabel("Valor")
plt.title("Métricas de Evaluación del Modelo")

plt.tight_layout()
plt.show()

# Guardar el modelo entrenado
df = pd.read_csv("../attrition_datasets/train_test/attrition_availabledata_07"
                 ".csv/attrition_availabledata_07.csv")
preprocesado = joblib.load("../results/processed_data.pkl")

X = df.drop(columns=["Attrition","EmployeeID", "EmployeeCount", "Over18",
                 "StandardHours"], errors='ignore')  # Eliminamos la
# columna objetivo

X_procesado = preprocesado.transform(X)

y = df["Attrition"].map({"Yes": 1, "No": 0})  # Convertimos 'Yes' y 'No' a valores numéricos

mejor_modelo = modelo_final.fit(X_procesado, y)

joblib.dump(mejor_modelo, "../results/modelo_final.pkl")

print("\n✅ ¡Entrenamiento completado! El modelo ha sido guardado como 'modelo_final.pkl'.")

# 📂 6. Cargar los datos de la competición (sin la variable objetivo)
df_comp = pd.read_csv("../attrition_datasets/train_test"
                      "/attrition_competition_07.csv/attrition_competition_07.csv")

# 🔄 Preprocesar los datos de la competición antes de predecir
X_comp = df_comp.drop(columns=["EmployeeCount", "Over18", "StandardHours"], errors='ignore')
X_comp_procesado = preprocesado.transform(X_comp)  # ✅ Aplicamos la transformación

# 🔍 7. Generar predicciones con el modelo ya entrenado
predicciones_competicion = modelo_final.predict(X_comp_procesado)

# 📊 8. Guardar las predicciones en un archivo CSV
predicciones_df = pd.DataFrame({
    "EmployeeID": df_comp["EmployeeID"],
    "Attrition": pd.Series(predicciones_competicion).map({1: "Yes", 0: "No"})
})
predicciones_df.to_csv("../results/predicciones.csv", index=False)
print("\n✅ ¡Predicciones generadas y guardadas en 'predicciones.csv'!")

# Hacemos un conteo de las predicciones y lo mostramos en una grafica
print("\n 📊 Creando gráfico de predicciones...")
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Attrition", hue="Attrition", data=predicciones_df, palette="viridis", legend=False)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')
plt.title("Predicciones de Attrition en Competición")
plt.show()


