import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 📂 Paths
DATA_PATH = "../attrition_datasets/train_test/attrition_availabledata_07.csv/attrition_availabledata_07.csv"
PROCESSED_DATA_FILE = "../results/processed_data.pkl"


# 📥 Cargar los datos
def load_data():
    print("📥 Cargando datos...")
    df = pd.read_csv(DATA_PATH)
    return df


# 📊 Preprocesamiento de datos
def preprocess_data(df):
    print("🔄 Preprocesando datos...")

    target_variable = "Attrition"
    X = df.drop(
        columns=[target_variable, "EmployeeID", "EmployeeCount", "Over18",
                 "StandardHours"], errors='ignore')
    y = df[target_variable].map({"Yes": 1, "No": 0})

    print("✂️ Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3,
                                                        random_state=16,
                                                        stratify=y)

    print("⚙️ Aplicando transformación de datos...")
    nominal_features = ["Gender", "Department", "JobRole", "MaritalStatus",
                        "EducationField"]
    ordinal_features = ["BusinessTravel"]
    numeric_features = [col for col in X.columns if
                        col not in nominal_features + ordinal_features]
    travel = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']

    imputer = ColumnTransformer(
        transformers=[
            ('imputer_num', SimpleImputer(strategy='mean'), numeric_features),
            ('imputer_cat', SimpleImputer(strategy='most_frequent'),
             nominal_features + ordinal_features),
        ], remainder='passthrough', verbose_feature_names_out=False
    ).set_output(transform="pandas")

    encoder = ColumnTransformer(
        transformers=[
            (
            'encoder_onehot', OneHotEncoder(drop='first', sparse_output=False),
            nominal_features),
            ('encoder_ordinal', OrdinalEncoder(categories=[travel]),
             ordinal_features),
        ], remainder='passthrough', verbose_feature_names_out=False
    ).set_output(transform="pandas")

    scaler = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_features),
        ], remainder='passthrough', verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipeline = Pipeline([
        ('imputer', imputer),
        ('encoder', encoder),
        ('scaler', scaler),
    ])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=pipeline.get_feature_names_out())
    X_test = pd.DataFrame(X_test, columns=pipeline.get_feature_names_out())

    joblib.dump(pipeline, PROCESSED_DATA_FILE)

    #Guardamos los datos preprocesados
    X_train.to_csv("../results/X_train.csv", index=False)
    X_test.to_csv("../results/X_test.csv", index=False)
    y_train.to_csv("../results/y_train.csv", index=False)
    y_test.to_csv("../results/y_test.csv", index=False)

    print("✅ Preprocesamiento completado.")
    return X_train, X_test, y_train, y_test


# 🏆 Evaluación interna con validación cruzada
def evaluate_models(X_train, y_train):
    print("📊 Evaluando modelos con validación cruzada...")

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=16),
        "K-Neighbors": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=20000,
                                                  solver='lbfgs',
                                                  random_state=16),
        "Random Forest": RandomForestClassifier(random_state=16,
                                                n_estimators=100),
        "SVM": SVC(random_state=16),
        "Neural Network": MLPClassifier(random_state=16, max_iter=500)
    }

    model_scores = []
    for name, model in models.items():
        print(f"🚀 Evaluando {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=5,
                                 scoring='balanced_accuracy')
        mean_score, std_score = scores.mean(), scores.std()
        model_scores.append((name, mean_score, std_score))

    model_scores.sort(key=lambda x: x[1], reverse=True)

    print("# 📊 Resultados de Evaluación Interna")
    for name, mean_score, std_score in model_scores:
        print(
            f"{name}: Balanced Accuracy = {mean_score:.4f} ± {std_score:.4f}")


# 🚀 Función principal
def main():
    print("🎯 Iniciando preprocesamiento y evaluación de modelos...")
    start_time = time.time()

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("📊 Datos preprocesados y divididos en entrenamiento y prueba.")
    evaluate_models(X_train, y_train)

    elapsed_time = time.time() - start_time
    print(f"🏁 Proceso finalizado en {elapsed_time:.2f} segundos.")


# 🔥 Ejecutar script
if __name__ == "__main__":
    main()