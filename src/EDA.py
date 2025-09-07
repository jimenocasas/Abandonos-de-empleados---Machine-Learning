import pandas as pd

# 🎨 ANSI Colors para salida en terminal
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# 📂 Paths
FILE_PATH_AVAILABLE = ("../attrition_datasets/train_test"
                       "/attrition_availabledata_07.csv/attrition_availabledata_07.csv")
FILE_PATH_COMPETITION = \
    "../attrition_datasets/train_test/attrition_competition_07.csv/attrition_competition_07.csv"


# 🏗️ Función para mostrar mensajes formateados
def print_status(message, status="info"):
    colors = {"info": BLUE, "success": GREEN, "warning": YELLOW, "error": RED}
    print(f"{colors.get(status, RESET)}{message}{RESET}")


# 📥 Cargar los datos
def load_data(file_path, name):
    print_status(f"📥 Cargando dataset: {name}...", "info")
    df = pd.read_csv(file_path)
    print_status(
        f"✅ Dataset {name} cargado con {df.shape[0]} filas y {df.shape[1]} columnas.",
        "success")
    return df


# 🔍 Información general del dataset
def dataset_info(df, name):
    print_status(f"📊 Analizando dataset: {name}...", "info")
    print(f"\n# Información del dataset: {name}")
    print(f"- 📌 Número de instancias: {df.shape[0]}")
    print(f"- 📌 Número de variables: {df.shape[1]}\n")

    print("🗂️ Tipos de datos:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"- {dtype} ({count} columnas)")

    print_status("✅ Información general mostrada.", "success")


# 🔎 Identificación de tipos de variables
def variable_types(df):
    print_status("📊 Identificando tipos de variables...", "info")
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print("\n# 🔍 Tipos de variables")
    print("\n📂 Variables categóricas:", ", ".join(categorical))
    print("\n📈 Variables numéricas:", ", ".join(numerical))

    print_status("✅ Identificación de variables completada.", "success")
    return categorical, numerical


# 📉 Identificación de valores faltantes
def missing_values(df):
    print_status("🔍 Buscando valores faltantes...", "info")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\n# ⚠️ Valores faltantes")
    if missing.empty:
        print("No se encontraron valores faltantes.")
    else:
        for col, count in missing.items():
            print(f"- {col}: {count} valores faltantes")

    print_status("✅ Análisis de valores faltantes completado.", "success")


# 🏷️ Variables con alta cardinalidad
def high_cardinality(df, categorical_vars, threshold=10):
    print_status("📊 Buscando variables con alta cardinalidad...", "info")
    high_card = [col for col in categorical_vars if df[col].nunique() > threshold]

    print("\n# 🔢 Variables con alta cardinalidad")
    if high_card:
        print(", ".join(high_card))
    else:
        print("No se encontraron variables con alta cardinalidad.")

    print_status("✅ Análisis de cardinalidad completado.", "success")


# 🚨 Identificación de columnas constantes o identificadores
def constant_or_id_columns(df):
    print_status("🔍 Identificando columnas constantes o de ID...", "info")
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    id_cols = [col for col in df.columns if "id" in col.lower()]

    print("\n# 🏷️ Columnas constantes e identificadores")
    print("📌 Columnas constantes:", ", ".join(constant_cols) if constant_cols else "Ninguna")
    print("🆔 Columnas de ID:", ", ".join(id_cols) if id_cols else "Ninguna")

    print_status("✅ Análisis de columnas constantes y ID completado.", "success")


# 📊 Determinar si es clasificación o regresión y verificar desbalance
def problem_type(df, target_variable="Attrition"):
    print_status("📌 Determinando tipo de problema...", "info")
    print("\n# 🤖 Tipo de problema: Clasificación o Regresión")

    if target_variable in df.columns:
        unique_values = df[target_variable].nunique()
        if unique_values <= 10:
            print("📌 Es un problema de **clasificación**.\n")
            class_counts = df[target_variable].value_counts(normalize=True)
            print("📊 **Distribución de clases:**")
            for cls, proportion in class_counts.items():
                print(f"- {cls}: {proportion:.2%}")

            if class_counts.min() < 0.3:
                print("\n⚠️ **El conjunto de datos está desbalanceado.**")
        else:
            print("📌 Es un problema de **regresión**.")
    else:
        print("⚠️ No se encontró la variable objetivo en el dataset.")

    print_status("✅ Determinación del tipo de problema completada.", "success")


# 🚀 Función principal
def main():
    print_status("🎯 Iniciando análisis exploratorio de datos (EDA)...", "info")

    # 📥 Cargar datasets
    df_available = load_data(FILE_PATH_AVAILABLE, "Available Data")
    df_competition = load_data(FILE_PATH_COMPETITION, "Competition Data")

    # 📊 Análisis del dataset
    dataset_info(df_available, "Available Data")
    dataset_info(df_competition, "Competition Data")

    # 📂 Tipos de variables
    categorical_vars, numerical_vars = variable_types(df_available)

    # ⚠️ Valores faltantes
    missing_values(df_available)

    # 🔢 Variables con alta cardinalidad
    high_cardinality(df_available, categorical_vars)

    # 🏷️ Columnas constantes y de identificación
    constant_or_id_columns(df_available)

    # 🤖 Tipo de problema y balance de clases
    problem_type(df_available)

    print_status("🏁 Análisis exploratorio finalizado.", "success")


# 🔥 Ejecutar script
if __name__ == "__main__":
    main()
