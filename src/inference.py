# import sys
# import os
# # Agregar la raíz del proyecto al PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import pandas as pd
# from src.model_loader import load_model
# from src.data_preprocessor import preprocessor_data
# # Ocultar advertencias de FutureWarning
# import warnings
# warnings.filterwarnings("ignore")

# def main():
#     # Ruta del modelo guardado
#     model_path = "models/trained_model_2025-01-10.joblib"
#     model = load_model(model_path)
#     print("Cargó el modelo exitosamente")
#         # Obtener las columnas que espera el modelo
#     expected_columns = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
#     if expected_columns is None:
#         print("Error: No se encontraron columnas esperadas en el modelo.")
#         return

#     # Datos de entrada para hacer inferencia
#     input_data = {
#         'citric acid': 0.25,
#         'residual sugar': 10.0,
#         'chlorides': 0.04,
#         'free sulfur dioxide': 20.0,
#         'total sulfur dioxide': 120.0,
#         'sulphates': 0.6
#     }

#     # Columnas utilizadas en el modelo entrenado
#     columns_to_use = ['citric acid', 'residual sugar', 'chlorides', 
#                       'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']

#     # Aplicar preprocesamiento a los datos de entrada
#     try:
#         # Preprocesar datos de entrada
#         preprocessed_data = preprocessor_data(input_data=input_data, columns_to_impute=columns_to_use)
#         print("Preprocesamiento de datos completado con éxito")
#     except Exception as e:
#         print(f"Error durante el preprocesamiento: {e}")
#         sys.exit(1)

#     # Realizar predicción con el modelo
#     try:
#         prediction = model.predict(preprocessed_data)
#         print(f"Predicción: {prediction}")
#     except Exception as e:
#         print(f"Error durante la inferencia: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data
# Ocultar advertencias de FutureWarning
import warnings
warnings.filterwarnings("ignore")


def main():
    # Ruta del modelo guardado
    model_path = "models/trained_model_2025-01-10.joblib"
    model = load_model(model_path)
    print("Cargó el modelo exitosamente")
    
    # Obtener las columnas que espera el modelo
    expected_columns = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
    if expected_columns is None:
        print("Error: No se encontraron columnas esperadas en el modelo.")
        return

    print(f"Columnas esperadas por el modelo: {expected_columns}")

    # Datos de entrada para hacer inferencia
    input_data = {
        'citric acid': 0.25,
        'residual sugar': 10.0,
        'chlorides': 0.04,
        'free sulfur dioxide': 20.0,
        'total sulfur dioxide': 120.0,
        'sulphates': 0.6
    }

    # Crear un DataFrame con los datos de entrada
    df_input = pd.DataFrame([input_data])

    # Asegurarse de que todas las columnas necesarias estén presentes
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0  # Rellenar columnas faltantes con un valor predeterminado

    # Ordenar las columnas en el orden esperado por el modelo
    df_input = df_input[expected_columns]
    
    print(f"Datos de entrada ajustados para el modelo:\n{df_input}")
    #print(model.feature_names_in_)


    # Realizar predicción con el modelo
    try:
        prediction = model.predict(df_input)
        print(f"Predicción: {prediction}")
    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
