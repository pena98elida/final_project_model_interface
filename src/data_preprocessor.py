import pandas as pd
import numpy as np

def preprocessor_data(input_data: dict, columns_to_impute: list) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada para la inferencia:
    - Reemplaza valores faltantes implícitos (como ceros) con la mediana calculada previamente.
    - Identifica y trata valores atípicos (outliers) usando el rango intercuartílico (IQR).
    
    Args:
        input_data (dict): Diccionario con los datos de entrada.
        columns_to_impute (list): Lista de columnas donde los valores de 0 se consideran faltantes.
    
    Returns:
        pd.DataFrame: DataFrame preprocesado.
    """
    try:
        # Convertir input_data a DataFrame
        df = pd.DataFrame([input_data])
        
        # Imputar valores implícitos de 0 con la mediana
        for col in columns_to_impute:
            if col in df.columns:
                median_value = df[col].median()  # Puedes ajustar para usar la mediana calculada del entrenamiento
                df[col] = df[col].replace(0., np.nan)  # Reemplazar ceros con NaN
                df[col].fillna(median_value, inplace=True)  # Imputar con la mediana
        
        # Tratamiento de outliers usando IQR
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Aplicar capping
            df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
        
        return df

    except Exception as e:
        print(f"Error en preprocessor_data: {e}")
        raise
