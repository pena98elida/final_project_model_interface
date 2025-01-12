# Usa la imagen oficial de Python 3.12
FROM python:3.12-slim

# Instala Poetry
RUN pip install poetry

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY pyproject.toml poetry.lock README.md /app/

# Instala las dependencias sin crear entornos virtuales
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Copia el resto del proyecto
COPY . /app/

# Establece el comando para ejecutar el script de inferencia
CMD ["poetry", "run", "python", "src/inference.py"]

