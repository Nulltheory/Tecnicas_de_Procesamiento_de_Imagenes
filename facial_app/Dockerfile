# Usa Python 3.11
FROM python:3.11-slim
WORKDIR /app

# Instala dependencias del sistema (CON LA LIBRERÍA GL CORREGIDA)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    # Dependencias de OpenCV (libgl1 es la clave)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de tu proyecto
COPY requirements.txt ./
COPY src/ ./src/
COPY app.py ./
COPY .streamlit/ ./.streamlit/

# Instala las dependencias de Python
RUN pip3 install -r requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Ejecuta TU archivo (app.py)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]