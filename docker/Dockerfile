FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de Jupyter
EXPOSE 8888

# Comando para iniciar Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
