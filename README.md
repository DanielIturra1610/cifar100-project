# Proyecto de Clasificación CNN - Los Simpson

Este proyecto implementa un clasificador de imágenes basado en Redes Neuronales Convolucionales (CNN) para identificar 18 personajes de la serie Los Simpson.

## Descripción

El sistema utiliza TensorFlow/Keras para desarrollar y entrenar un modelo CNN capaz de reconocer personajes de Los Simpson con una precisión superior al 85%. Todo el entorno está dockerizado para facilitar la reproducibilidad y colaboración.

## Estructura del Proyecto

```
cifar100-project/
├── docker/                   # Configuración de Docker
│   └── Dockerfile            # Configuración del entorno
├── notebooks/                # Notebooks y código
│   ├── Entrega2/             # Entrega 2 - CNN para Los Simpson
│   │   ├── checkpoints/      # Modelos guardados
│   │   ├── setup_data.py     # Script para descargar datos
│   │   └── Entrega_2_CNN_Solucion.ipynb  # Notebook principal
├── docker-compose.yml        # Configuración de servicios
├── requirements.txt          # Dependencias de Python
└── README.md                 # Este archivo
```

## Requisitos Previos

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Aproximadamente 3GB de espacio libre para imágenes Docker y datasets

## Configuración Rápida

Sigue estos pasos para poner en marcha el proyecto:

### 1. Clonar el repositorio

```bash
git clone https://github.com/[usuario]/cifar100-project.git
cd cifar100-project
```

### 2. Iniciar el entorno Docker

```bash
docker-compose up
```

Este comando:
- Construye la imagen de Docker con todas las dependencias
- Inicia los servicios de Jupyter Notebook
- Mapea el puerto 8888 para acceder a Jupyter

### 3. Acceder a Jupyter Notebook

Una vez que los contenedores estén funcionando, abre tu navegador y ve a:

```
http://localhost:8888
```

No se requiere token o contraseña para acceder.

### 4. Trabajar con el Notebook

1. Navega a `notebooks/Entrega2/Entrega_2_CNN_Solucion.ipynb`
2. Al ejecutar el notebook, el dataset se descargará automáticamente (~1.16GB)
3. Sigue las instrucciones del notebook para entrenar y evaluar el modelo

## Dataset

El dataset contiene aproximadamente:
- 19,000 imágenes de entrenamiento
- 890 imágenes de prueba
- 18 clases (personajes principales de Los Simpson)

La descarga y configuración del dataset es automática mediante el script `setup_data.py` que se ejecuta desde el notebook.

## Solución de problemas

- **Error de OpenCV**: Si aparecen errores relacionados con libGL.so, estos ya están solucionados en la imagen Docker
- **Problemas de memoria**: Reduce el batch size si tu GPU tiene limitaciones de memoria

## Colaboración

Si quieres colaborar en este proyecto:
1. Crea un fork del repositorio
2. Crea una rama para tu contribución (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request