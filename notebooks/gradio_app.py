import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import os
import glob
from tensorflow.keras.models import load_model

# Mapeo de personajes
MAP_CHARACTERS = {
    0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
    3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
    7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
    11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
    14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'
}

# Tamaño de imágenes
IMG_SIZE = 64

# Cargar el modelo
model_path = '/app/notebooks/Entrega2/checkpoints'

# Función para buscar el modelo más reciente
def get_latest_model(model_dir):
    model_files = glob.glob(os.path.join(model_dir, '*_best.h5'))
    if not model_files:
        return None
    # Ordenar por fecha de modificación (el más reciente primero)
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def load_latest_model():
    model_file = get_latest_model(model_path)
    if model_file:
        print(f"Cargando modelo: {model_file}")
        print(f"Fecha de modificación: {os.path.getmtime(model_file)}")
        return load_model(model_file)
    else:
        print("No se encontró ningún modelo. Asegúrate de entrenar un modelo primero.")
        return None

# Preparar modelo
model = load_latest_model()

def preprocess_image(image):
    """Preprocesa la imagen para la predicción."""
    # Redimensionar
    if image is not None:
        # Convertir de RGB a BGR si viene de Gradio
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Redimensionar
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Normalizar
        image = image.astype('float32') / 255.0
        
        # Expandir dimensiones para que coincida con el input del modelo
        return np.expand_dims(image, axis=0)
    return None

def predict_character(image):
    """Predice el personaje en la imagen."""
    if image is None:
        return {p: 0.0 for p in MAP_CHARACTERS.values()}
    
    # Preprocesar imagen
    processed_image = preprocess_image(image)
    
    # Predecir
    preds = model.predict(processed_image)[0]
    
    # Crear diccionario de resultados
    results = {MAP_CHARACTERS[i]: float(preds[i]) for i in range(len(preds))}
    
    return results

def predict_and_display(image):
    """Función para Gradio que muestra la imagen y sus predicciones."""
    if image is None:
        return None, {p: 0.0 for p in MAP_CHARACTERS.values()}
    
    # Obtener predicciones
    predictions = predict_character(image)
    
    return image, predictions

# Ejemplos de imágenes para la galería
test_images_dir = '/app/notebooks/Entrega2/simpsons_data/kaggle_simpson_testset/kaggle_simpson_testset'
examples = []
if os.path.exists(test_images_dir):
    for char_name in ["homer_simpson", "bart_simpson", "lisa_simpson", "marge_simpson", "ned_flanders"]:
        char_files = glob.glob(f"{test_images_dir}/{char_name}_*.jpg")
        if char_files:
            examples.extend(char_files[:2])  # Agregar 2 ejemplos de cada personaje

# Crear interfaz Gradio
with gr.Blocks(title="Clasificador de Personajes de Los Simpson") as demo:
    gr.Markdown("# Clasificador de Personajes de Los Simpson")
    gr.Markdown("Sube una imagen o selecciona un ejemplo para ver la predicción del modelo CNN.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Imagen de entrada", type="numpy")
            predict_btn = gr.Button("Predecir", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Imagen procesada")
            output_label = gr.Label(label="Predicciones")
    
    predict_btn.click(
        fn=predict_and_display,
        inputs=input_image,
        outputs=[output_image, output_label]
    )
    
    gr.Examples(
        examples=examples,
        inputs=input_image
    )
    
    gr.Markdown("""
    ## Instrucciones para el video
    
    1. Selecciona o sube diferentes imágenes de personajes
    2. Haz clic en "Predecir" para ver los resultados
    3. Graba la pantalla mientras interactúas con la aplicación
    4. Comenta brevemente sobre la precisión del modelo
    """)

# Lanzar app
if __name__ == "__main__":
    if model is not None:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("No se pudo cargar el modelo. Asegúrate de que haya un modelo entrenado en la carpeta de checkpoints.")
