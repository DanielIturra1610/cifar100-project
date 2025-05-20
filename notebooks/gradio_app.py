import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import os
import glob
import tempfile
import time
from tensorflow.keras.models import load_model
from pathlib import Path
import math

# Mapeo de personajes
MAP_CHARACTERS = {
    0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
    3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
    7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
    11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
    14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'
}

# Para acceder más fácilmente a los nombres de los personajes como lista
CHARACTER_NAMES = list(MAP_CHARACTERS.values())

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

# Cargar detector optimizado para velocidad pero con buena precisión
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')  # El mejor para caricaturas

# Función para procesar videos
def is_overlapping(box1, box2, threshold=0.5):
    """Determina si dos cuadros delimitadores se superponen significativamente."""
    # Extraer coordenadas
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calcular área de cada caja
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Calcular coordenadas de intersección
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap
    
    # Calcular ratio de superposición con respecto al área menor
    smaller_area = min(area1, area2)
    if smaller_area == 0:
        return False
    
    overlap_ratio = overlap_area / smaller_area
    return overlap_ratio > threshold

# Diccionario para mantener la identidad de personajes entre frames
tracked_characters = {}

def create_tracker():
    """Crea un tracker para seguimiento de objetos"""
    # Usar CSRT que es más preciso aunque un poco más lento
    return cv2.TrackerCSRT_create()

def process_video(video_path):
    """Procesa un video y devuelve la ruta del video procesado."""
    if model is None:
        return None, "No se pudo cargar el modelo. Asegúrate de que haya un modelo entrenado."
    
    # Crear archivo temporal para la salida con nombre más informativo
    output_filename = f"simpson_detected_{int(time.time())}.mp4"
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    print(f"El video procesado se guardará en: {output_path}")
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Error: No se pudo abrir el video {video_path}"
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_stats = {name: 0 for name in CHARACTER_NAMES}
    total_detections = 0
    
    # Procesar cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Copia del frame original
        output_frame = frame.copy()
        
        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste con ecualización de histograma para mejor detección
        gray = cv2.equalizeHist(gray)
        
        # Parámetros optimizados específicamente para personajes de Los Simpson
        # minNeighbors más bajo para detectar caricaturas
        # minSize ajustado para evitar falsos positivos pequeños
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # Más preciso para detectar diferentes tamaños
            minNeighbors=2,     # Menos restrictivo para caricaturas
            minSize=(30, 30),   # Tamaño mínimo razonable
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Primero actualizar trackers existentes
        tracking_ids_to_remove = []
        for track_id, tracker_info in tracked_characters.items():
            tracker = tracker_info['tracker']
            success, box = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                character = tracker_info['character']
                confidence = tracker_info['confidence']
                label = f"{character.replace('_', ' ').title()}: {confidence:.1f}%"
                
                # Dibujar rectángulo y etiqueta con colores más visibles
                box_color = (255, 0, 255)  # Magenta en BGR
                text_color = (255, 255, 255)  # Blanco para el texto
                
                # Dibujar rectángulo más grueso
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), box_color, 3)
                
                # Añadir fondo semi-transparente para el texto
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_y = y - 15 if y - 15 > 15 else y + h + 25
                text_x = x
                
                # Dibujar rectángulo de fondo
                cv2.rectangle(output_frame, 
                            (text_x-5, text_y-label_size[1]-5),
                            (text_x + label_size[0] + 5, text_y + 5),
                            (0, 0, 0), -1)  # Rectángulo negro relleno
                
                # Dibujar el texto
                cv2.putText(output_frame, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                # Actualizar estadísticas de ese personaje
                if character in processed_stats:
                    processed_stats[character] += 1
                    total_detections += 1
                    
                # Actualizar contador de vida del tracker
                tracker_info['life'] += 1
            else:
                # Si fallan demasiados frames seguidos, marcar para eliminar
                tracker_info['failures'] += 1
                if tracker_info['failures'] > 5:  # Tolerancia a fallas
                    tracking_ids_to_remove.append(track_id)
        
        # Eliminar trackers fallidos
        for track_id in tracking_ids_to_remove:
            del tracked_characters[track_id]
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer el rostro
            face_region = frame[y:y+h, x:x+w]
            
            # Preprocesar la imagen
            preprocessed = preprocess_image(face_region)
            
            if preprocessed is not None:
                # Predecir
                predictions = model.predict(preprocessed, verbose=0)[0]
                top_idx = np.argmax(predictions)
                confidence = predictions[top_idx] * 100
                
                # Solo mostrar si la confianza es mayor a un umbral (reducido para capturar más personajes)
                if confidence > 25:
                    character = MAP_CHARACTERS.get(top_idx, "desconocido")
                    label = f"{character.replace('_', ' ').title()}: {confidence:.1f}%"
                    
                    # Dibujar rectángulo y etiqueta con colores más visibles
                    # Usar color magenta (255, 0, 255) que destaca bien sobre fondos claros y oscuros
                    box_color = (255, 0, 255)  # Magenta en BGR
                    text_color = (255, 255, 255)  # Blanco para el texto
                    
                    # Dibujar rectángulo más grueso (3 pixels)
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), box_color, 3)
                    
                    # Añadir fondo semi-transparente para el texto
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_y = y - 15 if y - 15 > 15 else y + h + 25
                    text_x = x
                    
                    # Dibujar rectángulo de fondo
                    cv2.rectangle(output_frame, 
                                (text_x-5, text_y-label_size[1]-5),
                                (text_x + label_size[0] + 5, text_y + 5),
                                (0, 0, 0), -1)  # Rectángulo negro relleno
                    
                    # Dibujar el texto más grande
                    cv2.putText(output_frame, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    
                    # Crear un tracker para este rostro
                    track_id = len(tracked_characters) + 1
                    tracker = create_tracker()
                    bbox = (x, y, w, h)
                    success = tracker.init(frame, bbox)
                    
                    if success:
                        tracked_characters[track_id] = {
                            'tracker': tracker,
                            'character': character,
                            'confidence': confidence,
                            'life': 0,       # Cuántos frames ha durado
                            'failures': 0    # Contador de fallos consecutivos
                        }
                    
                    # Actualizar estadísticas
                    if character in processed_stats:
                        processed_stats[character] += 1
                        total_detections += 1
        
        # Progreso
        frame_count += 1
        if frame_count % 10 == 0:  # Mostrar progreso más frecuentemente
            print(f"Procesado {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
        
        # Guardar el frame en el video de salida
        out.write(output_frame)
    
    # Liberar recursos
    cap.release()
    out.release()
    
    # Generar resumen de detecciones
    summary = "Resumen de detecciones:\n"
    if total_detections > 0:
        # Ordenar personajes por frecuencia de aparición
        sorted_chars = sorted(processed_stats.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars:
            if count > 0:
                percentage = (count / total_detections) * 100
                summary += f"- {char.replace('_', ' ').title()}: {count} detecciones ({percentage:.1f}%)\n"
    else:
        summary += "No se detectaron personajes en el video."
    
    print(f"Procesamiento completado: {total_detections} detecciones totales")
    return output_path, summary

def video_interface(video):
    """Función para procesar videos en la interfaz de Gradio.""" 
    if video is None:
        return None, "Por favor, sube un video."
    
    # Verificar si video es una ruta (string) o bytes
    if isinstance(video, str):
        # Si es una ruta, usarla directamente
        video_path = video
        # Crear un directorio temporal para la salida
        temp_dir = tempfile.mkdtemp()
        output_path, summary = process_video(video_path)
    else:
        # Si son bytes, guardar en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            temp_input_path = temp_input.name
        
        # Escribir el video a disco
        with open(temp_input_path, 'wb') as f:
            f.write(video)
        
        # Procesar el video
        output_path, summary = process_video(temp_input_path)
        
        # Limpiar el archivo temporal de entrada
        try:
            os.unlink(temp_input_path)
        except:
            pass
    
    if output_path:
        return output_path, summary
    else:
        return None, summary

# Crear interfaz Gradio con pestañas
with gr.Blocks(title="Clasificador de Personajes de Los Simpson") as demo:
    gr.Markdown("# Clasificador de Personajes de Los Simpson")
    
    with gr.Tabs():
        # Pestaña de clasificación de imágenes
        with gr.TabItem("Clasificación de Imágenes"):
            gr.Markdown("### Sube una imagen o selecciona un ejemplo para ver la predicción del modelo CNN.")
            
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
        
        # Pestaña de procesamiento de video
        with gr.TabItem("Detector en Video"):
            gr.Markdown("### Sube un video de Los Simpson para detectar personajes")
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video de entrada")
                    process_btn = gr.Button("Procesar Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="Video con detecciones")
                    summary_output = gr.Textbox(label="Resumen de detecciones", lines=10)
            
            process_btn.click(
                fn=video_interface,
                inputs=video_input,
                outputs=[video_output, summary_output]
            )
            
            gr.Markdown("""
            #### Instrucciones para el video
            
            1. Sube un video corto de Los Simpson (idealmente menos de 1 minuto)
            2. Haz clic en "Procesar Video" y espera mientras se analiza cada frame
            3. Explora el video resultante con los personajes identificados
            4. El sistema funciona mejor con rostros frontales y bien iluminados
            """)
    
    # Instrucciones generales
    with gr.Accordion("Instrucciones para la demo", open=False):
        gr.Markdown("""
        ## Cómo usar esta demostración:
        
        ### Para imágenes individuales:
        1. Selecciona la pestaña "Clasificación de Imágenes"
        2. Sube una imagen o usa uno de los ejemplos
        3. Haz clic en "Predecir" para ver los resultados
        
        ### Para videos:
        1. Selecciona la pestaña "Detector en Video"
        2. Sube un video corto de Los Simpson
        3. Haz clic en "Procesar Video" y espera el resultado
        4. El procesamiento puede tardar dependiendo de la duración del video
        
        ### Para grabar una demostración:
        1. Usa software de grabación de pantalla (OBS, Loom, etc.)
        2. Muestra cómo funciona el modelo con diferentes imágenes y un video
        3. Menciona la precisión del modelo (86.55%)
        4. Comenta sobre fortalezas y limitaciones observadas
        """)

# Directorio de salida para videos procesados
OUTPUT_PATH = '/app/notebooks/output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Lanzar app
if __name__ == "__main__":
    if model is not None:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("No se pudo cargar el modelo. Asegúrate de que haya un modelo entrenado en la carpeta de checkpoints.")
