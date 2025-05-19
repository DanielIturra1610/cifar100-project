import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gradio as gr

# Carga el modelo entrenado
model = load_model("/app/notebooks/Entrega2/checkpoints/SimpleCNN_3conv_32_64_128filters_0.5drop_Adam_lr0.0010000000474974513_19-05-2025_0414_final.h5")

# Lista de clases (orden debe coincidir con entrenamiento)
class_names = ['abraham_grampa_simpson', 'apu_nahasapeemapetilon', 'bart_simpson', 'charles_montgomery_burns', 'chief_wiggum', 'comic_book_guy',
                'edna_krabappel', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lisa_simpson', 'marge_simpson', 'milhouse_van_houten',
                'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'principal_skinner', 'sideshow_bob']  

# Preprocesamiento de imagen
def preprocess_image(image):
    image = image.convert("RGB")              # Asegura 3 canales
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [64, 64])  # Asegura tamaño correcto
    image = tf.cast(image, tf.float32) / 255.0
    return image

def predict(image):
    img = preprocess_image(image)
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    # Si el modelo YA devuelve probabilidades, no aplicar softmax otra vez
    # probs = tf.nn.softmax(preds).numpy()
    probs = preds

    # Devolver valores como floats, no strings
    class_probabilities = {
        class_names[i]: float(prob) for i, prob in enumerate(probs)
    }

    # Ordenamos de mayor a menor
    sorted_probs = dict(sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True))

    return sorted_probs


# Interfaz de Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),  # Muestra las 5 más probables
    title="Clasificador de Simpsons",
    description="Sube una imagen de un personaje de Los Simpsons para predecir quién es."
)

# Lanzamiento (importante para docker)
demo.launch(server_name="0.0.0.0", server_port=7860)
