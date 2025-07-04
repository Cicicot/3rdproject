# backend/predictor.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Cargar modelo entrenado
modelo_path = os.path.join(os.path.dirname(__file__), "vacas_model.h5")
modelo = tf.keras.models.load_model(modelo_path)

class_names = ['brahman', 'cholistani', 'dhani', 'fresian', 'kankarej', 'sahiwal', 'sibbi']  

def predecir_raza(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicciones = modelo.predict(img_array)
    pred_index = np.argmax(predicciones)
    probabilidad = np.max(predicciones)

    if probabilidad < 0.7:
        return "Desconocida"

    return class_names[pred_index]
