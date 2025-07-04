# Clasificación de Vacas por Raza

Este proyecto usa TensorFlow y un modelo entrenado en Python para identificar razas de ganado bovino mediante imágenes.

# Estructura
- `backend/`: API para cargar imágenes y predecir la raza
- `modelo_entrenamiento/`: Código y curvas del entrenamiento
- `vacas_model.h5`: Modelo entrenado
- `requirements.txt`: Librerías necesarias

## Cómo ejecutar
1. Crear entorno virtual:
   python -m venv venv
2. Activar entorno:
- Windows: `venv\Scripts\activate`

3. Instalar dependencias:
pip install -r requirements.txt

# Una vez activado el entorno virtual, levantar el servidor, desde backend/, ejecutar:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
