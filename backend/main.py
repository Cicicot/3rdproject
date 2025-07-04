
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predictor import predecir_raza
import os
import uuid

app = FastAPI()

# Para permitir peticiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta temporal para subir imágenes
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/clasificar/")
async def clasificar(file: UploadFile = File(...)):
    # Guardar imagen temporalmente
    ext = file.filename.split('.')[-1]
    nombre_archivo = f"{uuid.uuid4()}.{ext}"
    ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)

    with open(ruta_imagen, "wb") as buffer:
        buffer.write(await file.read())

    # Predecir
    resultado = predecir_raza(ruta_imagen)

    # Eliminar imagen
    os.remove(ruta_imagen)

    return {"raza": resultado}
