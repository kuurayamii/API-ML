from fastapi import FastAPI, File, UploadFile
from PIL import Image
import onnxruntime as ort
import numpy as np
from utils import Preprocesado_Imagen

api = FastAPI()
ort_session = ort.InferenceSession("modelo/modelo_reconocimiento_digimon.onnx")

digimons = {
    "0": "Gatomon",
    "1": "Wizardmon",
    "2": "Galgomon",
    "3": "Lopmon",
    "4": "Terriermon"
}


@api.get("/")
async def root():
    return{"mensaje": "Fast API en funcionamiento"}


@api.post("/prediccion/")
async def Predicciones(file: UploadFile = File()):
    
    imagen_request = Image.open(file.file).convert("RGB")

    input_tensor = Preprocesado_Imagen(imagen_request)

    input_name = ort_session.get_inputs()[0].name
    
    output = ort_session.run(None, {input_name: input_tensor})[0]

    prediccion = np.argmax(output, axis=1)
    

    return {
        "Nombre": int(prediccion)
    }