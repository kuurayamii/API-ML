#Las librerias utilizadas para el desarrollo de la API
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort
import numpy as np

#Archivo Utils del cual se extrae la imagen ya procesada, con el objetivo de ocupar menos lineas de codido
from utils import Preprocesado_Imagen


#Se inicializa FastAPI para poder realizar lo necesario para esta
api = FastAPI()

origins = [
    "*"
]

api.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#Se inicializa onnx runtime con la url de donde se encuentra actualmente el modelo
ort_session = ort.InferenceSession("modelo/modelo_reconocimiento_digimon.onnx")

#Se creo un diccionario para luego poder cambiar los valores predichos al final, correspondientes a 5 Digimons
digimons = {
    0: "Gatomon",
    1: "Wizardmon",
    2: "Galgomon",
    3: "Lopmon",
    4: "Terriermon"
}

#url base
@api.get("/")
async def root():
    """
        Se utiliza principalmente para verificar el funcionamiento de la API


        Returns:
            Lo que se retorna es 'Fast API en funcionamiento' lo que indica que, efectivamente, la API funciona como corresponde
    """
    return{"mensaje": "Fast API en funcionamiento"}

#url a la cual dirigirse para poder realizar la prediccion
@api.post("/prediccion/")
async def Predicciones(file: UploadFile = File()):

    """
        Se utiliza para realizar predicciones, en base a un modelo ya entrenado con relacion a Digimon.
        Este modelo permite predecir si, una imagen corresponde o no a 5 Digimons, los cuales son:
        'Gatomon', 'Wizardmon', 'Galgomon', 'Lopmon' y 'Terriermon'. En caso de que la imagen no corresponda a
        los Digimons mencionados, esta retorna 'La imagen indicada o no es un digimon, o es uno fuera de los 5'


        Args:
            params: file (File): Imagen

        Return:
            (int) con cambio a (str) segun (dict) mencionado anteriormente
    
    """
    
    imagen_request = Image.open(file.file).convert("RGB")

    input_tensor = Preprocesado_Imagen(imagen_request)

    input_name = ort_session.get_inputs()[0].name
    
    output = ort_session.run(None, {input_name: input_tensor})[0]

    prediccion = np.argmax(output, axis=1)

    prediccion = int(prediccion)

    digimon_predicho = digimons.get(prediccion, "La imagen indicada o no es un digimon, o es uno fuera de los 5")
    
    return {
        "Nombre": "Es un " + digimon_predicho
    }