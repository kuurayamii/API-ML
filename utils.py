#Importaciones de las librerias a utilizar
import numpy as np
from PIL import Image

def Preprocesado_Imagen(imagen: Image.Image, input_size=(224,224)):

    """
        Funcion la cual permite el procesamiento de las imagenes de tal forma que, pueda usarse con el modelo
        ya entrenado. Retorna la imagen ya procesada


        Args:
            imagen (Image)
            input_size (tuple): (224,244)

        Return:
            image_array
    
    """

    imagen = imagen.resize(input_size)

    image_array = np.array(imagen).astype(np.float32)
    
    image_array /= 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array
