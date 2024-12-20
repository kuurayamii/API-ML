import numpy as np
from PIL import Image

def Preprocesado_Imagen(imagen: Image.Image, input_size=(224,224)):

    imagen = imagen.resize(input_size)

    image_array = np.array(imagen).astype(np.float32)
    
    image_array /= 255.0

    image_array = np.transpose(image_array, (2,0,1))

    image_array = np.expand_dims(image_array, axis=0)

    return image_array
