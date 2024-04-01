#### Importar modelo entrenado con data augmentation e implementado para GradCAM
#### Removemos las últimas capas que no son convolucionales, para dejar sólo las capas
##### encargadas de realizar la reducción de dimensiones mediante la aplicación de filtros de 
###   convolution, strides y maxpooling


import numpy as np
import tensorflow as tf
from tensorflow import keras



### Importar modelo entrenado, con el que hemos estado haciendo inferencias



model = keras.models.load_model('.../DataAugmentation/modelo_para_GradCAM2')



##### --->>>> Retirar capas que no son convolucionales y usar la reducción de dimensiones dada por las capas 
#### de convolución.




model2 = keras.Sequential()

for layer in model.layers[:-3]: # remover 
    model2.add(layer)    


# Freeze layers ### remover la posibilidad de seguir entrenando las capas. 
for layer in model2.layers:
    layer.trainable = False




