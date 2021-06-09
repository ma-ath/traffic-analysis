import os, sys
import numpy as np

# Adiciona a pasta TrafficSoundAnalysis para o 'PATH', para que o python reconheça nosso pacote
sys.path.append(os.path.dirname(os.path.abspath('')))

# Importa a classe DataHandler da nossa biblioteca de trafego
from TrafficSoundAnalysis.DataHandler.DataHandler import *
from TrafficSoundAnalysis.ModelHandler.ModelHandler import *

# Isso aqui é para resolve um problema de 'Out Of Memory' na minha gpu (GTX 1650)
# https://stackoverflow.com/questions/59873568/unknownerror-failed-to-get-convolution-algorithm
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


# dataset_dir = '/home/.../traffic-analysis/Examples/dataset'
dataset_dir = os.path.join(os.path.abspath(''), "dataset")

#   Cria a pasta 'raw' em 'Examples/dataset/raw'
if not os.path.isdir(os.path.join(dataset_dir, "raw")):
    raise Exception("Por favor, complete antes o tutorial para extração do dataset!")

#   Cria o modelo a ser treinado a partir de um dicionario python

from network import *
from folds import *

model = ModelHandler()

model.LoadModelFromDictionary(network)

model.CompileModel()

model.ShowModel()

### Treinamento
#   Cria o objecto dataHandler
dataHandler = DataHandler(dataset_directory = dataset_dir, image_format = [64, 64, 3])

[
    x_train,
    y_train,
    x_val,
    y_val
] = dataHandler.LoadDataset(folds[0], CNN_offline=False)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

model.Train(x=x_train, y=y_train, validation_data=(x_val, y_val))