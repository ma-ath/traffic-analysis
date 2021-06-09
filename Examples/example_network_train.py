import os, sys
import numpy as np

# Adiciona a pasta TrafficSoundAnalysis para o 'PATH', para que o python reconheça nosso pacote
sys.path.append(os.path.dirname(os.path.abspath('')))

# Importa a classe DataHandler da nossa biblioteca de trafego
from TrafficSoundAnalysis.DataHandler.DataHandler import *
from TrafficSoundAnalysis.Model.Model import *

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

#   Cria o objecto dataHandler
dataHandler = DataHandler(dataset_directory = dataset_dir)

#   Cria o modelo a ser treinado a partir de um dicionario python

from network import *

model = Model()

model.BuildModelFromDictionary(network)

tf.keras.utils.plot_model(
    model.model,
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

