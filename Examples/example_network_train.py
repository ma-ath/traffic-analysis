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

#   Cria o objecto dataHandler
dataHandler = DataHandler(dataset_directory = dataset_dir)

#   Cria o modelo a ser treinado a partir de um dicionario python

from network import *

model = ModelHandler()

model.BuildModelFromDictionary(network)

## Geração de um modelo mobilenet para nosso projeto
input_format = (64, 64, 3)

# Baixa a rede mobilenet do keras.applications e congela seus pesos
convolutional_layer = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_format)
for layer in convolutional_layer.layers[:]:
    layer.trainable = False

# Cria um modelo do tipo sequencial
mobile_net = tf.keras.Sequential(name='mobile_net_test')

mobile_net.add(tf.keras.layers.Input(input_format))
mobile_net.add(convolutional_layer)
mobile_net.add(tf.keras.layers.GlobalAveragePooling2D(data_format=None))

mobile_net.add(tf.keras.layers.Dense(128, activation='tanh'))
mobile_net.add(tf.keras.layers.Dropout(0.2))
mobile_net.add(tf.keras.layers.Dense(1, activation='linear'))

# Summarize
mobile_net.summary()

model.BuildModelFromKeras(mobile_net)

#model.ShowModel()
