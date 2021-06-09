"""
/--------------------------------/
Created on June 1, 2021
@authors:
    Matheus S. Lima
@company: Federal University of Rio de Janeiro - Polytechnic School - Analog and Digital Signal Processing Laboratory (PADS)
/--------------------------------/
"""

import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from TrafficSoundAnalysis.utils.utils import *
from TrafficSoundAnalysis.utils.consts import *

class ModelHandler:
    """
    This class is the model used for defining and training neural networks 
    """
    #   Keras model inside this Model Class
    model = None

    def __init__(self):
        pass

    def Train(self, optimizer = tf.keras.optimizers.Adam, loss = tf.keras.losses.mean_squared_error, batchsize = 32, epochs = 20, callbacks = None):
        """
        Method that receives an image capture, a model and estimates the sound pressure information
        """
        pass

    def Evaluate():
        """
        Method that receives an image capture, a model and estimates the sound pressure information
        """
        pass

    def Save():
        """
        Method that receives an image capture, a model and estimates the sound pressure information
        """
        pass

    def Load():
        """
        Method that receives an image capture, a model and estimates the sound pressure information
        """
        pass

    def BuildModelFromDictionary(self, network):
        """
        Function that loads the desired model from a python dictionary
        Updates the model.model paramether, and also returns the builded model
        """

        model = tf.keras.Model()

        try:
            #   Checks if the network should include the cnn layers or not (offline or online training)
            #   If its offline, we assume that the user has already extracted all its features
            if not network['cnn_offline']:
                #   Input format of network
                input_format = (network['input_format'][0], network['input_format'][1], network['input_format'][2])
                layer_input = tf.keras.layers.Input(input_format)

                #   ------------------  CNN  ------------------
                #   Load the correct convolutional layer
                if network['cnn'].lower() == "vgg16":
                    convolutional_layer = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_format)
                elif network['cnn'].lower() == "inceptionv3":
                    convolutional_layer = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_format)
                elif network['cnn'].lower() == "resnet50":
                    convolutional_layer = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_format)
                else:
                    raise Exception('Non supported CNN Model: '+str(network('cnn')))

                #   Freeze (or not) all convolutional layers
                for layer in convolutional_layer.layers[:]:
                    layer.trainable = not network['cnn_freeze_imagenet_weights']

                output_shape = convolutional_layer.output_shape
                convolution_output = convolutional_layer(layer_input)

                #   ------------------  Pooling  ------------------
                if network['pooling'].lower() == "none":
                    pooling_output = convolution_output
                elif network['pooling'].lower() == "gap":
                    pooling_output = tf.keras.layers.GlobalAveragePooling2D(data_format=None)(convolution_output)
                elif network['pooling'].lower() == "gmp":
                    pooling_output = tf.keras.layers.GlobalMaxPooling2D(data_format=None)(convolution_output)

                cnn_output = pooling_output
            else:
                #   ------------------  CNN  ------------------
                if network['cnn'].lower() == "vgg16":
                    input_format = (CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[0], CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[1], CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2])
                elif network['cnn'].lower() == "inceptionv3":
                    input_format = (CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[0], CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[1], CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2])
                elif network['cnn'].lower() == "resnet50":
                    input_format = (CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[0], CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[1], CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2])
                else:
                    raise Exception('Non supported CNN Model: '+str(network('cnn')))

                #   ------------------  Pooling  ------------------
                if network['pooling'].lower() != "none":
                    input_format = input_format[2]

                layer_input = tf.keras.layers.Input(input_format)
                cnn_output = layer_input
            
            #   At the end of this block, we have an 'layer_input' variable of type tf.keras.layers.Input, and cnn_output of type tf.keras.layers...
            # print(input_format)
            # print(layer_input)
            # print(cnn_output)

            #   Case of no RNN network
            if network['rnn'].lower() == 'none':
                #   No RNN on network:
                #   Check if the network has a hidden fc layer; If not, it's very simple indeed
                if not network['hiddenfc']:
                    #   Output layer
                    layer_output = tf.keras.layers.Dense(1, activation='linear')(cnn_output)
                else:
                    layer_hiddenfc = tf.keras.layers.Dense(network['hiddenfc_size'], activation=network['hiddenfc_activation'], activity_regularizer=network['hiddenfc_regularizer'])(cnn_output)
                    layer_hiddenfc_dropout = tf.keras.layers.Dropout(network['hiddenfc_dropout'])(layer_hiddenfc)
                    #   Output layer
                    layer_output = tf.keras.layers.Dense(1, activation='linear')(layer_hiddenfc_dropout)
                
                #   Here, we have a sucessefuly created a network, with input and output
                #   We just need to define the model
                model = tf.keras.models.Model(inputs=layer_input, outputs=layer_output)
                self.model = model
                return model
            elif network['rnn'].lower() == 'lstm':
                # Change the input format to (timesteps, input_format)
                if type(input_format) is int:
                    new_input_format = (network['rnn_timesteps'], input_format)
                else:
                    new_input_format = (network['rnn_timesteps'],)+input_format

                # Creates a new, TimeDistributed model of the CNN. Adds the adequate input layer to this model
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Input(shape=new_input_format))

                #   ---------------------------- CNN
                if not network['cnn_offline']:
                    # Puts the whole CNN inside TimeDistribute,
                    # Extremelly heavy model, as the CNN is repeated 'rnn_timesteps' times

                    # Model of only 1 CNN.
                    cnn_model = tf.keras.Model(inputs=layer_input, outputs=cnn_output, name="CNN")

                    # Adds every layer of the CNN after its input
                    for layer in cnn_model.layers[1:]:
                        model.add(tf.keras.layers.TimeDistributed(layer))

                #   ---------------------------- RNN
                # Add the LSTM to the model
                model.add(tf.keras.layers.LSTM(network['rnn_outputsize'], dropout=network['rnn_dropout'], stateful=network['rnn_isstateful']))

                if not network['hiddenfc']:
                    #   Output layer
                    model.add(tf.keras.layers.Dense(1, activation='linear'))
                else:
                    #   hidden fc
                    model.add(tf.keras.layers.Dense(network['hiddenfc_size'], activation=network['hiddenfc_activation'], activity_regularizer=network['hiddenfc_regularizer']))
                    #   dropout
                    model.add(tf.keras.layers.Dropout(network['hiddenfc_dropout']))
                    #   Output layer
                    model.add(tf.keras.layers.Dense(1, activation='linear'))

                self.model = model
                return model
            else:
                raise Exception('Non supported RNN Model: '+str(network('rnn')))


        except Exception as e:
            print_error('An error has occurred')
            print_error(str(e))
    
    def BuildModelFromKeras(self, model):
        """
        Method that simply updates the self.model paramether with an externally defined model
        """
        valid_model = tf.keras.Model()
        valid_sequential =tf.keras.Sequential()
        if type(model) != type(valid_model) and type(model) != type(valid_sequential):
            raise Exception("You must pass a valid keras.Model or keras.Sequential model")
        self.model = model
    
    def ShowModel(self, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True):
        if self.model is not None:
            tf.keras.utils.plot_model(
                self.model,
                to_file=to_file,
                show_shapes=show_shapes,
                show_dtype=show_dtype,
                show_layer_names=show_layer_names,
                expand_nested=expand_nested
            )
        else:
            print_error('You must build a model before attempting to show it')
        pass
    pass