#   Modelo de dicionario python para definicao de uma rede
network = dict()

#   Model and training configuration
network['model_name'] = 'training_example'
network['input_format'] = [64, 64, 3]

#   Convolutional layer
network['cnn'] = 'vgg16' #or inceptionv3, or resnet50
network['cnn_offline'] = False
network['cnn_freeze_imagenet_weights'] = True

#   Pooling layer
network['pooling'] = 'gap'  #or 'none', or 'gmp'

#   RNN Layer
network['rnn'] = 'lstm' #   or 'lstm'
network['rnn_timesteps'] = 10
network['rnn_outputsize'] = 64
network['rnn_dropout'] = 0.2
network['rnn_isstateful'] = False

#   Hidden FC layer
network['hiddenfc'] = True
network['hiddenfc_size'] = 64
network['hiddenfc_activation'] = 'tanh'
network['hiddenfc_regularizer'] = None
network['hiddenfc_dropout'] = 0

#   Auxiliary faster R-CNN support [EXPERIMENTAL]
# network['fasterRCNN_support'] = False
# network['fasterRCNN_type'] = 'counting'    #'counting' or 'onehot'
# network['fasterRCNN_dense_size'] = 64

#   Dataset related things
network['dataset_overlapwindows'] = True
network['dataset_causalprediction'] = False