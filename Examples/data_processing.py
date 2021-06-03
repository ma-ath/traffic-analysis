"""
/--------------------------------/
Created on June 2, 2021
@authors:
    Matheus S. Lima
@company: Federal University of Rio de Janeiro - Polytechnic School - Analog and Digital Signal Processing Laboratory (PADS)
/--------------------------------/
"""
#
#   Example of extraction of data from video and creation of a dataset
#
import os, sys
from cv2 import data
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from TrafficSoundAnalysis.DataHandler.DataHandler import *

#
#   https://stackoverflow.com/questions/59873568/unknownerror-failed-to-get-convolution-algorithm
#
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
   
test_dataset = os.path.join(os.path.dirname(__file__), "dataset")

#   Create 'dataset' folder on Examples/dataset
if not os.path.isdir(test_dataset):
    os.makedirs(test_dataset)
#   Create 'dataset/raw' folder  on Examples/dataset/raw
if not os.path.isdir(os.path.join(test_dataset, "raw")):
    os.makedirs(os.path.join(test_dataset, "raw"))

#   Object DataHandler
dataHandler = DataHandler(dataset_directory = test_dataset)

#   Uses a datascraper to extract all data
dataHandler.ExtractDataFromRawVideos(downsampling_factor=600, image_format = [64, 64, 3])

#   Loads data from one fold
[
    input_train_data,
    output_train_data,
    input_test_data,
    output_test_data,
    train_number_of_frames,
    test_number_of_frames
] = dataHandler.CreateFoldFromExtractedData(train_videos = ['road.mp4', 'road2.mp4','road3.mp4'], test_videos = ['road4.mp4','road5.mp4','road6.mp4'])

#   Saves all folds to disk

from folds import *

dataHandler.CreateAndSaveFoldsToDisk(folds)

dataHandler.ExtractImageFeaturesFromFoldsOnDisk(folds, image_format = [64, 64, 3], cnn='vgg16', pooling='gap')

[
    x_train,
    y_train,
    x_test,
    y_test
] = dataHandler.LoadDatasetFromFoldOnDisk("fold_0",
                                CNN="vgg16",
                                Pooling="gap",
                                LSTM=True,
                                time_steps=3,
                                overlap_windows=True,
                                causal_prediction=True,
                                stateful=True,
                                batch_size=32,
                                image_format = [64,64,3])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)