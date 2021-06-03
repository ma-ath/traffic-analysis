"""
/--------------------------------/
Created on June 1, 2021
@authors:
    Matheus S. Lima
@company: Federal University of Rio de Janeiro - Polytechnic School - Analog and Digital Signal Processing Laboratory (PADS)
/--------------------------------/
"""

import os

"""
This file contains constants used throughout the code 
"""
#   -----------------------------------------------
#   Constants related to file structure

from os.path import expanduser

CONST_STR_NNOISE_BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONST_STR_DATASET_BASEPATH = os.path.join(expanduser("~"), "dataset") #os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),"dataset")           #Datapath for the dataset
CONST_STR_DATASET_BASEPATH_RAW = "raw"
CONST_STR_DATASET_BASEPATH_FRCNN = "fasterRCNN_features"
CONST_STR_DATASET_BASEPATH_FOLDS = "folds"
CONST_STR_DATASET_BASEPATH_FOLDS_VGG16 = "vgg16"
CONST_STR_DATASET_BASEPATH_FOLDS_RESNET50 = "resnet50"
CONST_STR_DATASET_BASEPATH_FOLDS_INCEPTIONV3 = "inceptionV3"

CONST_STR_DATASET_CONFIG_FILENAME = "configuration_file.pickle"        #this is the name of the config file for raw videos
CONST_STR_DATASET_NMBOFFRAMES_FILENAME = "number_of_frames.pickle"
CONST_STR_DATASET_AUDIOFILE_FILENAME = "audio_data.wav"
CONST_STR_DATASET_TARGETDATA_FILENAME = "output_targets.npy"
CONST_STR_DATASET_STATISTICS_FILENAME = "video_statistics.pickle"
CONST_STR_DATASET_STACKEDFRAMES_FILENAME = "stacked_video_frames.npy"

CONST_STR_RESULTS_BASEPATH = "results"

#   -----------------------------------------------
#   Constants related to network structure

CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE = (7, 7, 512)
CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE = (5, 5, 2048)
CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE = (7, 7, 2048)

if __name__ == "__main__":
    print("File path structure:")
    print("---------------------------------")
    print("NNoise basepath:", CONST_STR_NNOISE_BASEPATH)
    print("---------------------------------")
    print("Dataset basepath:", CONST_STR_DATASET_BASEPATH)
    print("Dataset structure:")
    print("\t", os.path.join(CONST_STR_DATASET_BASEPATH, CONST_STR_DATASET_BASEPATH_RAW))
    print("\t", os.path.join(CONST_STR_DATASET_BASEPATH, CONST_STR_DATASET_BASEPATH_FRCNN))
    print("\t", os.path.join(CONST_STR_DATASET_BASEPATH, CONST_STR_DATASET_BASEPATH_FOLDS))
    print("\t", os.path.join(CONST_STR_DATASET_BASEPATH, CONST_STR_DATASET_BASEPATH_FOLDS, CONST_STR_DATASET_BASEPATH_FOLDS_VGG16))
    print("\t", os.path.join(CONST_STR_DATASET_BASEPATH, CONST_STR_DATASET_BASEPATH_FOLDS, CONST_STR_DATASET_BASEPATH_FOLDS_RESNET50))
    print("\t", os.path.join(CONST_STR_DATASET_BASEPATH, CONST_STR_DATASET_BASEPATH_FOLDS, CONST_STR_DATASET_BASEPATH_FOLDS_INCEPTIONV3))
    
