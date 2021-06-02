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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from TrafficAnalysis.NNoise.DataHandler.DataHandler import *



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
dataHandler.scraper.ExtractData(downsampling_factor=500, image_format = [224,224,3])

#   Builds 1 fold
input_train_data, output_train_data, input_test_data, output_test_data, train_number_of_frames, test_number_of_frames = dataHandler.scraper.BuildFoldFromExtractedData(train_videos = ['road.mp4', 'road2.mp4','road3.mp4'], test_videos = ['road4.mp4','road5.mp4','road6.mp4'])

#   ...