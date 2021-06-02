"""
/--------------------------------/
Created on June 1, 2021
@authors:
    Matheus S. Lima
@company: Federal University of Rio de Janeiro - Polytechnic School - Analog and Digital Signal Processing Laboratory (PADS)
/--------------------------------/
"""

import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from TrafficAnalysis.NNoise.DataHandler.DataScraper.DataScraper import DataScraper
from TrafficAnalysis.NNoise.utils.utils import *
from TrafficAnalysis.NNoise.utils.consts import *

class DataHandler:
    """
    Implements the code for handling data used in training and evaluation of sound pressure inference models using images
    """

    __dataset_directory = ""

    def __init__(self, dataset_directory = CONST_STR_DATASET_BASEPATH):
        if type(dataset_directory) != str:
            print_error("DataHandler constructor expects a string as argument, not "+str(type(dataset_directory)))
            raise TypeError("DataHandler constructor expects a string as argument, not "+str(type(dataset_directory)))

        self.__dataset_directory = dataset_directory
        self.scraper = DataScraper(dataset_directory)

        if not self.__CheckIfPathExists(dataset_directory):
            print_error("Cound not find the dataset base folder: "+str(dataset_directory))
            raise Exception("Cound not find the dataset base folder")
        if not self.__CheckIfPathExists(os.path.join(dataset_directory, CONST_STR_DATASET_BASEPATH_RAW)):
            print_error("Cound not find the dataset raw folder: "+os.path.join(dataset_directory, CONST_STR_DATASET_BASEPATH_RAW))
            raise Exception("Cound not find the dataset raw folder")
        pass

    def __CheckIfPathExists(self, path):
        return os.path.exists(path)

    pass