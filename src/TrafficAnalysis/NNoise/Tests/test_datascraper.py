import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from TrafficAnalysis.NNoise.DataHandler.DataHandler import *

test_dataset = os.path.join(os.path.dirname(__file__), "dataset")

dataHandler = DataHandler(dataset_directory = test_dataset)

dataHandler.scraper.ExtractImages(downsampling_factor=10, image_format = [240,240,3])