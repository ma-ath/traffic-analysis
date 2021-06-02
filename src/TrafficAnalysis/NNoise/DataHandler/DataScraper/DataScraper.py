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
import pickle
import numpy as np
import scipy.io.wavfile
import math
import re
from imutils import paths

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from TrafficAnalysis.NNoise.utils.utils import *
from TrafficAnalysis.NNoise.utils.consts import *

class DataScraper:
    """
    Implements the code for scraping data to be used for training and evaluation of sound pressure inference models using images.

    This class is responsible for extracting video images, building folds and 
    """

    #   Default values
    __dataset_directory = ""
    __downsampling_factor = 1
    __image_format = [224,224,3]

    def __init__(self, dataset_directory = CONST_STR_DATASET_BASEPATH):
        """
        Class constructor:
        ---------------------------
        dataset_directory : string
        Contains the path to the dataset files
        """
        if type(dataset_directory) == str:
            self.__dataset_directory = dataset_directory
        else:
            print_error("DataScraper constructor expects a string as argument, not "+str(type(dataset_directory)))
            raise Exception("DataScraper constructor expects a string as argument")

    def ExtractData(self, downsampling_factor, image_format, video = None, dont_extract_soundtargets = False):
        """
        Method that extract images in a given directory
        ---------------------------
        downsampling_factor : int
        downsampling factor for frames extractions

        video
        extract images from a specific video [None = all videos on raw datapath]
        """

        #   Check methods inputs
        if type(downsampling_factor) != int:
            print_error("downsampling_factor is expected to be an integer, not "+str(type(downsampling_factor)))
            raise TypeError("downsampling_factor is expected to be an integer, not "+str(type(downsampling_factor)))

        if video != None and type(video) != str:
            print_error("video is expected to be a string, not "+str(type(video)))
            raise TypeError("video is expected to be a string, not "+str(type(video)))
        
        if type(image_format) != list and len(image_format) != 3:
            print_error("image_format is expected to be a list of size 3, not "+str(type(image_format)))
            raise TypeError("image_format is expected to be a list of size 3, not "+str(type(image_format)))

        self.__downsampling_factor = downsampling_factor
        self.__image_format = image_format

        print_info("Starting image extraction process")
        print_info("Basepath of this dataset: "+self.__dataset_directory)
        print_info("Downsampling factor of extraction: "+str(self.__downsampling_factor))
        print_info("Format of extracted images: "+str(self.__image_format))
        print_info("Reading filepath off all videos")

        #   Loads all paths from all videos
        try:
            dataset_raw_datapath = [f for f in os.listdir(os.path.join(self.__dataset_directory, CONST_STR_DATASET_BASEPATH_RAW)) if os.path.isfile(os.path.join(os.path.join(self.__dataset_directory, CONST_STR_DATASET_BASEPATH_RAW), f))]
        except:
            print_error("Could not find the dataset raw path")
            raise Exception("Could not find the dataset raw path")

        #   The "dataset configuration file" is a file used to track which videos were already processed (already have be extracted), and which were not
        try:    #Tries loading the configuration file
            with open(os.path.join(self.__dataset_directory, CONST_STR_DATASET_CONFIG_FILENAME), "rb") as fp:
                dataset_config_file = pickle.load(fp)

            #   Here we see what videos were already processed and which were not
            unprocessed_videos = [item for item in dataset_raw_datapath if item not in dataset_config_file]

            #   ..and we add the unprocessed videos to the dataset_config_file (because they will be processed at the end of the script)
            dataset_config_file += unprocessed_videos

        except:     #   A new dataset, configuration file does not exist
            print_warning("Could not find dataset configuration file. Creating one")
            #   ... there is no config file
            dataset_config_file = dataset_raw_datapath
            #   ... therefore all data is still unprocessed
            unprocessed_videos = dataset_raw_datapath

        #   If there are no unprocessed videos, all videos were already processed an there is nothing else to do
        if not unprocessed_videos:
            print_info("There are no new videos to be processed in the dataset")
            return
        else:
            print_info("The following new videos were added to the dataset")
            for video_name in unprocessed_videos:
                print('\t'+video_name)
                if self.__CheckForSpaces(video_name):
                    print_error("This method is known for giving errors when the raw video name has spaces. Please delete all spaces and try again")
                    raise Exception("Raw video name has spaces")

        #extract frames and sound for each video in train data!
        for video_name in unprocessed_videos:
            #   Datapath of raw video (where the raw video is)
            video_raw_datapath = os.path.join(self.__dataset_directory, CONST_STR_DATASET_BASEPATH_RAW, video_name)
            #   Datapath of the processed information of video

            video_datapath = os.path.join(self.__dataset_directory, video_name.replace(".", ""))
            #   Make directory to hold all extracted information from the video
            try:
                if not self. __CheckIfPathExists(video_datapath):
                    os.makedirs(video_datapath)
            except OSError:
                print_error("Could not make directory for video '"+str(video_name)+"'")
                raise Exception("Could not make directory for video '"+str(video_name)+"'")

            cap = cv2.VideoCapture(video_raw_datapath)

            currentFrame = 0    # This variable counts the frame in the extracted video
            videoFrame = 0      # This variable counts the actual frame in the raw video
                                # I use those 2 variables so that I can change the fps of extraction
                                # by decimation

            total_number_of_video_frames = 0    # Total number of extracted video frames

            print_info("Extracting frames from video "+video_name)

            #   ------------------- Extraction of frames from video

            #   Resize the raw video to the desired resolution
            #   Extact frames from resized video
            #   I stoped using ffmpeg for image extraction after noticing undesireble compression artifacts 
            #   Using cv2.resize() is a much better alternative
            while cap.isOpened():
                # Capture frame-by-frame
                ret, frame = cap.read()
                if videoFrame% self.__downsampling_factor == 0:
                    if ret:
                        # Saves image of the current frame in png file
                        frame_name = os.path.join(video_datapath,str(currentFrame)+'.png')
                        #   Frame resize
                        frame = cv2.resize(frame, (self.__image_format[0], self.__image_format[1]))
                        cv2.imwrite(frame_name, frame)
                        # To stop duplicate images
                        currentFrame += 1
                    else:
                        break

                videoFrame += 1
                total_number_of_video_frames = currentFrame

            # Save the number of frames in this video on the frames clfolder
            with open(os.path.join(video_datapath, CONST_STR_DATASET_NMBOFFRAMES_FILENAME), "wb") as fp:
                pickle.dump(total_number_of_video_frames, fp)

            print_info(str(total_number_of_video_frames)+" frames were extracted from video "+video_name)
            # When everything done, release the capture
            cap.release()

            #   ------------------- Extraction of audio from video
            if not dont_extract_soundtargets:
                if not is_tool("ffmpeg"):
                    print_error("ffmpeg is not installed on your system or is not in the PATH environment variable. Please install ffmpeg for audio operations")
                    raise Exception("ffmpeg is not installed on your system or is not in the PATH environment variable. Please install ffmpeg for audio operations")
                print_info("Extracting audio information from video "+video_name)

                #   Execute command to extract only audio from video
                audio_filepath = os.path.join(video_datapath, CONST_STR_DATASET_AUDIOFILE_FILENAME)
                os_command = "ffmpeg -loglevel quiet -i "+video_raw_datapath+" "+audio_filepath
                os.system(os_command)

                #   Get samples from audio file generated
                print_info("Reading audio file ...")
                FSample, samples = scipy.io.wavfile.read(audio_filepath)
                samples = np.array(samples)
                original_audio = samples

                #   ------------------- Calculate the total power for each frame

                M = math.floor(samples.shape[0]/total_number_of_video_frames)    #Number of Samples used for each frame calculatiom
                St = np.zeros((total_number_of_video_frames, 2))             #Array of audio power in each frame

                print_info("Calculating audio power for each frame ...")

                #   Square and divide all samples by M
                samples = np.square(samples, dtype='int64')
                samples = np.divide(samples, M)

                #   Do the partial sum of everything
                for i in range(0, total_number_of_video_frames):
                    St[i] = np.sum(samples[i*M:(i+1)*M], axis=0)

                #   Clip the zeros to a minor value, and log everything
                St = np.clip(St, 1e-12, None)
                St = np.log(St)

                # save numpy array as .npy file
                np.save(os.path.join(video_datapath, CONST_STR_DATASET_TARGETDATA_FILENAME), St)
            else:
                print_warning("Not extracting audio targets from video "+video_name)

        for video_name in unprocessed_videos:
            print_info("Stacking images frames for video "+video_name)

            first_frame = True

            #   Datapath of raw video (where the raw video is)
            video_raw_datapath = os.path.join(self.__dataset_directory, CONST_STR_DATASET_BASEPATH_RAW, video_name)
            #   Datapath of the processed information of video
            video_datapath = os.path.join(self.__dataset_directory, video_name.replace(".", ""))

            # grab all image paths and order it correctly
            frame_datapaths = list(paths.list_images(video_datapath))
            frame_datapaths.sort(key=lambda f: int(re.sub('\D', '', f)))

            for frame_path in frame_datapaths:
                #   Read image frame
                frame = cv2.imread(frame_path)

                # We then stack all frames on top of each other
                # Image stacking is now what consumes the most time in processing
                if first_frame:
                    stacked_frames_array = np.expand_dims(frame, axis=0)
                    first_frame = False
                else:
                    stacked_frames_array = np.vstack((stacked_frames_array, np.expand_dims(frame, axis=0)))

            #   Save the stacked frames numpy to the corresponding video folder
            print_info("Stacked frames tensor format: "+str(stacked_frames_array.shape)+" of type np.uint8")
            print_info("Saving stacked frames data to "+os.path.join(video_datapath, CONST_STR_DATASET_STACKEDFRAMES_FILENAME))
            np.save(os.path.join(video_datapath, CONST_STR_DATASET_STACKEDFRAMES_FILENAME), stacked_frames_array)

        #Last step: Calculate mean and std for each video in dataset. Save this information in disk
        for video_name in unprocessed_videos:
            print_info("Loading numpy dataset for mean and std calculation")

            #   Load numpy array
            video_datapath = os.path.join(self.__dataset_directory, video_name.replace(".", ""))

            video_data = np.load(os.path.join(video_datapath, CONST_STR_DATASET_STACKEDFRAMES_FILENAME))

            #   Calculate mean and std of video
            mean = np.mean(video_data, axis=(0,1,2)).astype(float)
            std = np.std(video_data, axis=(0,1,2)).astype(float)

            statistics = [mean, std]

            print("mean: "+'\t'+str(statistics[0]))
            print("std: " +'\t'+str(statistics[1]))

            #   Save it to a file
            with open(os.path.join(video_datapath, CONST_STR_DATASET_STATISTICS_FILENAME), "wb") as fp:
                pickle.dump(statistics, fp)

        # Save the information of all videos on file
        with open(os.path.join(self.__dataset_directory, CONST_STR_DATASET_CONFIG_FILENAME), "wb") as fp:
            pickle.dump(dataset_config_file, fp)

        print_info("Dataset extraction process ended successfully")

        pass

    def BuildFoldFromExtractedData(self, train_videos, test_videos):
        #   Load configuration file containing all processed videos on disk

        try:    #Load the config file
            with open(os.path.join(self.__dataset_directory, CONST_STR_DATASET_CONFIG_FILENAME), "rb") as fp:
                processed_videos = pickle.load(fp)

        except:     #   new dataset, configuration file does not exist
            print_error("There is no configuration file on system. You should extract the data from the videos by running ExtractData before atempting to build a fold")
            raise Exception("There is no configuration file on system. You should extract the data from the videos by running ExtractData before atempting to build a fold")

        #   Check if i'm passing a video that is not processed in the dataset
        unknown_videos = [item for item in train_videos if item not in processed_videos]
        if unknown_videos != []:
            print_error("The following videos were not processed in the dataset:")
            for unknown_video in unknown_videos:
                print('\t'+unknown_video)
            print_error("You are trying to build a fold using a video that has not been extracted yet. You should extract the data from all videos by running ExtractData before atempting to build a fold")
            raise Exception("You are trying to build a fold using a video that has not been extracted yet. You should extract the data from all videos by running ExtractData before atempting to build a fold")

        unknown_videos = [item for item in test_videos if item not in processed_videos]
        if unknown_videos != []:
            print_error("The following videos were not processed in the dataset:")
            for unknown_video in unknown_videos:
                print('\t'+unknown_video)
            print_error("You are trying to build a fold using a video that has not been extracted yet. You should extract the data from all videos by running ExtractData before atempting to build a fold")
            raise Exception("You are trying to build a fold using a video that has not been extracted yet. You should extract the data from all videos by running ExtractData before atempting to build a fold")


        #   Check if a video is beeing used for both test and train dataset
        common_videos = [item for item in test_videos if item in train_videos]
        if common_videos != []:
            print_error("The following videos are being used for both train and evaluation datasets. You should not train and evaluate on same videos!")

            for common_video in common_videos:
                print('\t'+common_video)
            print_error("Please reevaluate your data")
            raise Exception("The following videos are being used for both train and evaluation dataset. You should not train and evaluate on same videos!")

        #   --------------  Build starting
        #   This Loop:
        #
        #       get all number of frames:   train_number_of_frames
        #       get all means and std:      train_mean, train_std
        #       get all frames for dataset: input_train_data
        #       get all outputs on dataset: output_train_data
        #
        #       Test and Train codes are the same, just duplicated
        #
        #   --------------   Train dataset
        #

        print_info("Starting building process for train dataset")

        train_number_of_frames = []     #   Vector with the total number of frames in each video. This is necessary to calculate a number of things such as total mean, std, data loading, etc...
        first_video = True
        for video_name in train_videos:

            video_datapath = os.path.join(self.__dataset_directory ,video_name.replace(".", ""))

            #   Get total number of frames in video
            with open(os.path.join(video_datapath, CONST_STR_DATASET_NMBOFFRAMES_FILENAME), "rb") as fp:
                number_of_frames = pickle.load(fp)
            train_number_of_frames.append(number_of_frames)

            #   Get the statistics in the video
            #   Put those in the train_mean and train_std ndarrays
            with open(os.path.join(video_datapath, CONST_STR_DATASET_STATISTICS_FILENAME), "rb") as fp:
                video_statistics = pickle.load(fp)
            
            video_statistics = np.asarray(video_statistics)
            if first_video:
                train_mean = video_statistics[0]
                train_std = video_statistics[1]
            else:
                train_mean = np.vstack((train_mean, video_statistics[0]))
                train_std = np.vstack((train_std, video_statistics[1]))
       
            #   Load numpy video data. 
            video_data = np.load(os.path.join(video_datapath, CONST_STR_DATASET_STACKEDFRAMES_FILENAME))

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                input_train_data = video_data
            else:
                input_train_data = np.vstack((input_train_data, video_data))

            #   Load numpy output data. 
            audio_data = np.load(os.path.join(video_datapath, CONST_STR_DATASET_TARGETDATA_FILENAME))

            # If audio file is Stereo, I take the mean between both chanels and concatenate in one channel
            try:
                if audio_data.shape[1] == 2:
                    print_warning("Audio file from video "+video_name+" is stereo. Taking the average of both channels as output")
                    audio_data = np.mean(audio_data, axis=1)
            except:
                #   Nothing to do here
                #   is this try: necessary?
                pass

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                output_train_data = audio_data
            else:
                output_train_data = np.concatenate((output_train_data, audio_data))

            #   Last of all, switch of the "first video" variable
            if first_video:
                first_video = False

        #   Calculate mean and std for all dataset:
        #   Dataset preprossesing. We normalize the dataset by subtracting mean and dividing by std

        train_combined_mean, train_combined_std = self.__CalculateFoldStatistics(train_mean, train_std, train_number_of_frames)

        input_train_data = self.__NormalizeFold(input_train_data, train_combined_mean, train_combined_std)

        # For some reason, axis 3 (colour) is fliped
        input_train_data = np.flip(input_train_data, axis=3)

        #
        #   --------------   Test dataset
        #

        print_info("Starting building process for test dataset")

        test_number_of_frames = []     #   Vector with the total number of frames in each video. This is necessary to calculate a number of things such as total mean, std, data loading, etc...
        first_video = True
        for video_name in test_videos:

            video_datapath = os.path.join(self.__dataset_directory, video_name.replace(".", ""))

            #   Get total number of frames in video
            with open(os.path.join(video_datapath, CONST_STR_DATASET_NMBOFFRAMES_FILENAME), "rb") as fp:
                number_of_frames = pickle.load(fp)
            test_number_of_frames.append(number_of_frames)

            #   Get the statistics in the video
            #   Put those in the train_mean and train_std ndarrays
            with open(os.path.join(video_datapath, CONST_STR_DATASET_STATISTICS_FILENAME), "rb") as fp:
                video_statistics = pickle.load(fp)
            
            video_statistics = np.asarray(video_statistics)
            if first_video:
                test_mean = video_statistics[0]
                test_std = video_statistics[1]
            else:
                test_mean = np.vstack((test_mean, video_statistics[0]))
                test_std = np.vstack((test_std, video_statistics[1]))
       
            #   Load numpy video data. 
            video_data = np.load(os.path.join(video_datapath, CONST_STR_DATASET_STACKEDFRAMES_FILENAME))

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                input_test_data = video_data
            else:
                input_test_data = np.vstack((input_test_data, video_data))

            #   Load numpy output data.
            audio_data = np.load(os.path.join(video_datapath, CONST_STR_DATASET_TARGETDATA_FILENAME))

            # If audio file is Stereo, I take the mean between both chanels and concatenate in one channel
            try:
                if audio_data.shape[1] == 2:
                    print_warning("Audio file from video "+video_name+" is stereo. Taking the average of both channels as output")
                    audio_data = np.mean(audio_data, axis=1)
            except:
                pass

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                output_test_data = audio_data
            else:
                output_test_data = np.concatenate((output_test_data, audio_data))

            #   Last of all, switch of the "first video" variable
            if first_video:
                first_video = False

        #   Calculate mean and std for all dataset:
        #   Dataset preprossesing. We normalize the dataset by subtracting mean and dividing by std
        #
        test_combined_mean, test_combined_std = self.__CalculateFoldStatistics(test_mean, test_std, test_number_of_frames)

        input_test_data = self.__NormalizeFold(input_test_data, test_combined_mean, test_combined_std)

        # For some reason, axis 3 (colour) is fliped
        #input_test_data = np.flip(input_test_data, axis=3)

        #   Return the builded dataset and the vectors with train and test video sizes
        
        print_info("Fold built is complete")
        
        return input_train_data, output_train_data, input_test_data, output_test_data, train_number_of_frames, test_number_of_frames

    def __CheckIfPathExists(self, path):
        """Private function that checks if the given path exists"""
        return os.path.exists(path)
    
    def __CheckForSpaces(self, string):
        return " " in string

    def __CalculateFoldStatistics(self, data_mean, data_std, data_size):
        """
            This function is used to calculate mean and standart deviantion of the dataset
            without loading the full dataset into memory, as it would be loaded astype float64


            The way used to calculate mean and std was:
                1 - instead of calculating mean and std for all videos in the dataset, 
                calculate mean and std for each video in the dataset

                2 - After this, calculate the joined mean and std of all videos, by
                    2.1 -  combined_mean = weighted average of all means
                    2.2 -  combined_std using the formula on https://www.statstodo.com/CombineMeansSDs_Pgm.php
        """
        #   If there is only on video, there is no meaning in combining multiple statistics
        if len(data_size) == 1:
            return data_mean, data_std

        combined_mean = np.average(data_mean, axis=(0), weights=data_size)

        tn = np.sum(data_size)

        tx = np.array(data_mean)
        for i in range(data_mean.shape[0]):
            tx[i] = tx[i] * data_size[i]
        tx = np.sum(tx, axis=0)

        txx = np.square(data_std)
        for i in range(data_std.shape[0]):
            txx[i] = txx[i] * data_size[i]-1
            A = (np.square(data_mean)[i] * data_size[i])
            txx[i] = txx[i] + A
        txx = np.sum(txx, axis=0)

        combined_std = np.sqrt((txx - (np.square(tx) / tn))/(tn))

        return combined_mean, combined_std

    def __NormalizeFold(self, data, mean, std):
        """
            Reshape dataset from (n,224*224*3) to (n,224,224,3)
            Normalize dataset by subtracting mean and dividing by std
        """
        #data = np.reshape(data, (data.shape[0],)+self.__image_format).astype("float32")

        data = data.astype("float32")

        data[:, :, :, 0] -= (mean[0])
        data[:, :, :, 1] -= (mean[1])
        data[:, :, :, 2] -= (mean[2])

        data[:, :, :, 0] /= (std[0])
        data[:, :, :, 1] /= (std[1])
        data[:, :, :, 2] /= (std[2])

        return data