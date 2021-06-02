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
    Implements the code for scraping data to be used for training and evaluation of sound pressure inference models using images
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

    def ExtractImages(self, downsampling_factor, image_format, video = None, dont_extract_soundtargets = False):
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

        # Save the information of all videos on file
        with open(os.path.join(self.__dataset_directory, CONST_STR_DATASET_CONFIG_FILENAME), "wb") as fp:
            pickle.dump(dataset_config_file, fp)

        print_info("Dataset extraction process ended successfully")

        pass

    def __CheckIfPathExists(self, path):
        """Private function that checks if the given path exists"""
        return os.path.exists(path)
    
    def __CheckForSpaces(self, string):
        return " " in string