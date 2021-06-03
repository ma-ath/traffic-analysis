"""
    This file contains a
    Python dictionaries with all training and testing folds
    Each entry has the name of training and testing videos
"""
number_of_folds = 3
folds = [dict() for i in range(number_of_folds)]

for i in range(number_of_folds):
    folds[i]["name"] = "fold_"+str(i)
    folds[i]["number"] = i

# Fold 0
folds[0]["training_videos"] = [
    "BuzzingMalaysianRoadTraffic.mp4",
    "Objectdetectionusingdeeplearningdatasetcctvroadvideo.mp4"
]
folds[0]["testing_videos"] = [
    "Relaxinghighwaytraffic.mp4"
]

# Fold 1
folds[1]["training_videos"] = [
    "BuzzingMalaysianRoadTraffic.mp4",
    "Relaxinghighwaytraffic.mp4"

]
folds[1]["testing_videos"] = [
    "Objectdetectionusingdeeplearningdatasetcctvroadvideo.mp4"
]

# Fold 2
folds[2]["training_videos"] = [
    "Relaxinghighwaytraffic.mp4",
    "Objectdetectionusingdeeplearningdatasetcctvroadvideo.mp4"
]
folds[2]["testing_videos"] = [
    "BuzzingMalaysianRoadTraffic.mp4"
]

if __name__ == "__main__":
    print("The following new videos were added to the dataset")
    for fold in folds:
        print(fold['name'])
        print('training_videos')
        for video in fold['training_videos']:
            print('\t'+video)
        print('testing_videos')
        for video in fold['testing_videos']:
            print('\t'+video)