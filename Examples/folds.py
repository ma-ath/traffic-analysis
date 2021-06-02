"""
    This file contains a
    Python dictionaries with all training and testing folds
    Each entry has the name of training and testing videos
"""
number_of_folds = 2
folds = [dict() for i in range(number_of_folds)]

for i in range(number_of_folds):
    folds[i]["name"] = "fold_"+str(i)
    folds[i]["number"] = i

folds[0]["training_videos"] = [
    "road.mp4",
    "road2.mp4",
    "road3.mp4"
]

folds[0]["testing_videos"] = [
    "road4.mp4",
    "road5.mp4",
    "road6.mp4"
]

folds[1]["training_videos"] = [
    "road2.mp4",
    "road3.mp4",
    "road4.mp4"
]

folds[1]["testing_videos"] = [
    "road5.mp4",
    "road6.mp4",
    "road.mp4"
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