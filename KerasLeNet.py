import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Conv2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt 

images = []
measurements = []

# myTrainingData is uploaded from my local machine
# so need to change the path
def get_training_data_label_pairs(directory):
    doFlip = False
    if directory == "trainingDataRecovery":
        print("Doing flip for: ", directory)
        doFlip = False
    lines = []
    log_file_path = directory + '/driving_log.csv'
    with open(log_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        path = line[0] # only centre image for now
        if (len(path) < 8):
            continue
        fileName = path.split('/')[-1]
        imagePath = directory + '/IMG/'
        currentPath = imagePath + fileName
        image = ndimage.imread(currentPath)
        measurement = float(line[3])
        images.append(image)
        measurements.append(measurement)
        if doFlip == True:
            flipped_image = np.fliplr(image)
            measurement = -1 * measurement
            images.append(image)
            measurements.append(measurement)
            

# get training data
get_training_data_label_pairs('behaviorCloningSampleData')
get_training_data_label_pairs('myTrainingData')
get_training_data_label_pairs('trainingData2Lap')
get_training_data_label_pairs('trainingDataClockWise')
get_training_data_label_pairs('toughTrackTrainingData')
get_training_data_label_pairs('trainingDataRecovery')
get_training_data_label_pairs('trainingDataRecovery')

print(len(images))
print(len(measurements))
rangemax = print(max(measurements))
rangemin = print(min(measurements))
X_train = np.array(images)
Y_train = np.array(measurements)

width = 0.1
angels = np.arange(-1, 1, width)
#hist = np.histogram(Y_train, bins=angels)
plt.hist(Y_train, bins=angels)
plt.show()

# Build a basic network to see if evrything works
# Disable it as we are trying to get data stats
# input_shape = [160,320,3]
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
# model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=input_shape))
# model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.30))
# model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.30))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dropout(0.30))
# model.add(Dense(84))
# model.add(Dense(1))

# model.compile(loss='mse', optimizer = 'adam')
# model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

# model.save('model.h5')

