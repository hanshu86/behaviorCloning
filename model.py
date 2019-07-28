import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Activation
from keras.layers import Conv2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D

images = []
measurements = []

# myTrainingData is uploaded from my local machine
# so need to change the path
def get_training_data_label_pairs(directory):
    doFlip = False
    if directory == "trainingDataRecovery":
        print("Doing flip for: ", directory)
        # Lets not flip image as it is not improving the model for some reason
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
# Use it again as this data is proving to be critical
get_training_data_label_pairs('trainingDataRecovery')

X_train = np.array(images)
Y_train = np.array(measurements)

# Build a Nvidia Network
input_shape = [160,320,3]
model = Sequential()
model.add(Lambda(lambda x: ((x  - 128.0)/ 128.0) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=input_shape))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), padding='Valid', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), padding='Valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2,2), padding='Valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='Valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='Valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.30))
model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.30))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

# Use default learning rate of 0.001 for adam optimizer
model.compile(loss='mse', optimizer = 'adam')
# Use default batch size of 32. Use 2 ecpochs
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, epochs = 2)
# save the model
model.save('model.h5')

