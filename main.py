import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
# myTrainingData is uploaded from my local machine
# so need to change the path
with open('myTrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    path = line[0] # only centre image for now
    fileName = path.split('/')[-1]
    currentPath = 'myTrainingData/IMG/'+ fileName
    image = ndimage.imread(currentPath)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)
    
X_train = np.array(images)
Y_train = np.array(measurements)

# Build a basic network to see if evrything works
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)

model.save('model.h5')

