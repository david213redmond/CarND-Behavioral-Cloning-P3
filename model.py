import csv
import numpy as np
import random as rd
from PIL import Image
import tensorflow as tf

# params
data_dir = "./data/run_sample/"

images = []
measurements = []
from skimage import io

log_path = data_dir+"driving_log.csv"
num_lines = sum(1 for line in open(log_path))

lines = []
with open(log_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # form adjusted stirring record.
        new_line = [
            line[0],
            line[1],
            line[2],
            str(steering_center),
            str(steering_left),
            str(steering_right),
            line[6]
        ]
        lines.append(new_line)

from skimage import io
import scipy.misc
images = []
measurements = []
## data augmentation.
for line in lines:
    throttle = line[6]
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = data_dir + "IMG/" + filename
        image = io.imread(current_path)
        measurement = float(line[i+3])
        # original image
        images.append(image)
        measurements.append(measurement)
        # flipped image
        flipped_image = np.fliplr(image)
        images.append(flipped_image)
        measurements.append(-measurement)

X_train = np.asarray(images)
y_train = np.asarray(measurements)
print("training data shape:", X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.backend import set_value, get_value
from keras.optimizers import Adam

model = Sequential()
# preprocessing.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # normalization
model.add(Cropping2D(cropping=((70,25), (0,0))))
# modeling.
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
# train model.
adam = Adam(lr=0.00025)
model.compile(loss="mse", optimizer=adam)
model.fit(
    X_train, 
    y_train, 
    validation_split=0.2,
    shuffle=True,
    nb_epoch=5
)
# save model.
model.save('model.h5')
