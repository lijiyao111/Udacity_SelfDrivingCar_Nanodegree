import os
import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle


def getData(center=True, left=True, right=True, correction=0.2):

    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    data_samples = []

    for i, line in enumerate(samples):
        # Skip the header line
        if i == 0:
            continue

        # Read from three cameras   
        if center == True:
            imgName = 'data/IMG/'+line[0].split('/')[-1]
            angle = float(line[3]) 
            data_samples.append([imgName, angle])
        if left == True:
            imgName = 'data/IMG/'+line[1].split('/')[-1]
            angle = float(line[3]) + correction
            data_samples.append([imgName, angle])
        if right == True:
            imgName = 'data/IMG/'+line[2].split('/')[-1]
            angle = float(line[3]) - correction
            data_samples.append([imgName, angle])    

    return data_samples



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = batch_sample[0]
                img = mpimg.imread(image_name)
                angle = float(batch_sample[1])

                images.append(img)
                angles.append(angle)
                # flip image
                images.append(cv2.flip(img,1))
                angles.append(angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = Sequential()
    # Normalize image
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop image to only see road
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    # Added Dropout layer
    model.add(Dropout(0.40))
    model.add(Dense(50))
    # Added Dropout layer
    model.add(Dropout(0.40))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

samples = getData(center=True, left=True, right=True, correction=0.2)

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Model creation
model = nVidiaModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')

trainLoss = history_object.history['loss']
valLoss = history_object.history['val_loss']

print(history_object.history.keys())
print('Loss')
print(trainLoss)
print('Validation Loss')
print(valLoss)

plt.figure()
plt.plot(range(1, len(trainLoss)+1), trainLoss, label='Train loss' )
plt.plot(range(1, len(trainLoss)+1), valLoss, label='Validation loss' )
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean square error')
plt.grid()
plt.savefig('loss.png')

