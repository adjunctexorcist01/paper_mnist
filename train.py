from keras.utils import np_utils
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import os
import numpy as np
from keras.callbacks import ModelCheckpoint

#Training params
batchsize = 128
epochs = 20

#Loading the Dataset
(X_train, y_train), (X_test, y_test)  = mnist.load_data()

#Converting into float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

#Normalize our data
X_train /= 255
X_test /= 255

#Reshaping the data
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
input_shape =(28,28,1)

#Onehot Encoding the outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
checkpoint = ModelCheckpoint(os.getcwd()+"/models/"+"emnist_digits_20ep.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(y_test.shape[1], activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train,
          batch_size=batchsize,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[checkpoint])
# history = model.fit(X_train, y_train,
#           batch_size=batchsize,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(X_test, y_test),
#           callbacks=[checkpoint])

# model.save("emnist_letters_20ep.h5")