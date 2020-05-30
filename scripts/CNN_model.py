

import keras, os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np


class CNN_hb_model():

    def __init__(self, X_train, y_train, X_val, y_val, epoch, batch_size, hidden_size):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_size = hidden_size


    def model_lstm(self):
        '''Define the CNN network'''

        model = Sequential()
        model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(193, 1)))

        model.add(Conv1D(128, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(2))

        model.add(Conv1D(256, kernel_size=5, activation='relu'))

        model.add(Dropout(0.3))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(6, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        history = LossHistory()
        weights = ModelCheckpoint(filepath='../models/CNNbeatclassification.h5')

        model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=70,
                            batch_size=200,
                            verbose=1,
                            callbacks=[early_stopping, weights, history])




# customize an History class that save losses to a file for each epoch
class LossHistory(Callback):

    def on_train_begin(self, logs=None):

        if os.path.exists('../models/losses/CNNloss.npz'):
            self.loss_array = np.load('../models/losses/CNNloss.npz')['loss']
        else:
            self.loss_array = np.empty([2, 0])

    def on_epoch_end(self, epoch, logs=None):
        # append new losses to loss_array
        loss_train = logs.get('loss')
        loss_test = logs.get('val_loss')

        loss_new = np.array([[loss_train], [loss_test]])  # 2 x 1 array
        self.loss_array = np.concatenate((self.loss_array, loss_new), axis=1)

        # save model to disk
        np.savez_compressed('../models/losses/CNNloss.npz', loss=self.loss_array)


