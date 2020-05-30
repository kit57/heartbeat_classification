
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,  LSTM
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,Callback
import itertools, os

CLASSES = ['artifact','murmur','normal']



class LSTM_hb_model():

    def __init__(self, X_train, y_train, X_val, y_val, epoch, batch_size, hidden_size):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def model_lstm(self):
        '''Define the LSTM network'''


        model = Sequential()
        model.add(LSTM(self.hidden_size, return_sequences=True, input_shape=(40, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(len(CLASSES), activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc', 'mse', 'mae'])
        model.summary()

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        history = LossHistory()
        weights = ModelCheckpoint(filepath='../models/LSTMbeatclassification.h5')

        model.fit(self.X_train, self.y_train,
                    batch_size=self.batch_size,
                    epochs=self.epoch,
                    verbose=0,
                    validation_data=(self.X_val, self.y_val),
                    callbacks=[early_stopping, weights, history])


# customize an History class that save losses to a file for each epoch
class LossHistory(Callback):

    def on_train_begin(self, logs=None):

        if os.path.exists('../models/losses/LSTMloss.npz'):
            self.loss_array = np.load('../models/losses/LSTMloss.npz')['loss']
        else:
            self.loss_array = np.empty([2, 0])

    def on_epoch_end(self, epoch, logs=None):
        # append new losses to loss_array
        loss_train = logs.get('loss')
        loss_test = logs.get('val_loss')

        loss_new = np.array([[loss_train], [loss_test]])  # 2 x 1 array
        self.loss_array = np.concatenate((self.loss_array, loss_new), axis=1)

        # save model to disk
        np.savez_compressed('../models/losses/LSTMloss.npz', loss=self.loss_array)


