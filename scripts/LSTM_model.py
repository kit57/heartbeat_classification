
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
from sklearn.model_selection import train_test_split

from utils.load_data import load_dataset_from_folders
from utils.evaluate import calc_accuracy, test_model_unlabelled, test_model_testdata
from keras.callbacks import ModelCheckpoint,Callback
import os, random

CLASSES = ['artifact','murmur','normal']


class LSTM_hb_model():

    '''

    Trains a LSTM model with the heartbeat sounds of the dataset.
    Saves the model and the model's losses in ./models folder

    '''

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

        return model

class LossHistory(Callback):

    '''

     Customize an History class that save losses to a file for each epoch

    '''

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


if __name__ == "__main__":

    random.seed(0)
    epochs = 100
    batch_size = 54
    hidden_size = 64

    x_data, y_data, test_x, test_y = load_dataset_from_folders() # This step might take some time

    # train_test_split train data
    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.20, random_state=0)

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
    y_val = np.array(keras.utils.to_categorical(y_val, len(CLASSES)))
    test_y = np.array(keras.utils.to_categorical(test_y, len(CLASSES)))

    model = LSTM_hb_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epoch=epochs,
                         batch_size=batch_size, hidden_size=hidden_size)

    model = model.model_lstm() # Trains LSTM model

    calc_accuracy(model=model, X_val=X_val, y_val=y_val)

    print()
    print('Training is finished')

    print()
    print('Predicting labels of test data')
    test_model_testdata(model=model, test_x=test_x)

    print()
    print('Predicting labels from new sound files')
    test_model_unlabelled(path_model='../models/LSTMbeatclassification.h5', folder_unlabelled='../data/test/')

    print()
    print('Done :)')
