

import keras, os, random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ['artifact', 'murmur', 'normal', 'extrastole', 'extrahls']

label_to_int = {k: v for v, k in enumerate(CLASSES)}
int_to_label = {v: k for k, v in label_to_int.items()}

class CNN_hb_model():

    '''

    Trains a CNN model using images extracted from audio files by running ../beatsoundtoimage.py
    Saves the model and the model's losses in ./models folder.

    '''

    def __init__(self, train_it, val_it, epochs, hidden_size):

        self.train_it=train_it
        self.val_it = val_it
        self.epochs = epochs
        self.hidden_size = hidden_size


    def model_cnn(self):

        '''

        Define the CNN network

        '''

        model = Sequential()
        model.add(Conv2D(self.hidden_size, kernel_size=2, activation='relu',
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))  #193, 1

        model.add(Conv2D(128, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, kernel_size=2, activation='relu'))

        model.add(Dropout(0.3))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(5, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        history = LossHistory()
        weights = ModelCheckpoint(filepath='../models/CNNbeatclassification.h5')

        total_train = 500
        total_val = 85

        model.fit_generator(self.train_it,
                            steps_per_epoch=total_train//batch_size,
                            epochs=self.epochs,
                            validation_data=self.val_it,
                            validation_steps=total_val//batch_size,
                            verbose=1,
                            callbacks=[early_stopping, weights, history])

        return model

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


# Loading dataset
def load_datasets(path_train, path_val):
    # create a data generator
    datagen = ImageDataGenerator(rescale=1. / 255)

    # load and iterate training dataset
    train_it = datagen.flow_from_directory(path_train, batch_size=batch_size, class_mode='binary',
                                           shuffle=True,
                                           target_size=(IMG_HEIGHT, IMG_WIDTH))
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(path_val, batch_size=batch_size, class_mode='binary',
                                         target_size=(IMG_HEIGHT, IMG_WIDTH))

    print()
    # confirm the iterator works
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    return train_it, val_it

if __name__ == '__main__':

    random.seed(0)
    epochs = 60
    batch_size = 54
    hidden_size = 64
    IMG_HEIGHT = 10
    IMG_WIDTH = 40

    print('Collecting the data...')

    train_it, val_it = load_datasets('../images/train/', '../images/val/')

    print()
    print('Starting training...')

    model = CNN_hb_model(train_it=train_it, val_it=val_it, epochs=epochs,
                         hidden_size=hidden_size)

    model = model.model_cnn() # Train CNN model with images

    print()
    print('Training is finished')


    print()
    print('Loading model to run a test')
    loaded_model = load_model('../models/CNNbeatclassification.h5', compile=True)

    test_loss, test_acc = loaded_model.evaluate(val_it, verbose=5)
    y_pred = model.predict_generator(val_it)
    y_pred = np.argmax(y_pred, axis=1)
    print('\nValidation accuracy: ', test_acc)
    print('Confusion matrix: ', confusion_matrix(val_it.classes, y_pred))
    print(classification_report(val_it.classes, y_pred,
                                target_names=CLASSES))

    print()
    print('Predict on new data from test folder')

    image_path = r'../images/test/'

    for filename in os.listdir(image_path):
        filename = os.path.join(image_path, filename)
        image = keras.preprocessing.image.load_img(filename, color_mode='rgb', target_size=(10, 40), interpolation='nearest')
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = model.predict_classes(input_arr)
        print('Prediction on test folder return :', int_to_label[int(predictions)])
