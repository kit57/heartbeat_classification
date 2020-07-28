
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random, librosa, fnmatch, os, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.load_data import load_file_data

# We are going to use just 3 classes for this project
#CLASSES = ['artifact', 'murmur', 'normal']
CLASSES = ['artifact','murmur','normal', 'extrastole', 'extrahls']

label_to_int = {k: v for v, k in enumerate(CLASSES)}
int_to_label = {v: k for k, v in label_to_int.items()}

class RandomForest_model():

    def __init__(self, X_train, y_train, X_val, y_val, max_depth, max_features, min_samples_split, n_estimators):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators


    def RF_model(self):

        # Instantiate and train the model on training data
        rf = RandomForestClassifier(max_depth=self.max_depth,
                                   max_features=self.max_features,
                                   min_samples_split=self.min_samples_split,
                                   n_estimators=self.n_estimators,
                                   n_jobs=-1,
                                   verbose=1).fit(self.X_train, self.y_train)

        return rf

def ses_df(audio_folders, columns, class_list):

    '''
    Computes rolloff, chroma and centroids of audios using librosa library.

    :param audio_folders: list of paths where to find audios which we are going to use for training
    :param columns: list of column names we are going to use to create the dataframe
    :param class_list: list of the classes in the training dataset
    :return: pandas dataframe containing the dataset
    '''

    lista=[]
    addim=0

    for folder in audio_folders:
        for CLASS in class_list:
            folders = fnmatch.filter(os.listdir(folder),CLASS)
            label = CLASS.split("*")[0]

            for file in folders:
                x, sr = librosa.load(folder+file, duration=5, res_type='kaiser_fast')
                lista.append([np.mean(x) for x in librosa.feature.mfcc(x,sr=sr)])
                lista[addim].append(sum(librosa.zero_crossings(x)))
                lista[addim].append(np.mean(librosa.feature.spectral_centroid(x)))
                lista[addim].append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))
                lista[addim].append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))
                lista[addim].append(label)
                lista[addim].append(file)
                addim += 1

    return pd.DataFrame(lista, columns=columns)


if __name__ == '__main__':

    print('Starting process...')
    random.seed(0)
    max_depth = 8
    max_features = 5
    min_samples_split = 5
    n_estimators = 500

    #  This might take a while
    audio_folders = ["../data/set_a/", "../data/set_b/"]
    columns = ["mfkk" + str(i) for i in range(20)]
    for isim in ["zero", "centroid", "rolloff", "chroma", "class", "file"]:
        columns.append(isim)

    class_list = ["normal*.wav", "artifact*.wav", "murmur*.wav", "extrahls*.wav", "extrastole*.wav"]
    dfheartbeat = ses_df(audio_folders, columns, class_list)
    #print(dfmusic.head())

    # Defines X and y from dataframe
    X = dfheartbeat.iloc[:, 0:24]
    y = dfheartbeat["class"] # Label is column class

    # LabelEncoder for classes
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    # train_test_split train data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)

    model = RandomForest_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, max_depth=max_depth,
                               max_features=max_features, min_samples_split=min_samples_split, n_estimators=n_estimators)

    print('Starting training')

    print(y_val)

    model = model.RF_model()  # Trains LSTM model

    print(model)

    decision_tree_model_pkl = open('../models/RadomForestClassifier.pkl', 'wb')
    pickle.dump(model, decision_tree_model_pkl)
    decision_tree_model_pkl.close()

    print()
    print('Training is finished')
    print()

    file_model = open('../models/RadomForestClassifier.pkl', 'rb')
    model = pickle.load(file_model)
    print('Predicting labels of test data')
    y_pred = model.predict(X_val)
    print(y_pred)

    print()
    print('Model\'s accuracy')
    print(accuracy_score(y_true=y_val, y_pred=y_pred))


    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_val, y_pred)
    print(conf_matrix)

    print("Classification Report:", )
    class_report = classification_report(y_val, y_pred)
    print(class_report)


    print()
    print('Predicting labels from new sound files')

    #  This again might take a while. Be patient...
    audio_folders = ["../data/test/"]
    columns = ["mfkk" + str(i) for i in range(20)]
    for isim in ["zero", "centroid", "rolloff", "chroma", "class", "file"]:
        columns.append(isim)

    class_list = ["Bunlabelledtest*.wav"]
    dfheartbeat = ses_df(audio_folders, columns, class_list)

    # Example predict on test data
    y_pred_test = model.predict(dfheartbeat.iloc[:, 0:24])
    for pred in y_pred_test:
        print("prediction test return :", pred, "-", int_to_label[pred])

    print('Process is finished')
