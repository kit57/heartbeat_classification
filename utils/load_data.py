
import numpy as np
import os, fnmatch, random
import librosa

SAMPLE_RATE = 16000
MAX_SOUND_CLIP_DURATION = 12 # seconds

random.seed(0)

def audio_norm(data):

    '''
    Audio normalization
    '''

    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

def load_file_data_without_change(folder,file_names, duration=3, sr=16000):

    '''
    Get audio data without padding highest qualify audio. Best option for training if you want high quality.
    It needs more computational power on the other hand.

    :param folder: path to folder where we will find files for our dataset
    :param file_names: you can apply a filter to extract certain files.
    Eg: fnmatch.filter(os.listdir('../data/set_a'), 'artifact*.wav')
    :param duration: int for the seconds
    :param sr: target sampling rate
    :return: data for our dataset
    '''

    input_length=sr*duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file=folder+file_name
            print ("load file ",sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load( sound_file, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)
    return data


def load_file_data (folder, file_names, duration=12, sr=16000):

    '''

    Loads files from folder. Get audio data with a fix padding may also chop off some file.

    :param folder: path to folder where we will find files for our dataset
    :param file_names: you can apply a filter to extract certain files.
    Eg: fnmatch.filter(os.listdir('../data/set_a'), 'artifact*.wav')
    :param duration: int for the seconds
    :param sr: target sampling rate
    :return: data for our dataset

    '''

    input_length = sr*duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file=folder+file_name
            print ("load file ",sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load( sound_file, sr=sr, duration=duration,res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
            if (round(dur) < duration):
                print ("fixing audio lenght :", file_name)
                y = librosa.util.fix_length(X, input_length)
            #normalized raw audio
            # y = audio_norm(y)
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)
    return data


def load_dataset_from_folders():

    '''

    Loads set_a and set_b and joins then into a dataset for our training

    '''

    # load dataset-a, keep them separate for testing purpose

    INPUT_DIR = "../data"
    A_folder = INPUT_DIR + '/set_a/'
    # set-a
    A_artifact_files = fnmatch.filter(os.listdir('../data/set_a'), 'artifact*.wav')
    A_artifact_sounds = load_file_data(folder=A_folder, file_names=A_artifact_files, duration=MAX_SOUND_CLIP_DURATION)
    A_artifact_labels = [0 for items in A_artifact_files]

    A_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'normal*.wav')
    A_normal_sounds = load_file_data(folder=A_folder, file_names=A_normal_files, duration=MAX_SOUND_CLIP_DURATION)
    A_normal_labels = [2 for items in A_normal_sounds]

    A_extrahls_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'extrahls*.wav')
    A_extrahls_sounds = load_file_data(folder=A_folder, file_names=A_extrahls_files, duration=MAX_SOUND_CLIP_DURATION)
    A_extrahls_labels = [1 for items in A_extrahls_sounds]

    A_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'murmur*.wav')
    A_murmur_sounds = load_file_data(folder=A_folder, file_names=A_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
    A_murmur_labels = [1 for items in A_murmur_files]

    # test files
    A_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'Aunlabelledtest*.wav')
    A_unlabelledtest_sounds = load_file_data(folder=A_folder, file_names=A_unlabelledtest_files,
                                             duration=MAX_SOUND_CLIP_DURATION)
    A_unlabelledtest_labels = [-1 for items in A_unlabelledtest_sounds]

    # load dataset-b, keep them separate for testing purpose
    B_folder = INPUT_DIR + '/set_b/'
    # set-b
    B_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'normal*.wav')  # include noisy files
    B_normal_sounds = load_file_data(folder=B_folder, file_names=B_normal_files, duration=MAX_SOUND_CLIP_DURATION)
    B_normal_labels = [2 for items in B_normal_sounds]

    B_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'murmur*.wav')  # include noisy files
    B_murmur_sounds = load_file_data(folder=B_folder, file_names=B_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
    B_murmur_labels = [1 for items in B_murmur_files]

    B_extrastole_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'extrastole*.wav')
    B_extrastole_sounds = load_file_data(folder=B_folder, file_names=B_extrastole_files,
                                         duration=MAX_SOUND_CLIP_DURATION)
    B_extrastole_labels = [1 for items in B_extrastole_files]

    # test files
    B_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'Bunlabelledtest*.wav')
    B_unlabelledtest_sounds = load_file_data(folder=B_folder, file_names=B_unlabelledtest_files,
                                             duration=MAX_SOUND_CLIP_DURATION)
    B_unlabelledtest_labels = [-1 for items in B_unlabelledtest_sounds]

    # Concatenate train and test data
    x_data = np.concatenate((A_artifact_sounds, A_normal_sounds, A_extrahls_sounds, A_murmur_sounds,
                             B_normal_sounds, B_murmur_sounds, B_extrastole_sounds))

    y_data = np.concatenate((A_artifact_labels, A_normal_labels, A_extrahls_labels, A_murmur_labels,
                             B_normal_labels, B_murmur_labels, B_extrastole_labels))

    test_x = np.concatenate((A_unlabelledtest_sounds, B_unlabelledtest_sounds))
    test_y = np.concatenate((A_unlabelledtest_labels, B_unlabelledtest_labels))

    print("Training data records: ", len(x_data))
    print("Testing data records: ", len(test_x))
    print("Total records: ", len(x_data) + len(test_x))

    return x_data, y_data, test_x, test_y

