
# Imports the necessary dependencies
import numpy as np
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib as plt
import seaborn as sns
import fnmatch, os
from keras.models import load_model

from utils.load_data import load_file_data

# We are going to use just 3 classes for this project
CLASSES = ['artifact', 'murmur', 'normal']

label_to_int = {k: v for v, k in enumerate(CLASSES)}
int_to_label = {v: k for k, v in label_to_int.items()}


def calc_accuracy(model, X_val, y_val):

    '''
    Obtains model's accuracy and saves plot as an image

    :param model: trained model
    :param X_val: X_val from split of train and test data
    :param y_val: y_val from split of train and test data
    :return: None
    '''

    scores = model.evaluate(X_val, y_val, verbose=0)

    preds = model.predict(X_val)  # label scores
    classpreds = np.argmax(preds, axis=1)  # predicted classes
    y_testclass = np.argmax(y_val, axis=1)  # true classes

    # Classification Report
    print(classification_report(y_testclass, classpreds, target_names=CLASSES))

    print("Model evaluation accuracy: ", round(scores[1] * 100), "%")



def compute_ROC_curve(preds, y_val, n_classes):
    '''
    Compute ROC curve and ROC area for each class

    :param preds: Predictions of the model
    :param y_val: Actual correct labels
    :param n_classes: int of number of classes
    :return: None
    '''

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Each Class')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], linewidth=3, label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], CLASSES[i]))
    ax.legend(loc="best", fontsize='x-large')
    ax.grid(alpha=.4)
    sns.despine()

    # Save the plot as an image
    fig.savefig('static/auc_plot.jpg', bbox_inches='tight', dpi=150)
    plt.show()


def test_model_testdata(model, test_x):

    '''
     This function loads a trained model makes inference from unlabelled audio files placed in folder

    :param model: (str) path where is your model
    :param test_x: (numpy.ndarray) test_x that you get using load_dataset_from_folders() function. It has all tests samples
    :return: None
    '''


    # Example predict on test data
    y_pred = model.predict_classes(test_x, batch_size=32)
    print("prediction test return :", y_pred[1], "-", int_to_label[y_pred[1]])
    print()
    print('All predicted labels: ', y_pred)



def test_model_unlabelled(path_model, folder_unlabelled):

    '''

    This function loads a trained model makes inference from unlabelled audio files placed in folder

    :param path_model:  (str) path where is your model
    :param folder_unlabelled: (str) path to folder where there are all the unlabelled audio files
    :return: None

    '''

    MAX_SOUND_CLIP_DURATION = 12  # seconds

    # Loads the model from path
    model = load_model(path_model)

    test_files = fnmatch.filter(os.listdir(folder_unlabelled), '*.wav')
    test_sounds = load_file_data(folder=folder_unlabelled, file_names=test_files, duration=MAX_SOUND_CLIP_DURATION)
    print('Testing record files: ', len(test_sounds))

    test = np.array(test_sounds).reshape((len(test_sounds), -1, 1))

    # Example predict on test data
    y_pred = model.predict_classes(test)
    for pred in y_pred:
        print("prediction test return :", pred, "-", int_to_label[pred])
