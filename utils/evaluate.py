
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib as plt
import seaborn as sns

CLASSES = ['artifact','murmur','normal']

def calc_accuracy(model, X_val, y_val):

    scores = model.evaluate(X_val, y_val, verbose=0)

    preds = model.predict(X_val)  # label scores
    classpreds = np.argmax(preds, axis=1)  # predicted classes
    y_testclass = np.argmax(y_val, axis=1)  # true classes
    # Classification Report
    print(classification_report(y_testclass, classpreds, target_names=CLASSES))

    print("Model evaluation accuracy: ", round(scores[1] * 100), "%")

def compute_ROC_curve(preds, y_val, n_classes):
    # Compute ROC curve and ROC area for each class
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
    plt.show()






