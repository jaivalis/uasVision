import numpy as np

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from modules.classifier.medalon.m_classifier import MedalonClassifier
from modules.classifier.trainingStream import TrainingStream
from modules.classifier.feature_extractors.haar import HaarHolder
from modules.util.serializer import *
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split


# synthetic data
training_path = '../dataset/MADELON/madelon_train.data'
labels_path = '../dataset/MADELON/madelon_train.labels'
valid_path = '../dataset/MADELON/madelon_valid.data'
valid_labels_path = '../dataset/MADELON/madelon_valid.labels'
# our data
vidPaths = ['../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4']
annPaths = ['../dataset/annotations/sanitized/COW809_1_sane.txt']

# scikit = False
scikit = True
# synthetic = True
synthetic = False

if __name__ == '__main__':
    if synthetic:
        training = np.loadtxt(training_path)
        labels = np.transpose(np.loadtxt(labels_path))
        validation = np.loadtxt(valid_path)
        valid_labels = np.transpose(np.loadtxt(valid_labels_path))

        # training = load_digits(10)
        # x,y      = training.data, training.target
        # training, validation, labels, valid_labels = train_test_split(x, y, test_size=0.5)

    else:  # real data
        ts = TrainingStream(vidPaths[0], annPaths[0])
        hh = HaarHolder((24, 24))
        # training = np.loadtxt('training.data')
        # labels = np.loadtxt('labels.data')
        validation = np.loadtxt('validation.data')
        valid_labels = np.loadtxt('valid_labels.data')
        training, labels = ts.get_training_set(haar_holder=hh, patch_count=750)
        # np.savetxt('training.data', training, delimiter=' ')
        # np.savetxt('labels.data', labels, delimiter=' ')
        # validation, valid_labels = ts.get_training_set(haar_holder=hh, patch_count=250)
        np.savetxt('validation.data', validation, delimiter=' ')
        np.savetxt('valid_labels.data', valid_labels, delimiter=' ')

    sample_count, feature_count = np.shape(training)
    print "Shape training", np.shape(training)
    print "Shape labels", np.shape(labels)

    alpha = .0005
    beta = 0.0
    gamma = .8

    if scikit:
        # By default, weak learners are decision stumps.
        # http://algorithm.daqwest.com/vinay?search=AdaBoost
        ada_real = AdaBoostClassifier(
            learning_rate=1,
            n_estimators=100,
            algorithm="SAMME.R"
        )
        ada_real.fit(training, labels)
        print ada_real.feature_importances_
        #print ada_real.feature_importances_ #print weights of weak classifiers
        ada_r_err = 1.0 - ada_real.score(validation, valid_labels)
        print "Real Adaboost Classifier error: %f" % ada_r_err

        ada = AdaBoostClassifier(
            learning_rate=1,
            n_estimators=300,
            algorithm="SAMME"
        )
        ada.fit(training, labels)
        print len(ada.feature_importances_)
        #print ada.feature_importances_ #print weights of weak classifiers
        ada_err = 1.0 - ada.score(validation, valid_labels)
        print "Adaboost Classifier error: %f" % ada_err
        # Real 350: 0.4 error
        # Discrete 350 0.365
        # Real 500: 0.372
        # Discrete 500 0.405
    else:
        clf = MedalonClassifier(training, labels, alpha, beta, gamma, layers=100, algorithm='adaboost')
        serialize(clf)
        ada_err = 1.0 - clf.score(validation, valid_labels)
        print "Adaboost Classifier error: %f" % ada_err
