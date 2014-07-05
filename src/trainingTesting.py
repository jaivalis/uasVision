import pylab as pl
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier

from modules.classifier.trainingStream import TrainingStream
from modules.classifier.feature_extractors.haar import HaarHolder
from modules.classifier.classifier import *
from modules.util.serializer import *

vidPaths = ['../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4']
annPaths = ['../dataset/annotations/sanitized/COW809_1_sane.txt']

if __name__ == '__main__':
    negative_count = 10

    ts = TrainingStream(vidPaths[0], annPaths[0])
    hh = HaarHolder((24, 24))

    alpha = .0005
    beta = 0.0
    gamma = .8

    # training_set = ts.get_minimal_training_set(hh, 100)
    # print training_set


    # dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    # dt_stump.fit(training_data[0], training_data[1])
    # dt_stump_err = 1.0 - dt_stump.score(training_data[0], training_data[1])

    # ada_real = AdaBoostClassifier(
    #     base_estimator=dt_stump,
    #     learning_rate=1,
    #     n_estimators=500,
    #     algorithm="SAMME.R")
    # ada_real.fit(training_data[0], training_data[1])