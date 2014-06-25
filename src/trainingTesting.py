from modules.classifier.trainingStream import TrainingStream
from modules.classifier.feature_extractors.haar import HaarHolder
from modules.classifier.classifier import StrongClassifier
from modules.util.serializer import *

vidPaths = ['../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4']
annPaths = ['../dataset/annotations/sanitized/COW809_1_sane.txt']

if __name__ == '__main__':
    negative_count = 10

    ts = TrainingStream(vidPaths[0], annPaths[0])
    hh = HaarHolder((24, 24))

    alpha = .9
    beta = 0.1
    gamma = .8

    # get_pickles()
    classifier = StrongClassifier(ts, hh, alpha, beta, gamma, layers=20, sample_count=500, algorithm='wald')

    # for p in patches:
    #     # if p.label == -1:
    #     p.imshow()
    #     h.update_patch_haar(p)