import cv2
import numpy as np


class Patch(object):

    def __init__(self, crop, annotation):
        self.crop = crop
        self.frame_id = annotation.frame
        self.xmin = annotation.xmin
        self.ymin = annotation.ymin
        self.xmax = annotation.xmax
        self.ymax = annotation.ymax
        self.haar = None

        assert(not np.array_equal(np.ndarray([]), self.crop))

    def update_haar_features(self, haar):
        self.haar = haar

    # def extractHOGFeatures(self):
    #     pass

    def imshow(self):
        show = np.divide(self.crop, 255.0)
        cv2.imshow('Patch: imshow()', show)
        cv2.waitKey()

    def __str__(self):
        ret = 'Frame: ' + str(self.frame_id)
        ret += ' xmin: ' + str(self.xmin)
        ret += ' ymin: ' + str(self.ymin)
        ret += ' xmax: ' + str(self.xmax)
        ret += ' ymax: ' + str(self.ymax)
        return ret