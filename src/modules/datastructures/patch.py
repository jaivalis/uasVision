import cv2
import numpy as np


class Patch(object):

    def __init__(self, crop, frame_id, (xmin, ymin, xmax, ymax), label):
        self.crop = crop
        self.frame_id = frame_id
        self.xmin = xmin
        self.ymin = ymin
        self.ymax = ymax
        self.xmax = xmax
        self.label = label
        self.haar = None

        assert(not np.array_equal(np.ndarray([]), self.crop))

    def update_haar_features(self, haar):
        self.haar = haar

    # def extractHOGFeatures(self):
    #     pass

    def overlap(self, other):
        # p1_self = (self.xmin, self.ymin)
        # p2_self = (self.xmin, self.ymax)
        # p3_self = (self.xmax, self.ymin)
        # p4_self = (self.xmax, self.xmax)
        # p1_other = (other.xmin, other.ymin)
        # p2_other = (other.xmin, other.ymax)
        # p3_other = (other.xmax, other.ymin)
        # p4_other = (other.xmax, other.xmax)
        p1 = (self.xmin, self.ymin)
        p2 = (self.xmax, self.xmax)
        p3 = (other.xmin, other.ymin)
        p4 = (other.xmax, other.xmax)
        overlap = (p2[1] < p3[1] or p1[1] > p4[1] or p2[0] < p3[0] or p1[1] > p4[1])
        return not overlap

    def size(self):
        return self.xmax - self.xmin, self.ymax - self.ymin

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
        ret += ' label: ' + str(self.label)
        return ret