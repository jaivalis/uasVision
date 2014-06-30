__author__ = 'T'
from featureHolder import FeatureHolder
from modules.util.image_f import get_downsampled, get_integral_image
from cv2 import rectangle, imshow, waitKey, cvtColor, COLOR_GRAY2RGB
from scipy.misc import imresize


class HaarFeature(object):
    def __init__(self, type, h, w, i, j):
        self.type = type
        self.h = int(h)
        self.w = int(w)
        self.i = int(i)
        self.j = int(j)

    def __str__(self):
        return "Type: %d, height: %d, width: %d, x: %d, y: %d" % (self.type, self.h, self.w, self.i, self.j)

    def get_id(self):

        return -1

    def apply(self, img):  # TODO change signature to apply(self, patch)
        ret = None

        img = get_downsampled(img)
        img = get_integral_image(img)

        if self.type == 0:
            a1 = img[self.i, self.j]
            b1 = img[self.i, self.j + self.w - 1]
            c1 = img[self.i + self.h - 1, self.j]
            d1 = img[self.i + self.h - 1, self.j + self.w - 1]
            S1 = a1 - b1 - c1 + d1
            a2 = img[self.i, self.j + self.w]
            b2 = img[self.i, self.j + 2 * self.w - 1]
            c2 = img[self.i + self.h - 1, self.j + self.w]
            d2 = img[self.i + self.h - 1, self.j + 2 * self.w - 1]
            S2 = a2 - b2 - c2 + d2
            ret = S1 - S2
        if self.type == 1:
            a1 = img[self.i, self.j]
            b1 = img[self.i, self.j + self.w - 1]
            c1 = img[self.i + self.h - 1, self.j]
            d1 = img[self.i + self.h - 1, self.j + self.w - 1]
            S1 = a1 - b1 - c1 + d1
            a2 = img[self.i, self.j + self.w]
            b2 = img[self.i, self.j + 2 * self.w - 1]
            c2 = img[self.i + self.h - 1, self.j + self.w]
            d2 = img[self.i + self.h - 1, self.j + 2 * self.w - 1]
            S2 = a2 - b2 - c2 + d2
            a3 = img[self.i, self.j + 2 * self.w]
            b3 = img[self.i, self.j + 3 * self.w - 1]
            c3 = img[self.i + self.h - 1, self.j + 2 * self.w]
            d3 = img[self.i + self.h - 1, self.j + 2 * self.w]
            S3 = a3 - b3 - c3 + d3
            ret = S1 - S2 + S3
        if self.type == 2:
            S1 = img[self.i, self.j] - img[self.i, self.j + self.w - 1] \
                - img[self.i + self.h - 1, self.j] + img[self.i + self.h - 1, self.j + self.w - 1]
            S2 = img[self.i + self.h, self.j] - img[self.i + self.h, self.j + self.w - 1] \
                - img[self.i + 2 * self.h - 1, self.j] + img[self.i + 2 * self.h - 1, self.j + self.w - 1]
            ret = S1 - S2
        if self.type == 3:
            S1 = img[self.i, self.j] - img[self.i, self.j + self.w - 1] \
                 - img[self.i + self.h - 1, self.j] + img[self.i + self.h - 1, self.j + self.w - 1]
            S2 = img[self.i + self.h, self.j] - img[self.i + self.h, self.j + self.w - 1] \
                 - img[self.i + 2 * self.h - 1, self.j] + img[self.i + 2 * self.h - 1, self.j + self.w - 1]
            S3 = img[self.i + 2 * self.h, self.j] - img[self.i + 2 * self.h, self.j + self.w - 1] \
                 - img[self.i + 3 * self.h - 1, self.j] + img[self.i + 3 * self.h - 1, self.j + self.w - 1]
            ret = S1 - S2 + S3
        if self.type == 4:
            S1 = img[self.i, self.j] - img[self.i, self.j + self.w - 1] \
                 - img[self.i + self.h - 1, self.j] + img[self.i + self.h - 1, self.j + self.w - 1]
            S2 = img[self.i + self.h, self.j] - img[self.i + self.h, self.j + self.w - 1] \
                 - img[self.i + 2 * self.h - 1, self.j] + img[self.i + 2 * self.h - 1, self.j + self.w - 1]
            S3 = img[self.i, self.j + self.w] - img[self.i, self.j + 2 * self.w - 1] \
                 - img[self.i + self.h - 1, self.j + self.w] + img[self.i + self.h - 1, self.j + 2 * self.w - 1]
            S4 = img[self.i + self.h, self.j + self.w] - img[self.i + self.h, self.j + 2 * self.w - 1] \
                 - img[self.i + 2 * self.h - 1, self.j + self.w] + img[self.i + 2 * self.h - 1, self.j + 2 * self.w - 1]
            ret = S1 - S2 - S3 + S4
        return ret

    def __eq__(self, other):
        return self.h == other.h and self.w == other.h

    def visualize(self, crop):
        # crop = to_rgb(crop)
        if self.type == 0:
            rectangle(crop, (self.i, self.j), (self.i + self.h - 1, self.j + self.w - 1), (1, 0, 0))
            rectangle(crop, (self.i, self.j + self.w), (self.i + self.h - 1, self.j + 2 * self.w - 1), (0, 1, 0))
        elif self.type == 1:
            rectangle(crop, (self.i, self.j), (self.i + self.h - 1, self.j + self.w - 1), (1, 0, 0))
            rectangle(crop, (self.i, self.j + self.w), (self.i + self.h - 1, self.j + 2 * self.w - 1), (0, 1, 0))
            rectangle(crop, (self.i, self.j + 2 * self.w), (self.i + self.h - 1, self.j + 2 * self.w), (0, 1, 0))
        elif self.type == 2:
            rectangle(crop, (self.i, self.j), (self.i + self.h - 1, self.j + self.w - 1), (1, 0, 0))
            rectangle(crop, (self.i + self.h, self.j), (self.i + 2 * self.h - 1, self.j + self.w - 1), (0, 1, 0))
        elif self.type == 3:
            rectangle(crop, (self.i, self.j), (self.i + self.h - 1, self.j + self.w - 1), (1, 0, 0))
            rectangle(crop, (self.i + self.h, self.j), (self.i + 2 * self.h - 1, self.j + self.w - 1), (0, 1, 0))
            rectangle(crop, (self.i + 2 * self.h, self.j), (self.i + 3 * self.h - 1, self.j + self.w - 1), (1, 0, 0))
        elif self.type == 4:
            rectangle(crop, (self.i, self.j), (self.i + self.h - 1, self.j + self.w - 1), (1, 0, 0))
            rectangle(crop, (self.i, self.j + self.w), (self.i + self.h - 1, self.j + 2 * self.w - 1), (0, 1, 0))
            rectangle(crop, (self.i + self.h, self.j), (self.i + 2 * self.h - 1, self.j + self.w - 1), (0, 1, 0))
            rectangle(crop, (self.i + self.h, self.j + self.w), (self.i + 2 * self.h - 1, self.j + 2 * self.w - 1),
                      (1, 0, 0))
        crop = imresize(crop, (100, 100))
        imshow('image', crop)
        waitKey(0)


class HaarHolder(FeatureHolder):
    def __init__(self, (h, w)):
        self.features = []
        for i in range(w):
            for j in range(h):
                height = 1
                while i + height - 1 < h:
                    width = 1
                    while j + 2 * width - 1 < w:
                        feat = HaarFeature(0, height, width, i, j)
                        self.features.append(feat)
                        width += 1
                    height += 1
                height = 1
                while i + height - 1 < h:
                    width = 1
                    while j + 3 * width - 1 < w:
                        feat = HaarFeature(1, height, width, i, j)
                        self.features.append(feat)
                        width += 1
                    height += 1
                width = 1
                while j + width - 1 < w:
                    height = 1
                    while i + 2 * height - 1 < h:
                        feat = HaarFeature(2, height, width, i, j)
                        self.features.append(feat)
                        height += 1
                    width += 1
                width = 1
                while j + width - 1 < w:
                    height = 1
                    while i + 3 * height - 1 < h:
                        feat = HaarFeature(3, height, width, i, j)
                        self.features.append(feat)
                        height += 1
                    width += 1
                height = 1
                while i + 2 * height - 1 < h:
                    width = 1
                    while j + 2 * width - 1 < w:
                        feat = HaarFeature(4, height, width, i, j)
                        self.features.append(feat)
                        width += 1
                    height += 1

    def __len__(self):
        return len(self.features)

    def get_features(self):
        return self.features

    def get(self):
        pass


if __name__ == '__main__':
    hh = HaarHolder((24, 24))
    types = [0, 0, 0, 0, 0]
    for f in hh.features:
        types[f.type] += 1
    print len(hh.features)
    print types