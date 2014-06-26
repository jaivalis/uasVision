import random
import numpy as np
from scipy.misc import imresize
from cv2 import integral

from modules.datastructures.patch import Patch


def get_downsampled(img):
    ret = imresize(img, (24, 24))
    return ret


def get_integral_image(img):
    return integral(img)


def to_rgb(im):
    # as 3a, but we add an extra copy to contiguous 'C' order
    # data
    return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')


def get_random_patch(img, size, frame_id):
    h, w = np.shape(img)

    randx = randy = -1
    while True:
        randx = random.randint(0, h - 1)
        if randx + size[0] < h:
            break
    while True:
        randy = random.randint(0, w - 1)
        if randy + size[1] < w:
            break
    xmin = randx
    xmax = randx + size[0]
    ymin = randy
    ymax = randy + size[1]
    crop = img[xmin:xmax, ymin:ymax]
    return Patch(crop, frame_id, (xmin, ymin, xmax, ymax), -1)


def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def overlap(positives, patch):
    ret = False
    for p in positives:
        ret = ret or p.overlap(patch)
    return ret