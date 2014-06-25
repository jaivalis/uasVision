import random
import numpy as np
from scipy.misc import imresize
from cv2 import integral

from modules.datastructures.patch import Patch


def get_downsampled(img):
    ret = imresize(img, (60, 60))
    return ret


def get_integral_image(img):
    return integral(img)


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


def overlap(positives, patch):
    ret = False
    for p in positives:
        ret = ret or p.overlap(patch)
    return ret