import random
import numpy as np

from modules.datastructures.patch import Patch


def get_integral_image(img):
    w, h = img.shape
    ret = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            ret[x, y] = np.sum(img[0:x+1, 0:y+1])
    return ret


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