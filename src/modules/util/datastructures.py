__author__ = 'T'
import random


def shuffle_dict(dict):
    values = dict.values()
    print values
    random.shuffle(values)
    y = dict(zip(dict.keys(), values))
    print values