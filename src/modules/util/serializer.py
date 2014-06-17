import cPickle
import os

"""
Mainly used to serialize StrongClassifier objects
"""


def serialize(object):
    id = object.get_id()
    with open(id + '.pickle', 'wb') as f:
        cPickle.dump(object, f)


def deserialize(path):
    with open(path, 'rb') as f:
        entry = cPickle.load(f)
    return entry


def get_pickles(path='.'):
    ret = []
    files = [f for f in os.listdir(path) if os.path.isfile(f)]
    for f in files:
        if f[-7: len(f)] == '.pickle':
            print f
            ret.append(f)
    return ret