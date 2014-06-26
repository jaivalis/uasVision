import cPickle
import os

from modules.classifier.classifier import MinimalStrongClassifier


"""
Used to serialize objects of the class StrongClassifier
"""


def serialize(object):
    id_ = object.get_id()
    with open(id_ + '.pickle', 'wb') as f:
        cPickle.dump(object, f)
    print "%s has been serialized and saved to file." % id_.title()


def deserialize(path):
    with open(path, 'rb') as pickle_file:
        entry = cPickle.load(pickle_file)
    print "%s has been deserialized and loaded to memory." % entry.get_id().title()
    return entry


def get_pickles(path='.'):
    ret = []
    files = [f for f in os.listdir(path)]
    for f in files:
        if f[-7: len(f)] == '.pickle':
            ret.append(path + '/' + f)
    return ret