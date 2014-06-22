import random


def random_sample(arr, sample_count):
    """ Returns a random subsample of arr of size 'sample_count'
    :param arr: List to sample from
    :param sample_count: Size of returned dictionary
    :return: a random subsample of d of size 'sample_count'
    """
    assert sample_count < len(arr)
    ret = []
    indexes = range(len(arr))
    indexes = indexes[0: sample_count]
    random.shuffle(indexes)
    for i in indexes:
        ret.append(arr[i])
    return ret


def binning(seq, bin_count):
    avg = len(seq) / float(bin_count)
    ret = []
    last = 0.0

    while last < len(seq):
        ret.append(seq[int(last):int(last + avg)])
        last += avg

    return ret