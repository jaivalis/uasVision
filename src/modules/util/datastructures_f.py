import random


def random_sample_weighted_patches(lst, sample_count):
    # TODO make sure that some samples are contained
    """ Returns a random subsample of arr of size 'sample_count' containing 50% positive and 50% negative samples
    :param lst: List containing [patch, weight]
    :param sample_count: Size of returned dictionary
    :return: a random subsample of d of size 'sample_count'
    """
    sample_count = min(sample_count, len(lst))
    ret = []
    indexes = range(len(lst))
    random.shuffle(indexes)
    pos_count = 0
    neg_count = 0
    for i in indexes:
        patch = lst[i][0]
        if pos_count < 1 + sample_count / 2. and patch.label == +1:
            pos_count += 1
            ret.append(lst[i])
        if neg_count < 1 + sample_count / 2. and patch.label == -1:
            neg_count += 1
            ret.append(lst[i])

        if len(ret) == sample_count:
            break
    return ret