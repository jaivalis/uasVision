import random


def random_sample_weighted_patches(lst, sample_count):
    # TODO assert that the sampling is feasible raise exception otherwise fucking idiot
    """ Returns a random subsample of arr of size 'sample_count' containing both positive and negative samples
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

    i = 0
    while i < len(indexes):
        if lst[i][0].label == +1:
            pos_count += 1
        if lst[i][0].label == -1:
            neg_count += 1

        ret.append(lst[i])
        i += 1
        if len(ret) == sample_count:
            if pos_count > .2 * len(ret) and neg_count > .2 * len(ret):
                break
            else:  # re-sample
                ret = []
                i = 0
                pos_count = 0
                neg_count = 0
                random.shuffle(indexes)
    print "Sampling of %d samples [DONE]" % sample_count
    return ret