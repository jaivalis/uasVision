import random


def split_weighted_patches(weighted_patches):
    pos = []
    neg = []
    for patch, w in weighted_patches:
        if patch.label == +1:
            pos.append([patch, w])
        elif patch.label == -1:
            neg.append([patch, w])
    return pos, neg


def random_sample_weighted_patches(weighted_patches, sample_count):
    """ Returns a random subsample of arr of size 'sample_count' containing both positive and negative samples
    :param weighted_patches: List containing [patch, weight]
    :param sample_count: Size of returned dictionary
    :return: a random subsample of d of size 'sample_count'
    """
    pos, neg = split_weighted_patches(weighted_patches)
    pos_count = len(pos)
    neg_count = len(neg)
    sample_count = min(sample_count, len(weighted_patches), 2*pos_count, 2*neg_count)
    if neg_count == pos_count == 0:
        print "Weighted patches is empty, cannot be sampled from."
        return []
    if pos_count == 0:
        print "Weighted patches do not contain any negative examples, cannot be sampled from."
        return []
    if neg_count == 0:
        print "Weighted patches do not contain any positive examples, cannot be sampled from."
        return []
    print "Sampling %d samples out of sample pool of size %d " % (sample_count, len(weighted_patches)),
    ret = []

    if sample_count == 2*pos_count:
        ret.extend(pos)
        random.shuffle(neg)
        ret.extend(neg[0:pos_count])
    elif sample_count == 2*neg_count:
        ret.extend(pos)
        random.shuffle(pos)
        ret.extend(neg[0:neg_count])
    else:
        random.shuffle(pos)
        random.shuffle(neg)
        ret.extend(pos[0:sample_count/2])
        ret.extend(neg[0:sample_count/2])
    print "[DONE]"
    return ret