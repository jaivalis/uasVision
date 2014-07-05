from modules.classifier.medalon.m_weakClassifier import MedalonWeakClassifier
from modules.util.math_f import *
from sys import maxint
import copy
import time


class MedalonClassifier(object):
    def __init__(self,
                 training_data,
                 labels,
                 alpha,
                 beta,
                 gamma,
                 layers=10,
                 algorithm='wald'):
        s_count, feature_count = np.shape(training_data)
        l_count = np.size(labels)
        if s_count != l_count:
            raise ValueError("Invalid classifier input data (sample count: %d, label_count: %d)" % (s_count, l_count))
        if algorithm not in ['adaboost', 'wald']:
            raise ValueError('Unsupported learning algorithm')
        self.training_data = training_data
        self.labels = labels
        self.A = (1. - beta) / alpha
        self.B = beta / (1. - alpha)
        self.gamma = gamma
        self.layers = layers
        self.all_classifiers = []  # len(classifiers) = T

        # initial_weights = np.ones((1, s_count)) / s_count
        for i in range(feature_count):
            wc_responses = training_data[:, i]
            wc = MedalonWeakClassifier(wc_responses, self.labels, i)
            self.all_classifiers.append(wc)

        self.algorithm = algorithm

        self.classifiers = []
        tic = time.clock()

        weights = self._get_weights()

        for t in range(self.layers):  # choose the weak classifier with the minimum error
            print "Learn with bootstrapping using %s, layer #%d" % (self.algorithm.title(), t+1)
            h_t = self._fetch_best_weak_classifier(weights)
            self.classifiers.append(copy.deepcopy(h_t))    # add it to the strong classifier
            self.all_classifiers.remove(h_t)

            self.classifiers[-1].update_alpha()
            weights = self._adaboost_reweight(weights, t)
            toc = time.clock()
            print toc - tic
            print self.classifiers[t]

    def _get_weights(self):
        s_count, feature_count = np.shape(self.training_data)
        return np.ones(s_count) / s_count

    def _adaboost_reweight(self, weighted_patches, t):
        ret = np.zeros(len(weighted_patches))
        wc = self.classifiers[t]
        a_t = wc.alpha
        index = 0

        sum_ = 0.0
        for w in weighted_patches:
            pred = wc.classify(wc.annotated_responses[index, 0])
            true_label = wc.annotated_responses[index, 1]
            w_prime = w * np.exp(-a_t * true_label * pred)
            ret[index] = w_prime
            sum_ += w_prime
            index += 1
        for i in xrange(len(ret)):
            ret[i] /= sum_
        return ret

    def _reweight_and_discard_irrelevant(self, weighted_sample_pool, t):
        """ Throws away training samples that fall in the predefined thresholds and reweighs the patches
        :param weighted_sample_pool: A set of training samples
        :param t: layer number
        :return: The filtered set of training samples
        """
        tmp = []
        ret = []
        wc = self.classifiers[t]
        theta_a = wc.theta_a
        theta_b = wc.theta_b

        norm_factor = 0
        discarded = 0
        for patch, w in weighted_sample_pool:
            response = self.h_t(patch, t)
            # if t > 3:
            # if response < theta_a or response > theta_b:  # throw it away
            #     discarded += 1
            #     continue
            r = self.classify(patch)
            label = patch.label
            new_weight = w * np.exp(-label * r)

            tmp.append([patch, new_weight])
            norm_factor += new_weight
        for patch, w in tmp:  # normalize weights
            normalized_weight = w / norm_factor
            ret.append([patch, normalized_weight])
        print "Discarded %d training samples" % discarded
        return ret

    def _estimate_ratios(self, weighted_patches, t):
        """
        Returns the KDEs along with the linear spaces used for visualizing
        :param weighted_patches:
        :param t: layer number
        :return:
        """
        pos_weighted_patches = []
        pos = []
        neg_weighted_patches = []
        neg = []
        for patch, w in weighted_patches:  # TODO speedup
            if patch.label == +1:
                pos_weighted_patches.append([patch, w])
                pos.append(self.h_t(patch, t))
            elif patch.label == -1:
                neg_weighted_patches.append([patch, w])
                neg.append(self.h_t(patch, t))

        # Compute Cumulative conditional probabilities of classes
        # compute gaussians for negative and positive classes
        all_patches = np.append(pos, neg)
        sigma = np.std(all_patches)
        h = 1.144 * sigma * len(all_patches) ** (-0.2)

        kde_n, kde_p, xs_n, xs_p = get_two_kdes(pos, neg, h)
        plot_gaussians(neg, pos, sigma, h)  # TESTING
        return kde_n, kde_p, xs_n, xs_p

    def _tune_thresholds(self, kde_n, kde_p, xs_n, xs_p, t):
        """ Update the threshold of the weak classifier using this algorithm:
        http://personal.ee.surrey.ac.uk/Personal/Z.Kalal/Publications/2007_MSc_thesis.pdf pages 37, 38
        :param pos_gaussian: Sum of positive class gaussians
        :param neg_gaussian: Sum of negative class gaussians
        :param t: layer number
        """
        index = 0
        r = []

        best_ratio_a = 0
        best_ratio_b = 0
        for theta_candidate in np.linspace(min(min(xs_n), min(xs_p)), max(max(xs_n), max(xs_p)), 1000):
            # if neg_gaussian[index] < .5 and pos_gaussian[index] < .5:
            #     continue
            rat = kde_n[index] / kde_p[index]
            r.append([theta_candidate, rat])
            if rat > self.A and self.classifiers[t].theta_a == -maxint:
                if rat > best_ratio_a:
                    best_ratio_a = rat
                    self.classifiers[t].theta_a = theta_candidate
            if rat < self.B and self.classifiers[t].theta_b == maxint:
                # if and self.classifiers[t].theta_b == +maxint:
                if rat > best_ratio_b:
                    best_ratio_b = rat
                    self.classifiers[t].theta_b = theta_candidate
            index += 1
        # assert self.classifiers[t].theta_a != -maxint
        # assert self.classifiers[t].theta_b != +maxint
        # assert self.classifiers[t].theta_a < self.classifiers[t].theta_b
        plot_ratios(r, self.classifiers[t].theta_a, self.classifiers[t].theta_b)

    def score(self, validation_set, labels):
        correct = 0.0
        i = 0
        for sample in validation_set:
            resp = self.classify(sample)

            correct += (resp == labels[i])
            i += 1
        return correct / np.size(labels)

    def _fetch_best_weak_classifier(self, weights):
        """ Returns the weak classifier that produces the least error for a given training set
        :param weights: Weighted training set
        :return: The best classifier
        """
        min_error = 2.
        print "Training and measuring error for %d classifiers" % len(self.all_classifiers),
        dec = .05
        i = 0
        for wc in self.all_classifiers:
            i += 1
            wc.train(weights)
            if wc.error < min_error:
                min_error = wc.error
                ret = wc
            if i > dec * len(self.all_classifiers):
                dec += .05
                print ".",
        print "[DONE]"
        return ret

    def h_t(self, x, t):
        """ H_t(x) returns the summation of the responses of the first t weak classifiers.
        :param t: Layer count
        :param x: Patch to be classified
        :return: H_t(x)
        """
        ret = 0
        strong_classifier = self.classifiers[0:t+1]
        for wc in strong_classifier:
            ret += wc.classify(x)
        return ret

    def classify(self, sample):
        """ Implementation of Algorithm 1 Sochman.
        :param sample: Patch to be classified
        :return: +1, -1
        """
        if self.algorithm == 'adaboost':
            ret = 0
            for t in range(0, len(self.classifiers)):
                wc = self.classifiers[t]
                a = wc.alpha
                ret += wc.classify(sample[self.classifiers[t].column]) * a
            return np.sign(ret)
        elif self.algorithm == 'wald':
            for t in range(0, len(self.classifiers)):
                h_ = self.h_t(sample, t)
                wc = self.classifiers[t]
                if h_ >= wc.theta_b:
                    return +1
                if h_ <= wc.theta_a:
                    return -1
            if self.h_t(sample, self.layers) > self.gamma:
                return +1
            else:
                return -1

    def get_id(self):
        return "%s_" % self.algorithm

    def __str__(self):
        ret = "%s Strong classifier A:%.2f, B:%.2f {\n" % (self.algorithm.title(), self.A, self.B)
        cl = 1
        for wc in self.classifiers:
            ret += "\tWeak classifier #%s:\n\t\t%s\n" % (cl, wc)
            cl += 1
        return ret + "}"