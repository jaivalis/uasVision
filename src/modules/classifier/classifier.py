from modules.classifier.weakClassifier import WeakClassifier
from modules.util.math_f import *
from modules.util.datastructures_f import *
from sys import maxint
import copy
import time


class StrongClassifier(object):
    def __init__(self,
                 training_stream,
                 feature_holder,
                 alpha,
                 beta,
                 gamma,
                 layers=10,
                 sample_count=1000,
                 algorithm='wald'):
        self.training_stream = training_stream
        self.feature_holder = feature_holder
        self.A = (1. - beta) / alpha
        self.B = beta / (1. - alpha)
        self.gamma = gamma
        self.layers = layers
        self.sample_count = sample_count
        self.all_classifiers = []  # len(classifiers) = T
        if algorithm not in ['adaboost', 'wald']:
            raise ValueError('Unsupported learning algorithm')
        self.algorithm = algorithm

        self.classifiers = []

        #########    Parzen window technique    #########
        # Phase1:  Create all possible weak classifiers #
        for feature in feature_holder.get_features():
            wc = WeakClassifier(feature)
            self.all_classifiers.append(wc)
        # self.all_classifiers = self.all_classifiers[-50:len(self.all_classifiers)]  # TODO remove this, testing

        print "Initialized %d weak classifiers." % (len(self.all_classifiers))

        # Phase2: Algorithm2: Learning with bootstrapping
        self.learn_with_bootstrapping(sample_count)

    def learn_with_bootstrapping(self, sample_count=10000):
        """ Algorithm2 Sochman
        Outputs the strong classifier with the theta_a, theta_b of the weak classifiers updated
        """
        tic = time.clock()
        training_set_size = 150  # TODO: change to 1000, 500 or something
        sample_pool = self.training_stream.extract_training_patches(sample_count, negative_ratio=1.)
        # initialize weights
        weighted_patches = []
        for patch in sample_pool:                              # weight all patches: training pool P
            weighted_patches.append([patch, 1. / len(sample_pool)])
            # if patch.label == +1:
            #     pos_patch = patch                              # PRESENTATION, REPORT
        # shuffle training pool
        weighted_patches = random_sample_weighted_patches(weighted_patches, len(weighted_patches))

        if self.algorithm == 'adaboost':  # Shuffle the training data
            training_data = random_sample_weighted_patches(weighted_patches, len(weighted_patches))
        elif self.algorithm == 'wald':    # Sample training_set_size samples
            training_data = random_sample_weighted_patches(weighted_patches, training_set_size)

        for t in range(self.layers):  # choose the weak classifier with the minimum error
            print "Learn with bootstrapping using %s, layer #%d" % (self.algorithm.title(), t+1)

            if self.algorithm == 'adaboost':
                h_t = self._fetch_best_weak_classifier(weighted_patches)
            elif self.algorithm == 'wald':
                h_t = self._fetch_best_weak_classifier(training_data)
            # h_t.visualize(pos_patch)                       # PRESENTATION, REPORT
            self.classifiers.append(copy.deepcopy(h_t))    # add it to the strong classifier

            if self.algorithm == 'adaboost':
                self.classifiers[-1].update_alpha(weighted_patches)
                weighted_patches = self._adaboost_reweight(weighted_patches, t)
            elif self.algorithm == 'wald':
                kde_n, kde_p, xs_n, xs_p = self._estimate_ratios(training_data, t)
                # find decision thresholds for the strong classifier
                self._tune_thresholds(kde_n, kde_p, xs_n, xs_p, t)
                # throw away training samples that fall in our thresholds
                weighted_patches = self._reweight_and_discard_irrelevant(weighted_patches, t)
                # sample new training data
                training_data = random_sample_weighted_patches(weighted_patches, training_set_size)
                if len(training_data) == 0:
                    print "no more training data!"
                    break
            toc = time.clock()
            print toc - tic
            print self

    def _adaboost_reweight(self, weighted_patches, t):
        ret = []
        wc = self.classifiers[t]
        z_t = wc.z
        a_t = wc.alpha
        for patch, w in weighted_patches:
            pred = wc.classify(patch)
            true_label = patch.label
            w_prime = w * np.exp(-a_t * true_label * pred) / z_t

            ret.append([patch, w_prime])
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

    def _fetch_best_weak_classifier(self, weighted_patches):
        """ Returns the weak classifier that produces the least error for a given training set
        :param weighted_patches: Weighted training set
        :return: The best classifier
        """
        min_error = 2.
        print "Training and measuring error for %d classifiers" % len(self.all_classifiers),
        dec = .05
        i = 0
        for wc in self.all_classifiers:
            i += 1
            wc.train(weighted_patches)
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

    def classify(self, patch):
        """ Implementation of Algorithm 1 Sochman.
        :param patch: Patch to be classified
        :return: +1, -1
        """
        if self.algorithm == 'adaboost':
            ret = 0
            for t in range(0, len(self.classifiers)):
                wc = self.classifiers[t]
                a = wc.alpha
                ret += wc.classify(patch) * a
            return np.sign(ret)
        elif self.algorithm == 'wald':
            for t in range(0, len(self.classifiers)):
                h_ = self.h_t(patch, t)
                wc = self.classifiers[t]
                if h_ >= wc.theta_b:
                    return +1
                if h_ <= wc.theta_a:
                    return -1
            if self.h_t(patch, self.layers) > self.gamma:
                return +1
            else:
                return -1

    def __str__(self):
        ret = "%s Strong classifier A:%.2f, B:%.2f {\n" % (self.algorithm.title(), self.A, self.B)
        cl = 1
        for wc in self.classifiers:
            ret += "\tWeak classifier #%s:\n\t\t%s\n" % (cl, wc)
            cl += 1
        return ret + "}"


class MinimalStrongClassifier(object):
    def __init__(self, strong_classifier):
        self.A = strong_classifier.A
        self.B = strong_classifier.B
        self.gamma = strong_classifier.gamma
        self.layers = strong_classifier.layers
        self.sample_count = strong_classifier.sample_count
        self.all_classifiers = []  # len(classifiers) = T
        self.algorithm = strong_classifier.algorithm

        self.classifiers = strong_classifier.classifiers

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

    def classify(self, patch):
        """ Implementation of Algorithm 1 Sochman.
        :param patch: Patch to be classified
        :return: +1, -1
        """
        if self.algorithm == 'adaboost':
            ret = 0
            for t in range(0, len(self.classifiers)):
                wc = self.classifiers[t]
                a = wc.alpha
                ret += wc.classify(patch) * a
            return np.sign(ret)
        elif self.algorithm == 'wald':
            for t in range(0, len(self.classifiers)):
                h_ = self.h_t(patch, t)
                wc = self.classifiers[t]
                if h_ >= wc.theta_b:
                    return +1
                if h_ <= wc.theta_a:
                    return -1
            if self.h_t(patch, self.layers) > self.gamma:
                return +1
            else:
                return -1

    def get_id(self):
        """ Used by cPickle to serialize/de-serialize
        :rtype : basestring
        :return: A unique identifier
        """
        return "%s_%.2f_%f.2_%f.1_%d_%d" % (self.algorithm, self.A, self.B, self.gamma, self.layers, self.sample_count)