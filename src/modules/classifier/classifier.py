from modules.classifier.weakClassifier import WeakClassifier
from modules.util.math_f import *
from modules.util.datastructures_f import *
import copy


class StrongClassifier(object):
    def __init__(self, training_stream, feature_holder, alpha, beta, gamma, layers=10, sample_count=20):
        self.training_stream = training_stream
        self.feature_holder = feature_holder
        self.A = (1. - beta) / alpha
        self.B = beta / (1. - alpha)
        self.gamma = gamma
        self.layers = layers
        self.all_classifiers = []  # len(classifiers) = T

        self.classifiers = []

        #########    Parzen window technique    #########
        # Phase1:  Create all possible weak classifiers #
        for feature in feature_holder.get_features():
            wc = WeakClassifier(feature)
            self.all_classifiers.append(wc)
        self.all_classifiers = self.all_classifiers[-1000:len(self.all_classifiers)]  # TODO remove this, testing
        print "Initialized %d weak classifiers." % (len(self.all_classifiers))

        # Phase2: Algorithm2: Learning with bootstrapping
        self.learn_with_bootstrapping(sample_count)

    def learn_with_bootstrapping(self, sample_count=10000):
        """ Algorithm2 Sochman
        Outputs the strong classifier with the theta_a, theta_b of the weak classifiers updated
        """
        training_set_size = 90  # TODO: change to 1000, 500 or something
        sample_pool = self.training_stream.extract_training_patches(sample_count)
        # initialize weights
        weighted_patches = []
        for patch in sample_pool:                              # weight all patches: training pool P
            weighted_patches.append([patch, 1. / len(sample_pool)])

        # shuffle training pool
        weighted_patches = random_sample(weighted_patches, len(weighted_patches))

        training_data = random_sample(weighted_patches, training_set_size)  # sample 'training_set_size' many samples

        for t in range(self.layers+1):
            # choose the weak classifier with the minimum error
            h_t = self._fetch_best_weak_classifier(training_data)
            self.classifiers.append(copy.deepcopy(h_t))    # add it to the strong classifier

            print self

            neg, pos = self._estimate_ratios(training_data, t)
            # find decision thresholds for the strong classifier
            self._tune_thresholds(pos, neg, t)

            # throw away training samples that fall in our thresholds
            weighted_patches = self._reweight_and_discard_irrelevant(weighted_patches, t)

            # sample new training data
            training_data = random_sample(weighted_patches, training_set_size)

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

        for patch, w in weighted_sample_pool:
            response = self.h_t(patch, t)
            if response < theta_a or response > theta_b:  # throw it away
                continue
            r = wc.classify(patch)
            label = patch.label
            new_weight = w * np.exp(-label * r)

            tmp.append([patch, new_weight])
            norm_factor += new_weight
        for patch, w in tmp:  # normalize weights
            normalized_weight = w / norm_factor
            ret.append([patch, normalized_weight])
        return ret

    def _estimate_ratios(self, weighted_patches, t):
        """ Real Adaboost for feature selection, right ?
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
        h = 1.144 * sigma * len(all_patches) ** -0.2
        lin_space = np.linspace(-1, 1, num=1000)
        neg_sum_of_gaussians = sum_of_gaussians(neg, lin_space, h)
        pos_sum_of_gaussians = sum_of_gaussians(pos, lin_space, h)
        plot_gaussians(neg, pos, sigma, h)
        return neg_sum_of_gaussians, pos_sum_of_gaussians

    def _tune_thresholds(self, pos_gaussian, neg_gaussian, t):
        """ Update the threshold of the weak classifier using this algorithm:
        http://personal.ee.surrey.ac.uk/Personal/Z.Kalal/Publications/2007_MSc_thesis.pdf pages 37, 38
        :param pos_gaussian: Sum of positive class gaussians
        :param neg_gaussian: Sum of negative class gaussians
        :param t: layer number
        """
        index = 0
        threshold_found = 0
        r_a_theta = []
        for theta_a_candidate in np.linspace(-1, 1, num=1000):
            if neg_gaussian[index] < theta_a_candidate and pos_gaussian[index] < theta_a_candidate:
                r_a = neg_gaussian[index] / pos_gaussian[index]
                r_a_theta.append([theta_a_candidate, r_a])
                if r_a > self.A and threshold_found == 0:
                    self.classifiers[t].theta_a = theta_a_candidate
                    threshold_found = 1
            index += 1
        lin_space = np.linspace(1, -1, num=1000)  # from right to left
        index = 0
        threshold_found = 0
        r_b_theta = []
        for theta_b_candidate in lin_space:
            if neg_gaussian[index] > theta_b_candidate and pos_gaussian[index] > theta_b_candidate:
                r_b = neg_gaussian[index] / pos_gaussian[index]
                r_b_theta.append([theta_b_candidate, r_b])
                if r_b < self.B and threshold_found == 0:
                    self.classifiers[t].theta_b = theta_b_candidate
                    threshold_found = 1
            index += 1
        plot_ratios(r_b_theta, r_a_theta, self.classifiers[t].theta_b, self.classifiers[t].theta_a)

    def _fetch_best_weak_classifier(self, weighted_patches):
        """ Returns the weak classifier that produces the least error for a given training set
        :param weighted_patches: Weighted training set
        :return: The best classifier
        """
        min_error = 2.
        for wc in self.all_classifiers:
            wc.train(weighted_patches)
            if wc.error < min_error:
                ret = wc
        # ret.plot_gaussian()
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

    def _classify(self, patch):
        wc = self.classifiers[len(self.classifiers)-1]
        return wc.classify(patch)

    def classify(self, patch):
        """ Implementation of Algorithm 1 Sochman.
        :param patch: Patch to be classified
        :return: +1, -1
        """
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
        :return: A unique identifier
        """
        return ""

    def __str__(self):
        ret = ""
        cl = 1
        ret += "{\n"
        for wc in self.classifiers:
            ret += "\tWeak classifier #" + str(cl) + " - " + str(wc) + "\n"
            cl += 1
        return ret + "}"