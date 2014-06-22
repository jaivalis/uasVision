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
        self.all_classifiers = self.all_classifiers[-100:len(self.all_classifiers)]  # TODO remove this, testing
        print "Initialized %d weak classifiers." % (len(self.all_classifiers))

        # # Phase2: Compute the classifier response for all the training patches
        # samples = 0
        # cll = 0
        # for classifier in self.all_classifiers:
        #     cll += 1
        #     while True:
        #         for patch in training_stream.get_random_patches():
        #             samples += 1
        #             classifier.store_response(patch)
        #
        #         if samples > sample_count + 1:
        #             break
        #     print "Done extracting feature #%d: " % cll + str(classifier)
        #     classifier.plot_gaussian()

        # Phase3: Algorithm2: Learning with bootstrapping
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

            neg, pos = self._estimate_ratios(weighted_patches, t)
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
            #response = self.h_t(patch, t)
            #if response < theta_a or response > theta_b:  # throw it away
            #    continue
            r = wc.classify(patch)
            label = patch.label
            new_weight = w * np.exp(-label * r)

            tmp.append([patch, new_weight])
            norm_factor += new_weight
        for patch, w in tmp:  # normalize weights
            normalized_weight = w / norm_factor
            ret.append([patch, normalized_weight])
        return ret

    def classify_batch(self, weighted_sample_pool):
        """ Classifies a sample of weighted samples drawn from the sample pool
        :param weighted_sample_pool: Weighted training set, subsample of the sample pool
        :return: Ratio
        """
        correct = 0.
        incorrect = 0.
        for patch, w in weighted_sample_pool:    # fetch response for patch
            true_label = patch.label
            ret_label = self._classify(patch)
            if true_label == ret_label:
                correct += w
            else:
                incorrect += w
        epsilon = 1. / (2. * len(weighted_sample_pool))
        ret = .5 * np.log((correct + epsilon) / (incorrect + epsilon))
        # print "correct classifications: %f, incorrect classifications: %f, ret: %f" % (correct, incorrect, ret)
        return .5 * np.log((correct + epsilon) / (incorrect + epsilon))

    def _estimate_ratios(self, weighted_patches, t):
        """ Real Adaboost for feature selection, right ?
        :param weighted_patches:
        :param t: layer number
        :return:
        """
        # split weighted_patches into j bins
        bin_count = 3

        pos_weighted_patches = []
        neg_weighted_patches = []
        for patch, w in weighted_patches:  # TODO speedup
            if patch.label == +1:
                pos_weighted_patches.append([patch, w])
            elif patch.label == -1:
                neg_weighted_patches.append([patch, w])
        pos_bins = binning(pos_weighted_patches, bin_count)
        neg_bins = binning(neg_weighted_patches, bin_count)

        pos_ratios = []
        neg_ratios = []
        for i in range(bin_count):
            if i < len(pos_bins):
                pos_ratios.append(self.classify_batch(pos_bins[i]))
            if i < len(neg_bins):
                neg_ratios.append(self.classify_batch(neg_bins[i]))
        if pos_ratios:
            plt.hist(pos_ratios, bins=bin_count, alpha=.5, color='blue')
        if neg_ratios:
            plt.hist(neg_ratios, bins=bin_count, alpha=.5, color='red')
        plt.show()

        # Compute Cumulative conditional probabilities of classes
        # compute gaussians for negative and positive classes
        sigma_neg = np.std(neg_ratios)
        sigma_pos = np.std(pos_ratios)
        h_neg = 1.144 * sigma_neg * len(neg_ratios) ** -0.2
        h_pos = 1.144 * sigma_pos * len(pos_ratios) ** -0.2
        lin_space = np.linspace(-1, 1, num=1000)
        neg_sum_of_gaussians = sum_of_gaussians(neg_ratios, lin_space, h_neg)
        pos_sum_of_gaussians = sum_of_gaussians(pos_ratios, lin_space, h_pos)
        plot_gaussians(neg_ratios, pos_ratios, sigma_neg, sigma_pos, h_neg, h_pos)
        return neg_sum_of_gaussians, pos_sum_of_gaussians

    def _tune_thresholds(self, pos_gaussian, neg_gaussian, t):
        """ Update the threshold of the weak classifier using this algorithm:
        http://personal.ee.surrey.ac.uk/Personal/Z.Kalal/Publications/2007_MSc_thesis.pdf pages 37, 38
        :param pos_gaussian: Sum of positive class gaussians
        :param neg_gaussian: Sum of negative class gaussians
        :param t: layer number
        """
        index = 0
        for theta_a_candidate in np.linspace(-1, 1, num=1000):
            if neg_gaussian[index] < theta_a_candidate and pos_gaussian[index] < theta_a_candidate:
                r_a = neg_gaussian[index] / pos_gaussian[index]
                if r_a > self.A:
                    self.classifiers[t].theta_a = theta_a_candidate
                    break
            index += 1
        lin_space = np.linspace(1, -1, num=1000)  # from right to left
        index = 0
        for theta_b_candidate in lin_space:
            if neg_gaussian[index] > theta_b_candidate and pos_gaussian[index] > theta_b_candidate:
                r_b = neg_gaussian[index] / pos_gaussian[index]
                if r_b < self.B:
                    self.classifiers[t].theta_b = theta_b_candidate
                    break
            index += 1

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