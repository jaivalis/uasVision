from modules.classifier.weakClassifier import WeakClassifier
from modules.util.math_f import *
from modules.util.datastructures_f import *


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
        self.all_classifiers = self.all_classifiers[-10:len(self.all_classifiers)]  # TODO remove this, testing
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
        training_set_size = 50  # TODO: change to 1000, 500 or something
        sample_pool = self.training_stream.extract_training_patches(sample_count)
        # initialize weights
        weighted_patches = []
        for patch in sample_pool:                              # weight all patches
            weighted_patches.append([patch, 1. / len(sample_pool)])
        training_data = random_sample(weighted_patches, training_set_size)  # sample 'training_set_size' many samples

        for t in range(self.layers+1):
            # choose the weak classifier with the minimum error
            h_t = self._fetch_best_weak_classifier(training_data)
            self.classifiers.append(h_t)    # add it to the strong classifier

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
        ret = []
        wc = self.classifiers[t]
        theta_a = wc.theta_a
        theta_b = wc.theta_b
        for patch, w in weighted_sample_pool:
            response = self.h_t(patch, t)
            if response < theta_a or response > theta_b:
                continue
            else:
                r = wc.classify(patch)

                label = patch.label
                new_weight = w * np.exp(-label * r)

                ret.append([patch, new_weight])
        # normalize weights
        for patch, w in ret:
            pass
        return ret

    def classify_batch(self, weighted_sample_pool):
        """ Classifies a sample of weighted samples drawn from the sample pool
        :param weighted_sample_pool: Weighted training set, subsample of the sample pool
        :return: Ratio
        """
        correct = incorrect = 0
        for patch, w in weighted_sample_pool:    # fetch response for patch
            true_label = patch.label
            ret_label = self.classify(patch)
            if true_label == ret_label:
                correct += w
            else:
                incorrect += w
        epsilon = 1. / (2. * len(weighted_sample_pool))
        return .5 * np.log((correct + epsilon) / (incorrect + epsilon))

    def _estimate_ratios(self, weighted_patches, t):
        """ Real Adaboost for feature selection, right ?
        :param weighted_patches:
        :param t: layer number
        :return:
        """
        ret = []
        # split weighted_patches into j bins
        bin_count = 10
        bins = binning(weighted_patches, bin_count)

        for bin in bins:
            r = self.classify_batch(bin)
            ret.append(r)

        pos_responses = neg_responses = None
        for p, w in weighted_patches:
            response = self.h_t(p, t)
            true_label = p.label
            # append to the responses
            if p.label == +1:
                if pos_responses is None:
                    pos_responses = np.array([response, true_label, w])
                else:
                    pos_responses = np.vstack((pos_responses, [response, true_label, w]))
            elif p.label == -1:
                if neg_responses is None:
                    neg_responses = np.array([response, true_label, w])
                else:
                    neg_responses = np.vstack((neg_responses, [response, true_label, w]))
        # Compute Cumulative conditional probabilities of classes
        #plot_histograms(pos_responses, neg_responses)
        data = np.append(pos_responses, neg_responses, axis=0)
        plot_gaussians(data, 0.5, 0.5)
        # compute gaussians for negative and positive classes
        sigma_neg = np.std(neg_responses)
        sigma_pos = np.std(pos_responses)
        h_neg = 1.144 * sigma_neg * len(neg_responses) ** -0.2
        h_pos = 1.144 * sigma_pos * len(pos_responses) ** -0.2
        lin_space = np.linspace(-1, 1, num=1000)
        neg_sum_of_gaussians = sum_of_gaussians(neg_responses, lin_space, h_neg)
        pos_sum_of_gaussians = sum_of_gaussians(pos_responses, lin_space, h_pos)
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
                    break
            index += 1
        lin_space = np.linspace(1, -1, num=1000)  # from right to left
        index = 0
        for theta_b_candidate in lin_space:
            if neg_gaussian[index] > theta_b_candidate and pos_gaussian[index] > theta_b_candidate:
                r_b = neg_gaussian[index] / pos_gaussian[index]
                if r_b < self.B:
                    break
            index += 1
        self.classifiers[t].theta_a = theta_a_candidate
        self.classifiers[t].theta_b = theta_b_candidate

    def _fetch_best_weak_classifier(self, weighted_patches):
        """ Returns the weak classifier that produces the least error
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