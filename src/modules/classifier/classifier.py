from modules.classifier.weakClassifier import WeakClassifier
from modules.util.math_f import *
from modules.util.datastructures import *


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
        self.theta_a = None
        self.theta_b = None

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
        training_set_size = 500.
        sample_pool = self.training_stream.extract_training_patches(sample_count)
        # initialize weights
        weighted_patches = {}
        for patch in sample_pool:                              # weight all patches
            weighted_patches[patch] = 1. / len(sample_pool)
        training_data = weighted_patches[0:training_set_size]  # sample 'training_set_size' many samples

        for t in range(self.layers+1):
            # choose the weak classifier with the minimum error
            h_t = self._fetch_best_weak_classifier(training_data)
            self.classifiers.append(h_t)    # add it to the strong classifier

            neg, pos = self._estimate_ratio(weighted_patches, t)
            # find decision thresholds for the strong classifier
            self._tune_thresholds(pos, neg, t)

            # throw away training samples that fall in our thresholds
            sample_pool = self._reweight_and_discard_irrelevant(sample_pool, t)
            
            # sample new training data
            training_data = random_dict_sample(weighted_patches, training_set_size)
        return

    def _reweight_and_discard_irrelevant(self, weighted_sample_pool, t):
        """ Throws away training samples that fall in the predefined thresholds and re weights the patches
        :param weighted_sample_pool: A set of training samples
        :param t: layer number
        :return: The filtered set of training samples
        """
        ret = {}
        wc = self.classifiers[t]
        theta_a = wc.theta_a
        theta_b = wc.theta_b
        for patch, w in weighted_sample_pool.iteritems():
            response = h_t(patch, t)
            if response < theta_a or response > theta_b:
                continue
            else:
                r = wc.classify(patch)

                label = patch.label
                new_weight = w * np.exp(-label * r)

                ret[patch] = new_weight
        # normalize weights
        for patch, w in ret.iteritems():
            pass
        return ret

    def _estimate_ratio(self, weighted_patches, t):
        """ Real Adaboost for feature selection, right ?
        :param weighted_patches:
        :param t: layer number
        :return:
        """
        pos_responses = neg_responses = None
        for p, w in weighted_patches.iteritems():
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
        # compute gaussians for negatve and positive classes
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
        http://personal.ee.surrey.ac.uk/Personal/Z.Kalal/Publications/2007_MSc_thesis.pdf
        :param pos_gaussian: Sum of positive class gaussians
        :param neg_gaussian: Sum of negative class gaussians
        :param t: layer number
        """
        #
        # page 37, 38
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
        for wc in self.classifiers[0:t+1]:
            ret += wc.classify(x)
        return ret

    def classify(self, patch):
        """ Implementation of Algorithm 1 Sochman.
        :param patch: Patch to be classified
        :return: +1, -1
        """
        for t in range(0, self.layers):
            h_t = h_t(t, patch)
            wc = self.classifiers[t]
            if h_t >= wc.theta_b:
                return +1
            if h_t <= wc.theta_a:
                return -1
        if self.h_t(self.layers, patch) > self.gamma:
            return +1
        else:
            return -1

    def get_id(self):
        """
        Used by cPickle to serialize/de-serialize
        :return: A unique identifier
        """
        return ""