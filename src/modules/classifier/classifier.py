from modules.classifier.weakClassifier import WeakClassifier


class StrongClassifier(object):
    def __init__(self, training_stream, feature_holder, alpha, beta, gamma, layers=10, sample_count=20):
        self.training_stream = training_stream
        self.feature_holder = feature_holder
        self.alpha = alpha
        self.beta = beta
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
        self.all_classifiers = self.all_classifiers[-10:len(self.all_classifiers)]  #TODO remove this, testing
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

    def learn_with_bootstrapping(self, sample_count=1000):
        """ Algorithm2 Sochman
        Outputs the strong classifier with the theta_a, theta_b of the weak classifiers updated
        """
        s = self.training_stream.extract_training_patches(sample_count)

        # initialize weights
        weighted_patches = {}
        for patch in s:
            weighted_patches[patch] = 1. / len(s)

        A = (1 - self.beta) / self.alpha
        B = self.beta / (1 - self.alpha)

        for t in range(self.layers+1):
            # choose the weak classifier with the minimum error
            h_t = self._fetch_best_weak_classifier(weighted_patches)
            self.classifiers.append(h_t)    # add it to the strong classifier

            # find decision thresholds for the strong classifier
            self._tune_thresholds(t)

            # throw away training samples that fall in our thresholds

            # sample new training data
        return

    def _tune_thresholds(self, layer):
        """ Update the threshold of the classifier """
        pass

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

    def h_t(self, t, x):
        """ H_t(x) returns the summation of the responses of the first t weak classifiers.
        :param t: Layer count
        :param x: Patch to be classified
        :return: H_t(x)
        """
        ret = 0
        for wc in self.classifiers[0:t]:
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

    def get_r(self, patches, t):
        """

        :param patches:
        :param t:
        :return:
        """
        for t in range(0, self.layers):
            pass

        return self.error

    def train(self, frame_count):
        frames_processed = 0

        while frames_processed < frame_count:
            training = self.training_stream.get_random_patches(10)
            for patch in training:
                self.train_on_patch(patch)
            frames_processed += 1

        self.training_stream.getT()
        return None

    def train_on_patch(self, patch):
        l = patch.label
        pass

    def get_id(self):
        """
        Used by cPickle to serialize/de-serialize
        :return: A unique identifier
        """
        return ""