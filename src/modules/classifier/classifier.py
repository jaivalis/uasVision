from modules.classifier.weakClassifier import WeakClassifier


class StrongClassifier(object):
    def __init__(self, training_stream, feature_holder, alpha, beta, gamma, layers=10, sample_count=20):
        self.training_stream = training_stream
        self.feature_holder = feature_holder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.layers = layers
        self.classifiers = []  # len(classifiers) = T

        #########    Parzen window technique    #########
        # Phase1:  Create all possible weak classifiers #
        for feature in feature_holder.get_features():
            wc = WeakClassifier(feature)
            self.classifiers.append(wc)
        self.classifiers = self.classifiers[-10:len(self.classifiers)]  #TODO remove this, testing
        print "Initialized %d weak classifiers." % (len(self.classifiers))

        # # Phase2: Compute the classifier response for all the training patches
        # samples = 0
        # cll = 0
        # for classifier in self.classifiers:
        #     cll += 1
        #     while True:
        #         for patch in training_stream.get_random_patches():
        #             samples += 1
        #             classifier.train(patch)
        #
        #         if samples > sample_count + 1:
        #             break
        #     print "Done extracting feature #%d: " % cll + str(classifier)
        #     classifier.plot_gaussian()

    # Phase3: Algorithm2: Learning with bootstrapping
        self.learn_with_bootstrapping(sample_count)

    def learn_with_bootstrapping(self, sample_count):
        """
        Algorithm2 Sochman
        ==================
        Outputs the strong classifier with the theta_a, theta_b of the weak classifiers updated
        """
        tr = self.training_stream.extract_training_patches(sample_count)

        # initialize weights
        weighted_patches = {}
        for patch in tr:
            weighted_patches[patch] = 1. / len(tr)

        A = (1 - self.beta) / self.alpha
        B = self.beta / (1 - self.alpha)

        for t in range(self.layers):
            # choose the weak classifier with the minimum error
            h_t = self._find_optimal_weak_classifier(weighted_patches)

            # estimate the likelihood ratio R_t eq. 19

            # find thresholds for chosen classifier

            # throw away training samples for which

            # sample new training data
        return

    def _find_optimal_weak_classifier(self, weighted_patches):
        ret = None
        for wc in self.classifiers:
            patches = weighted_patches.keys()

            wc.train(patches)

        self.classifiers.sort()
        assert ret is not None
        return self.classifiers[0]

    def classify(self, patch):
        """
        Implementation of Algorithm 1 Sochman
        :param patch: Patch to be classified
        :return: +1, -1
        """
        for cl in self.classifiers:
            label = cl.classify(patch)
            if label != 0:
                return label
        if self.H(patch) > self.gamma:
            return +1
        return -1

    def H(self, patch):
        ret = 0
        for wc in self.classifiers:
            ret += wc.h(patch)

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