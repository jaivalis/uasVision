class StrongClassifier(object):

    def __init__(self, training_stream, alpha, beta, gamma, layers):
        self.training_stream = training_stream
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classifiers = []  # len(classifiers) = T

        # Parzen window
        total_frame_cont = self.training_stream.size()
        for i in xrange(total_frame_cont / layers):
            patches = self.training_stream.get_random_patches()

            for h in xrange(layers):
                wc = WeakClassifier()

        self.train()

    def classify(self, patch):

        return None

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


class WeakClassifier(object):
    def __init__(self):
        self.theta_a = -1
        self.theta_b = -1

    def classify(self, patch):
        """
        Returned values: -1, 0, +1
        """
        return None

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return False