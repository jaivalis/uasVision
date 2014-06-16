import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity


class StrongClassifier(object):
    def __init__(self, training_stream, feature_holder, alpha, beta, gamma, sample_count):
        self.training_stream = training_stream
        self.feature_holder = feature_holder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classifiers = []  # len(classifiers) = T

        # Parzen window

        # Phase1: Create all possible weak classifiers
        for feature in feature_holder.get_features():
            wc = WeakClassifier(feature)
            self.classifiers.append(wc)
        print "Initialized %d weak classifiers." % (len(self.classifiers))

        # Phase2: Compute the classifier response for all the training patches
        samples = 0
        cll = 0
        for classifier in self.classifiers:
            cll += 1
            while True:
                for patch in training_stream.get_random_patches():
                    samples += 1
                    classifier.train(patch)

                if samples > sample_count + 1:
                    break
            print "Done extracting feature #%d: " % cll + str(classifier)
            classifier.get_gaussian()

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
    def __init__(self, feature):
        self.feature = feature
        self.theta_a = -1
        self.theta_b = -1
        self.w = -1  # weight

        # used to calculate the standard deviation
        self.responses = None  # TODO: split this into positive and negative?

    def classify(self, patch):
        """
        :return: values: -1, 0, +1
        """

        return None

    def train(self, patch):
        response = self.feature.apply(patch.crop)
        label = patch.label

        # TODO switch on label ?
        if self.responses is None:
            self.responses = np.array([response, label])
        else:
            self.responses = np.vstack((self.responses, [response, label]))

    def get_gaussian(self):
        """
        :return: Two Gaussians functions, one per class. Each of those consists of the sum of Gaussians per class.
        """
        # calculate \sigma
        sigma = np.std(self.responses)
        n = len(self.responses)
        h = 1.144 * sigma * n ** (-1/5)

        gaussian = KernelDensity(kernel='gaussian', bandwidth=h).fit(self.responses)

        # plot Gaussian
        X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
        bins = np.linspace(-5, 10, 10)
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        # histogram 1
        # ax[0, 0].hist(self.responses[:, 0], bins=bins, fc='#AAAAFF', normed=True)
        # ax[0, 0].text(-3.5, 0.31, "Histogram")
        #
        # # histogram 2
        # ax[0, 1].hist(self.responses[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
        # ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")
        #
        # # gaussian KDE
        # kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(self.responses)
        # log_dens = kde.score_samples(X_plot)
        # ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        # ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")
        #
        # for axi in ax.ravel():
        #     axi.plot(self.responses[:, 0], np.zeros(self.responses.shape[0]) - 0.01, '+k')
        #     axi.set_xlim(-4, 9)
        #     axi.set_ylim(-0.02, 0.34)
        #
        # for axi in ax[:, 0]:
        #     axi.set_ylabel('Normalized Density')
        #
        # for axi in ax[1, :]:
        #     axi.set_xlabel('x')


    # plt.plot(gaussian)
        # plt.show()
        return gaussian


    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return False

    def __str__(self):
        ret = "Feature: {" + str(self.feature) + "}"
        ret += " theta_A:" + str(self.theta_a) + " theta_B:"
        ret += str(self.theta_b) + " w:" + str(self.w)
        return ret

        # calculation of the true positives:
        #
        # for every data point
        # Gaussian centered on the feature return value with variance 1.144\sigma n^(-1/5)