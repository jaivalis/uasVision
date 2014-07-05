import numpy as np
from modules.util.math_f import pdf_gaussian, plot_gaussians, plot_wc
from sys import maxint
from sklearn.tree import DecisionTreeClassifier
import operator


class MedalonWeakClassifier(object):
    def __init__(self, responses, labels, column):
        self.error = np.float64(1.)
        # used to calculate the standard deviation
        self.annotated_responses = None

        self.stored_responses = np.transpose(np.vstack((responses, labels)))
        self.threshold = None

        # for Adaboost:
        self.column = column
        self.alpha = None
        self.z = None

        ## unused...
        self.theta_a = -maxint
        self.theta_b = maxint
        # The confidence of either side of the decision stump to be class +1
        self.conf_left = None
        self.conf_right = None

        self.tree = None

    def classify(self, sample):
        """ Implementation of the decision stump
        :return: Probability of the patch being of class +1
        """
        ret = self.tree.predict(sample)
        return ret

    def train(self, weights):
        self.annotated_responses = None  # clear the previous training samples
        self.responses = self.stored_responses[:, 0]
        self.labels = self.stored_responses[:, 1]
        self.weights = weights
        # print np.shape(weights), np.shape(self.labels), np.shape(self.responses)
        self.annotated_responses = np.vstack((self.responses, self.labels, weights)).T

        response_values = np.zeros((np.size(self.weights), 1))
        response_values[:, 0] = self.stored_responses[:, 0]
        weights = self.annotated_responses[:, 2]

        self.tree = DecisionTreeClassifier(max_depth=1)
        self.tree.fit(response_values, self.labels, sample_weight=weights)
        pred = self.tree.predict(response_values)
        self.error = np.sum((pred != self.labels) * weights)

        # # https://code.google.com/p/pyclassic/source/browse/trunk/decision_stump.py?spec=svn14&r=14
        # sorted_xyw = np.array(sorted(zip(response_values, labels, weights), key=operator.itemgetter(0)))
        # xsorted = sorted_xyw[:, 0]
        #
        # wy = sorted_xyw[:, 1]*sorted_xyw[:, 2]
        # score_left = np.cumsum(wy)
        # score_right = np.cumsum(wy[::-1])
        # score = -score_left[0:-1:1] + score_right[-1:0:-1]
        # Idec = np.where(xsorted[:-1] < xsorted[1:])[0]
        #
        # if len(Idec)>0:  # determine the boundary
        #     ind, maxscore = max(zip(Idec, abs(score[Idec])), key=operator.itemgetter(1))
        #     self.error = 0.5-0.5*maxscore  # compute weighted error
        #     self.threshold = (xsorted[ind] + xsorted[ind+1]) / 2.  # threshold
        #     self.s = np.sign(score[ind])  # direction of -1 -> 1 change
        # else:  # all identical;
        #     self.error = 0.5
        #     self.threshold = 0
        #     self.s = 1
        # self._eval_confidences()
        # self._eval_Z(self.error)

    # def get_classifications_string(self):  # used for debugging
    #     pos = self.annotated_responses[self.annotated_responses[:, 1] == +1]
    #     neg = self.annotated_responses[self.annotated_responses[:, 1] == -1]
    #
    #     smaller_pos = pos[pos[:, 0] < self.threshold]
    #     smaller_pos_w = np.sum(smaller_pos[:, 2])
    #     bigger_pos = pos[pos[:, 0] > self.threshold]
    #     bigger_pos_w = np.sum(bigger_pos[:, 2])
    #     smaller_neg = neg[neg[:, 0] < self.threshold]
    #     smaller_neg_w = np.sum(smaller_neg[:, 2])
    #     bigger_neg = neg[neg[:, 0] > self.threshold]
    #     bigger_neg_w = np.sum(bigger_neg[:, 2])
    #     return "+{ %.2f < [threshold] < %.2f } | -{ %.2f < [threshold] < %.2f }" \
    #            % (smaller_pos_w, bigger_pos_w, smaller_neg_w, bigger_neg_w)

    def _eval_Z(self, misclassified_weight):
        """ Evaluates self.z, weight normalizing factor used by Adaboost.
        :param misclassified_weight: Weights of misclassified patches
        """
        self.z = 2. * np.sqrt(misclassified_weight * (1. - misclassified_weight))

    def _eval_confidences(self):
        """ Updates the confidences of the classifier in order to avoid calculating them every time """
        epsilon = 1e-5

        positives = self.annotated_responses[self.annotated_responses[:, 1] == +1]
        pos_left = len(positives[positives[:, 0] < self.threshold])
        all_left = len(self.annotated_responses[self.annotated_responses[:, 0] < self.threshold])
        pos_right = len(positives[positives[:, 0] > self.threshold])
        all_right = len(self.annotated_responses[self.annotated_responses[:, 0] > self.threshold])

        self.conf_left = .5 * np.log((pos_left + epsilon) / (all_left + epsilon))
        self.conf_right = .5 * np.log((pos_right + epsilon) / (all_right + epsilon))

    def update_alpha(self):
        """ Used by Adaboost to update the weight of the classifier """
        # r = 0
        # index = 0
        # for w in weighted_patches:
        #     predicted_label = self.classify(self.annotated_responses[index, 0])
        #     r += w * self.annotated_responses[index, 1] * predicted_label
        #     index += 1
        self.alpha = np.log((1. - self.error) / self.error)

    # def plot_gaussian(self):
    #     """ Plots the mixture of Gaussians split into two classes (+, -) """
    #     sigma = np.std(self.annotated_responses)
    #     n = len(self.annotated_responses)
    #     h = 1.144 * sigma * n ** (-1 / 5)
    #     # plot_gaussians(self.annotated_responses, sigma, h)

    def visualize(self, patch):
        crop = patch.crop / 255
        self.feature.visualize(crop)

    def __gt__(self, other):
        return self.error > other.error

    def __eq__(self, other):
        return np.array_equal(self.annotated_responses, other.annotated_responses) and self.threshold == other.threshold

    def __str__(self):
        theta_a = self.theta_a  # for readability
        theta_b = self.theta_b
        alpha = self.alpha
        if alpha is None:
            alpha = -999
        # plot_wc(self)
        return "error: %.4f, alpha: %.2f" % \
               (self.error, alpha)