import numpy as np
from modules.util.math_f import pdf_gaussian, plot_gaussians, plot_wc
from sys import maxint


class WeakClassifier(object):
    def __init__(self, feature):
        self.feature = feature
        self.error = np.float64(1.)
        # used to calculate the standard deviation
        self.annotated_responses = None
        self.threshold = None

        self.theta_a = -maxint
        self.theta_b = maxint
        # The confidence of either side of the decision stump to be class +1
        self.conf_left = None
        self.conf_right = None
        # for Adaboost:
        self.alpha = None
        self.z = None

    def classify(self, patch):
        """ Implementation of the decision stump
        :return: Probability of the patch being of class +1
        """
        response = self.feature.apply(patch.crop)
        if response < self.threshold:
            return -1
        else:
            return +1

    def train(self, weighted_patches):
        self.annotated_responses = None  # clear the previous training samples
        for p, w in weighted_patches:  # Evaluate responses for all patches
            response = self.feature.apply(p.crop)
            true_label = p.label
            # append to the responses
            if self.annotated_responses is None:
                self.annotated_responses = np.array([response, true_label, w])
            else:
                self.annotated_responses = np.vstack((self.annotated_responses, [response, true_label, w]))

        response_values = self.annotated_responses[:, 0]
        pos = self.annotated_responses[self.annotated_responses[:, 1] == +1]
        neg = self.annotated_responses[self.annotated_responses[:, 1] == -1]
        summed_weights = sum(self.annotated_responses[:, 2])

        best_ratio = 0.0
        for t in response_values:
            thr = t + .5

            pos_above_threshold = sum(pos[pos[:, 0] > thr][:, 2]) / summed_weights  # responses
            pos_below_threshold = sum(pos[pos[:, 0] < thr][:, 2]) / summed_weights  # responses

            neg_above_threshold = sum(neg[neg[:, 0] > thr][:, 2]) / summed_weights  # responses
            neg_below_threshold = sum(neg[neg[:, 0] < thr][:, 2]) / summed_weights  # responses

            ratio1 = pos_above_threshold + neg_below_threshold  # sum of weights
            ratio2 = pos_below_threshold + neg_above_threshold  # sum of weights

            if ratio1 > best_ratio or ratio2 > best_ratio:
                best_ratio = max(ratio1, ratio2)
                self.threshold = thr
                self.error = 1 - best_ratio
        self._eval_confidences()
        self._eval_Z(self.error)

    def get_classifications_string(self):  # used for debugging
        pos = self.annotated_responses[self.annotated_responses[:, 1] == +1]
        neg = self.annotated_responses[self.annotated_responses[:, 1] == -1]

        smaller_pos = len(pos[pos[:, 0] < self.threshold])
        bigger_pos = len(pos[pos[:, 0] > self.threshold])
        smaller_neg = len(neg[neg[:, 0] < self.threshold])
        bigger_neg = len(neg[neg[:, 0] > self.threshold])
        return "+{ %d < [threshold] < %d } | -{ %d < [threshold] < %d }" \
               % (smaller_pos, bigger_pos, smaller_neg, bigger_neg)

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

    def update_alpha(self, weighted_patches):
        """ Used by Adaboost to update the weight of the classifier """
        r = 0
        for patch, w in weighted_patches:
            predicted_label = self.classify(patch)
            r += w * patch.label * predicted_label
        self.alpha = .5 * np.log((1. + r) / (1. - r))

    def plot_gaussian(self):
        """ Plots the mixture of Gaussians split into two classes (+, -) """
        sigma = np.std(self.annotated_responses)
        n = len(self.annotated_responses)
        h = 1.144 * sigma * n ** (-1 / 5)
        plot_gaussians(self.annotated_responses, sigma, h)

    def visualize(self, patch):
        crop = patch.crop / 255
        self.feature.visualize(crop)

    def __gt__(self, other):
        return self.error > other.error

    def __eq__(self, other):
        return self.w == other.w and self.error == other.error

    def __str__(self):
        theta_a = self.theta_a  # for readability
        theta_b = self.theta_b
        alpha = self.alpha
        if theta_a == -maxint:
            theta_a = -999
        if theta_b == maxint:
            theta_b = +999
        if alpha is None:
            alpha = -999
        plot_wc(self)
        return "Feature: {%s} threshold: %.1f, error: %.2f, alpha: %.2f, theta_a: %.2f, theta_b: %.2f\n\t\t%s" % \
               (self.feature, self.threshold, self.error, alpha, theta_a, theta_b, self.get_classifications_string())