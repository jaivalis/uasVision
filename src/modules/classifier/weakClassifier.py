import numpy as np
from modules.util.math_f import pdf_gaussian, plot_gaussians
from sys import maxint


class WeakClassifier(object):
    def __init__(self, feature):
        self.feature = feature
        self.error = np.float64(1.)
        # used to calculate the standard deviation
        self.annotated_responses = None
        self.threshold = None
        # The sign of the dominant class left of the threshold
        self.dominant_left = None

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
            return self.conf_left
        else:
            return self.conf_right

    def train(self, weighted_patches):
        """ Given a training set T, finds the threshold that produces the lowest error and updates the error value of
        the classifier.
        :param weighted_patches: Weighted training set
        """
        for p, w in weighted_patches:
            response = self.feature.apply(p.crop)
            true_label = p.label
            # append to the responses
            if self.annotated_responses is None:
                self.annotated_responses = np.array([response, true_label, w])
            else:
                self.annotated_responses = np.vstack((self.annotated_responses, [response, true_label, w]))

        response_values = self.annotated_responses[:, 0]
        pos_response_values = self.annotated_responses[self.annotated_responses[:, 1] == +1]
        neg_response_values = self.annotated_responses[self.annotated_responses[:, 1] == -1]

        # find which class occupies the left and which the right portion of the responses
        median = np.median(response_values) + 0.5

        weight_pos_left = 0
        weight_pos_right = 0
        weight_neg_left = 0
        weight_neg_right = 0
        weight_total_pos = weight_total_neg = 0
        for [response, true_label, w] in self.annotated_responses:
            if true_label == +1:
                weight_total_pos += w
                if response < median:
                    weight_pos_left += w
                else:
                    weight_pos_right += w
            elif true_label == -1:
                weight_total_neg += w
                if response < median:
                    weight_neg_left += w
                else:
                    weight_neg_right += w
        left_pos_concentration = weight_pos_left / weight_total_pos
        left_neg_concentration = weight_neg_left / weight_total_neg
        if left_neg_concentration > left_pos_concentration:
            self.dominant_left = -1
            left = neg_response_values
            right = pos_response_values
        else:
            self.dominant_left = +1
            left = pos_response_values
            right = neg_response_values

        for t in response_values:
            thr = t + .5
            if thr > max(response_values):
                continue

            misclassified_left = left[left[:, 0] > thr]
            misclassified_right = right[right[:, 0] < thr]
            misclassified = sum(misclassified_left[:, 2]) + sum(misclassified_right[:, 2])
            err = misclassified / sum(self.annotated_responses[:, 2])

            if err < self.error:
                self.error = err
                self.threshold = thr

        self._eval_confidences()

        misclassified_left = left[left[:, 0] > self.threshold]
        misclassified_right = right[right[:, 0] < self.threshold]
        self._eval_Z(misclassified=sum(misclassified_left[:, 2]) + sum(misclassified_right[:, 2]))

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

        self.conf_left = .5 * np.log((pos_left + epsilon)/(all_left + epsilon))
        self.conf_right = .5 * np.log((pos_right + epsilon)/(all_right + epsilon))

    def update_alpha(self, weighted_patches):
        """ Used by Adaboost to update the weight of the classifier """
        error = 0
        for patch, w in weighted_patches:
            predicted_label = self.classify(patch)
            e = .5 * patch.label * predicted_label
            error += e
        a = .5 * np.log((1. + error) / (1. - error))
        self.alpha = a

    def plot_gaussian(self):
        """ Plots the mixture of Gaussians split into two classes (+, -) """
        sigma = np.std(self.annotated_responses)
        n = len(self.annotated_responses)
        h = 1.144 * sigma * n ** (-1/5)
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
        if theta_a == -maxint:
            theta_a = -999
        if theta_b == maxint:
            theta_b = +999
        return "Feature: {%s} threshold: %.4f, dominant_left: %d, error: %.2f, theta_a: %.2f, theta_b: %.2f" %\
            (self.feature, self.threshold, self.dominant_left, self.error, theta_a, theta_b)
