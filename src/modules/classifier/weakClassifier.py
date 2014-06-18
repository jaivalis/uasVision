import numpy as np
from modules.util.math_f import pdf_gaussian, plot_gaussians


class WeakClassifier(object):
    def __init__(self, feature):
        self.feature = feature
        self.best_thresh_sm = -1
        self.best_thresh_gr = -1
        self.w = -1
        self.h = None
        self.error = None
        # used to calculate the standard deviation
        self.annotated_responses = None  # TODO get rid of this ?
        self.pos_gmm = None  # Gaussian Mixture model for the +1 class
        self.neg_gmm = None  # Gaussian Mixture model for the -1 class

    def classify(self, patch):
        """
        :return: values: -1, 0, +1
        """
        response = self.feature.apply(patch.crop)
        if response < self.theta_a:
            return 1
        elif response > self.theta_b:
            return -1
        return 0

    def train(self, patch):
        """ For a given patch saves the response in self.responses
        """
        response = self.feature.apply(patch.crop)
        label = patch.label

        # append to the responses (used for standard deviation)
        if self.annotated_responses is None:
            self.annotated_responses = np.array([response, label])
        else:
            self.annotated_responses = np.vstack((self.annotated_responses, [response, label]))
        sigma = np.std(self.annotated_responses)

        n = len(self.annotated_responses)
        h = 1.144 * sigma * n ** (-1/5)
        xs = np.linspace(0, 255)

        # gaussian = pdf_gaussian(xs, mu=response, sigma=sigma)
        # if patch.label == 1:
        #     self.pos_gmm = append_gaussian(self.pos_gmm, gaussian, np.linspace(-20, 300))
        # elif patch.label == -1:
        #     self.neg_gmm = append_gaussian(self.pos_gmm, gaussian, np.linspace(-20, 300))

    def train(self, patches):
        """ Find the threshold that produces the lowest error
        """
        for p in patches:
            response = self.feature.apply(p.crop)
            true_label = p.label
            # append to the responses
            if self.annotated_responses is None:
                self.annotated_responses = np.array([response, true_label])
            else:
                self.annotated_responses = np.vstack((self.annotated_responses, [response, true_label]))

        smallest = 0
        greatest = 0
        best_thresh_sm = best_thresh_gr = -1
        response_values = self.annotated_responses[:, 0]
        for t in range(min(response_values)-1, max(response_values)+1):
            thresh_pos_candidate = t
            responses_below_threshold = sum(response_values < thresh_pos_candidate) / len(response_values)
            responses_above_threshold = sum(response_values > thresh_pos_candidate) / len(response_values)
            if responses_below_threshold > smallest:
                smallest = responses_below_threshold
                best_thresh_sm = thresh_pos_candidate
                self.error = 1 - smallest
            if responses_above_threshold > greatest:
                greatest = responses_above_threshold
                best_thresh_gr = thresh_pos_candidate
                self.error = 1 - greatest

            if 1-smallest != 1-greatest:
                print 'not the same'

        assert best_thresh_sm != -1 and best_thresh_gr != -1
        self.best_thresh_sm = best_thresh_sm
        self.best_thresh_gr = best_thresh_gr

    def get_error(self):
        return self.error

    def plot_gaussian(self):
        """
        Plots the mixture of Gaussians split into two classes (+, -)
        """
        sigma = np.std(self.annotated_responses)

        n = len(self.annotated_responses)
        h = 1.144 * sigma * n ** (-1/5)

        plot_gaussians(self.annotated_responses, sigma, h)

    def __gt__(self, other):
        return self.error < other.error

    def __eq__(self, other):
        return self.w == other.w and self.error == other.error

    def __str__(self):
        ret = "Feature: {" + str(self.feature) + "}"
        ret += " theta_A:" + str(self.theta_a) + " theta_B:"
        ret += str(self.theta_b) + " w:" + str(self.w)
        return ret