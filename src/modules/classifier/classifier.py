class StrongClassifier(object):

    def __init__(self, trainingStream, alpha, beta, gamma):
        self.trainingStream = trainingStream
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def classify(self, patch):
        # TODO
        return None

    def train(self, patch):
        # TODO
        self.trainingStream.getT()

        return None


class WeakClassifier(object):
    def __init__(self):
        # TODO
        self.thetaA = -1
        self.thetaB = -1

    def classify(self, patch):
        # TODO
        return None