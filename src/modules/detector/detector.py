import numpy as np
from modules.datastructures.patch import Patch


class Detector(object):
    def __init__(self, classifier, frame_feed):
        self.classifier = classifier
        self.frame_feed = frame_feed

    def classify_all(self):
        frame = self.frame_feed.get_next_frame()
        while frame is not None:
            detections = self.detect(frame, self.frame_feed.frame_id)

            self.output_detections(detections)

            frame = self.frame_feed.get_next_frame()
        print "Classification process done."

    def detect(self, frame, frame_id):
        h, w = np.shape(frame)
        ret = []

        window_sizes = [(24, 24)]
        for x in range(h):  # sliding window
            for y in range(w):
                for window in window_sizes:
                    x_max = x + window[0]
                    y_max = y + window[1]
                    if x_max > h-1 or y_max > w-1:
                        continue
                    crop = frame[y:y_max, x:x_max]
                    p = Patch(crop, frame_id, (x, y, x_max, y_max), None)

                    label = self.classifier.classify(p)

                    if label == +1:
                        ret.append(p)
        return ret

    def output_detections(self, detections):
        """
        :param detections: List containing the detections for a given frame
        """
        for d in detections:
            print d